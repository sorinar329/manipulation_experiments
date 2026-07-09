from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
)
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Dict

from segmind.players.data_player import FilePlayer, FrameData

logger = logging.getLogger(__name__)

TIME_COLUMN = "sim_time"


@dataclass(eq=False)
class MujocoCSVEpisodePlayer(FilePlayer):
    """
    Plays an episode recorded from a MuJoCo simulation onto a world parsed from
    the same MJCF scene.

    The recording has one row per frame, with a self-describing flat schema::

        episode, frame, sim_time, phase          -- per-frame metadata
        body.<name>.{px,py,pz,qw,qx,qy,qz}       -- world pose of every link
        joint.<name>.qpos.<c>                    -- joint configuration
        joint.<name>.qvel.<c>                    -- joint velocity

    which is unrelated to the ``<obj>:position_i`` layout that
    :class:`~segmind.players.csv_player.CSVEpisodePlayer` reads. It is also
    richer: the robot's joint configuration is recorded, so the arm is
    articulated during replay rather than only the free-floating objects moving.

    The world is expected to come from
    :class:`~semantic_digital_twin.adapters.mjcf.MJCFParser`, which turns every
    MuJoCo free joint into a ``Connection6DoF`` and every hinge/slide into an
    ``ActiveConnection1DOF`` named after the recorded joint.
    """

    data_frames: pd.DataFrame = field(
        default=None, init=False, hash=False, compare=False
    )
    """
    The data frames of the CSV file.
    """

    free_body_connections: Dict[str, Connection6DoF] = field(
        default_factory=dict, init=False, hash=False, compare=False
    )
    """
    The connections of the free-floating bodies, by body name.
    """

    joint_connections: Dict[str, ActiveConnection1DOF] = field(
        default_factory=dict, init=False, hash=False, compare=False
    )
    """
    The 1-DoF connections driven by the recording, by joint name.
    """

    def get_frame_data_generator(self):
        """
        Reads the CSV file and generates the frame data.
        """
        logger.debug(f"Reading CSV file {self.file_path}")
        self.data_frames = pd.read_csv(self.file_path)
        self._bind_to_world()
        for i, (_, objects_data) in enumerate(self.data_frames.iterrows()):
            yield FrameData(
                time=float(objects_data[TIME_COLUMN]),
                objects_data=objects_data.to_dict(),
                frame_idx=i,
            )

    def _bind_to_world(self):
        """
        Matches the recorded columns against the world's connections.

        Only the connections the recording actually covers are driven. Bodies
        held by a fixed connection -- the arm's links, a mocap marker -- are
        left to forward kinematics.
        """
        columns = set(self.data_frames.columns)
        self.free_body_connections = {}
        self.joint_connections = {}
        for connection in self.world.connections:
            if isinstance(connection, Connection6DoF):
                name = connection.child.name.name
                if f"body.{name}.px" in columns:
                    self.free_body_connections[name] = connection
            elif isinstance(connection, ActiveConnection1DOF):
                name = connection.name.name
                if f"joint.{name}.qpos.q" in columns:
                    self.joint_connections[name] = connection
        logger.debug(
            f"Replaying {len(self.free_body_connections)} free bodies and "
            f"{len(self.joint_connections)} joints"
        )

    def get_objects_poses(self, frame_data: FrameData) -> Dict[Body, Pose]:
        """
        Reads the world pose of every free-floating body from the frame.

        :param frame_data: The frame data.
        :return: The poses of the objects.
        """
        objects_data = frame_data.objects_data
        objects_poses: Dict[Body, Pose] = {}
        for name, connection in self.free_body_connections.items():
            pose = Pose.from_xyz_quaternion(
                pos_x=objects_data[f"body.{name}.px"],
                pos_y=objects_data[f"body.{name}.py"],
                pos_z=objects_data[f"body.{name}.pz"],
                quat_w=objects_data[f"body.{name}.qw"],
                quat_x=objects_data[f"body.{name}.qx"],
                quat_y=objects_data[f"body.{name}.qy"],
                quat_z=objects_data[f"body.{name}.qz"],
            )
            if self.position_shift:
                pose.x += self.position_shift.x
                pose.y += self.position_shift.y
                pose.z += self.position_shift.z
            pose.timestamp = frame_data.time
            objects_poses[connection.child] = pose
        return objects_poses

    def get_joint_states(self, frame_data: FrameData) -> Dict[str, float]:
        """
        Reads the configuration of every recorded 1-DoF joint from the frame.

        :param frame_data: The frame data.
        :return: The joint positions, by joint name.
        """
        return {
            name: float(frame_data.objects_data[f"joint.{name}.qpos.q"])
            for name in self.joint_connections
        }

    def process_objects_data(self, frame_data: FrameData):
        """
        Writes one recorded frame into the world state.

        Overrides the base implementation, which assigns each world pose
        straight to ``parent_connection.origin``. Assigning to
        ``Connection6DoF.origin`` writes the connection's *degrees of freedom*,
        while reading it back yields
        ``parent_T_connection @ dofs @ connection_T_child``. MJCFParser folds
        each free body's MJCF spawn position into ``parent_T_connection``, so a
        world pose has to be expressed relative to that frame first -- otherwise
        every object ends up displaced by its own spawn offset.

        The whole frame is written before the world is notified once, so frame
        callbacks never observe a half-applied frame.

        :param frame_data: The frame data.
        """
        for body, pose in self.get_objects_poses(frame_data).items():
            connection = self.free_body_connections[body.name.name]
            world_T_body = HomogeneousTransformationMatrix(data=pose.to_np())
            self._write_free_body_dofs(
                connection,
                connection.parent_T_connection_expression.inverse() @ world_T_body,
            )

        for name, position in self.get_joint_states(frame_data).items():
            self.joint_connections[name].position = position

        self.world.notify_state_change()

    def _write_free_body_dofs(
        self,
        connection: Connection6DoF,
        transformation: HomogeneousTransformationMatrix,
    ):
        """
        Writes a 6-DoF connection's degrees of freedom without notifying the world.

        ``Connection6DoF.origin``'s setter notifies on every assignment, and
        replaying a single frame touches a dozen of them.

        :param connection: The connection to write.
        :param transformation: The transform from the connection's parent frame.
        """
        position = transformation.to_position().to_np()
        orientation = transformation.to_rotation_matrix().to_quaternion().to_np()
        state = self.world.state
        state[connection.x.id].position = position[0]
        state[connection.y.id].position = position[1]
        state[connection.z.id].position = position[2]
        state[connection.qx.id].position = orientation[0]
        state[connection.qy.id].position = orientation[1]
        state[connection.qz.id].position = orientation[2]
        state[connection.qw.id].position = orientation[3]

    def _pause(self): ...

    def _resume(self): ...
