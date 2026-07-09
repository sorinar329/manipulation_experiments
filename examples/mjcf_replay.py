"""Shared machinery for replaying a recorded episode in a semantic_digital_twin world.

The demos compose their scenes at runtime with MjSpec, so there is no MJCF on
disk for semantic_digital_twin's MJCFParser to read. A ReplayScene exports the
spec to one XML, parses it into a World, and drives that world frame by frame
from the CSV written by recorder.FrameRecorder -- the arm's joints as well as
the free-floating props.

The replaying itself is done by MujocoCSVEpisodePlayer (see mujoco_csv_player),
so the episode can later be fed to segmind's event detectors.

Note this needs the environment that has semantic_digital_twin and segmind
installed (~/.virtualenvs/cram-env), not the one the demos record in.
"""

import argparse
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from semantic_digital_twin.adapters.mjcf import MJCFParser

from mujoco_csv_player import MujocoCSVEpisodePlayer

REPO_ROOT = Path(__file__).resolve().parent.parent
FPS = 30

# MJCFParser turns a MuJoCo plane into a zero-thickness Box, which RViz then
# refuses to draw, so the exported floor is given a real slab thickness instead.
FLOOR_THICKNESS = 0.02
# A plane with a 0 half-extent is infinite in that direction; the export needs a
# finite box, so such a plane is clamped to this much floor around the origin.
PLANE_HALF_EXTENT = 1.0
FLOOR_RGBA = "0.35 0.40 0.45 1"

# `world` is the root frame; `target` is the mocap IK marker, driven by
# mocap_pos rather than by any qpos the recording captures.
SKIP_BODIES = frozenset({"world", "target"})


def _strip_cameras(root: ET.Element) -> None:
    """Drop every camera from the exported XML.

    MJCFParser's parse_camera reads MjsCamera fields that only exist on newer
    mujoco, and the replay has no use for them.
    """
    for parent in root.iter():
        for camera in list(parent.findall("camera")):
            parent.remove(camera)


def _plane_to_box(geom: ET.Element) -> None:
    """Rewrite a plane geom in place as an equivalent thin box.

    A plane's `size` is (x half-extent, y half-extent, grid spacing) -- the
    third number is the checker texture's spacing, not a thickness, so it is
    dropped rather than reused. The box is sunk by half its thickness to keep
    its top face on the plane's surface.
    """
    size = [float(v) for v in (geom.get("size") or "").split()]
    half_x = size[0] if len(size) > 0 and size[0] > 0 else PLANE_HALF_EXTENT
    half_y = size[1] if len(size) > 1 and size[1] > 0 else PLANE_HALF_EXTENT

    pos = [float(v) for v in (geom.get("pos") or "0 0 0").split()]
    pos[2] -= FLOOR_THICKNESS / 2

    geom.set("type", "box")
    geom.set("size", f"{half_x} {half_y} {FLOOR_THICKNESS / 2}")
    geom.set("pos", " ".join(str(v) for v in pos))
    # parse_geom colors a shape from rgba and ignores `material`, so the floor
    # would otherwise come through as MuJoCo's default grey.
    geom.set("rgba", FLOOR_RGBA)


def _lift_worldbody_geoms(root: ET.Element) -> None:
    """Wrap each geom sitting directly on <worldbody> in a static body.

    MJCFParser.parse() only descends into worldbody's *bodies*, so a geom
    attached straight to the world -- the panda scene's floor plane, stacking's
    stack_pad -- never reaches the parsed World. Giving each one its own
    jointless body puts it back, fixed to the root.
    """
    worldbody = root.find("worldbody")
    if worldbody is None:
        return
    for i, geom in enumerate(list(worldbody.findall("geom"))):
        worldbody.remove(geom)
        name = geom.get("name") or f"worldgeom{i}"
        if geom.get("type") == "plane":
            _plane_to_box(geom)
        geom.set("name", name)
        body = ET.SubElement(worldbody, "body", name=name)
        body.append(geom)


@dataclass
class ReplayScene:
    """One demo's replay: its scene spec, its recording, and where to stage the MJCF."""

    build_spec: Callable[[], mujoco.MjSpec]
    csv_file: Path
    scene_file: Path
    meshdir: Path
    node_name: str

    def export_scene(self) -> Path:
        """Compile the demo's scene and write it out as a single MJCF.

        `meshdir` is made absolute so the XML resolves the Panda's meshes from
        the resources tree wherever it is written.
        """
        spec = self.build_spec()
        spec.meshdir = str(self.meshdir) + "/"
        spec.compile()

        root = ET.fromstring(spec.to_xml())
        _strip_cameras(root)
        _lift_worldbody_geoms(root)

        self.scene_file.parent.mkdir(parents=True, exist_ok=True)
        self.scene_file.write_text(ET.tostring(root, encoding="unicode"))
        return self.scene_file

    def build_player(self, **kwargs) -> MujocoCSVEpisodePlayer:
        world = MJCFParser(file_path=str(self.export_scene())).parse()
        return MujocoCSVEpisodePlayer(
            file_path=str(self.csv_file),
            world=world,
            time_between_frames=timedelta(seconds=1 / FPS),
            **kwargs,
        )

    def check(self) -> None:
        """Replay headless and cross-check the parsed world against MuJoCo.

        Only the free bodies and the joints are replayed; every other body's
        pose -- the whole arm -- follows from forward kinematics. Driving a
        MuJoCo model of the same MJCF with the same inputs and comparing the
        resulting body poses therefore tests that MJCFParser rebuilt the
        kinematic structure faithfully.

        Comparing against the recorded `body.*` columns instead would not work:
        the recorder captures data.xpos after mj_step without re-running forward
        kinematics, so each row's body poses lag its joint positions by one
        physics step. That skew is reported here, but it is a property of the
        recording.
        """
        player = self.build_player()
        world = player.world
        model = mujoco.MjModel.from_xml_path(str(self.scene_file))
        data = mujoco.MjData(model)

        worst_body, worst_error, worst_skew, frames = None, 0.0, 0.0, 0
        for frame_data in player.frame_data_generator:
            row = frame_data.objects_data
            player.process_objects_data(frame_data)
            frames += 1

            # Feed MuJoCo exactly what the player was given: recorded world poses
            # for the free bodies, recorded joint positions for the arm.
            for joint in range(model.njnt):
                address = model.jnt_qposadr[joint]
                if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                    name = model.body(model.jnt_bodyid[joint]).name
                    for i, component in enumerate(("px", "py", "pz", "qw", "qx", "qy", "qz")):
                        data.qpos[address + i] = row[f"body.{name}.{component}"]
                else:
                    data.qpos[address] = row[f"joint.{model.joint(joint).name}.qpos.q"]
            mujoco.mj_kinematics(model, data)

            for body in world.bodies:
                name = body.name.name
                if name in SKIP_BODIES:
                    continue
                actual = world.compute_forward_kinematics_np(world.root, body)[:3, 3]
                expected = data.xpos[model.body(name).id]
                error = float(np.linalg.norm(actual - expected))
                if error > worst_error:
                    worst_body, worst_error = name, error
                if f"body.{name}.px" in row:
                    recorded = np.array([row[f"body.{name}.p{a}"] for a in "xyz"])
                    worst_skew = max(worst_skew, float(np.linalg.norm(expected - recorded)))

        print(f"replayed {frames} frames")
        print(f"worst deviation from MuJoCo: {worst_error * 1e6:.3f} um on {worst_body!r}")
        print(f"recorder xpos/qpos skew (one physics step): up to {worst_skew * 1000:.3f} mm")
        assert worst_error < 1e-5, f"{worst_body} deviates {worst_error:.6f} m from MuJoCo"
        print("OK: parsed world reproduces MuJoCo's kinematics")

    def replay(self) -> None:
        import rclpy
        from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
            VizMarkerPublisher,
        )

        # use_realtime is left off: DataPlayer._run passes the cumulative sim time
        # as the per-frame target to _wait_to_maintain_frame_rate, so frame k
        # sleeps for k * time_between_frames and the replay becomes quadratic.
        player = self.build_player()

        rclpy.init()
        node = rclpy.create_node(self.node_name)
        viz_marker_publisher = VizMarkerPublisher(node=node, _world=player.world)
        viz_marker_publisher.with_tf_publisher()

        player.start()
        player.join()

    def main(self, description: str) -> None:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--check", action="store_true",
                            help="Verify the replay against MuJoCo instead of visualizing it.")
        args = parser.parse_args()
        self.check() if args.check else self.replay()
