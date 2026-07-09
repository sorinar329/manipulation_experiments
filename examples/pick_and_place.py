"""Pick-and-place demo for the Franka Panda arm, played live in the MuJoCo viewer.

The gripper (hand.xml) is attached to the arm (panda_nohand, via scene.xml) at
runtime with MjSpec, since the vendored scene.xml only includes the bare arm.
A damped-least-squares differential IK controller drives the gripper's pinch
site through a scripted sequence of waypoints to pick up a box and place it
at a marked target location.
"""

import time
from collections import namedtuple
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from recorder import FrameRecorder, launch_viewer

PANDA_DIR = Path(__file__).resolve().parent.parent / "resources/robot/robot_arms/franka_emika_panda"

BOX_HALF_SIZE = 0.02
BOX_START = np.array([0.5, -0.25, BOX_HALF_SIZE])
PLACE_XY = np.array([0.5, 0.25])
HOVER_HEIGHT = 0.25

GRIP_OPEN = 255.0
GRIP_CLOSED = 90.0

Phase = namedtuple("Phase", ["name", "target_pos", "grip", "duration"])


def build_scene():
    """Compose arm + gripper + a pickable box into one compiled MuJoCo model."""
    spec = mujoco.MjSpec.from_file(str(PANDA_DIR / "scene.xml"))
    hand_spec = mujoco.MjSpec.from_file(str(PANDA_DIR / "hand.xml"))
    spec.attach(hand_spec, site=spec.site("attachment_site"))

    hand_body = spec.body("/hand")
    hand_body.add_site(name="pinch_site", pos=[0, 0, 0.1034], size=[0.005] * 3, rgba=[1, 0, 1, 0.5])

    box = spec.worldbody.add_body(name="box", pos=BOX_START.tolist())
    box.add_freejoint()
    box.add_geom(
        name="box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[BOX_HALF_SIZE] * 3,
        rgba=[0.8, 0.2, 0.1, 1],
        friction=[1, 0.05, 0.001],
    )

    spec.worldbody.add_geom(
        name="place_pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[PLACE_XY[0], PLACE_XY[1], 0.001],
        size=[0.05, 0.05, 0.001],
        rgba=[0.1, 0.8, 0.1, 0.5],
        contype=0,
        conaffinity=0,
    )

    # The box is carried kinematically while grasped (see KinematicGrasp), so it
    # never needs to physically contact the fingers.
    spec.add_exclude(bodyname1="box", bodyname2="/hand")
    spec.add_exclude(bodyname1="box", bodyname2="/left_finger")
    spec.add_exclude(bodyname1="box", bodyname2="/right_finger")

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    _reset_box(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def _box_addrs(model, box_name="box"):
    body = model.body(box_name)
    jnt = body.jntadr[0]
    return model.jnt_qposadr[jnt], model.jnt_dofadr[jnt]


def _reset_box(model, data, box_start=None, box_name="box"):
    # attach() re-pads the "home" keyframe with zeros for the box's new dofs,
    # so the spawn pose has to be restored explicitly after every reset.
    box_start = BOX_START if box_start is None else box_start
    qpos_adr, dof_adr = _box_addrs(model, box_name)
    data.qpos[qpos_adr:qpos_adr + 3] = box_start
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
    data.qvel[dof_adr:dof_adr + 6] = 0


class DiffIKController:
    """Damped-least-squares differential IK for the gripper's pinch site.

    Holds a fixed downward orientation (captured at startup) and tracks a
    moving position target by solving for an arm joint-angle step each call.
    """

    def __init__(self, model, data, pinch_site_name="pinch_site", arm_joint_names=None,
                 gripper_actuator_name="actuator8"):
        arm_joint_names = arm_joint_names or [f"joint{i + 1}" for i in range(7)]
        self.model = model
        self.data = data
        self.pinch_id = model.site(pinch_site_name).id
        self.arm_act = np.array([model.actuator(n).id for n in arm_joint_names])
        self.arm_qpos_adr = np.array([model.jnt_qposadr[model.joint(n).id] for n in arm_joint_names])
        self.arm_dof_adr = np.array([model.jnt_dofadr[model.joint(n).id] for n in arm_joint_names])
        self.gripper_act = model.actuator(gripper_actuator_name).id
        self.home_quat = np.empty(4)
        mujoco.mju_mat2Quat(self.home_quat, data.site_xmat[self.pinch_id].copy())
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    @property
    def pinch_pos(self):
        return self.data.site_xpos[self.pinch_id].copy()

    def step(self, target_pos, gripper_ctrl, kp_pos=4.0, kp_ori=2.0, damping=0.15, max_dq=0.08):
        data, model, pinch_id = self.data, self.model, self.pinch_id

        pos_err = target_pos - data.site_xpos[pinch_id]

        cur_quat = np.empty(4)
        mujoco.mju_mat2Quat(cur_quat, data.site_xmat[pinch_id].copy())
        if np.dot(cur_quat, self.home_quat) < 0:
            cur_quat = -cur_quat  # keep quaternions in the same hemisphere
        ori_err_local = np.empty(3)
        mujoco.mju_subQuat(ori_err_local, self.home_quat, cur_quat)
        # mju_subQuat's result is expressed in the site's local frame; the
        # Jacobian below is expressed in world frame, so rotate it across.
        ori_err = np.empty(3)
        mujoco.mju_rotVecQuat(ori_err, ori_err_local, cur_quat)

        err = np.concatenate([kp_pos * pos_err, kp_ori * ori_err])

        mujoco.mj_jacSite(model, data, self._jacp, self._jacr, pinch_id)
        J = np.vstack([self._jacp[:, self.arm_dof_adr], self._jacr[:, self.arm_dof_adr]])
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + damping**2 * np.eye(6), err)
        dq = np.clip(dq, -max_dq, max_dq)

        data.ctrl[self.arm_act] = data.qpos[self.arm_qpos_adr] + dq
        data.ctrl[self.gripper_act] = gripper_ctrl


class KinematicGrasp:
    """Rigidly carries the box along with the gripper while engaged.

    Simpler and more reliable for a demo than tuning friction/contact for a
    real force-closure grasp: the box's pose is pinned to the pinch site's
    pose (at a fixed relative offset captured at grasp time) every step.
    """

    def __init__(self, model, data, pinch_id, box_name="box"):
        self.model = model
        self.data = data
        self.pinch_id = pinch_id
        self.qpos_adr, self.dof_adr = _box_addrs(model, box_name)
        self.active = False
        self._offset_pos = None
        self._offset_quat = None

    def engage(self):
        data = self.data
        site_pos = data.site_xpos[self.pinch_id].copy()
        site_quat = np.empty(4)
        mujoco.mju_mat2Quat(site_quat, data.site_xmat[self.pinch_id].copy())
        site_quat_inv = np.empty(4)
        mujoco.mju_negQuat(site_quat_inv, site_quat)

        box_pos = data.qpos[self.qpos_adr:self.qpos_adr + 3].copy()
        box_quat = data.qpos[self.qpos_adr + 3:self.qpos_adr + 7].copy()

        self._offset_pos = np.empty(3)
        mujoco.mju_rotVecQuat(self._offset_pos, box_pos - site_pos, site_quat_inv)
        self._offset_quat = np.empty(4)
        mujoco.mju_mulQuat(self._offset_quat, site_quat_inv, box_quat)
        self.active = True

    def release(self):
        self.active = False

    def update(self):
        if not self.active:
            return
        data = self.data
        site_pos = data.site_xpos[self.pinch_id].copy()
        site_quat = np.empty(4)
        mujoco.mju_mat2Quat(site_quat, data.site_xmat[self.pinch_id].copy())

        new_pos = np.empty(3)
        mujoco.mju_rotVecQuat(new_pos, self._offset_pos, site_quat)
        new_pos += site_pos
        new_quat = np.empty(4)
        mujoco.mju_mulQuat(new_quat, site_quat, self._offset_quat)

        data.qpos[self.qpos_adr:self.qpos_adr + 3] = new_pos
        data.qpos[self.qpos_adr + 3:self.qpos_adr + 7] = new_quat
        data.qvel[self.dof_adr:self.dof_adr + 6] = 0
        mujoco.mj_forward(self.model, data)


def build_phases(box_start=None, place_xy=None, hover_height=None, box_half_size=None, surface_z=0.0):
    box_start = BOX_START if box_start is None else box_start
    place_xy = PLACE_XY if place_xy is None else place_xy
    hover_height = HOVER_HEIGHT if hover_height is None else hover_height
    box_half_size = BOX_HALF_SIZE if box_half_size is None else box_half_size
    rest_z = surface_z + box_half_size

    hover_pick = [box_start[0], box_start[1], hover_height]
    grasp_pos = [box_start[0], box_start[1], rest_z]
    hover_place = [place_xy[0], place_xy[1], hover_height]
    release_pos = [place_xy[0], place_xy[1], rest_z]
    return [
        Phase("home_settle", None, GRIP_OPEN, 0.5),
        Phase("hover_pick", hover_pick, GRIP_OPEN, 1.5),
        Phase("descend_pick", grasp_pos, GRIP_OPEN, 1.2),
        Phase("grasp", grasp_pos, GRIP_CLOSED, 0.6),
        Phase("lift", hover_pick, GRIP_CLOSED, 1.2),
        Phase("transit", hover_place, GRIP_CLOSED, 1.8),
        Phase("descend_place", release_pos, GRIP_CLOSED, 1.2),
        Phase("release", release_pos, GRIP_OPEN, 0.6),
        Phase("retreat", hover_place, GRIP_OPEN, 1.2),
    ]


def play(model, data, controller, grasp, phases, target_mocap_name="target", on_cycle_end=None,
         num_cycles=None, recorder=None, headless=False):
    """Drive the MuJoCo viewer through repeated laps of the phase sequence.

    num_cycles: number of pick-and-place repetitions, or None to loop until
    the viewer window is closed.
    recorder: optional FrameRecorder; when given, one row is captured per frame.
    headless: run without opening a viewer window (for batch recording).
    """
    target_mocap_id = model.body(target_mocap_name).mocapid[0]
    if headless and num_cycles is None:
        num_cycles = 1

    cycle = 0
    with launch_viewer(model, data, headless=headless) as viewer:
        while viewer.is_running() and (num_cycles is None or cycle < num_cycles):
            frame = 0
            for phase in phases:
                start_pos = controller.pinch_pos
                end_pos = np.array(phase.target_pos) if phase.target_pos is not None else start_pos
                n_steps = max(1, int(phase.duration / model.opt.timestep))

                if phase.name == "grasp":
                    grasp.engage()
                elif phase.name == "release":
                    grasp.release()

                for i in range(n_steps):
                    if not viewer.is_running():
                        return
                    step_start = time.time()

                    target = start_pos + (i + 1) / n_steps * (end_pos - start_pos)
                    data.mocap_pos[target_mocap_id] = target

                    controller.step(target, phase.grip)
                    mujoco.mj_step(model, data)
                    grasp.update()

                    if recorder is not None:
                        recorder.record(cycle, frame, phase=phase.name)
                    frame += 1

                    viewer.sync()
                    if not headless:
                        remaining = model.opt.timestep - (time.time() - step_start)
                        if remaining > 0:
                            time.sleep(remaining)

            cycle += 1
            grasp.release()
            if on_cycle_end is not None:
                on_cycle_end()
            mujoco.mj_forward(model, data)


def run_demo(num_cycles=None, record=False, record_path=None, headless=False, fps=30):
    """Launch the MuJoCo viewer and play the pick-and-place loop on the plain floor scene.

    record: capture link poses + joint states and write a CSV on exit.
    record_path: explicit CSV path (default recordings/pick_and_place_<timestamp>.csv).
    headless: run without a viewer window (useful for batch recording).
    fps: recording rate; the ~500 Hz physics is decimated to this (None = every step).
    """
    model, data = build_scene()
    controller = DiffIKController(model, data)
    grasp = KinematicGrasp(model, data, controller.pinch_id)
    phases = build_phases()
    recorder = FrameRecorder(model, data, experiment="pick_and_place", fps=fps) if record else None
    play(model, data, controller, grasp, phases, on_cycle_end=lambda: _reset_box(model, data),
         num_cycles=num_cycles, recorder=recorder, headless=headless)
    if recorder is not None:
        path = recorder.save(record_path)
        print(f"Recorded {len(recorder)} frames to {path}")


if __name__ == "__main__":
    run_demo()
