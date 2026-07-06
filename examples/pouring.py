"""Pouring demo for the Franka Panda arm, played live in the MuJoCo viewer.

A source cup filled with small spheres is picked up, carried over an empty
target cup, and tilted so the spheres pour out under gravity into the target.

Built on the same pieces as pick_and_place.py -- arm + hand composed at runtime
with MjSpec and a damped-least-squares differential IK controller -- with two
changes needed to make pouring work:

  * The IK controller tracks a *commanded tilt orientation* rather than always
    holding the gripper pointing straight down; that tilt is what pours the cup.

  * The cup is held with a real welded joint to the hand rather than the
    kinematic "teleport the grasped body each step" trick used by the box demo.
    Teleporting a container does not carry loose dynamic contents with it (they
    are only held by contact, and the cup floor jumps out from under them), and
    a teleported, effectively-infinite-mass cup flings the spheres when it tips.
    A weld keeps the cup dynamic, so the spheres ride along by genuine contact
    and pour out gently when the cup tilts.
"""

import time
from collections import namedtuple
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Reuse the qpos/qvel address lookup verbatim; it is agnostic to which
# free-jointed body it is asked about.
from pick_and_place import _box_addrs

PANDA_DIR = Path(__file__).resolve().parent.parent / "resources/robot/robot_arms/franka_emika_panda"

# --- Cup geometry (a round, open-top cup approximated by a wall of thin box
# --- segments around a thin cylindrical bottom, since MuJoCo has no hollow
# --- cylinder primitive). ---
CUP_RADIUS = 0.035
CUP_HEIGHT = 0.075
WALL_THICK = 0.003
BOTTOM_THICK = 0.004
N_WALL_SEG = 16

SOURCE_XY = np.array([0.45, -0.20])   # cup that starts full, gets picked up
TARGET_XY = np.array([0.52, 0.22])    # empty cup, fixed to the world

# --- Spheres that start inside the source cup. ---
N_PARTICLES = 10
PARTICLE_RADIUS = 0.007

HOVER_HEIGHT = 0.28
# Where the gripper hovers to pour: offset back along -x from the target so that
# once the cup tilts about the pinch site the lip swings out over the target's
# mouth. Tune POUR_DX / POUR_HEIGHT / TILT_ANGLE together if the stream misses.
POUR_DX = -0.06
POUR_HEIGHT = 0.17
TILT_ANGLE = np.deg2rad(120.0)         # how far to tip the cup during the pour
TILT_AXIS = np.array([0.0, 1.0, 0.0])  # world axis to tilt about (tips lip toward +x)

GRIP_OPEN = 255.0
GRIP_CLOSED = 70.0

# name, target_pos (world xyz or None to hold), tilt (rad), grip, duration (s)
Phase = namedtuple("Phase", ["name", "target_pos", "tilt", "grip", "duration"])


def _add_cup(body, rgba):
    """Add bottom + ring-of-walls geoms forming an open-top round cup to `body`."""
    body.add_geom(
        name=f"{body.name}_bottom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[CUP_RADIUS, BOTTOM_THICK / 2, 0],
        pos=[0, 0, BOTTOM_THICK / 2],
        rgba=rgba,
        friction=[1, 0.05, 0.001],
    )
    wall_center_z = BOTTOM_THICK + CUP_HEIGHT / 2
    seg_half_len = np.pi * CUP_RADIUS / N_WALL_SEG * 1.15  # slight overlap so no gaps
    for i in range(N_WALL_SEG):
        ang = 2 * np.pi * i / N_WALL_SEG
        quat = np.empty(4)
        mujoco.mju_axisAngle2Quat(quat, [0, 0, 1], ang)
        body.add_geom(
            name=f"{body.name}_wall{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[WALL_THICK / 2, seg_half_len, CUP_HEIGHT / 2],
            pos=[CUP_RADIUS * np.cos(ang), CUP_RADIUS * np.sin(ang), wall_center_z],
            quat=quat.tolist(),
            rgba=rgba,
            friction=[1, 0.05, 0.001],
        )


def _particle_home(i):
    """Scatter position for sphere i, stacked in a small grid inside the source cup."""
    per_layer = 4
    layer = i // per_layer
    idx = i % per_layer
    ang = 2 * np.pi * idx / per_layer
    r = CUP_RADIUS * 0.45
    return np.array([
        SOURCE_XY[0] + r * np.cos(ang),
        SOURCE_XY[1] + r * np.sin(ang),
        BOTTOM_THICK + PARTICLE_RADIUS + 1.7 * PARTICLE_RADIUS * (2 * layer + 1),
    ])


def build_scene():
    """Compose arm + gripper + two cups + spheres into one compiled MuJoCo model."""
    spec = mujoco.MjSpec.from_file(str(PANDA_DIR / "scene.xml"))
    hand_spec = mujoco.MjSpec.from_file(str(PANDA_DIR / "hand.xml"))
    spec.attach(hand_spec, site=spec.site("attachment_site"))

    hand_body = spec.body("/hand")
    hand_body.add_site(name="pinch_site", pos=[0, 0, 0.1034], size=[0.005] * 3, rgba=[1, 0, 1, 0.5])

    # Source cup: free-jointed so the arm can lift it.
    source = spec.worldbody.add_body(name="source_cup", pos=[SOURCE_XY[0], SOURCE_XY[1], 0])
    source.add_freejoint()
    _add_cup(source, rgba=[0.2, 0.5, 0.85, 1])

    # Target cup: fixed to the world so it stays put while being poured into.
    target = spec.worldbody.add_body(name="target_cup", pos=[TARGET_XY[0], TARGET_XY[1], 0])
    _add_cup(target, rgba=[0.85, 0.55, 0.2, 1])

    # Spheres inside the source cup.
    for i in range(N_PARTICLES):
        p = spec.worldbody.add_body(name=f"particle{i}", pos=_particle_home(i).tolist())
        p.add_freejoint()
        p.add_geom(
            name=f"particle{i}_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[PARTICLE_RADIUS, 0, 0],
            rgba=[0.9, 0.2, 0.2, 1],
            friction=[0.6, 0.02, 0.001],
            condim=4,
        )

    # The cup is held by a weld to the hand (engaged at grasp time), so it never
    # needs to physically contact the fingers -- exclude those contacts.
    spec.add_exclude(bodyname1="source_cup", bodyname2="/hand")
    spec.add_exclude(bodyname1="source_cup", bodyname2="/left_finger")
    spec.add_exclude(bodyname1="source_cup", bodyname2="/right_finger")

    weld = spec.add_equality()
    weld.name = "cup_weld"
    weld.type = mujoco.mjtEq.mjEQ_WELD
    weld.objtype = mujoco.mjtObj.mjOBJ_BODY
    weld.name1 = "/hand"
    weld.name2 = "source_cup"
    weld.active = False
    weld.solref = [0.01, 1.0]                    # stiff, so the cup tracks the hand tightly
    weld.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    reset_props(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def _reset_free_body(model, data, name, pos):
    qpos_adr, dof_adr = _box_addrs(model, name)
    data.qpos[qpos_adr:qpos_adr + 3] = pos
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
    data.qvel[dof_adr:dof_adr + 6] = 0


def reset_props(model, data):
    # attach() re-pads the "home" keyframe with zeros for the new free dofs, so
    # every free body's spawn pose has to be restored after a reset.
    _reset_free_body(model, data, "source_cup", [SOURCE_XY[0], SOURCE_XY[1], 0])
    for i in range(N_PARTICLES):
        _reset_free_body(model, data, f"particle{i}", _particle_home(i))


class WeldGrasp:
    """Holds the source cup by welding it to the hand while engaged.

    The weld's relative pose is captured from the live state at engage time, so
    the cup is pinned wherever it happens to sit relative to the gripper when
    the grasp closes. Because the cup stays a dynamic body, the spheres inside
    ride along by contact and pour out naturally when the cup is tilted.
    """

    def __init__(self, model, data, eq_name="cup_weld", hand_body="/hand", cup_body="source_cup"):
        self.model = model
        self.data = data
        self.eq_id = model.equality(eq_name).id
        self.hand_id = model.body(hand_body).id
        self.cup_id = model.body(cup_body).id

    def engage(self):
        hand_pos = self.data.xpos[self.hand_id].copy()
        hand_quat = self.data.xquat[self.hand_id].copy()
        cup_pos = self.data.xpos[self.cup_id].copy()
        cup_quat = self.data.xquat[self.cup_id].copy()

        hand_quat_inv = np.empty(4)
        mujoco.mju_negQuat(hand_quat_inv, hand_quat)
        rel_pos = np.empty(3)
        mujoco.mju_rotVecQuat(rel_pos, cup_pos - hand_pos, hand_quat_inv)
        rel_quat = np.empty(4)
        mujoco.mju_mulQuat(rel_quat, hand_quat_inv, cup_quat)

        # eq_data (weld, 11): [anchor(3), relpose_pos(3), relpose_quat(4), torquescale(1)]
        ed = self.model.eq_data[self.eq_id]
        ed[0:3] = 0.0
        ed[3:6] = rel_pos
        ed[6:10] = rel_quat
        ed[10] = 1.0
        self.data.eq_active[self.eq_id] = 1

    def release(self):
        self.data.eq_active[self.eq_id] = 0


class TiltIKController:
    """Damped-least-squares differential IK that tracks position AND a commanded
    tilt of the gripper about a fixed world axis.

    Same DLS solve as the pick-and-place controller, but instead of always
    holding the startup ("home") orientation it tracks `home_quat` rotated by a
    per-step tilt angle -- that rotation is what tips the cup to pour it.
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
        # attach() namespaces the hand's actuator (e.g. "/actuator8"); accept
        # either the bare or prefixed name.
        try:
            self.gripper_act = model.actuator(gripper_actuator_name).id
        except KeyError:
            self.gripper_act = model.actuator("/" + gripper_actuator_name).id
        self.home_quat = np.empty(4)
        mujoco.mju_mat2Quat(self.home_quat, data.site_xmat[self.pinch_id].copy())
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    @property
    def pinch_pos(self):
        return self.data.site_xpos[self.pinch_id].copy()

    def target_quat(self, tilt):
        """home orientation rotated by `tilt` radians about TILT_AXIS (world frame)."""
        if tilt == 0.0:
            return self.home_quat.copy()
        dquat = np.empty(4)
        mujoco.mju_axisAngle2Quat(dquat, TILT_AXIS, tilt)
        out = np.empty(4)
        mujoco.mju_mulQuat(out, dquat, self.home_quat)  # world-frame pre-multiply
        return out

    def step(self, target_pos, target_quat, gripper_ctrl,
             kp_pos=4.0, kp_ori=2.0, damping=0.15, max_dq=0.08):
        data, model, pinch_id = self.data, self.model, self.pinch_id

        pos_err = target_pos - data.site_xpos[pinch_id]

        cur_quat = np.empty(4)
        mujoco.mju_mat2Quat(cur_quat, data.site_xmat[pinch_id].copy())
        if np.dot(cur_quat, target_quat) < 0:
            cur_quat = -cur_quat  # keep quaternions in the same hemisphere
        ori_err_local = np.empty(3)
        mujoco.mju_subQuat(ori_err_local, target_quat, cur_quat)
        # mju_subQuat's result is in the site's local frame; rotate it to world
        # to match the world-frame Jacobian below.
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


def build_phases():
    rim_z = BOTTOM_THICK + CUP_HEIGHT  # gripper grabs near the cup's rim
    hover_pick = [SOURCE_XY[0], SOURCE_XY[1], HOVER_HEIGHT]
    grasp_pos = [SOURCE_XY[0], SOURCE_XY[1], rim_z]
    pour_pos = [TARGET_XY[0] + POUR_DX, TARGET_XY[1], POUR_HEIGHT]
    return [
        Phase("home_settle",   None,        0.0,        GRIP_OPEN,   0.5),
        Phase("hover_pick",    hover_pick,  0.0,        GRIP_OPEN,   1.5),
        Phase("descend_pick",  grasp_pos,   0.0,        GRIP_OPEN,   1.2),
        Phase("grasp",         grasp_pos,   0.0,        GRIP_CLOSED, 0.6),
        Phase("lift",          hover_pick,  0.0,        GRIP_CLOSED, 1.2),
        Phase("transit",       pour_pos,    0.0,        GRIP_CLOSED, 2.0),
        Phase("pour",          pour_pos,    TILT_ANGLE, GRIP_CLOSED, 2.0),
        Phase("drain",         pour_pos,    TILT_ANGLE, GRIP_CLOSED, 1.5),
        Phase("upright",       pour_pos,    0.0,        GRIP_CLOSED, 1.5),
        Phase("retreat",       hover_pick,  0.0,        GRIP_CLOSED, 1.8),
    ]


def play(model, data, controller, grasp, phases, target_mocap_name="target",
         on_cycle_end=None, num_cycles=None):
    """Drive the viewer through repeated laps of the phase sequence, interpolating
    both the position target and the tilt angle within each phase."""
    target_mocap_id = model.body(target_mocap_name).mocapid[0]

    cycle = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and (num_cycles is None or cycle < num_cycles):
            prev_tilt = 0.0
            for phase in phases:
                start_pos = controller.pinch_pos
                end_pos = np.array(phase.target_pos) if phase.target_pos is not None else start_pos
                n_steps = max(1, int(phase.duration / model.opt.timestep))

                if phase.name == "grasp":
                    grasp.engage()

                for i in range(n_steps):
                    if not viewer.is_running():
                        return
                    step_start = time.time()

                    frac = (i + 1) / n_steps
                    target = start_pos + frac * (end_pos - start_pos)
                    tilt = prev_tilt + frac * (phase.tilt - prev_tilt)
                    data.mocap_pos[target_mocap_id] = target

                    controller.step(target, controller.target_quat(tilt), phase.grip)
                    mujoco.mj_step(model, data)

                    viewer.sync()
                    remaining = model.opt.timestep - (time.time() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)

                prev_tilt = phase.tilt

            cycle += 1
            grasp.release()
            if on_cycle_end is not None:
                on_cycle_end()
            mujoco.mj_forward(model, data)


def run_demo(num_cycles=None):
    """Launch the MuJoCo viewer and play the pouring loop on the plain floor scene."""
    model, data = build_scene()
    controller = TiltIKController(model, data)
    grasp = WeldGrasp(model, data)
    phases = build_phases()
    play(model, data, controller, grasp, phases,
         on_cycle_end=lambda: reset_props(model, data), num_cycles=num_cycles)


if __name__ == "__main__":
    run_demo()
