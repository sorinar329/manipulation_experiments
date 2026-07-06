"""Pick-and-place demo in the furnished apartment scene, played live in the
MuJoCo viewer.

The Panda arm (with gripper attached) is mounted directly on top of the
kitchen island, exactly the way the plain-floor demo in pick_and_place.py
mounts it on a tabletop -- only the world offset changes. Reuses the same
differential-IK controller and kinematic-grasp helper.
"""

from pathlib import Path

import mujoco
import numpy as np

from pick_and_place import DiffIKController, KinematicGrasp, build_phases, play, _reset_box

REPO_ROOT = Path(__file__).resolve().parent.parent
PANDA_DIR = REPO_ROOT / "resources/robot/robot_arms/franka_emika_panda"
APARTMENT_XML = REPO_ROOT / "resources/apartment/apartment.xml"

# Kitchen island countertop: a single collision slab spanning roughly
# x in [2.15, 3.15], y in [1.12, 3.18], top surface at z=0.9356 (verified by
# raycasting the compiled apartment model). The arm is mounted near the front
# edge so its 0.5m forward reach lands well inside the counter's footprint.
FRAME_POS = np.array([2.3, 2.15, 0.9356])

BOX_HALF_SIZE = 0.02
BOX_START = FRAME_POS + np.array([0.5, -0.25, BOX_HALF_SIZE])
PLACE_XY = FRAME_POS[:2] + np.array([0.5, 0.25])
HOVER_HEIGHT = FRAME_POS[2] + 0.25

ARM_JOINT_NAMES = [f"/joint{i + 1}" for i in range(7)]
GRIPPER_ACTUATOR_NAME = "//actuator8"
PINCH_SITE_NAME = "/pinch_site"
HAND_BODY_NAME = "//hand"
LEFT_FINGER_BODY_NAME = "//left_finger"
RIGHT_FINGER_BODY_NAME = "//right_finger"


def build_scene():
    """Compose the apartment + arm/gripper mounted on the kitchen island + a pickable box."""
    arm_spec = mujoco.MjSpec.from_file(str(PANDA_DIR / "panda_nohand.xml"))
    hand_spec = mujoco.MjSpec.from_file(str(PANDA_DIR / "hand.xml"))
    arm_spec.attach(hand_spec, site=arm_spec.site("attachment_site"))
    arm_spec.body("/hand").add_site(name="pinch_site", pos=[0, 0, 0.1034], size=[0.005] * 3,
                                     rgba=[1, 0, 1, 0.5])

    spec = mujoco.MjSpec.from_file(str(APARTMENT_XML))
    frame = spec.worldbody.add_frame(pos=FRAME_POS.tolist())
    spec.attach(arm_spec, frame=frame)

    # apartment.xml ships with no floor or lights -- it's meant to be dropped
    # into a scene that supplies both.
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                             size=[5, 5, 0.01], pos=[1.5, 2.2, 0], rgba=[0.3, 0.3, 0.32, 1])
    spec.worldbody.add_light(pos=[1.5, 2.2, 3], dir=[0, 0, -1])
    spec.worldbody.add_light(pos=[FRAME_POS[0], FRAME_POS[1], FRAME_POS[2] + 1.2], dir=[0, 0, -1])

    spec.worldbody.add_body(name="target", pos=FRAME_POS.tolist(), mocap=True).add_site(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01], rgba=[0, 0, 1, 1])

    box = spec.worldbody.add_body(name="box", pos=BOX_START.tolist())
    box.add_freejoint()
    box.add_geom(name="box_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BOX_HALF_SIZE] * 3,
                 rgba=[0.8, 0.2, 0.1, 1], friction=[1, 0.05, 0.001])

    spec.worldbody.add_geom(name="place_pad", type=mujoco.mjtGeom.mjGEOM_BOX,
                             pos=[PLACE_XY[0], PLACE_XY[1], FRAME_POS[2] + 0.001],
                             size=[0.05, 0.05, 0.001], rgba=[0.1, 0.8, 0.1, 0.5],
                             contype=0, conaffinity=0)

    spec.add_exclude(bodyname1="box", bodyname2=HAND_BODY_NAME)
    spec.add_exclude(bodyname1="box", bodyname2=LEFT_FINGER_BODY_NAME)
    spec.add_exclude(bodyname1="box", bodyname2=RIGHT_FINGER_BODY_NAME)

    model = spec.compile()
    # apartment.xml doesn't set an integrator, so attach() keeps its (parent)
    # default of Euler; the arm's position servos need implicitfast to be stable.
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    _reset_box(model, data, BOX_START, box_name="box")
    mujoco.mj_forward(model, data)
    return model, data


def run_demo(num_cycles=None):
    """Launch the MuJoCo viewer and play the pick-and-place loop in the apartment."""
    model, data = build_scene()
    controller = DiffIKController(model, data, pinch_site_name=PINCH_SITE_NAME,
                                   arm_joint_names=ARM_JOINT_NAMES,
                                   gripper_actuator_name=GRIPPER_ACTUATOR_NAME)
    grasp = KinematicGrasp(model, data, controller.pinch_id, box_name="box")
    phases = build_phases(box_start=BOX_START, place_xy=PLACE_XY, hover_height=HOVER_HEIGHT,
                          box_half_size=BOX_HALF_SIZE, surface_z=FRAME_POS[2])

    play(model, data, controller, grasp, phases,
         on_cycle_end=lambda: _reset_box(model, data, BOX_START, box_name="box"),
         num_cycles=num_cycles)


if __name__ == "__main__":
    run_demo()
