"""Cube-stacking demo for the Franka Panda arm, played live in the MuJoCo viewer.

Builds directly on pick_and_place.py: it reuses that demo's differential-IK
controller and kinematic grasp unchanged, and just drives them through three
pick-and-place laps -- one per cube -- placing each cube a little higher than
the last so the three end up stacked into a tower.

The cubes are carried kinematically (pinned to the gripper) like the box in the
pick-and-place demo, but unlike that demo they physically collide with each
other, so each placed cube rests on the one below and the stack holds up under
normal simulated contact.

    python examples/stacking.py                    # watch it build towers
    python examples/stacking.py --cubes 6          # a taller tower
    python examples/stacking.py --record           # write recording/stacking.csv

See replay_stacking.py for playing a recording back into a semantic_digital_twin
world.
"""

import argparse
import colorsys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Reuse the controller, the kinematic-carry helper, and its address lookup
# straight from the pick-and-place demo -- they need no changes for stacking.
from pick_and_place import DiffIKController, KinematicGrasp, _box_addrs, Phase
from recorder import FrameRecorder, launch_viewer

REPO_ROOT = Path(__file__).resolve().parent.parent
PANDA_DIR = REPO_ROOT / "resources/robot/robot_arms/franka_emika_panda"
RECORDING_FILE = REPO_ROOT / "recording/stacking.csv"

HALF = 0.02                       # cube half-size (edge = 0.04)
N_CUBES = 25                      # crank this up to see how tall a tower it can build

# Cubes recorded by --record. The full 25-cube tower is minutes of simulation and
# a CSV to match, while four levels already exercise every phase of the stack.
RECORD_CUBES = 4

STACK_XY = np.array([0.50, 0.15])

HOVER_HEIGHT = 0.30               # pick-side hover height (cubes start on the floor)
STACK_CLEARANCE = 0.12            # how far above the current tower top to hover
PLACE_GAP = 0.003                 # release each cube this far above its resting spot

GRIP_OPEN = 255.0
GRIP_CLOSED = 90.0

# Stiffer-than-default contacts so cubes rest crisply ON the floor/each other
# instead of visibly sinking in (MuJoCo's default solref=[0.02,1] lets a stacked
# or freshly-dropped cube penetrate ~0.1-0.5 mm, which reads as clipping and a
# mushy, "stuck" landing). These are stable at the 0.002 s timestep.
CONTACT_SOLREF = [0.008, 1.0]
CONTACT_SOLIMP = [0.96, 0.99, 0.001, 0.5, 2.0]

# Collision masks (contype/conaffinity bits) used to make picking physical:
#   bit 0 -> "world" contacts: cube<->cube, cube<->floor  (always on)
#   bit 1 -> "gripper-collidable": cube<->hand/fingers
# Every cube carries both bits; the hand/finger collision geoms carry only bit 1.
# When a cube is grasped we clear bit 1 on it, so it stops colliding with the
# gripper (the kinematic carry would otherwise fight the fingers) while still
# resting on / colliding with the other cubes and floor via bit 0.
WORLD_BIT = 1
HAND_BIT = 2
CUBE_MASK_ALL = WORLD_BIT | HAND_BIT   # 3: collides with world + gripper
CUBE_MASK_HELD = WORLD_BIT             # 1: collides with world only

# Pick-up grid on the floor, kept in front of the base (y < 0) and clear of the
# stack. Laid out in +x columns and -y rows from a near, comfortably reachable
# corner. Spacing is wide enough that the open gripper fits around one cube
# without hitting its neighbours (now that those contacts are real). Very large
# N eventually pushes cubes past the arm's reach -- itself one way the demo
# "stops working".
GRID_COLS = 4
GRID_X0, GRID_Y0 = 0.34, -0.14
GRID_DX, GRID_DY = 0.09, -0.09


def make_starts(n):
    """Floor (x, y) start positions for n cubes on the pick-up grid."""
    return np.array([[GRID_X0 + (i % GRID_COLS) * GRID_DX,
                      GRID_Y0 + (i // GRID_COLS) * GRID_DY] for i in range(n)])


def make_colors(n):
    """A rainbow of rgba colors so tower levels are visually distinguishable."""
    return [list(colorsys.hsv_to_rgb(i / max(1, n), 0.65, 0.9)) + [1.0] for i in range(n)]


CUBE_STARTS = make_starts(N_CUBES)
CUBE_COLORS = make_colors(N_CUBES)


def set_n_cubes(n):
    """Resize the cube set, keeping the derived start grid and colors in step.

    The scene is built from these module globals, so a replay has to resize them
    to whatever its recording captured before rebuilding the spec."""
    global N_CUBES, CUBE_STARTS, CUBE_COLORS
    N_CUBES = n
    CUBE_STARTS = make_starts(n)
    CUBE_COLORS = make_colors(n)


def _stack_center_z(level):
    """Resting center height of the cube at a given stack level (0 = bottom)."""
    return HALF * (2 * level + 1)


def build_spec():
    """Compose arm + gripper + pickable cubes + a place pad into an uncompiled MjSpec.

    Split out of build_scene() so the same scene can be exported to MJCF (see
    replay_stacking) rather than only compiled in place. Note the contact and
    collision-mask tuning build_scene() applies lives on the compiled *model*,
    not on the spec, so it does not survive an export -- the replay is purely
    kinematic and has no use for it."""
    spec = mujoco.MjSpec.from_file(str(PANDA_DIR / "scene.xml"))
    hand_spec = mujoco.MjSpec.from_file(str(PANDA_DIR / "hand.xml"))
    spec.attach(hand_spec, site=spec.site("attachment_site"))

    hand_body = spec.body("/hand")
    hand_body.add_site(name="pinch_site", pos=[0, 0, 0.1034], size=[0.005] * 3, rgba=[1, 0, 1, 0.5])

    for i in range(N_CUBES):
        cube = spec.worldbody.add_body(name=f"cube{i}", pos=[CUBE_STARTS[i][0], CUBE_STARTS[i][1], HALF])
        cube.add_freejoint()
        cube.add_geom(
            name=f"cube{i}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[HALF] * 3,
            rgba=CUBE_COLORS[i],
            friction=[1, 0.05, 0.001],
            contype=CUBE_MASK_ALL,
            conaffinity=CUBE_MASK_ALL,
            solref=CONTACT_SOLREF,
            solimp=CONTACT_SOLIMP,
        )

    spec.worldbody.add_geom(
        name="stack_pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[STACK_XY[0], STACK_XY[1], 0.001],
        size=[0.05, 0.05, 0.001],
        rgba=[0.9, 0.9, 0.2, 0.4],
        contype=0,
        conaffinity=0,
    )

    return spec


def build_scene():
    """Compile the stacking scene and put every cube at its spawn pose."""
    model = build_spec().compile()

    # Put the gripper's collision geoms on HAND_BIT only, so they collide with
    # cubes (which carry that bit) but not with the floor/arm. Visual geoms have
    # contype 0 and are left untouched.
    hand_bodies = {model.body(n).id for n in ("/hand", "/left_finger", "/right_finger")}
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] in hand_bodies and model.geom_contype[gid] != 0:
            model.geom_contype[gid] = HAND_BIT
            model.geom_conaffinity[gid] = HAND_BIT

    # Stiffen the floor's contacts to match the cubes (it comes from scene.xml
    # with MuJoCo's soft defaults), so cubes don't sink into it.
    floor_gid = model.geom("floor").id
    model.geom_solref[floor_gid] = CONTACT_SOLREF
    model.geom_solimp[floor_gid] = CONTACT_SOLIMP

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    reset_cubes(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def reset_cubes(model, data):
    # attach() re-pads the "home" keyframe with zeros for the cubes' new dofs, so
    # every spawn pose has to be restored explicitly after a reset.
    for i in range(N_CUBES):
        qpos_adr, dof_adr = _box_addrs(model, f"cube{i}")
        data.qpos[qpos_adr:qpos_adr + 3] = [CUBE_STARTS[i][0], CUBE_STARTS[i][1], HALF]
        data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
        data.qvel[dof_adr:dof_adr + 6] = 0
        # Restore full collisions (in case a cube was left in the "held" state).
        gid = model.geom(f"cube{i}_geom").id
        model.geom_contype[gid] = CUBE_MASK_ALL
        model.geom_conaffinity[gid] = CUBE_MASK_ALL


class CollisionAwareGrasp(KinematicGrasp):
    """Kinematic carry that also toggles the held cube's gripper-collision bit.

    While a cube is grasped it is carried by overwriting its pose each step, so
    it must not physically fight the fingers -- we clear its HAND_BIT on engage
    (leaving world contacts intact) and restore it on release. Every other cube
    keeps colliding with the gripper, so the arm can't ghost through them."""

    def __init__(self, model, data, pinch_id, box_name):
        super().__init__(model, data, pinch_id, box_name=box_name)
        self.geom_id = model.geom(f"{box_name}_geom").id

    def engage(self):
        super().engage()
        self.model.geom_contype[self.geom_id] = CUBE_MASK_HELD
        self.model.geom_conaffinity[self.geom_id] = CUBE_MASK_HELD

    def release(self):
        super().release()
        self.model.geom_contype[self.geom_id] = CUBE_MASK_ALL
        self.model.geom_conaffinity[self.geom_id] = CUBE_MASK_ALL


def build_cube_phases(level):
    """Pick-and-place phase sequence for the cube going onto stack `level`."""
    start = CUBE_STARTS[level]
    # Hover above the stack has to rise with the tower so the carried cube clears
    # the top on the way in and out.
    hover_z = max(HOVER_HEIGHT, _stack_center_z(level) + STACK_CLEARANCE)
    hover_pick = [start[0], start[1], HOVER_HEIGHT]
    grasp_pos = [start[0], start[1], HALF]
    hover_stack = [STACK_XY[0], STACK_XY[1], hover_z]
    place_pos = [STACK_XY[0], STACK_XY[1], _stack_center_z(level) + PLACE_GAP]
    return [
        Phase("hover_pick", hover_pick, GRIP_OPEN, 1.5),
        Phase("descend_pick", grasp_pos, GRIP_OPEN, 1.2),
        Phase("grasp", grasp_pos, GRIP_CLOSED, 0.6),
        Phase("lift", hover_pick, GRIP_CLOSED, 1.2),
        Phase("transit", hover_stack, GRIP_CLOSED, 2.0),
        Phase("descend_place", place_pos, GRIP_CLOSED, 1.4),
        Phase("release", place_pos, GRIP_OPEN, 0.6),
        Phase("retreat", hover_stack, GRIP_OPEN, 1.2),
    ]


def play_stack(model, data, controller, target_mocap_name="target", num_towers=None,
               recorder=None, headless=False):
    """Drive the viewer, stacking the three cubes into a tower each lap.

    recorder: optional FrameRecorder; when given, one row is captured per frame.
    headless: run without opening a viewer window (for batch recording).
    """
    target_mocap_id = model.body(target_mocap_name).mocapid[0]
    if headless and num_towers is None:
        num_towers = 1
    # One kinematic grasp per cube (each tracks its own free joint).
    grasps = [CollisionAwareGrasp(model, data, controller.pinch_id, box_name=f"cube{i}")
              for i in range(N_CUBES)]

    tower = 0
    with launch_viewer(model, data, headless=headless) as viewer:
        while viewer.is_running() and (num_towers is None or tower < num_towers):
            frame = 0
            for level in range(N_CUBES):
                grasp = grasps[level]
                for phase in build_cube_phases(level):
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
                            recorder.record(tower, frame, phase=f"cube{level}:{phase.name}")
                        frame += 1

                        viewer.sync()
                        if not headless:
                            remaining = model.opt.timestep - (time.time() - step_start)
                            if remaining > 0:
                                time.sleep(remaining)

            tower += 1
            for grasp in grasps:
                grasp.release()
            reset_cubes(model, data)
            mujoco.mj_forward(model, data)


def _gripper_actuator(model):
    """attach() namespaces the hand's actuator (e.g. "/actuator8"); find whichever
    name this compiled model actually uses."""
    for name in ("actuator8", "/actuator8"):
        try:
            model.actuator(name)
            return name
        except KeyError:
            continue
    raise KeyError("gripper actuator not found")


def run_demo(n_cubes=None, num_towers=None, record=False, record_path=None, headless=False, fps=30):
    """Launch the MuJoCo viewer and stack the cubes into a tower on the floor scene.

    record: capture link poses + joint states and write a CSV on exit.
    record_path: explicit CSV path (default recordings/stacking_<timestamp>.csv).
    headless: run without a viewer window (useful for batch recording).
    fps: recording rate; the ~500 Hz physics is decimated to this (None = every step).
    """
    if n_cubes is not None:
        set_n_cubes(n_cubes)
    model, data = build_scene()
    controller = DiffIKController(model, data, gripper_actuator_name=_gripper_actuator(model))
    recorder = FrameRecorder(model, data, experiment="stacking", fps=fps) if record else None
    play_stack(model, data, controller, num_towers=num_towers, recorder=recorder, headless=headless)
    if recorder is not None:
        path = recorder.save(record_path)
        print(f"Recorded {len(recorder)} frames to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack cubes into a tower with the Panda arm.")
    parser.add_argument("--cubes", type=int, help=f"cubes to stack (default: {N_CUBES})")
    parser.add_argument("--towers", type=int, help="stacking laps to run (default: until the viewer closes)")
    parser.add_argument("--record", action="store_true",
                        help=f"capture the episode to {RECORDING_FILE.relative_to(REPO_ROOT)}")
    parser.add_argument("--headless", action="store_true", help="run without a viewer window")
    parser.add_argument("--fps", type=int, default=30, help="recording rate (default: 30)")
    args = parser.parse_args()

    n_cubes, num_towers = args.cubes, args.towers
    if args.record:
        # A recording wants a bounded episode, so default it to a single short lap.
        n_cubes = n_cubes or RECORD_CUBES
        num_towers = num_towers or 1

    run_demo(n_cubes=n_cubes, num_towers=num_towers, record=args.record,
             record_path=RECORDING_FILE if args.record else None,
             headless=args.headless, fps=args.fps)
