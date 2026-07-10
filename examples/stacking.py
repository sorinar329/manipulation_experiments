"""Cube-stacking demo for the Franka Panda arm, played live in the MuJoCo viewer.

Builds directly on pick_and_place.py: it reuses that demo's differential-IK
controller unchanged, and just drives it through pick-and-place laps -- one
per cube -- placing each cube a little higher than the last so they stack
into a tower.

Unlike pick_and_place.py's KinematicGrasp, cubes here are held by real
force closure: the gripper actuator (hand.xml's "split" tendon) squeezes
the fingertip pads against the cube, and friction between them (see
CUBE_FRICTION) is what resists gravity and the arm's acceleration while
carrying it. Nothing pins the cube's pose, so a grip with too little of
that friction margin -- because GRASP_POSITION_NOISE happened to land
badly, or CUBE_FRICTION is too low, or GRIP_CLOSED does not squeeze hard
enough -- drops the cube mid-carry, same as a real robot would.

    python examples/stacking.py                    # watch it build towers
    python examples/stacking.py --cubes 6          # a taller tower
    python examples/stacking.py --record           # write recording/stacking_<N>.csv

See replay_stacking.py for playing a recording back into a semantic_digital_twin
world. It reads the fixed path recording/stacking.csv, not the auto-numbered
files this script writes -- point it at whichever run you want by copying or
symlinking that run's file to recording/stacking.csv.
"""

import argparse
import colorsys
import re
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Reuse the controller and its address lookup straight from the pick-and-place
# demo. Unlike that demo, cubes here are held by real friction rather than
# KinematicGrasp, so that class is not needed.
from pick_and_place import DiffIKController, _box_addrs, Phase
from recorder import FrameRecorder, launch_viewer

REPO_ROOT = Path(__file__).resolve().parent.parent
PANDA_DIR = REPO_ROOT / "resources/robot/robot_arms/franka_emika_panda"
RECORDING_DIR = REPO_ROOT / "recording"

HALF = 0.02                       # cube half-size (edge = 0.04)
N_CUBES = 4                      # crank this up to see how tall a tower it can build

# Cubes recorded by --record. The full 25-cube tower is minutes of simulation and
# a CSV to match, while four levels already exercise every phase of the stack.
RECORD_CUBES = 4

STACK_XY = np.array([0.50, 0.15])

HOVER_HEIGHT = 0.30               # pick-side hover height (cubes start on the floor)
STACK_CLEARANCE = 0.12            # how far above the current tower top to hover
PLACE_GAP = 0.003                 # release each cube this far above its resting spot

GRIP_OPEN = 255.0
# The gripper actuator is a *compliant* position servo (spring-like, not a hard
# force cap): squeeze force is the servo's stiffness times how far past the
# cube's actual surface its target sits, not the actuator's 100 N forcerange --
# that cap is nowhere close to binding here. Measured empirically (single cube,
# dead center, carried all the way to the stack pad, zero noise): 62 makes it
# every time, 63 drops it every time -- this compliant grip's holding margin
# collapses over a couple of ctrl units, not gradually. 61 sits just inside the
# reliable side of that edge, so GRASP_POSITION_NOISE below is what pushes some
# picks over it instead of every pick.
GRIP_CLOSED = 61.0

# Sliding/torsional/rolling friction for the cube geoms. This is what actually
# holds a cube up once it is squeezed between the fingertip pads -- there is no
# pose-pinning fallback. Push this down (or GRIP_CLOSED up) to make grips fail
# more often; push it up (or GRIP_CLOSED down, within its safe range above) to
# make them hold more reliably.
CUBE_FRICTION = [0.9, 0.05, 0.001]

# +-this much horizontal (x, y) noise, in meters, on the exact center of each
# cube when computing where to descend and close the gripper -- a stand-in for
# perception/positioning error a real robot would have. Combined with
# GRIP_CLOSED=61 this drops roughly one cube in six across a full 4-cube tower
# (empirically: 34/40 picks landed on the stack pad over 10 towers) -- some
# towers build cleanly, some lose a cube partway up, same as a marginal real
# grasp would.
GRASP_POSITION_NOISE = 0.004

# Stiffer-than-default contacts so cubes rest crisply ON the floor/each other
# instead of visibly sinking in (MuJoCo's default solref=[0.02,1] lets a stacked
# or freshly-dropped cube penetrate ~0.1-0.5 mm, which reads as clipping and a
# mushy, "stuck" landing). These are stable at the 0.002 s timestep.
CONTACT_SOLREF = [0.008, 1.0]
CONTACT_SOLIMP = [0.96, 0.99, 0.001, 0.5, 2.0]

# Collision masks (contype/conaffinity bits):
#   bit 0 -> "world" contacts: cube<->cube, cube<->floor  (always on)
#   bit 1 -> "gripper-collidable": cube<->hand/fingers     (always on)
# Every cube and every hand/finger collision geom carries both bits, so cubes
# always collide with the gripper -- that contact is the grasp, not something
# to be cleared while held. The split still keeps the hand/fingers off the
# floor and arm links, which only carry bit 0.
WORLD_BIT = 1
HAND_BIT = 2
CUBE_MASK_ALL = WORLD_BIT | HAND_BIT   # 3: collides with world + gripper

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
            friction=CUBE_FRICTION,
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


def build_cube_phases(level, rng=np.random):
    """Pick-and-place phase sequence for the cube going onto stack `level`.

    The descend/grasp target is offset from the cube's true center by up to
    GRASP_POSITION_NOISE, so the gripper does not always close on dead center --
    see the module docstring for why that is what lets a grip fail.
    """
    start = CUBE_STARTS[level]
    grasp_xy = start + rng.uniform(-GRASP_POSITION_NOISE, GRASP_POSITION_NOISE, size=2)
    # Hover above the stack has to rise with the tower so the carried cube clears
    # the top on the way in and out.
    hover_z = max(HOVER_HEIGHT, _stack_center_z(level) + STACK_CLEARANCE)
    hover_pick = [start[0], start[1], HOVER_HEIGHT]
    grasp_pos = [grasp_xy[0], grasp_xy[1], HALF]
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
               make_recorder=None, headless=False):
    """Drive the viewer, stacking the three cubes into a tower each lap.

    make_recorder: optional zero-arg callable returning a fresh FrameRecorder.
        Called once per tower lap and saved to its own file right after that
        lap finishes, so recording several laps in one run still produces one
        CSV per lap instead of combining them. None to skip recording.
    headless: run without opening a viewer window (for batch recording).
    """
    target_mocap_id = model.body(target_mocap_name).mocapid[0]
    if headless and num_towers is None:
        num_towers = 1

    tower = 0
    with launch_viewer(model, data, headless=headless) as viewer:
        while viewer.is_running() and (num_towers is None or tower < num_towers):
            recorder = make_recorder() if make_recorder is not None else None
            frame = 0
            for level in range(N_CUBES):
                for phase in build_cube_phases(level):
                    start_pos = controller.pinch_pos
                    end_pos = np.array(phase.target_pos) if phase.target_pos is not None else start_pos
                    n_steps = max(1, int(phase.duration / model.opt.timestep))

                    for i in range(n_steps):
                        if not viewer.is_running():
                            return
                        step_start = time.time()

                        target = start_pos + (i + 1) / n_steps * (end_pos - start_pos)
                        data.mocap_pos[target_mocap_id] = target

                        controller.step(target, phase.grip)
                        mujoco.mj_step(model, data)

                        if recorder is not None:
                            recorder.record(tower, frame, phase=f"cube{level}:{phase.name}")
                        frame += 1

                        viewer.sync()
                        if not headless:
                            remaining = model.opt.timestep - (time.time() - step_start)
                            if remaining > 0:
                                time.sleep(remaining)

            if recorder is not None:
                path = recorder.save(_next_recording_path())
                print(f"Recorded {len(recorder)} frames to {path}")

            tower += 1
            reset_cubes(model, data)
            # A fresh FrameRecorder's fps-decimation clock starts at sim time 0,
            # but data.time keeps climbing across laps unless reset here -- without
            # this, the next lap's recorder thinks it is miles behind schedule and
            # records every physics step until data.time catches back up to it.
            data.time = 0.0
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


def _next_recording_path():
    """recording/stacking_<N>.csv, N one past the highest existing index.

    Every run gets its own file this way instead of each execution silently
    overwriting the last one's recording/stacking.csv."""
    RECORDING_DIR.mkdir(parents=True, exist_ok=True)
    existing = [
        int(match.group(1))
        for path in RECORDING_DIR.glob("stacking_*.csv")
        if (match := re.fullmatch(r"stacking_(\d+)\.csv", path.name))
    ]
    next_index = max(existing, default=0) + 1
    return RECORDING_DIR / f"stacking_{next_index}.csv"


def run_demo(n_cubes=None, num_towers=None, record=True, headless=False, fps=30):
    """Launch the MuJoCo viewer and stack the cubes into a tower on the floor scene.

    record: capture link poses + joint states and write a CSV per tower lap
        (see play_stack's make_recorder), auto-numbered via _next_recording_path().
    headless: run without a viewer window (useful for batch recording).
    fps: recording rate; the ~500 Hz physics is decimated to this (None = every step).
    """
    if n_cubes is not None:
        set_n_cubes(n_cubes)
    model, data = build_scene()
    controller = DiffIKController(model, data, gripper_actuator_name=_gripper_actuator(model))

    def make_recorder():
        return FrameRecorder(model, data, experiment="stacking", fps=fps)

    play_stack(model, data, controller, num_towers=num_towers,
               make_recorder=make_recorder if record else None, headless=headless)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack cubes into a tower with the Panda arm.")
    parser.add_argument("--cubes", type=int, help=f"cubes to stack (default: {N_CUBES})")
    parser.add_argument("--towers", type=int, help="stacking laps to run (default: until the viewer closes)")
    parser.add_argument("--record", action="store_true",
                        help="capture the episode to recording/stacking_<N>.csv (auto-numbered)")
    parser.add_argument("--headless", action="store_true", help="run without a viewer window")
    parser.add_argument("--fps", type=int, default=30, help="recording rate (default: 30)")
    args = parser.parse_args()

    n_cubes, num_towers = args.cubes, args.towers
    if args.record:
        # A recording wants a bounded episode, so default it to a single short lap.
        n_cubes = n_cubes or RECORD_CUBES
        num_towers = num_towers or 1

    run_demo(n_cubes=n_cubes, num_towers=num_towers, record=True,
             headless=args.headless, fps=args.fps)
