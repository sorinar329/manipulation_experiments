"""Per-frame state recorder shared by every manipulation experiment.

Drops into any of the demo `play` loops and captures, once per simulated
frame, the world pose of every link (body) and the full state of every joint
(qpos + qvel). At the end of a run it writes a single flat CSV where each row is
one frame -- experiment-agnostic, since it reflects on whatever `model`/`data`
it is handed rather than knowing anything about cubes, cups or the arm.

Column layout (all in one flat table):
    episode, frame, sim_time, phase              -- per-frame metadata
    body.<name>.{px,py,pz,qw,qx,qy,qz}           -- one block per link
    joint.<name>.qpos.<c>                         -- joint configuration
    joint.<name>.qvel.<c>                         -- joint velocity

Joint component labels depend on the joint type (a free joint has 7 qpos / 6
qvel, a hinge or slide 1 / 1, a ball 4 / 3), so a free-jointed cube expands to
position + quaternion columns while an arm hinge expands to a single angle.

Typical use (see each demo's `run_demo`):

    rec = FrameRecorder(model, data, experiment="stacking")
    ...                                   # inside the step loop, per frame:
    rec.record(episode, frame, phase=phase.name)
    ...
    path = rec.save()                     # -> recordings/stacking_<timestamp>.csv
"""

import csv
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

# Per-joint-type labels for the qpos / qvel scalars, so each joint expands into
# self-describing columns instead of anonymous q0/q1/... indices.
_QPOS_LABELS = {
    mujoco.mjtJoint.mjJNT_FREE: ["px", "py", "pz", "qw", "qx", "qy", "qz"],
    mujoco.mjtJoint.mjJNT_BALL: ["qw", "qx", "qy", "qz"],
    mujoco.mjtJoint.mjJNT_SLIDE: ["q"],
    mujoco.mjtJoint.mjJNT_HINGE: ["q"],
}
_QVEL_LABELS = {
    mujoco.mjtJoint.mjJNT_FREE: ["vx", "vy", "vz", "wx", "wy", "wz"],
    mujoco.mjtJoint.mjJNT_BALL: ["wx", "wy", "wz"],
    mujoco.mjtJoint.mjJNT_SLIDE: ["v"],
    mujoco.mjtJoint.mjJNT_HINGE: ["v"],
}

_META_COLUMNS = ["episode", "frame", "sim_time", "phase"]


class FrameRecorder:
    """Accumulates one row of link poses + joint states per simulated frame."""

    def __init__(self, model, data, experiment="experiment", output_dir="recordings", fps=30):
        self.model = model
        self.data = data
        self.experiment = experiment
        self.output_dir = Path(output_dir)
        # Physics runs at 1/timestep (e.g. 500 Hz); `fps` decimates that down to a
        # target output rate by only keeping a frame once this much sim time has
        # elapsed. fps=None keeps every physics step.
        self.fps = fps
        self._interval = 1.0 / fps if fps else 0.0
        self._next_time = 0.0
        self._rows = []
        self._build_layout(model)

    def _build_layout(self, model):
        """Precompute the CSV header and the flat qpos/qvel index arrays used to
        pull each frame's values in one vectorized shot."""
        body_cols = []
        body_ids = []
        # Body 0 is the fixed world frame -- its pose is a constant identity, so
        # skip it and record every real link.
        for b in range(1, model.nbody):
            name = model.body(b).name or f"body{b}"
            body_ids.append(b)
            body_cols += [f"body.{name}.{c}" for c in ("px", "py", "pz", "qw", "qx", "qy", "qz")]

        qpos_cols, qvel_cols = [], []
        qpos_idx, qvel_idx = [], []
        for j in range(model.njnt):
            name = model.joint(j).name or f"joint{j}"
            jtype = model.jnt_type[j]
            qadr = model.jnt_qposadr[j]
            vadr = model.jnt_dofadr[j]
            for k, lbl in enumerate(_QPOS_LABELS[jtype]):
                qpos_cols.append(f"joint.{name}.qpos.{lbl}")
                qpos_idx.append(qadr + k)
            for k, lbl in enumerate(_QVEL_LABELS[jtype]):
                qvel_cols.append(f"joint.{name}.qvel.{lbl}")
                qvel_idx.append(vadr + k)

        self.columns = _META_COLUMNS + body_cols + qpos_cols + qvel_cols
        self._body_ids = np.array(body_ids, dtype=int)
        self._qpos_idx = np.array(qpos_idx, dtype=int)
        self._qvel_idx = np.array(qvel_idx, dtype=int)

    def record(self, episode, frame, phase=""):
        """Capture the current `data` state as one row.

        Call once per physics step; when `fps` is set, calls that fall between
        output frames are skipped so the CSV lands at the target rate. A tiny
        epsilon absorbs float error so e.g. 30 fps at a 0.002 s step doesn't
        drift by a frame."""
        data = self.data
        if self._interval and data.time + 1e-9 < self._next_time:
            return
        self._next_time += self._interval
        body_vals = np.hstack([data.xpos[self._body_ids], data.xquat[self._body_ids]]).ravel()
        qpos_vals = data.qpos[self._qpos_idx]
        qvel_vals = data.qvel[self._qvel_idx]
        self._rows.append(
            [episode, frame, float(data.time), phase, *body_vals, *qpos_vals, *qvel_vals]
        )

    def __len__(self):
        return len(self._rows)

    def save(self, path=None):
        """Write all recorded frames to a CSV and return the path.

        With no `path`, writes recordings/<experiment>_<timestamp>.csv."""
        if path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.output_dir / f"{self.experiment}_{stamp}.csv"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)
            writer.writerows(self._rows)
        return path


class _NullViewer:
    """Stand-in for the MuJoCo passive viewer for headless (windowless) runs, so
    the exact same `play` loop can drive an on-screen demo or a batch recording."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def is_running(self):
        return True

    def sync(self):
        pass


def launch_viewer(model, data, headless=False):
    """Return a passive MuJoCo viewer, or a no-op viewer when `headless`."""
    if headless:
        return _NullViewer()
    import mujoco.viewer
    return mujoco.viewer.launch_passive(model, data)
