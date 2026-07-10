"""Replay a recorded stacking episode in a semantic_digital_twin world.

Drives the arm's joints as well as the free-floating cubes from
recording/stacking.csv. See mjcf_replay for how the scene is exported and how
the replay is verified.

    python examples/stacking.py --record          # produce the recording
    python examples/replay_stacking.py            # replay into RViz
    python examples/replay_stacking.py --check    # headless, verify against MuJoCo

For RViz: add a MarkerArray plugin on /semworld/viz_marker, set its durability to
Transient Local, and set the fixed frame to `world`.

Note this needs the environment that has semantic_digital_twin and segmind
installed (~/.virtualenvs/cram-env), not the one the demos record in.
"""

import csv
import re

import stacking
from mjcf_replay import REPO_ROOT, ReplayScene

CSV_FILE = REPO_ROOT / "recording/stacking_3.csv"
_CUBE_COLUMN = re.compile(r"body\.cube(\d+)\.px$")


def recorded_cube_count(csv_file=CSV_FILE):
    """How many cubes the recording captured, read off its header.

    stacking builds its scene from module globals sized by N_CUBES, which the
    demo's --cubes flag varies. Rather than making the replay guess, count the
    per-cube column blocks so the exported scene always matches the CSV."""
    with open(csv_file, newline="") as f:
        header = next(csv.reader(f))
    return sum(1 for column in header if _CUBE_COLUMN.match(column))


def build_spec():
    stacking.set_n_cubes(recorded_cube_count())
    return stacking.build_spec()


SCENE = ReplayScene(
    build_spec=build_spec,
    csv_file=CSV_FILE,
    scene_file=REPO_ROOT / "resources/generated/stacking_scene.xml",
    meshdir=stacking.PANDA_DIR / "assets",
    node_name="stacking_replay",
)


if __name__ == "__main__":
    SCENE.main(__doc__)
