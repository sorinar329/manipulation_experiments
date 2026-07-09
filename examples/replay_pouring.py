"""Replay a recorded pouring episode in a semantic_digital_twin world.

Drives the arm's joints as well as the free-floating cups and particles from
recording/pouring.csv. See mjcf_replay for how the scene is exported and how the
replay is verified.

    python examples/replay_pouring.py            # replay into RViz
    python examples/replay_pouring.py --check    # headless, verify against MuJoCo

For RViz: add a MarkerArray plugin on /semworld/viz_marker, set its durability to
Transient Local, and set the fixed frame to `world`.

Note this needs the environment that has semantic_digital_twin and segmind
installed (~/.virtualenvs/cram-env), not the one the demos record in.
"""

import pouring
from mjcf_replay import REPO_ROOT, ReplayScene

SCENE = ReplayScene(
    build_spec=pouring.build_spec,
    csv_file=REPO_ROOT / "recording/pouring.csv",
    scene_file=REPO_ROOT / "resources/generated/pouring_scene.xml",
    meshdir=pouring.PANDA_DIR / "assets",
    node_name="pouring_replay",
)


if __name__ == "__main__":
    SCENE.main(__doc__)
