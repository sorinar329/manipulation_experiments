"""Segmind event-detection pipeline over a recorded stacking episode.

Replays recording/stacking.csv into the semantic_digital_twin world built by
replay_stacking, ticks a Segmind statechart of detectors over every frame, and
writes the detected event timeline to an interactive plot.

The detectors form the usual three tiers, with the coarse tier gated on the
spatial one so it only runs while its precondition holds:

    atomic   ContactDetector, LossOfContactDetector,
             TranslationDetector, StopTranslationDetector
    spatial  SupportDetector, LossOfSupportDetector
    coarse   PlacingDetector          <- support
             PickUpDetector           <- loss of support

Modelled on segmind/demo/tiago_demo_ready.py (branch safety_ai_ws). That demo
also wires HoldingDetector / LiftingDetector / OpeningDetector, which do not
exist on the branch installed here, so this pipeline stops at pick up and place.
It also gates every tier with start + end conditions; see build_pipeline for why
that wiring silently retires the detectors after their first event.

    python examples/segmind_stacking.py                  # analyse + write the plot
    python examples/segmind_stacking.py --max-frames 200 # quick smoke run
    python examples/segmind_stacking.py --show           # also open the plot
    python examples/segmind_stacking.py --rviz           # also publish markers

Needs the environment with semantic_digital_twin and segmind installed
(~/.virtualenvs/cram-env), and a sourced ROS 2 setup -- segmind's event types
import geometry_msgs even when nothing is published.
"""

import argparse
import time
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Set

from giskardpy.motion_statechart.context import MotionStatechartContext
from krrood.symbolic_math.symbolic_math import trinary_logic_not
from segmind.detectors.atomic_event_detectors_nodes import (
    ContactDetector,
    LossOfContactDetector,
    StopTranslationDetector,
    TranslationDetector,
)
from segmind.detectors.base import SegmindContext
from segmind.detectors.coarse_event_detector_nodes import PickUpDetector, PlacingDetector
from segmind.detectors.spatial_relation_detector_nodes import (
    LossOfSupportDetector,
    SupportDetector,
)
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

from mjcf_replay import REPO_ROOT
from replay_stacking import SCENE

PLOT_FILE = REPO_ROOT / "recording/stacking_events.html"
FPS_LOG_INTERVAL = 100
FLOOR_BODY = "floor"


@dataclass(eq=False, repr=False)
class RestrictedSupporters:
    """Limits a support detector to a fixed set of candidate supporting bodies.

    AbstractDetector.get_relation tests its predicate against every body in
    `world.bodies_with_collision`. For support that means asking whether each
    cube rests on each of the Panda's mesh links, which is both meaningless here
    -- a cube can only come to rest on the floor or on another cube -- and the
    most expensive thing in the loop: is_supported_by builds a bounding-box
    collection per pair. Dropping ~35 candidates to 5 takes the full episode from
    roughly 2.4 to 3.7 frames per second.

    Contact detection is deliberately left unrestricted: the gripper touching a
    cube is exactly the signal we want it to report.
    """

    supporters: List[Body] = field(default_factory=list, kw_only=True)

    def get_relation(self, context, tracked_objects: List[Body], predicate) -> Dict[Body, Set[Body]]:
        related_bodies: Dict[Body, Set[Body]] = {}
        for obj in tracked_objects:
            for body in self.supporters:
                if body is obj:
                    continue
                if predicate(obj, body):
                    related_bodies.setdefault(obj, set()).add(body)
        return related_bodies


@dataclass(eq=False, repr=False)
class CubeSupportDetector(RestrictedSupporters, SupportDetector):
    pass


@dataclass(eq=False, repr=False)
class CubeLossOfSupportDetector(RestrictedSupporters, LossOfSupportDetector):
    pass


def support_candidates(world) -> List[Body]:
    """The bodies a cube can rest on: the floor, plus the other free bodies."""
    return [
        body
        for body in world.bodies_with_collision
        if body.name.name == FLOOR_BODY or type(body.parent_connection) is Connection6DoF
    ]


def build_pipeline(world):
    """Wire the detector statechart and the executor that ticks it.

    The executor is built with `player=None` on purpose. EpisodeSegmenterExecutor
    .compile() starts its player on a background thread, which would race the
    frame loop below for the same generator; driving the player by hand keeps one
    frame and one tick in lockstep.
    """
    executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))

    supporters = support_candidates(world)
    contact = ContactDetector()
    loss_of_contact = LossOfContactDetector()
    translation = TranslationDetector()
    stop_translation = StopTranslationDetector()
    support = CubeSupportDetector(supporters=supporters)
    loss_of_support = CubeLossOfSupportDetector(supporters=supporters)
    placing = PlacingDetector()
    pick_up = PickUpDetector()

    # No detector is given a `tracked_object`, so each falls back to every body
    # held by a Connection6DoF -- exactly the four free-floating cubes. The arm's
    # links stay out of the tracked set but still take part as contact partners.
    statechart = SegmindStatechart().build_statechart([
        contact, loss_of_contact,
        translation, stop_translation,
        support, loss_of_support,
        placing, pick_up,
    ])

    # Conditions can only be wired once a node belongs to a statechart, since
    # observation_variable is minted on registration.
    #
    # The coarse tier is gated with start + *reset*, not the start + end the tiago
    # demo uses. An end_condition moves a node to DONE, which is terminal, so
    # `end = not support` would retire PlacingDetector one tick after the first
    # support burst and it would never fire again. A reset sends the node back to
    # NOT_STARTED, re-arming it every time its precondition returns. Gating is
    # worth it here because both coarse detectors rescan the entire event log on
    # every tick.
    #
    # The spatial tier is left ungated on purpose. A detector's
    # observation_variable is TRUE only on the ticks where it emitted a *new*
    # event, so gating support on contact would sample the support relation once
    # per contact change -- generally before the cube has settled, which loses
    # cube0's pick up and reports the rest seconds late. Support is cheap enough
    # to evaluate every tick once restricted to `supporters` (see
    # RestrictedSupporters).
    placing.start_condition = support.observation_variable
    placing.reset_condition = trinary_logic_not(support.observation_variable)

    pick_up.start_condition = loss_of_support.observation_variable
    pick_up.reset_condition = trinary_logic_not(loss_of_support.observation_variable)

    return executor, statechart


def retime_events_to_sim_clock(events, frame_marks):
    """Rewrite each event's timestamp from analysis wall-clock to episode sim time.

    DetectionEvent.timestamp defaults to datetime.now() at construction, so it
    records when the *detector* ran, not when the thing happened. The detector
    loop runs far slower than real time, which would stretch the plotted timeline
    from the episode's ~39 s out to however long the analysis took.

    `frame_marks` pairs the wall clock at the end of each tick with that frame's
    sim time. An event constructed during a tick therefore falls at or before
    that tick's mark, so the first mark at/after the event dates it. Events from
    the initial tick inside compile() predate every mark and land on frame 0.
    """
    if not frame_marks:
        return
    walls = [wall for wall, _ in frame_marks]
    sim_times = [sim for _, sim in frame_marks]
    epoch = datetime.fromtimestamp(0)
    for event in events:
        index = min(bisect_left(walls, event.timestamp), len(sim_times) - 1)
        event.timestamp = epoch + timedelta(seconds=sim_times[index])


def run(max_frames=None, show=False, rviz=False, plot_file=PLOT_FILE):
    print("=== Segmind stacking pipeline ===\n")

    print(f"Loading episode: {SCENE.csv_file}")
    player = SCENE.build_player()
    world = player.world

    print("Compiling statechart ...")
    executor, statechart = build_pipeline(world)
    segmind_context = executor.context.require_extension(SegmindContext)

    if rviz:
        import rclpy
        from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
            VizMarkerPublisher,
        )
        rclpy.init()
        node = rclpy.create_node("segmind_stacking")
        VizMarkerPublisher(node=node, _world=world).with_tf_publisher()

    executor.compile(statechart)

    print("Replaying episode frames ...")
    frame_marks = []
    start = interval_start = time.perf_counter()
    for frame_data in player.frame_data_generator:
        player.process_objects_data(frame_data)
        executor.tick()
        frame_marks.append((datetime.now(), frame_data.time))

        if len(frame_marks) % FPS_LOG_INTERVAL == 0:
            now = time.perf_counter()
            print(f"  frame {len(frame_marks):>5} -- {FPS_LOG_INTERVAL / (now - interval_start):.1f} fps")
            interval_start = now
        if max_frames is not None and len(frame_marks) >= max_frames:
            break

    elapsed = time.perf_counter() - start
    frames = len(frame_marks)
    print(f"\nProcessed {frames} frames ({frame_marks[-1][1]:.1f} s of sim) "
          f"in {elapsed:.1f} s -- {frames / elapsed:.1f} fps\n")

    events = segmind_context.logger.get_events()
    retime_events_to_sim_clock(events, frame_marks)
    events.sort(key=lambda event: event.timestamp)

    print("=== Detected events ===")
    for event in events:
        print(f"  {event}")
    print(f"\nTotal events: {len(events)}")

    segmind_context.logger.plot_events(show=show, save_path=str(plot_file))
    print(f"Timeline written to {plot_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int,
                        help="stop after this many frames (the full episode is slow)")
    parser.add_argument("--show", action="store_true", help="open the timeline in a browser")
    parser.add_argument("--rviz", action="store_true", help="also publish the world as markers")
    parser.add_argument("--plot", type=str, default=str(PLOT_FILE), help="where to write the timeline")
    args = parser.parse_args()
    run(max_frames=args.max_frames, show=args.show, rviz=args.rviz, plot_file=args.plot)


if __name__ == "__main__":
    main()
