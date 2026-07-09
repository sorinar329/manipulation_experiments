# manipulation_experiments

MuJoCo manipulation demos for the Franka Emika Panda, plus the machinery to
record an episode, replay it into a `semantic_digital_twin` world, and run
`segmind`'s event-detection pipeline over the replay. Both are subprojects of
[cognitive_robot_abstract_machine][cram].

Everything here lives outside the CRAM workspace on purpose — nothing in
`cognitive_robot_abstract_machine` is modified. Where upstream behaviour got in
the way, the workaround is local and is documented below.

[cram]: https://github.com/cram2/cognitive_robot_abstract_machine

## Layout

| Path | What it is |
| --- | --- |
| `examples/pick_and_place.py` | Differential-IK controller + kinematic grasp; the base the others build on |
| `examples/stacking.py` | Stacks cubes into a tower (`--record` writes `recording/stacking.csv`) |
| `examples/pouring.py` | Pours spheres between cups via a welded grasp |
| `examples/recorder.py` | Per-frame CSV recorder (link poses + joint states), demo-agnostic |
| `examples/mjcf_replay.py` | Exports an `MjSpec` scene to MJCF and replays a CSV into an SDT world |
| `examples/mujoco_csv_player.py` | `MujocoCSVEpisodePlayer`, a segmind `FilePlayer` for our CSV schema |
| `examples/replay_stacking.py`, `examples/replay_pouring.py` | Thin per-demo replay configs |
| `examples/segmind_stacking.py` | Segmind detector statechart over the stacking replay; writes an event timeline |

The demos record in a plain MuJoCo environment. Anything that touches
`semantic_digital_twin` or `segmind` needs `~/.virtualenvs/cram-env` **and** a
sourced ROS 2 setup — segmind's event types import `geometry_msgs` even when
nothing is published.

Verify a replay without a display:

```bash
python examples/replay_stacking.py --check   # cross-checks the SDT world against MuJoCo
```

## Upstream findings

Confirmed against the `main` checkout of the CRAM workspace. Each was hit while
building the replay and detection pipeline; none is worked around by editing
CRAM.

### 1. `MJCFParser` silently drops geoms attached directly to `<worldbody>`

`parse()` iterates `worldbody.bodies` (`adapters/mjcf.py:113`) and never looks at
`worldbody.geoms`. A geom parented straight to the world — the Panda scene's
`floor` plane, stacking's `stack_pad` — is absent from the parsed `World`. No
warning is emitted; the floor just isn't there.

**Workaround:** `mjcf_replay._lift_worldbody_geoms()` rewrites each worldbody
geom into its own jointless child body during export, so the parser sees it.

### 2. A MuJoCo plane parses to a zero-thickness `Box`, which RViz will not draw

`parse_geom` maps `mjGEOM_PLANE` to `Box(scale=Scale(*size[:2], 0.0))`
(`adapters/mjcf.py:276-281`). RViz drops markers with a zero scale component, so
even a plane that *does* reach the world is invisible. Compounding this,
`parse_geom` colours shapes from `geom.rgba` and ignores `material`, so a
textured floor comes through as MuJoCo's default grey.

**Workaround:** `mjcf_replay._plane_to_box()` converts the plane to a real slab
(2 cm thick, top face on z=0) and sets an explicit `rgba`. Note a plane's third
`size` component is the checker grid spacing, not a thickness — reusing it as
one would be a coincidence, so it is dropped.

### 3. Gating a detector with `end_condition` retires it permanently

`TransitionKind.END` moves a node to DONE (`giskardpy/motion_statechart/data_types.py:46`),
which is terminal — only a RESET returns it to NOT_STARTED. The published demo
(`segmind/demo/tiago_demo_ready.py`, branch `safety_ai_ws`) wires every tier as

```python
support_node.start_condition = contact_node.observation_variable
support_node.end_condition   = trinary_logic_not(contact_node.observation_variable)
```

so `SupportDetector` dies on the first tick where no *new* contact appears —
roughly one tick into the episode. Everything gated behind it never fires.
Reproduced here: all four `SupportEvent`s landed at t=0.03 s, with zero
`LossOfSupportEvent` and zero `PickUpEvent` over 1164 frames.

**Workaround:** gate with `start_condition` + `reset_condition` instead. The
node re-arms each time its precondition returns.

### 4. `observation_variable` is an edge, not a level — so it is a poor gate

`AbstractDetector.on_tick` returns TRUE only when it *emitted events this tick*
(`detectors/base.py:148`). Gating the spatial tier on the atomic tier therefore
evaluates the support relation once per contact *change*, typically before the
object has settled. In the stacking episode this lost `cube0`'s pick-up outright
and reported the remaining support transitions several seconds late.

**Workaround:** run `SupportDetector` / `LossOfSupportDetector` on every tick.
To pay for it, `segmind_stacking.RestrictedSupporters` narrows
`get_relation`'s candidate set from every collision body (~35, including each of
the arm's mesh links) to the 5 bodies a cube can actually rest on. Full episode:
2.4 → 3.7 fps.

### 5. `PickUpDetector` and `PlacingDetector` share one dedup set, and collide

Both call `_find_interaction_events`, which keys seen pairs on
`(secondary.tracked_object.id, secondary.with_object.id)` in the shared
`SegmindContext.placing_pairs` (`detectors/coarse_event_detector_nodes.py:77-81`).
Pick-up's secondary is a `LossOfSupportEvent`, placing's is a `SupportEvent` — so
picking an object *off* a surface and later placing it back *onto that same
surface* produce the identical key, and whichever runs second is suppressed.

Observed: `cube0` is lifted off the floor and placed back on the floor as the
tower's base. Its `PickUpEvent` fires and consumes `(cube0, floor)`; its
`PlacingEvent` never fires. Cubes 1–3 are placed on other cubes, so their keys
differ and they survive. This will bite the bottom object of any stack.

**Not worked around.** Only the bottom cube is affected.

### 6. The two motion detectors share a pose window and halve it

`MotionDetector.update_context_and_events` appends `obj.global_pose` to the
shared `segmind_context.latest_poses[obj]` on every tick
(`detectors/atomic_event_detectors_nodes.py:155`). With both `TranslationDetector`
and `StopTranslationDetector` registered, the list gains two poses per tick, so
the nominal `window_size = 4` window spans about two ticks rather than four.
Against `distance_threshold = 0.005` m the result flaps: our 1164-frame episode
produced 199 `TranslationEvent`s and 199 `StopTranslationEvent`s, largely
alternating on consecutive ticks.

Separately, `check_obj_movement` and `check_obj_rotation` both call
`_check_movement_and_trigger_event` (lines 179 and 191), so each detector gets two
chances to emit per object per tick.

**Not worked around.** It only adds noise to the atomic tier.

### 7. Event timestamps are analysis wall-clock, not episode time

`DetectionEvent.timestamp` defaults to `datetime.now()` at construction
(`datastructures/events.py:26`) — it records when the *detector ran*, not when
the event happened. `EventPlotter` normalizes these, so a timeline plotted
straight from a slow detector loop has an x-axis of CPU seconds (316 s here)
instead of episode seconds (38.8 s).

Worse, `AbstractInteractionDetector.shift_threshold` (15 s,
`detectors/coarse_event_detector_nodes.py:27`) compares those wall-clock stamps,
so pick-up/place correlation windows depend on how fast the machine ticks.

**Partial workaround:** `segmind_stacking.retime_events_to_sim_clock()` remaps
timestamps onto frame sim time before printing and plotting. This fixes the
output only — `shift_threshold` still sees wall-clock during the run.

### 8. `EpisodeSegmenterExecutor.compile()` starts the player thread

`compile()` calls `self.player.start()` (`episode_segmenter.py:80`), and
`EpisodePlayer` is a `PropagatingThread`. The published demo then *also* iterates
`file_player.frame_data_generator` on the main thread, so two consumers race the
same generator.

**Workaround:** build the executor with `player=None` and drive the player by
hand, keeping one frame and one tick in lockstep.

### 9. `DataPlayer._run` makes realtime replay quadratic

With `use_realtime=True`, `dt` is the *cumulative* sim time since the episode
started (`players/data_player.py:110`), and it is passed as the **per-frame**
wait target to `_wait_to_maintain_frame_rate` (`data_player.py:119`,
`episode_player.py:164-179`). Frame *k* therefore sleeps for roughly
`k × frame_period`, and total replay time grows with the square of the frame
count.

Note `_run` also sleeps `time_between_frames` unconditionally each frame, before
the `use_realtime` check.

**Workaround:** leave `use_realtime` off.

## API gotchas (not bugs)

- Detector `start_condition` / `reset_condition` can only be assigned **after**
  the node is added to a statechart; `observation_variable` is minted on
  registration and raises `NotInMotionStatechartError` before that.
- `MJCFParser.parse_camera` reads `MjsCamera` fields that only exist on newer
  `mujoco`. `mjcf_replay._strip_cameras()` removes cameras from the exported XML,
  which the replay has no use for anyway.

## Local quirks

- The recorder captures `data.xpos` after `mj_step` without re-running forward
  kinematics, so each row's body poses lag its joint positions by one physics
  step. `replay_*.py --check` reports this skew (≤2.5 mm) rather than treating it
  as an error. It is a property of the recording, not of the parsed world.
