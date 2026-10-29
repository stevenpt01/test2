# Unitree G1 — From Instruction to Grounded Action

**32-slide technical presentation — presenter notes**

This guide accompanies the editable PowerPoint deck. Model checkpoints, robot interfaces, and outcomes are proposed architecture components, not validated real-G1 benchmark results.

## 01. FROM INSTRUCTION

This is a proposed architecture and implementation design, not evidence of verified real-G1 task success. Focus on separating semantic task planning, active perception, navigation goals, and VLA pick/place execution. Two examples are studied: move a red cup to a kitchen counter; dispose of a user-authorized disposable cup in a trash bin.

## 02. The difficult part happens before grasping

A raw detector does not translate an instruction into where to look. The high-level planner must issue a grounded search specification; a separate active perception agent resolves that specification using memory, viewpoints, and the existing 3D detector. Only after the object is grounded may the task planner request navigation and manipulation.

## 03. Two tasks, one reusable execution contract

The architecture must not implicitly throw away reusable household property just because the task says clean the table. In the second case the target is a disposable cup explicitly designated for trash, or confirmed by an authorized user/approved disposal policy. The same seven-stage execution chain is reused.

## 04. Avoid asking one VLM to simultaneously reason, search, walk, grasp, and verify.

High-level planner typically uses RynnBrain-Plan-8B or a deterministic BT. Active perception can be a deterministic state machine. Semantic world is a separate persistent service. Navigation uses Nav2-style geometric planning plus a G1-compatible locomotion adapter. VLA manipulation uses a task-trained GR00T N1.7 policy with compatible embodiment. Safety is independent and cannot be bypassed by the VLM.

## 05. One shared world, three agent loops

The architecture intentionally separates three loops: (1) mission planning chooses the next semantic skill, (2) active perception searches for ungrounded objects, and (3) skill execution and verification update the world model. Shared world receives 3D detections, costmap, reconstruction references and robot localization, though costmap remains geometrically authoritative in navigation. Nav2 does not directly control G1 balance; the locomotion interface must be validated.

## 06. Seven steps — four distinct motion skills

This sequence is the core change requested. Stage 3 navigation ends at an approach pose with confirmed arrival; Stage 4 is independently invoked VLA pick. Stage 5 navigation transports the verified object while attached, Stage 6 is a separately invoked VLA place. Do not equate action goal acceptance with physical task completion. Observe→act→verify occurs at every stage.

## 07. Parse language into typed mission constraints

The structured mission should include target attributes, room/surface relation, explicit destination, permitted skills and task policies. Case A move cup to kitchen counter; Case B clear table by disposing of a specifically user-authorized disposable cup. Validate schema and semantic constraints before running any action.

## 08. Query the world before moving the robot

The semantic registry should track last_seen, pose covariance, source, observation count, association probability and object lifecycle. Fresh+confirmed still needs near-task verification. Stale triggers verification at known location, no match triggers search. The semantics of NOT_FOUND vs NOT_OBSERVED should be explicit.

## 09. A bounded search, not an unlimited wander

Implement ActivePerception as a ROS 2 action: goal target query, budget and authorized search region; feedback current viewpoint, coverage, candidate IDs, reason; terminal results FOUND / NOT_OBSERVED / UNSAFE / TIMEOUT / CANCELED. Search uses memory before motion. A viewpoint is an observation pose, different from a manipulation approach pose. Robots never move to unvalidated waypoints.

## 10. Plan observation viewpoints using the 3D scene

The illustrative 2D map stands for a 3D geometry-aware viewpoint planner. Candidate points around table are sampled in traversable free space; FOV and visibility estimated by raycasting or coverage; cost from Nav2; safety from torso and limb clearance. Reject camera viewpoints in inaccessible or unstable areas. Choose in order of expected information gain balanced with travel and risk, then stop after the budget.

## 11. A detection is not yet the requested object

Input may be vision_msgs/Detection3DArray from an existing 3D detector with synchronized image. The detector may not supply color; add an RGB crop classifier or projected point colors. Use 3D support geometry to test on_top_of(table_03) with tolerances; do not conflate proximity with support. Ground an object only if constraints and confidence pass.

## 12. A lightweight registry, not another SLAM pipeline

Object registry can initially be SQLite with objects, observations, relations and events; schema needs stable IDs, pose uncertainty, last_seen, deleted/stale flags, transform source, and event version. Do not duplicate full mesh. Store geometry_ref and map_version, let geometric service resolve voxels or mesh. Optional Spark-DSG scene graph later.

## 13. One cup ID across grasp, transport, and placement

Object ID must not change when cup relocates. New observations are gated by lifecycle and tracking: ATTACHED is derived from hand-to-object transform and not re-associated as a new scene object. After VLA place reports release, set PLACE_PENDING, then verify cup in destination. The original Semantic-nav-amr only label+distance merges and cannot handle mobile object lifecycle robustly.

## 14. Pick-up viewpoint ≠ manipulation approach pose

There are two distinct geometric problems: view selection for observing an object, and manipulation base pose selection for safely reaching it. Use arm IK/manipulability and contact feasibility, 3D body clearance, no-go zones, costmap path, balance and standing orientation. The approach goal is a candidate PoseStamped that can be validated and passed to Nav2; never simply use the detected object center or a fixed 0.7m offset.

## 15. Navigate to the cup — no manipulation yet

Nav2 NavigateToPose emits goal accepted, feedback and final result. ROS2 action semantics distinguish goal acceptance from physical completion. A separate G1 locomotion bridge must transform Nav2 path/pose commands into validated whole-body locomotion and ensure balance. Nav2 is NOT automatically a G1 balance controller. The handoff to VLA happens only after locomotion reports a stable terminal state.

## 16. VLA PICK — a separate skill from navigation

GR00T N1.7 G1 SONIC interface is documented as a post-training embodiment; you need compatible fine-tuned checkpoint, task demonstrations and tested observation / action mappings. Do not treat unadapted base checkpoint as guaranteed pick skill. VLA command is pick(cup_17), not navigation. Verify grasp using camera/hand sensors and target motion, with timeout. Only then transition object to ATTACHED.

## 17. Navigate to the destination while carrying

Now navigation starts a second, independent ROS action targeting the destination approach pose. Object stays attached to robot link, and original support relationship is removed. Carrying changes body envelope, dynamic constraints, safe gait/arm policy. If obstruction occurs, local route recovery may happen without task-level replan; if cup slips or collision risk increases, stop and escalate. G1 navigation interface still requires validated locomotion/WBC.

## 18. VLA PLACE — separate from arriving at the kitchen

G1 finishes navigation to kitchen counter before switching to manipulation policy. Goal should contain object ID, destination geometry, target placement zone, contact constraints, camera frame, and timeout. Policy places then releases; independent perception verifies object support, no slipping/drop, hand detachment. Registry transitions from PLACE_PENDING to AVAILABLE at new location only after evidence.

## 19. Verify the goal, update memory, then report success

An episode is complete if goal predicates satisfy the requested state, with fresh observations and no unresolved safety issue. Success metrics include actual object placement, no accidental disposal, stable final pose, and correct identity. Report pending/incomplete if camera cannot inspect final target, even if VLA returns done.

## 20. Red cup → kitchen counter: complete action trace

Sample trace is design illustration, not measured robot execution. Benchmark should log event IDs, pose transforms, object state, retry count, action results, last observation, costmap version and reference geometry for each step. Validate each transition independently.

## 21. “Clean the table” can mean object removal—not wiping

If the user explicitly says put this cup in the trash, the planner can respect the instruction subject to robot safety and any household-level policy. If the instruction is only clean the table, do not infer every cup is trash. Classify disposable or request approval. The end-to-end task is object removal, not wiping, disinfection, dirt estimation, or autonomous waste sorting beyond supported skills.

## 22. Table cleanup still uses navigation and VLA as separate actions

The trash example is deliberately handled as seven independent high-level skills. Active perception may need to find the bin as well as the cup, or use existing bin registry and fresh verification. The VLA PLACE skill should be trained for bin aperture and release, not just arbitrary counter placement. The robot must confirm disposal success with evidence, and avoid placing when the bin is full.

## 23. “Clean” is not permission to discard every object

Treat the instruction clean the table as underspecified. A reusable mug should not be thrown into a bin on an inferred policy. Require explicit disposal target or stored household rule; the user example of throwing a cup in trash should be treated as disposable or specifically authorized. In test scenarios, tag object truth labels; measure false disposal rate and uncertainty abstention.

## 24. Same interface scales to multiple objects and destinations

Long-horizon cleaning means object inventory, destination assignment, ordering, repeated independent skill execution. Do not equate object relocation with full cleaning tasks such as wiping; that requires a separate trained tool-use skill. Task graph can optimize carrying, accessibility, priority, and retry budget. Agents should maintain goal predicates for each object.

## 25. Failures should escalate to the smallest responsible agent

For unobservable cup, search next safe viewpoint. For ambiguous cup, do not choose one at random. For navigation blocked, Nav2 local recovery can replan. For failed grasp, VLA local bounded retry only after perception refresh. For drop, release attachment and re-ground identity. Bin full should trigger mission replan. Every recovery budget and action status must be logged.

## 26. Three ROS 2 communication types, three purposes

ROS 2 actions fit long-running active search, navigation and manipulation because they have Goal/Feedback/Result and cancellation. FindObjects is a fast world query service. Detections, tracked objects, TF2, costmap should be streamed. These names are proposals, not upstream published interface names. Standardize versions, map frame, timestamps, IDs, and terminal result codes.

## 27. Use typed requests, IDs, and terminal results

Create .action files with explicit goal, feedback and result. Keep an object_id as a stable identifier and references into world state instead of repeating large geometry. Include `mission_id`, `event_id` and `schema_version` for tracking. Validate allowed skills, map frame, freshness, safety and authorization. Avoid LLM output directly becoming low-level torques or joint commands.

## 28. Small deterministic managers around powerful models

Core logic is intentionally deterministic and testable; high-level model can propose a structured mission but it is not the authority on robot state. Active perception relies on safe views and budget. Real implementation needs ROS 2 action clients and server state handling, asynchronous cancellation, telemetry, persistence transaction boundaries, safety stops, and retry limits.

## 29. All runtime decisions can remain on your own machines

This is a logical deployment partition, not a promise that each model fits a specific GPU. Profile VLM and GR00T GPU requirements, camera bandwidth, ROS2 DDS latency and network reliability. Original RynnBrain weights are available, but benchmark against fixed deterministic planner too. Codex subscription may assist software development but must not be used as the robot runtime planner. GR00T N1.7 G1 SONIC requires fine-tuned checkpoint as documented.

## 30. Benchmark perception, navigation, manipulation, and safety separately

Proposed benchmark: first use oracle object and skill mocks to isolate planner, then closed loop physics with G1 and fixed low-level policies, then supervised hardware. Tasks: find cup, occluded cup, navigate pickup, pick, navigate while holding, place, room tidying, discard authorized cup. Stratify unseen layouts, illumination, dynamic obstacles, position noise, failed grasp, full bin. Report separate metrics with confidence intervals.

## 31. Build in acceptance-gated milestones

Milestones are logical rather than hard schedule promises: M0 contract design, M1 registry and persistence, M2 active perception using existing detector, M3 navigation using costmap and validated G1 controller, M4 fine-tuned GR00T N1.7 and state machine, M5 integration and benchmark. Codex can implement a phase only after approved design. Physical G1 motion always requires human authorization and safety review.

## 32. Upstream capabilities vs proposed G1 additions

Primary source URLs: https://github.com/Gukdoli/Semantic-nav-amr ; https://docs.nav2.org ; https://design.ros2.org/articles/actions.html ; https://github.com/ros-perception/vision_msgs ; https://github.com/MIT-SPARK/Spark-DSG ; https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html ; https://github.com/isaac-sim/IsaacLab-Arena ; https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-Plan-8B . Note licensed components separately. All G1_semantic_world interface names and examples in deck are proposed, not existing upstream interfaces.

## Engineering references

- [Semantic-nav-amr](https://github.com/Gukdoli/Semantic-nav-amr)
- [Nav2](https://docs.nav2.org)
- [ROS 2 Actions](https://design.ros2.org/articles/actions.html)
- [vision_msgs](https://github.com/ros-perception/vision_msgs)
- [Spark-DSG](https://github.com/MIT-SPARK/Spark-DSG)
- [GR00T N1.7 G1/SONIC VLA workflow](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html)
- [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena)
- [RynnBrain-Plan-8B](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-Plan-8B)
