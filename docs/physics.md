# Physics: what is simulated, and what is measured

PhyGS does not fake the hard parts of mobile manipulation. These are the choices a result depends on.

- **No attached grasps.** A grasped object is held by friction between the jaws; it can slip and drop. "Held" is measured as the object off its support and inside the hand.
- **Physical drawer strokes.** Drawers are prismatic joints with authored limits; opening one is a planned end-effector line on its handle, and it can jam or bounce.
- **A legged base.** Spot walks under a learned policy; it can stumble, fall (the run ends) and push furniture.
- **Contacts are measured, not prevented.** Every arm-link contact above a threshold is recorded with its peak force and the running command; nothing stops the robot from touching things, and the record goes beside the outcome.
- **Task evaluators read physics.** Success is read from object poses, joint states and the base pose, never from a skill's "succeeded". Skill feedback is for the agent.

Where each is read: ground truth (`Truth`) and the report in [world.md](world.md); the grip and the
arm's contacts in [manipulation_skills.md](manipulation_skills.md); the walking policy in
[locomotion.md](locomotion.md); drawers in scenes in [scenes.md](scenes.md#articulated-furniture).
