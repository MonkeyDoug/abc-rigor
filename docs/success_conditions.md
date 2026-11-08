# Revised success conditions

- **Ball tray:** sample and record one continuous uniform hold duration in
  `[5, 15]` seconds for each randomized episode. Keep the tray at least 5 cm
  above its initial height and the ball within the existing tray-local region
  for that duration without interruption. An invalid sample restarts the hold.
  Once the hold completes, ignore the ball and require a new tray–table contact
  at a later timestamp. Existing contact or repeated evaluation of the same
  timestamp does not count as lowering. Completion is latched. Physics-step updates
  prevent an interruption between control observations from being missed.
  `evaluate_model_data` uses simulation time; qpos-only callers must supply
  `time_s` explicitly.
- **Chess:** all pieces must be upright on their home squares, including pieces
  outside the displaced target subset. Pieces of the same color and type are
  interchangeable across that entire set. Match valid placements one-to-one;
  retain the existing board-relative position, height, and upright tolerances.
- **Relative placement:** allow zero separation along the requested direction.
  Keep the 30 cm upper limit, 8 cm perpendicular limit, and existing placement
  height check.
- **In-hand transfer:** require object–table contact and no object–gripper
  contact in place of the height limit. Keep the table footprint and opposite
  side requirements. Reset metadata remains the normal source of the start side.
- **LEGO sorting:** allow a block center up to 5 mm in Euclidean distance outside
  its matching bin's goal box. This is not a separate 5 mm extension on each
  axis. Preserve the exact wrong-bin exclusion.

The following overrides belong only to the separate historical success
calculation, not these persistent evaluator changes:

- Use the first mapped video frame to determine the in-hand transfer start side.
- For dish-rack scoring, ignore rack and gripper contact while retaining the
  upright and spatial tests.

CPU regressions: `python -m unittest discover -s tests -p test_success_conditions.py -v`.
