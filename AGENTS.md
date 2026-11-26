# Regression workflow

- Fix and validate the specific failing point first with the narrowest useful
  frame range.
- After the point passes, run one uninterrupted regression from the established
  beginning through the required end frame.
- Do not poll a long regression repeatedly. Estimate its duration from prior
  runs, wait for that interval, then inspect the completed log and point-history
  output once.
- A targeted pass is not a substitute for the final uninterrupted regression.
- Never add frame-specific production behavior; frame ranges are only for tests
  and diagnostics.
