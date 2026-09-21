---
name: doublebee-no-rewiring
description: "DoubleBee hardware is frozen for final testing — fix wheel sign/convention issues in software, not by rewiring"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4a1ea8f7-460d-4372-9a50-1f738c3051e4
  modified: 2026-08-20T21:20:42.817Z
---

As of 2026-08-20 the DoubleBee robot is in its final testing phase and the user
will not change wiring: "i wont play around with the wires anymore."

**Why:** the harness is settled and re-wiring mid-campaign invalidates the runs
already collected, on top of the risk of introducing a new fault.

**How to apply:** when a wheel turns the wrong way, an encoder counts backwards,
or M1/M2 map to the wrong side, resolve it with the software convention flags
(`--m1_is_right`, `--wheel_sign_left/right` in `db_wheels.py` and
`db_inference.py`), never by asking them to swap leads. The one exception that
software genuinely cannot fix is motor leads reversed while the encoder is not —
that makes the RoboClaw's velocity PID chase itself, and `db_wheels.py solve`
detects and names it specifically. See [[doublebee-paper-status]].
