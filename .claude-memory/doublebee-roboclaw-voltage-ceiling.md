---
name: doublebee-roboclaw-voltage-ceiling
description: RoboClaw settings need WriteNVM or they evaporate; the voltage ceiling fault and the 300 ms wheel lag both traced to this
metadata:
  type: project
---

**Every RoboClaw setting needs `rc.WriteNVM(0x80)` or it lives in RAM only.**
Verified 2026-09-04 the hard way. `SetMainVoltages` and `SetM1VelocityPID`
appeared to persist because unplugging USB does NOT reset the board (it runs off
the main battery), so a "power cycle" that only pulls USB proves nothing. A
later real power interruption reverted everything to defaults mid-session, which
disabled motor output and produced a flight where the wheels read exactly 0.00
rad/s for four seconds while -27 rad/s was commanded.

`WriteNVM` returns True and then the board REBOOTS, so the USB device
re-enumerates and the next call raises `[Errno 5]`. That is benign; reopen the
port and read back. A readback after that reboot is a genuine persistence test.

Settled configuration:
- `SetMainVoltages(a, 180, 300, 0)` = 18.0-30.0 V, auto ceiling OFF. The default
  auto offset of 20 sets the ceiling to detected pack voltage + 2.0 V, so a full
  6S pack at 25.2 V gave a 27.2 V ceiling and regen tripped `Main Battery
  Voltage Too High`. The threshold moved with charge state, which is why the
  fault looked random and unrelated to anything in software.
- `SetM1VelocityPID(a, 6.0, 1.0, 0.25, 7212)` and
  `SetM2VelocityPID(a, 2.0, 0.5, 0.25, 7212)`. Stock gains (1.0, 0.5) gave a
  300 ms closed-loop lag at 0.31 amplitude ratio against a 102 ms pendulum fall
  constant, so the balance loop had no authority and the robot fell in 2.1 s.
  Retuned, the bench step reaches 90% in ~150 ms with tail sd ~0.1%.
  **The two motors need DIFFERENT gains**: M1 has real mechanical drag (spins
  noticeably harder by hand), M2 is new and free, and P=4 that suits M1 makes M2
  limit-cycle at 64% overshoot. Match response, not numbers.
- `SetM1MaxCurrent(a, 320, 0)` signature is (max, min) in 0.01 A.

Diagnosis tools written 2026-09-04, in `modified_mav_ros_src/.../srv/`:
- `hw_lag.py <inference log>` is the go/no-go metric: closed-loop command to
  measured lag and amplitude ratio over live-control samples.
- `rc_step_metrics.py <bench csv> --setpoint N` gives rise90, overshoot and tail
  sd per motor.

Port trap: `/dev/ttyACM1` is the CubeBlack, `/dev/ttyACM2` the RoboClaw, and they
swap between boots. `policy_inference_4d.py` now defaults `--roboclaw_port auto`
which resolves the `/dev/serial/by-id/` symlink itself.

See [[doublebee-icra-plan]], [[doublebee-hardware-climbs]], [[doublebee-no-rewiring]].
