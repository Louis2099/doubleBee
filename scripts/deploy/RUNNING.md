# Hardware deployment

`db_inference.py` is the node that runs the learned policy on the real robot. It
is the same file used for the ICRA hardware trials, kept here under version
control.

## Important: this copy is not runnable on its own

The node executes from inside the MAVROS workspace, not from this repository:

```
~/doublebee_PID_JAI/modified_mav_ros_src/src/mavros/mavros/srv/db_inference.py
```

It depends on custom additions to that workspace which are **not** in this repo:

| component | path in the workspace |
| --- | --- |
| JAI output plugin | `mavros/src/plugins/jai_out_plugin.cpp`, `mavros/include/mavros/jai_out_plugin.h` |
| JAI raw plugin | `mavros/src/plugins/jai_raw_plugin.cpp`, `mavros/include/mavros/jai_raw_plugin.h` |
| messages | `mavros_msgs/msg/JAIOut.msg`, `mavros_msgs/msg/JaiRaw.msg` |
| service | `mavros_msgs/srv/JAISET.srv` |

Treat this copy as the authoritative record of the policy node. To actually fly
it, keep the workspace copy in sync.

The checkpoint is not in the repo either, since `.gitignore` excludes `*.pt`.

## Start order

Three things come up before the policy. Order matters, because the node expects
mocap and the FCU link to already be live.

### 1. Motion capture (NatNet)

```bash
cd ~/doublebee_PID_JAI/modified_mav_ros_src
source devel/setup.bash
roslaunch natnet_ros_cpp natnet_ros.launch
```

Check the IPs in the launch file first and confirm the Intel ethernet interface
is the one selected in network settings. The stream is **multicast**, so a
misconfigured interface gives a silent failure rather than an error.

### 2. MAVROS

```bash
roslaunch mavros apm.launch
```

`FCU: balanced ground to uncoupled` on startup is expected and not an error.

### 3. Verify the JAI link

```bash
rostopic echo /Jai_command
```

Nothing should publish until the policy runs, but the topic must exist. If it
does not, the JAI plugins above are not built into the workspace.

### 4. Policy

RoboClaw is **not** a separate service. The node drives it directly over serial,
configured by `--roboclaw_port`, `--roboclaw_baud` (default 38400),
`--roboclaw_address` (default 0x80) and `--roboclaw_accel`. Pass `--no_roboclaw`
to run without the wheels.

## Frozen command

This is the exact invocation that produced the first clean two-step climb
(`hw_final/pseudo_bias2_021316.csv`, 2026-09-09). Copy it, change one flag, and
log to a different prefix rather than editing it in place.

```bash
cd ~/doublebee_PID_JAI/modified_mav_ros_src/src/mavros/mavros/srv && python3 db_inference.py \
  --model_path ~/doublebee_PID_JAI/ckpts/hE4/model_5899.pt \
  --wheel_action_scale 23.6 --wheel_scale 1.0 \
  --wheel_scale_post 0.5 --post_climb_s 1.5 --post_climb_rise 0.03 \
  --max_wheel_diff 0 \
  --heading_hold_kp 0 \
  --wheel_lpf_alpha 0.3 \
  --wheel_ramp_s 0.3 \
  --roboclaw_accel 13139 \
  --rc_timeout 3.0 \
  --sim_servo_limit_rad 0.7854 --servo_slew_rad_s 2.0 \
  --servo_obs_source lowpass \
  --servo_attitude_hold --servo_hold_sign -1.0 \
  --servo_hold_blend 0.5 \
  --servo_lpf_alpha 0.3 \
  --servo_step_bias 0.3 --servo_bias_dist 0.23 \
  --servo_hold_damping 0.15 --servo_hold_slew_rad_s 10.0 \
  --contact 1.0 \
  --action_scale 1.0 --prop_scale 2.0 --servo_scale 1.0 \
  --prop_scale_step 5.0 --prop_step_relief 0.05 \
  --prop_boost_dist 0.3 --prop_min_frac 0.5 \
  --prop_map sim_damped --prewarm \
  --step -0.55 9.7821 -1.6494 0.6227 0.0650 \
  --step -0.15 10.1821 -1.6328 0.6393 0.1250 \
  --target 1.15 -0.71 \
  --base_z_offset 0.06575 \
  --log_path ~/doublebee_PID_JAI/hw_final/trial_$(date +%H%M%S).csv
```

### Why those flags

Recorded at the time, in the order the problems were found.

1. **Servo command clipped to ±45°.** The attitude-hold blend could exceed the
   action range and drive the servo to ±70°, outside anything seen in training.
2. **`--servo_lpf_alpha 0.3`.** The policy's servo action flips sign every tick
   on hardware, shaking the servo 12° peak to peak at 25 Hz. This mirrors
   `--wheel_lpf_alpha`, which fixed the identical thrash on the wheels.
3. **`--step` map corrected to the real 40 cm tread.** It previously said 70 cm,
   so the policy met step two with a height scan 30 cm out of date.
4. **`--servo_step_bias 0.3` with its own `--servo_bias_dist 0.23`.** Thrust sat
   26 to 41° off vertical at step two, wasting a third of it sideways. The bias
   aims it into the step at contact and releases past it. It needs a *shorter*
   gate than `--prop_boost_dist`; sharing 0.30 fired the tilt before the wheels
   reached step one and killed the climb entirely.

## Other terrain configurations

Per-terrain variants of the command live outside this repo in
`hw_final/terrain_commands/`. Only `trained_2x6cm.sh` matches the geometry the
policy was trained on. The others adjust `--prop_scale_step` and, for the ramp,
`--servo_hold_blend`, so the network weights are unchanged but the interface
gains are not identical across terrains.
