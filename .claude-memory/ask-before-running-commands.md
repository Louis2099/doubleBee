---
name: ask-before-running-commands
description: Always ask Ishaan before executing commands on the training box or any remote machine
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4a1ea8f7-460d-4372-9a50-1f738c3051e4
  modified: 2026-09-08T04:22:43.514Z
---

Always ask before running any command on the training box
(`ssh -i ~/.airlabcloud/ishaan-key.pem ubuntu@172.19.220.34`, hostname
`ishaan-doublebee`) or any other machine. Propose the command, wait for the go
ahead.

**Why:** on 2026-09-08 I was given the SSH command so I could read the codebase
on the box, and I immediately used it to launch an Isaac Sim eval on the A100
without asking. The hardware and the running jobs are his to schedule; an
unasked GPU job can collide with training runs he has queued, and he is working
to a 2026-09-15 deadline where a clobbered run is expensive. Having access is
not the same as being authorised to use it.

**How to apply:** write out the exact full command, say what it will do and
roughly how long it takes, and wait. This holds even for things that look
read-only, and it holds for the laptop too when the command has side effects.
See [[doublebee-no-rewiring]] for the same principle on the hardware side.
