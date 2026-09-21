---
name: doublebee-iros-reviews
description: IROS 3947 was rejected; the open criticism is the missing mode-switching baseline, and the full response plan lives in REVIEWER_RESPONSE_PLAN.md
metadata:
  type: project
---

The IROS 2026 submission (3947) was rejected by R1, R3 and R4. The current
RA-L/ICRA draft closes most structural complaints, but the criticism that
actually killed it is still open: R1 asked how much better the learned policy is
than a *well-designed mode-switching controller*. The constant-thrust ablation
arms are not that -- they are one mode held constant, never switched.

Full analysis, ranked ablation list, hardware-day checklist and the factual
landmines are in `/home/airlab/doublebee_PID_JAI/REVIEWER_RESPONSE_PLAN.md`.
Read that file before doing paper work, rather than re-deriving it.

**Why:** three reviewers converged on weak baselines and thin hardware
evidence, so every remaining hour should buy a baseline or a trial, not prose.

**How to apply:** when asked for paper changes, check the plan first; prefer the
eval-only items (switched-thrust baseline, oracle trigger, open-loop thrust
replay) over anything needing retraining. See [[doublebee-paper-status]],
[[doublebee-eval-variance]], [[doublebee-paper-terminology]].

**2026-09-11 finding:** `hE4` (the learned arm) was warm started from a
balancing policy and trained to ~5899 iterations; the `ct*` and actuator
ablation arms were trained cold to 4000. Figure 4 is therefore an unmatched
comparison. The switched-thrust baselines launched that night match hE4's
warm start and budget exactly. The rest of Figure 4 still needs a decision.
