# Gray-Box Solar Flare Forecasting — Continuity Handoff

Updated: 6 October 2026

## Research objective
Develop and validate a Gray-Box operational trust framework for M/X-class solar-flare forecasting. The published prediction work asks **can we predict?**; this project asks **when should we trust, degrade, reroute, or withhold the prediction?**

Working title from Prof. Rami:
**A Gray-Box Operational Framework for Multi-Horizon Solar Flare Forecasting: Calibration, Uncertainty, and Prospective Evaluation**

## Supervisor requirements
Prof. Rami requested:
1. probability calibration on temporally earlier data using raw/Platt/isotonic and Brier/Brier-skill/log-loss/reliability/slope/intercept/ECE;
2. uncertainty quantification using task-appropriate conformal methods;
3. OOD/applicability checks using feature-space distance, conformal nonconformity, ensemble disagreement and quality flags;
4. explicit NORMAL / DEGRADED / ABSTAIN-or-fallback states;
5. realistic missing-data protocols, masks/imputation sensitivity, complete-case sensitivity and historical block/channel outages;
6. chronological plus AR-disjoint and event-disjoint evaluation;
7. rolling evaluation using only information available at each issue time.

## Architecture boundary
Five conceptual layers:
1. data-driven predictor;
2. physics-informed/physically interpretable magnetic component;
3. multimodal fusion;
4. uncertainty/applicability;
5. operational safety.

Current implementation is physics-informed through SHARP magnetic variables; it is **not** a first-principles MHD solver or validated PINN. A PINN/full solver is optional later work, not required before the operational protocol is completed.

## Cadence clarification
Original AIA project:
- target/label horizon: **48 h** for the original AIA baseline;
- AIA sampling / natural issue cadence: **96 minutes**;
- extraction block length around 24 h is an engineering detail, not the forecast horizon/cadence.

Current Gray-Box primary development horizon: **72 h**.
Original multi-horizon plan remains: **72 h first, then 24 h, then 3 h**.

## Pre-Gray-Box data work already completed
Do not redo from scratch:
- 153,366 candidate records spanning May 2010–Aug 2026;
- 113,433 matched AIA–SHARP cases;
- inherited input cohorts: Cycle 24 = 55,871; primary Cycle 25 = 45,403; supplementary 2026 = 12,159;
- candidate 48 h and 72 h outcome tables built;
- NOAA/SWPC event-source and region adjudication performed;
- unresolved event/region cases retained explicitly rather than silently forced negative;
- historical 72 h baseline/predictions audited and reproduced.
The canonical operational multimodal master used for the completed 72 h policy work has 71,010 non-training rows.

## Frozen 72 h Gray-Box chronology
1. earlier probability calibration: raw retained for SHARP, AIA and fixed 0.5/0.5 fusion;
2. marginal conformal v1 failed rare-flare coverage;
3. q90 v1 hard-abstention policy withheld too many flare cases;
4. Mondrian/class-conditional v2 repaired calibration-period class balance;
5. Cycle-25 evaluation showed nominal flare coverage did not survive temporal shift and hard abstention remained unsafe;
6. Cycle 25 was reclassified as development diagnostic;
7. fallback-first v3 frozen before 2026:
   - NORMAL = trusted SHARP+AIA fusion;
   - DEGRADED = issue SHARP when fusion is not NORMAL but SHARP is available;
   - ABSTAIN = SHARP itself unavailable;
8. supplementary-2026 one-shot evaluation remained clean for v3;
9. locked bootstrap/McNemar statistics followed with no tuning;
10. detected AIA outage/degradation stress tests were post-hoc;
11. SHARP-failure / dual-outage tests were post-hoc;
12. reviewer-ready results synthesis completed.

## Key clean 2026 result
11,118 cases, 992 positives:
- fusion: TSS 0.2922, recall 0.3397, specificity 0.9525;
- fallback-first v3 q90: TSS 0.2964, recall 0.3458, specificity 0.9506;
- SHARP: TSS 0.3229, recall 0.4244, specificity 0.8985.
v3 vs fusion TSS delta ~+0.0042, bootstrap CI crosses zero; do not claim significant TSS superiority.
v3 vs SHARP has lower TSS/recall but higher specificity/precision and far fewer false alarms.

## Key uncertainty result
Aggregate conformal coverage can look strong while flare-class coverage collapses under temporal shift. Do not claim fixed conformal guarantees under later non-exchangeable solar regimes.

## Robustness result
- detected AIA loss: v3 routes to SHARP and preserves service while SHARP remains healthy;
- SHARP loss: current fusion and fallback both fail, so ABSTAIN is unavoidable;
- dual loss: no validated route remains.
AIA-only emergency fallback has **not** yet been validated and must not be silently added.

## Current repository checkpoint
Main branch contains notebooks 01–16 for the completed 72 h pipeline and reviewer-ready synthesis. Local Mac file `docs/EXECUTION_PLAN.md` has historically been kept unstaged; do not overwrite/commit the user's local edits accidentally.

## Revised A–G extension roadmap
A. **Cross-cycle reliability audit** — Cycle-24 earlier development support -> Cycle-25 diagnostic -> 2026; branch calibration/skill/conformal coverage/disagreement/state mix.
B. **Complete multimodal calibration diagnostics** — Brier Skill Score, ECE, calibration slope/intercept, reliability diagrams.
C. **Multimodal OOD/applicability** — extend earlier SHARP feature-distance work with AIA quality/OOD, branch disagreement, ensemble spread and error/coverage association.
D. **AR-disjoint + event-disjoint** — unseen-AR analysis, unique flare-event detection, event-level recall/lead time/fallback.
E. **Historical outage replay** — mine real AIA acquisition/recovery/channel/block failures and compare with synthetic stress tests.
F. **Fallback completion** — test whether AIA-only emergency mode is sufficiently useful when SHARP is unavailable; otherwise retain ABSTAIN.
G. **Rolling multimodal operational replay** — frozen predictors, past-only calibration/trust updates, declared issue cadence; call it retrospective rolling operational replay, not live prospective validation.

Only after A–G decide whether a PINN/explicit physics branch adds enough value. Then replicate the frozen higher-level architecture at 24 h and 3 h; do not redesign from scratch per horizon.

## Immediate next notebook
`notebooks/72h_graybox/17_cross_cycle_reliability_audit.ipynb`

Scientific status:
- no model fitting;
- no threshold tuning;
- no policy selection/redesign;
- Cycle-24 rows are development/descriptive, not independent;
- Cycle-25 is diagnostic/post-hoc for v3;
- 2026 is already-spent locked evidence and must not be used to redesign.

## Canonical frozen artifacts on VM
- master: `/home/abmoses2000/graybox_aia72_master_predictions_v2_20261005/graybox_aia72_master_predictions.csv.gz`
  - SHA256: `17300093b88b7c61ecadf5eb28acba0d6d023b340d7740807852dd4352b22b46`
- SHARP predictions: `/home/abmoses2000/sharp72_predictions_frozen_20261002.csv.gz`
  - SHA256: `7368c002a978321cdc67221564f3b7901687badc7aba28ab9df615179d1e017b`
- v3: `/home/abmoses2000/graybox_aia72_policy_v3_fallback_first_freeze_20261005`
- clean 2026: `/home/abmoses2000/graybox_aia72_supplementary2026_v3_eval_20261005`
- results synthesis: `/home/abmoses2000/graybox_aia72_results_synthesis_v2_20261005`

## Writing status
Drafted in chat:
- Introduction;
- Related Work / operational motivation;
- Gray-Box framework;
- Data and experimental design;
- complete 72 h Results;
- Discussion;
- Limitations;
- provisional Conclusion.
Do not finalise Abstract/Conclusion until extension analyses and 24 h / 3 h replication decisions are complete.

## Project separation
Keep this project separate from the main AIA paper. AIA acquisition/model-development work belongs in `Watchman77/solar-flare-aia-training`; this repository owns the operational Gray-Box framework.

## Notebook 17 executed — cross-cycle reliability findings
Executed 6 October 2026 on the VM with no fitting/tuning/redesign.

Support:
- Cycle-24 earlier development support: 24,046 cases, 1,406 positives, prevalence 5.85%.
- Cycle-25 diagnostic: 35,846 cases, 3,860 positives, prevalence 10.77%.
- Supplementary 2026 locked: 11,118 cases, 992 positives, prevalence 8.92%.
- 2010–2013 remain training-only and are not scored from in-sample predictions.
- 2020 is absent from the accepted multimodal prediction package.

Main shift:
- SHARP TSS: 0.689 -> 0.541 -> 0.323.
- AIA TSS: 0.333 -> 0.340 -> 0.074.
- Fusion TSS: 0.630 -> 0.488 -> 0.292.
- SHARP flare conformal coverage: 0.806 -> 0.735 -> 0.435.
- AIA flare conformal coverage: 0.812 -> 0.620 -> 0.064.
- Fusion flare conformal coverage: 0.838 -> 0.692 -> 0.346.
- ECE worsens across periods for all three branches.

Important new operational finding:
The frozen v3 q90 state logic becomes **more NORMAL** over time despite worsening predictive skill and flare-class conformal coverage:
- Cycle-24 earlier development: NORMAL 66.0%, DEGRADED 34.0%.
- Cycle-25 diagnostic: NORMAL 77.7%, DEGRADED 22.3%.
- 2026: NORMAL 92.5%, DEGRADED 7.5%.
At the same time, SHARP–AIA probability gap, fusion seed spread and fusion entropy all decline. Therefore the current disagreement/spread/entropy gates can become reassuring under shift even while real rare-flare reliability worsens. This strongly motivates explicit OOD / feature-space applicability analysis rather than relying on predictive disagreement alone.

Evidence boundary remains:
- Cycle 24 = development/descriptive.
- Cycle 25 = diagnostic/post-hoc for v3.
- 2026 = already-spent clean evidence; do not redesign from it.


## Notebook 18 executed — complete multimodal calibration diagnostics
Executed 6 October 2026 with no model fitting, threshold tuning, policy selection, or redesign.

Frozen Brier-skill reference:
- training climatology = 0.0576877980145392.

Regime-level findings:
- SHARP Brier Skill Score: 0.336 (Cycle-24 earlier development) -> 0.215 (Cycle-25 diagnostic) -> 0.160 (2026).
- AIA Brier Skill Score: -0.001 -> 0.013 -> -0.056.
- Fusion Brier Skill Score: 0.247 -> 0.176 -> 0.097.
- ECE increases across regimes for all branches:
  - SHARP: 0.0056 -> 0.0307 -> 0.0474.
  - AIA: 0.0217 -> 0.0674 -> 0.0793.
  - Fusion: 0.0303 -> 0.0373 -> 0.0597.
- 2026 calibration slopes are all < 1:
  - SHARP 0.656, intercept -0.022.
  - AIA 0.663, intercept +0.840.
  - Fusion 0.810, intercept +0.791.
Interpretation: later-period forecasts are increasingly miscalibrated; slope < 1 indicates probability spread is too extreme relative to outcomes, while the positive AIA/fusion intercepts in 2026 indicate an additional systematic probability-level shift. Do not refit on 2026.
- AIA becomes worse than the frozen training-climatology reference in 2026 (negative BSS), while SHARP and fusion remain better than climatology but with substantially reduced skill.

Annual diagnostics caveat:
- 2018 contains zero positives and therefore calibration slope/intercept are undefined.
- Very low-positive years such as 2019 should not be over-interpreted; regime-level summaries are the primary evidence.

Phase B status: complete.
Next scientific priority: Phase C multimodal OOD/applicability, explicitly testing feature-space/domain-shift indicators because Notebook 17 showed disagreement/spread/entropy can become more reassuring while true rare-flare reliability worsens.
