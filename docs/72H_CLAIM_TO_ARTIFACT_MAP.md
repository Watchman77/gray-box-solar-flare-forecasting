# 72 h manuscript claim-to-artifact map

Status: **working reproducibility map — 6 October 2026**

This table links each headline manuscript claim to the frozen or post-freeze evidence that supports it. It is intended to prevent unsupported wording during journal revision.

| Claim | Evidence status | Primary artifact(s) | Notebook / protocol | Key frozen source hashes / notes |
|---|---|---|---|---|
| SHARP/AIA/fusion degrade across later regimes | supported | cross-cycle branch metrics | cross-cycle reliability audit | master `17300093...`; SHARP ledger `7368c002...` |
| Calibration worsens later | supported | calibration diagnostics, reliability bins | calibration diagnostics v2 | BSS reference is frozen training prevalence |
| Flare-class conformal coverage degrades | supported | cross-cycle branch metrics / conformal audits | Mondrian conformal + cross-cycle audit | empirical transfer result, not universal guarantee |
| Entropy/seed spread/modality gap tighten while reliability worsens | supported | H1c physical-vs-statistical agreement table + rolling UQ monitor | Notebook 27; rolling replay | descriptive/post-hoc for later regimes |
| Real AIA failures are clustered | supported | empirical outage episodes / exposure tables | real outage structure v1 | engineering incidents, not live-service uptime |
| SHARP fallback preserves issuance during mapped AIA failures | supported | final replay summary | real outage replay + rolling replay | mapped incident replay; SHARP available |
| AIA-only automatic fallback is not supported | supported | branch feasibility + AIA uncertainty feasibility | AIA-only fallback feasibility v1 | post-hoc; no new threshold selected |
| v3 NORMAL/DEGRADED/ABSTAIN routing is frozen | supported | policy_family_v3.json / protocol.json | Notebook 11 | Cycle 25 informed semantic redesign; 2026 unopened at freeze |
| 2026 clean v3 evaluation | supported | supplementary-2026 policy cases/results/summary | Notebook 12 | no threshold tuning/reselection on 2026 |
| Rolling replay is prospective-style, not live prospective | supported with caution | operational replay log / cadence audit | rolling operational replay v1 | 72 h horizon + 24 h reporting delay |
| Latest-state logistic is competitive | supported | baseline_metrics.csv + AR-block CIs | Notebook 31 | 25,586 train rows; threshold on model_validation |
| SHARP has modest Cycle-25 TSS advantage over latest-state logistic | supported | paired bootstrap deltas | Notebook 31 | delta +0.039, 95% CI [0.0003, 0.0819] |
| Fusion is not universally superior to SHARP | supported | paired bootstrap deltas | Notebook 31 | Cycle-25 TSS/AP significantly favor SHARP |
| Explicit magnetic-state applicability layer exists | supported | H1b full physical state / coefficients | Notebooks 25–26 | exact recovered SHARP tensor `c9921be...` |
| 2026 physical-distance elevation is gradient-dominated | supported | leave-family-out applicability | Notebook 28 | post-hoc Phase H |
| Physical distance is not a case-level error score | supported | H1b/H1c risk association + class-conditioned distance | Notebooks 26–27 | negative Spearman relation to fusion squared error |
| JSOC paired SHARP summary keywords are identical across CCD/CEA series on queried pairs | supported | paired_feature_parity_summary.csv | Notebook 30 | 120 pairs, 240/240 successful queries |
| Local April-2026 gradient shift is astrophysical | **not supported** | — | — | solar-versus-pipeline origin unresolved |
| Physical distance should trigger ABSTAIN | **not supported** | — | — | Phase H rejected promotion to routing gate |
| Universal deep-learning superiority | **not supported** | — | — | simple logistic is competitive |
| True prospective validation | **not supported** | — | — | use “prospective-style chronological replay” |
| First-principles MHD/PINN Gray-Box | **not supported** | — | — | explicit physical-state monitor only |

## Canonical manuscript-level source objects

- Multimodal master predictions: `17300093b88b7c61ecadf5eb28acba0d6d023b340d7740807852dd4352b22b46`
- Frozen SHARP 72 h ledger: `7368c002a978321cdc67221564f3b7901687badc7aba28ab9df615179d1e017b`
- Exact recovered SHARP tensor: `c9921be7b7fe381b43436a0e927d15d84de83e15376025ae0d8a79a6ef917639`

## Writing rule

If a manuscript sentence is stronger than the wording in this map, verify the supporting artifact before retaining it. Negative and provenance-limited findings must remain visible.
