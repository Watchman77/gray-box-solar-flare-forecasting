# 72 h manuscript claim-to-artifact map

Status: **submission record — 6 October 2026**

This map links each frozen manuscript claim to one repository result folder and one repository folder hash. The hash in the table is the Git tree SHA for that result folder, so the cited record can be checked without relying on a local `/home/abmoses2000` path. Anchor files identify the result object within the folder.

## Frozen claim map

| ID | Manuscript claim | Evidence status | Repository result folder | Git tree SHA | Anchor file |
|---|---|---|---|---|---|
| C01 | SHARP, AIA and fixed fusion degrade when transferred from earlier development support into later Cycle-25/2026 regimes. | supported | `results/72h_graybox/20261006/cross_cycle_reliability_audit` | `379e766fe8afbdb0fcc54908bfa3a74d7c13a43f` | `cross_cycle_branch_metrics.csv` |
| C02 | Probability calibration worsens in later regimes; later BSS/ECE are not repaired by refitting on later outcomes. | supported | `results/72h_graybox/20261006/calibration_diagnostics_v2` | `97ba7a6ea8f77eac62ae0cca2345aa4f538acdb6` | `calibration_diagnostics.csv` |
| C03 | Flare-class conformal coverage degrades under later temporal transfer. | supported | `results/72h_graybox/20261006/cross_cycle_reliability_audit` | `379e766fe8afbdb0fcc54908bfa3a74d7c13a43f` | `cross_cycle_branch_metrics.csv` |
| C04 | Statistical agreement can tighten while rare-flare reliability worsens; the rolling evidence is prospective-style, not live prospective deployment. | supported with chronology qualification | `results/72h_graybox/20261006/rolling_operational_replay_v1` | `e1fcf50ed640c4017e5cdfc3eb2c21bb76ccfd1d` | `past_only_uq_monitor.csv` |
| C05 | Recorded AIA acquisition/recovery failures are strongly temporally clustered rather than independent masks. | supported | `results/72h_graybox/20261006/real_outage_structure_v1` | `643bd21a68d2b92b74d02a9da85c79778e6185e6` | `empirical_outage_episodes.csv` |
| C06 | When SHARP remains available, mapped AIA failures route forecasts to SHARP fallback and preserve issuance. | supported | `results/72h_graybox/20261006/real_outage_replay_v1` | `acd070ba9d4825800831fc7230ae7a15f6f95039` | `real_outage_replay_summary.csv` |
| C07 | The frozen AIA branch is not supported as an automatic degraded fallback when SHARP is unavailable. | supported, post-hoc | `results/72h_graybox/20261006/aia_only_fallback_feasibility_v1` | `8cd0306eb05d57f4c2dc64f09765bd1721d9c7b1` | `branch_feasibility_summary.csv` |
| C08 | The final fallback-first v3 semantics were frozen after Cycle-25 diagnostics, without Cycle-25 numerical threshold tuning and before supplementary-2026 outcomes were opened. | supported | `results/72h_graybox/20261005/fallback_v3` | `72c7915ce643f87a6fbe24c47c43c7b6c975f727` | `protocol.json` |
| C09 | Supplementary 2026 is the clean future evaluation of the frozen v3 policy; later Phase-H analyses of 2026 are post-hoc. | supported | `results/72h_graybox/20261005/supplementary2026_v3` | `b9ae10e3fd841c0985565a63871774b601072d9e` | `summary.json` |
| C10 | The post-hoc simple baseline is competitive; at validation-chosen thresholds SHARP has a modest Cycle-25 TSS advantage over the latest-state logistic, and fusion is worse than SHARP for Cycle-25 TSS/AP; 2026 paired intervals include zero. | supported, post-hoc comparator | `results/72h_graybox/20261006/baseline_suite_v2` | `1569f727348c46e09adcf4c4fde1af7e54493f1c` | `baseline_metrics.csv`, `baseline_paired_bootstrap_deltas_key.csv` |
| C11 | The explicit physical layer is an applicability/provenance monitor; the 2026 elevation is gradient-dominated, physical distance is not a case-level abstention score, and the solar-versus-pipeline origin remains unresolved. | supported as post-freeze diagnostic | `results/72h_graybox/20261006/phase_h_physical_layer_freeze` | `5ee4a2655740852938d12d19395e8830df6d1166` | `README.md` |

## Canonical frozen source objects

The post-freeze analyses read these objects; they do not overwrite them.

| File | Frozen size/shape | SHA-256 |
|---|---|---|
| `graybox_aia72_master_predictions.csv.gz` | 71,010 non-training rows | `17300093b88b7c61ecadf5eb28acba0d6d023b340d7740807852dd4352b22b46` |
| `sharp72_predictions_frozen_20261002.csv.gz` | 113,433 prediction-ledger rows | `7368c002a978321cdc67221564f3b7901687badc7aba28ab9df615179d1e017b` |
| `sharp.npy` | shape `(113433, 3, 16)` | `c9921be7b7fe381b43436a0e927d15d84de83e15376025ae0d8a79a6ef917639` |

Phase H and the baseline suite read these frozen sources. They do not refit the neural models and do not change the v3 policy.

## Baseline threshold record

Primary baseline TSS uses the validation-chosen/frozen thresholds below, not a generic 0.5 threshold:

- latest-state SHARP logistic: `0.10265333871104586`
- frozen SHARP: `0.0752000237504641`
- frozen AIA: `0.025101790286788`
- frozen fusion: `0.0905992648354184`

The baseline active-region bootstrap resamples `region_component_id`, uses 2,000 deterministic paired draws within each role, and retains all windows from each sampled active region.

## Explicit non-claims

The record does **not** support:
- a completed multi-horizon Gray-Box result;
- universal fusion superiority;
- universal deep-learning superiority;
- a physical H2 routing gate;
- a causal Solar-Cycle-25 explanation of the 2026 gradient shift;
- closure of the solar-versus-pipeline provenance question;
- first-principles MHD/PINN modelling;
- true live prospective validation.

## Writing rule

If a manuscript sentence is stronger than this map, the supporting artifact must be checked before that sentence is retained. Negative, post-hoc and provenance-limited findings remain visible.
