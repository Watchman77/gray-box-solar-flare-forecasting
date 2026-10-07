# 72-hour Gray-Box evidence reconciliation

Reviewed 7 October 2026. This is a same-assistant repository review of the two workspace evidence streams, not external scientific validation.

## What is covered

| Agreed analysis | Evidence | Disposition |
|---|---|---|
| SHARP, AIA and fixed 0.5/0.5 fusion comparison | `docs/72H_BASELINE_RESULTS_20261006.md`; `results/72h_graybox/20261006/baseline_suite_v2/` | Covered on the canonical non-training AIA master. Fusion is not universally better than SHARP. |
| Probability calibration, reliability, BSS and ECE | `results/72h_graybox/20261006/calibration_diagnostics_v2/` | Covered as a diagnostic analysis. It is post-hoc and uses the canonical master prediction table; it must be cited with its protocol and source hashes. |
| Split/Mondrian conformal uncertainty | `results/72h_graybox/20261005/mondrian_conformal_v2_primary_alpha.csv`, `results/72h_graybox/20261006/cross_cycle_reliability_audit/` | Covered for the frozen branches, with later class-wise degradation reported. The independently reviewed partial archive reproduces the key Cycle-25 flare-coverage values (SHARP/AIA/fusion approximately 0.735/0.620/0.692). |
| Three operational states | `results/72h_graybox/20261005/fallback_v3/` and `docs/72H_FINAL_SCIENTIFIC_NARRATIVE.md` | Covered as a frozen routing architecture: NORMAL=fusion, DEGRADED=SHARP fallback, ABSTAIN when SHARP is unavailable. This is an evaluated research policy, not a deployed safety certification. |
| Realistic AIA missingness and fallback | `results/72h_graybox/20261006/real_outage_structure_v1/`, `real_outage_replay_v1/`, `aia_only_fallback_feasibility_v1/` | Covered as retrospective/post-hoc replay. Engineering acquisition incidents are not equivalent to independently verified live-service outages. |
| AR/event-disjoint sensitivity | `results/72h_graybox/20261006/ar_event_disjoint_v1/` | Covered as an audit, subject to its event-artifact and region-overlap limitations. |
| Rolling past-only monitoring | `results/72h_graybox/20261006/rolling_operational_replay_v1/` | Covered as prospective-style historical replay, not live prospective deployment. |

## Compatibility limits

The original v3 GPU inference package was built for 96,596 candidate cases and its relay ended after 116 verified blocks. Its supplementary-2026 role is incomplete (2,560/11,118 cases) and its train-role inference was never reached. The newer canonical evidence uses a 71,010-row non-training AIA master (SHA256 recorded in `docs/72H_BASELINE_SUITE_PROTOCOL.md`) and separately preserved frozen prediction ledgers. Those are compatible for the paper’s frozen-branch comparison only where the source hashes and case support in each protocol agree; they must not be described as a completed rerun of the failed v3 relay.

The partial conformal metrics in `results/aia72_partial_analysis_20261007/` were independently rehashed and numerically recomputed from the verified conformal and Cycle-25 archives. This is a separate same-assistant review pass, not independent scientific validation. It confirms the key flare-class coverage values but does not supersede the canonical 2026 evidence.

## Remaining work ledger

1. Keep the failed v3 relay and `cleanup_verified=false` boundary visible in provenance. No retry is authorized by the consumed allowance.
2. Use the canonical 72-hour evidence folders for the manuscript claim ledger, with their actual protocols and source hashes.
3. Do not claim that the failed v3 package generated a complete 2026 or train-role archive.
4. Before extending to 24-hour or 3-hour horizons, freeze equivalent support, calibration, conformal and routing protocols and perform the same provenance review.
5. A true prospective deployment and independently verified operational availability remain future work.

## VM state check

A narrow read-only SSH/IAP probe was attempted on 7 October and could not connect to port 22 (`failed to connect to backend`). Therefore current VM process, lock and GPU state are **unknown**, not inferred to be free or occupied. No lease was acquired and no VM mutation or relaunch was attempted.
