# Canonical 72-hour lineage remediation and validation plan

Reviewed 7 October 2026. This plan does not rerun inference, consume a GPU allowance, or convert reconstructed evidence into an original execution receipt.

## Acceptance gates still missing

The canonical 71,010-row master is currently usable as protocol-scoped reported evidence, but its model-generation lineage is not accepted. Acceptance requires all of the following:

1. The original successful creation receipt for `/home/abmoses2000/graybox_aia72_master_predictions_v2_20261005/graybox_aia72_master_predictions.csv.gz`.
2. A byte-level hash and size record for the canonical file, with the transfer or archive receipt that moved it to the analysis environment.
3. The exact case manifest and role schedule used to create it, including the 3,905 model-validation, 2,831 probability-calibration, 4,168 conformal-calibration, 13,142 policy-validation, 35,846 Cycle-25 and 11,118 supplementary-2026 rows.
4. Checkpoint-generation pins for SHARP, AIA and fusion, including model file hashes, selected seed/epoch, architecture and probability-mean rule.
5. The training-only normalization hash and preprocessing/channel-order specification bound to the AIA rows.
6. Prediction-column and failure-status schema, with evidence that missing inputs were retained rather than silently excluded.
7. A successful source-transfer and cleanup receipt. The failed v3 relay receipt remains separate and cannot satisfy this gate.

## Existing pinned sources

The repository already preserves the following source pins:

- Canonical master hash: `17300093b88b7c61ecadf5eb28acba0d6d023b340d7740807852dd4352b22b46`.
- Frozen SHARP prediction ledger hash: `7368c002a978321cdc67221564f3b7901687badc7aba28ab9df615179d1e017b`.
- Frozen SHARP tensor hash: `c9921be7b7fe381b43436a0e927d15d84de83e15376025ae0d8a79a6ef917639`.
- Earlier calibration selection hash: `8570bfdfdc6583490d85f4a67c3ebcf73458a59ffa25cd9e57ecf455ca5aaedd`.
- v3 inference contract, bundle, case manifest, block manifest, normalization and three saved AIA model hashes are preserved under `outputs/aia72_inference_v3_20261004/`.
- The v3 complete-role archives independently verified locally are probability calibration, conformal calibration, policy validation and Cycle 25.

These pins establish reproducibility references. They do not by themselves prove that the canonical master was generated from the same v3 inputs.

## Future read-only VM inspection

When SSH/IAP access returns, use one bounded read-only probe only. It should inspect, without acquiring the research lock or changing files:

```text
sha256sum /home/abmoses2000/graybox_aia72_master_predictions_v2_20261005/graybox_aia72_master_predictions.csv.gz
stat -c '%s %i %y' /home/abmoses2000/graybox_aia72_master_predictions_v2_20261005/graybox_aia72_master_predictions.csv.gz
find /home/abmoses2000/graybox_aia72_master_predictions_v2_20261005 -maxdepth 2 -type f -print
find /home/abmoses2000 -maxdepth 2 -type f \( -name '*receipt*.json' -o -name '*manifest*.json' -o -name '*provenance*.json' \) -print
ps -eo pid,ppid,stat,cmd
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
```

The probe should compare discovered receipts against the repository pins, inspect the owner pointer and common-lock inode read-only, and save a dated output. It must not relaunch a worker, create a new lease, unlink a lock, alter the canonical file, or claim cleanup success from a process listing.

## Work that is already feasible on the Mac

The following analyses can continue without the missing canonical transfer receipt:

- Source and numerical review of the verified v3 complete-role archives.
- Recalculation of conformal thresholds and coverage for SHARP, AIA and fixed fusion on the complete conformal and Cycle-25 roles.
- Comparison of those values with the canonical reported artifacts, with any discrepancy recorded rather than silently harmonized.
- Review of the existing canonical operational-policy, outage, AR/event-disjoint and rolling-replay outputs as protocol-scoped results.
- Manuscript claim-ledger updates that label canonical lineage acceptance as pending and keep the failed v3 boundary visible.

The following remain blocked until the missing lineage evidence is supplied or a new, separately reviewed experiment is authorized:

- Treating the canonical master as an accepted reproduction of the v3 inference run.
- Claiming that the missing 8,558 supplementary rows were produced by the same v3 worker/models.
- Any new GPU inference or completion of the expired v3 relay.

## Bounded preflight for genuinely new work

If a new inference phase is later approved, preflight must first freeze the exact source tree and notebook, contract and block manifests, checkpoint/normalization hashes, expected case IDs, output schema, runtime versions, CPU-only staging tests, scratch and transfer ceilings, common-lock inode, owner pointer and cleanup procedure. The preflight must finish with a fresh one-use authorization and a fresh resource receipt. A new run must not reuse the consumed 5 October allowance or silently append to the failed collector directory.
