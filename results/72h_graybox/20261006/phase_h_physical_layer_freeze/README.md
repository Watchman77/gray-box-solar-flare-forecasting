# Phase H physical-layer freeze — repository record

This folder is a repository-level trail for the already-completed post-freeze Phase H diagnostics. **No new experiment is opened by this record.**

Canonical input:
- `sharp.npy`, shape `(113433, 3, 16)`, SHA-256 `c9921be7b7fe381b43436a0e927d15d84de83e15376025ae0d8a79a6ef917639`.

Frozen interpretation:
- the explicit MPSV physical layer is an applicability/provenance monitor;
- the 2026 distance elevation is gradient-family dominated;
- physical distance is not a valid case-level abstention score;
- paired JSOC comparison used 120 identical HARP/T_REC pairs and showed keyword parity on those queried pairs;
- that parity was **not shown to cover every local April–August 2026 CEA-era tensor row**;
- the solar-versus-pipeline origin of the local April-2026 gradient shift remains unresolved;
- no H2 physical gate was promoted into NORMAL / DEGRADED / ABSTAIN.

Primary supporting records:
- `docs/PHASE_H1B_RESULTS_20261006.md`
- `docs/PHASE_H1C_RESULTS_20261006.md`
- `docs/PHASE_H1D_RESULTS_20261006.md`
- `docs/PHASE_H1E_RESULTS_20261006.md`
- `docs/PHASE_H1F_RESULTS_20261006.md`
- `docs/PHASE_H_CONCLUSION_20261006.md`

Phase H reads the frozen source objects; it does not replace them, refit the neural branches, or change v3.
