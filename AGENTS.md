# Repository guidance

## Git authorship

For new commits made on behalf of repository owner Bamidele Akinwumi, use:

- Author and committer name: `BAMIDELE AKINWUMI`
- Author and committer email: `119769750+Watchman77@users.noreply.github.com`
- GitHub account: `Watchman77`

Do not add Codex, OpenAI or another assistant as a commit author, committer or co-author. Check the effective author/committer before committing and the saved metadata afterward. Preserve attribution on existing third-party commits. Do not rewrite published history unless explicitly requested. This Git preference does not replace any publication disclosure or contribution requirements.

## Research conventions

- Distinguish proposals, implemented methods, executed experiments and verified results.
- Preserve original labels, source data, evaluation splits and frozen results; corrections require versioned artefacts and an explanation.
- Keep 48-hour AIA results distinct from 3-, 24- and 72-hour forecasting.
- Split by declared issue-time boundaries and handle overlapping outcome windows. Apply the AR/event separation required by each named evaluation.
- Fit preprocessing, calibration, fusion and decision policies only on the designated earlier data. Preserve natural evaluation prevalence.
- Record actual observation times, time scales, data availability, source/processing versions, region identity and label follow-up.
- Compare models on matched forecast cases; report missing cases and full-population operational availability separately.
- Do not describe feature fusion as a validated physical simulator or report planned UQ as completed experiments.
- Keep large data, credentials, checkpoints and private correspondence out of Git. Reference versioned artefacts in manifests.
- Preserve the upstream repositories. Port only the components needed here with their provenance and applicable licensing recorded.
- Follow the user's requested scope for acquisition, training and deployment. Repository setup alone is not a request to launch expensive experiments.

The repository has executable baseline-audit and candidate-dataset scripts with focused scientific boundary tests, but no Gray-Box model implementation. Run `python -m unittest discover -s tests -v` for audit-code changes, and validate documentation links and JSON when editing the protocol. Preserve the distinction between historical array reconciliation and full model reproduction. Candidate outcome labels are not final training truth; read `docs/DATASET_BUILD_20261001.md` before using them. Never reuse upstream 48-hour roles as new 72-hour experiment roles without an explicit split audit.

Read `docs/EVENT_RECONCILIATION_20261001.md` before relying on inherited 48-hour labels or interpreting their cross-cycle evaluation. Source-supported correction proposals are versioned separately and have not been applied to upstream experiments. Do not report a repaired dataset, revised model metrics or complete negative-label coverage from the bounded positive-evidence audit alone.

The current candidate outcome version is documented in `docs/REGION_ADJUDICATION_20261001.md`. It uses conservative daily-source region decisions and retains unresolved associations as unknown. Preserve class-regime distinctions, original fields and earlier versions; do not merge reference-only region proposals into this version. A full computational verification receipt does not certify continuous event coverage, historical availability or final training eligibility.
