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

This initial scaffold has no executable model implementation or test suite. Validate documentation links and JSON when editing the scaffold; add meaningful tests alongside scientific code when it is introduced.
