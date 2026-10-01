# Gray-Box execution plan

Updated 1 October 2026. This is the working research plan, not a frozen experimental protocol or a claim of completed model evaluation.

## The contribution to test

Can an availability-aware forecast policy identify when to issue a normal forecast, use a validated degraded mode, or abstain, while making the resulting missed-flare and false-alarm burden explicit under temporal shift and realistic outages?

The existing ASR paper supplies temporal SHARP prediction. The separate AIA project supplies multimodal forecasting and already includes calibration, uncertainty and robustness work. Gray-Box must add a demonstrably useful decision policy and its evaluation. Merely adding AIA, calibration or dropout is insufficient justification for another paper. Reuse upstream artifacts with attribution; consolidate the paper scope if the operational study adds no distinct evidence.

Start with one complete 72-hour SHARP experiment. Add matched AIA-only and fusion branches after their **72-hour** target data and models exist. Existing 48-hour weights/results are not 72-hour forecasts. Keep 24/3-hour extension behind the successful completion of the first horizon. Met Office feedback can refine operational requirements later; it is not a prerequisite for the initial public-data study.

## Work in order

| Stage | Concrete work | Evidence required to advance | Current position |
|---|---|---|---|
| 1. Recover the baseline | Hash original data and notebook, reconstruct candidate sequence membership, reconcile table/split provenance, locate saved probabilities/scaler/checkpoints. | Versioned baseline manifest with case IDs and exact model/score lineage. | Cohort and saved headline probabilities reconciled; original case IDs and backbone weights remain unresolved. |
| 2. Build operational cases | Resolve physical region identities and time scales; define daily issue times, maximum permissible history gaps, label convention, reporting delay and source availability. Preserve missing-input cases. | Audited master-case table, event catalogue coverage, immutable split manifests. | Specification in progress; historical CSV cannot establish these on its own. |
| 3. Fit a compact SHARP baseline | Use an interpretable reference and the justified temporal backbone; fit preprocessing/model only on earlier development data. | Reproducible per-case raw scores on distinct calibration, policy and evaluation blocks. | Not executed here. |
| 4. Compare reliability methods | Uncalibrated reference, Platt, isotonic; uncertainty baselines and classification conformal candidate described below. | Earlier-data selection with locked method versions and probability/coverage diagnostics. | Not executed here. |
| 5. Freeze the decision policy | Input eligibility, normal/degraded/abstain rules, validated fallback, reason codes and thresholds; compare against simpler policies. | Signed-off numerical configuration and dated hashes before confirmatory predictions. | No numerical thresholds chosen. |
| 6. Evaluate decisions | Historical rolling replay, real or documented simulated outages, grouped sensitivities, and future logged forecasts when feasible. | All-case and accepted-case evaluation with dependence-aware intervals and honest evaluation history. | Not executed here. |

The executed cohort and prediction audit is documented in [BASELINE_72H_AUDIT.md](BASELINE_72H_AUDIT.md). Preserve the old experiment; source corrections produce a new baseline version and new results.

## Initial design choices and unresolved fields

- **Target:** combined same-region M/X occurrence within `(t, t+72h]`. Event start time is the proposed new-label convention; verify catalogue semantics before adopting it. Do not silently reinterpret `label_MX_3d`.
- **Cadence:** daily 12:00 UTC is the initial design candidate, matching the daily scale of the SHARP study. Existing CSV values at 12:00 are timezone-naive and do not establish UTC or delivery time.
- **Physics-informed layer:** the verified SHARP parameters and their physical interpretation, consistent with the supervisor's direction. No separate physical simulator is required for this stage.
- **Information cutoff:** observation end and actual availability must precede issuance. Where historical availability is unknown, report explicitly assumed-latency retrospective replay; do not call it demonstrated operational availability.
- **Roles:** ordered development/model selection, probability-calibration, conformal-calibration if used, policy-validation, then evaluation. Select probability-calibration method on earlier internal validation, leaving the final conformal block untouched by selection. Labels may enter any fit/update only after the outcome window and reporting delay have matured.
- **Splits:** choose explicit timestamp boundaries after support profiling; never split equal issue times by row count. Purge earlier outcome windows that extend past the next role's information cutoff. Additional region/event-disjoint sensitivities require actual reconciled identities and event IDs, not only chronological order.
- **New evaluation:** the already-inspected 2021–2026 AIA periods can support retrospective analyses, but cannot establish untouched confirmation of policies motivated by their failures. Reserve a genuinely uninspected period or log future forecasts after freezing. Historical rolling CV is already present in the ASR paper; it is not itself a new contribution.
- **Horizon follow-up:** the accepted AIA extension ends on 17 August 2026. For forecasts issued near that date, a new 72-hour target needs catalogue completeness through the corresponding times on 20 August, or an earlier issue cutoff. The existing 48-hour target is insufficient.

Exact dates, gap tolerance, label availability delay, permitted source products and numerical policy thresholds remain unresolved. Do not label the experiment frozen until these fields and case manifests exist.

## Bounded uncertainty study

Start with **calibrated probability plus predictive entropy** as a simple comparator, and **split conformal classification** as the first set-valued uncertainty candidate. Use separate earlier fitting blocks. Compare MC dropout only where a suitable frozen upstream model exists; consider a small deep ensemble only if it answers a residual question and compute is justified.

For binary occurrence, conformal outputs are sets drawn from `{no flare, flare}`. Record coverage overall and by true class, singleton/ambiguous/empty-set rates, support and performance by horizon/time regime. Rare positive events can be hidden by good overall coverage. Ordinary split-conformal guarantees rely on exchangeability; observed cross-cycle/time-series performance is empirical evidence, not an automatic coverage guarantee. Consider class-conditional or time-adaptive extensions after the baseline and their assumptions are specified. See [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511).

Conformalized quantile regression belongs to a separately defined continuous target, such as flare magnitude; it does not automatically give a valid confidence interval for the binary flare probability. It is outside this first implementation. A 2026 [solar-flare regression study](https://arxiv.org/abs/2603.06712) already investigates conformal, quantile and conformalized quantile methods. Earlier [probabilistic flare forecasting](https://arxiv.org/abs/2308.15410) already uses calibrated ensembles. These establish relevant overlap; they are not an exhaustive novelty review. Complete the closest-work comparison, including [ARCAFF outputs](https://cordis.europa.eu/project/id/101082164/results), before making a novelty claim.

## Comparisons that answer the operational question

1. Always issue the eligible calibrated baseline forecast.
2. Quality/availability checks only, with a separately validated fallback.
3. The same checks plus probability-entropy selection.
4. The same checks plus the selected uncertainty/OOD policy.

Use the same backbone, intended issue population and outage realizations. Develop thresholds on earlier policy-validation data; do not enforce a desired retained fraction by sorting future test scores. Compare actual retained coverage and the entire policy-validation-selected operating curve. Training a fallback and testing it under missingness are necessary before its use; a masked fusion branch is not automatically a trained single-modality model.

For probabilities, report Brier/Brier skill against a past-fitted climatology, log loss, reliability with bin support, calibration slope/intercept and declared ECE bins. Report TSS and **PR-AUC (average precision/AP)** together, with prevalence, ROC-AUC, HSS, precision, recall, F1 and confusion counts at frozen thresholds.

For decisions, report normal/degraded/abstain fractions, accepted-case error and flare recall, fallback outcomes, false alerts, and positive windows/events among abstentions. Show the full eligible-population denominator; abstentions cannot count as correct negative forecasts. Event-level missed-flare rates require an explicit event-to-issued-forecast mapping. Utility claims require a declared false-alarm/missed-event/abstention cost model; until stakeholder costs are agreed, show the trade-offs rather than declaring an optimal policy.

Use paired region-component resampling for model/policy differences and a temporal-block sensitivity, with the scheme frozen from development support. Report positive component/event counts and undefined metrics. These intervals describe evaluation sampling uncertainty; they do not replace training-seed variability. Compare supported outage states including contiguous time blocks and correlated channels; give sources for outage distributions or label them stress scenarios.

## Coordination and the next deliverable

The AIA project owns its data acquisition, corrected sixteen-feature training and required GOES experiments. This project consumes versioned outputs and does not launch duplicate GPU training. The next deliverable here is **the reconciled 72-hour baseline manifest and a new operational case contract**, followed by calibration. Checkpoint access and verified label/source-time provenance are needed before claiming model reproduction.
