# Manuscript V2 — working draft

## Working title

**A Gray-Box Trust Framework for 72-Hour Solar Flare Forecasting Under Temporal Shift, Uncertainty Degradation, and Data Failure**

## Central contribution

This paper is not positioned as a fusion-superiority paper. Its central contribution is an experimentally validated operational trust framework for deciding when a solar-flare forecast should be issued normally, degraded to a robust fallback, or withheld under temporal shift, uncertainty degradation, and realistic input failures.

The framework combines learned SHARP and AIA predictors with calibration, conformal uncertainty, statistical agreement signals, an explicit interpretable magnetic-state applicability monitor, data-quality/provenance checks, and a frozen NORMAL / DEGRADED / ABSTAIN routing policy.

---

## Abstract — draft v1

Reliable solar-flare forecasting requires more than high discrimination under a fixed retrospective test set. Operational systems must remain useful when probability calibration drifts, uncertainty estimates degrade, input modalities fail, and the physical feature distribution moves away from the development regime. We present a Gray-Box trust framework for 72-hour M/X-class solar-flare forecasting that combines frozen SHARP and AIA predictors with calibration, conformal uncertainty, statistical agreement measures, an interpretable magnetic-state applicability layer, and an explicit NORMAL / DEGRADED / ABSTAIN routing policy. All model fitting, calibration, conformal estimation and numeric trust thresholds were restricted to earlier time-ordered support. Cycle-25 diagnostic evidence exposed a failure mode in an earlier abstention design and motivated the final fallback-first routing semantics; the resulting v3 policy was then frozen without new numeric tuning before the supplementary-2026 evaluation.

Across regimes, predictive reliability degraded despite increasing model agreement. For SHARP, TSS declined from 0.689 on earlier Cycle-24 development support to 0.541 on Cycle-25 evaluation and 0.323 in 2026. Equal-weight SHARP–AIA fusion did not provide universal predictive gains and was significantly worse than SHARP in Cycle-25 TSS and average precision under paired active-region block bootstrap. A simple 16-feature latest-state SHARP logistic baseline was also competitive, confirming that the contribution is not raw predictive superiority. Conformal flare coverage degraded substantially in later regimes, while entropy, ensemble spread and SHARP–AIA disagreement became smaller, showing that internal statistical agreement can increase even as rare-event reliability worsens.

The frozen routing policy preserved forecast availability during realistic AIA acquisition failures by degrading to SHARP rather than relying automatically on an unstable AIA-only fallback. AIA outage analysis showed clustered empirical failure episodes, while AIA-only fallback performance deteriorated sharply in 2026. The explicit physical layer revealed a gradient-family-dominated shift in the locally stored SHARP feature distribution that was not detected by statistical confidence measures; matched same-record JSOC queries showed exact parity between the 16 SHARP summary keywords in the CCD and CEA series, although the local solar-versus-pipeline origin of the April 2026 shift remains unresolved. Physical applicability was therefore retained as an independent monitoring dimension rather than a direct abstention rule.

These results support a trust-oriented view of operational flare prediction: calibration, uncertainty, physical applicability, provenance, and graceful degradation should be treated as complementary system properties rather than inferred from predictive confidence alone.

---

## Core claim set for manuscript

1. **Operational trust contribution.** The main contribution is a reproducible issue/degrade/withhold framework under temporal shift and realistic AIA failures.
2. **Reliability paradox.** Statistical agreement tightened while rare-flare reliability and conformal coverage deteriorated.
3. **Fallback finding.** SHARP preserved availability under AIA failures; frozen AIA-only fallback was not stable enough for automatic use.
4. **Physical applicability finding.** An interpretable magnetic-state monitor identified a gradient-dominated distribution shift that was not reflected by statistical agreement, but physical distance was not a valid case-level abstention score.
5. **Baseline finding.** A simple latest-state SHARP logistic was competitive; temporal SHARP showed only a modest Cycle-25 TSS advantage, and equal-weight fusion was not universally superior.
6. **Claim boundary.** The work does not claim MHD/PINN physics, universal predictive superiority, true live prospective validation, or a causal solar-cycle explanation of the 2026 gradient shift.

---

## 1. Introduction

Solar flares are among the most consequential manifestations of solar magnetic activity. Their impulsive radiative output can disturb the near-Earth environment on timescales too short for mitigation after onset, which makes reliable forecasting a central objective of operational space weather. Yet the forecasting problem remains unusually difficult: major flares are rare, the active-region population evolves over the solar cycle, multiple observations can be unavailable or degraded, and forecast users require more than a binary measure of discrimination. They need probabilities that remain interpretable, evidence that a model is operating within a domain it understands, and a defined response when part of the forecasting system fails.

Solar-flare prediction has consequently developed along several complementary directions. Statistical approaches based on flare occurrence history established early probabilistic baselines and emphasized that more complex methods should demonstrate skill beyond simple climatological or persistence information (Wheatland, 2005). The availability of continuous SDO/HMI vector magnetic-field observations then enabled machine-learning approaches based on physically motivated active-region parameters, with Bobra and Couvidat (2015) showing that a comparatively small subset of SHARP-derived quantities can provide substantial discrimination for M/X-class events. Subsequent work introduced temporal neural models that explicitly encode active-region evolution rather than relying on a single magnetic snapshot (Liu et al., 2019), while benchmark resources such as SWAN-SF helped standardize multivariate time-series data, flare association and comparison protocols (Angryk et al., 2020). More recent studies have expanded the input space to UV/EUV imagery, video and multimodal observations, reflecting the expectation that flare-relevant information is distributed across several layers of the solar atmosphere (Francisco et al., 2024; Riggi et al., 2026).

Increasing predictive complexity, however, does not remove the need for rigorous verification. Community comparisons have repeatedly shown that apparent gains depend strongly on the event definition, test population, verification metric and evaluation protocol, and that no single forecasting method dominates under all conditions (Barnes et al., 2016; Leka et al., 2019a,b). Ensemble forecasting can improve some attributes of a forecast, but the benefit depends on member bias, combination strategy and decision threshold (Guerra et al., 2015). This is directly relevant to multimodal systems: combining two informative branches does not guarantee that the resulting probability is better calibrated, more robust under temporal shift, or more useful when one modality becomes unavailable.

The distinction between discrimination and operational reliability is especially important for probabilistic forecasts. Solar-flare forecast verification has long used reliability diagrams, Brier scores and Brier skill scores to assess whether stated probabilities correspond to observed frequencies (Crown, 2012). A recent 26-year verification of NOAA/SWPC forecasts further demonstrated that operational forecasts can exhibit substantial calibration and false-alarm problems and may fail to outperform simple statistical baselines on important metrics (Camporeale & Berger, 2025). These results reinforce two requirements for data-driven flare prediction: strong models should be compared with simple baselines, and categorical skill alone is insufficient evidence of trustworthy probabilistic behaviour.

Operational evaluation also changes the scientific question. Nishizuka et al. (2021) demonstrated the value of chronological and genuinely operational testing through Deep Flare Net, while Goodwin et al. (2024) showed in simulated real-time experiments that forecast behaviour is influenced by training-window design and solar-cycle conditions. Large research-to-operations efforts such as FLARECAST have likewise emphasized rigorous training/testing practice and the gap between retrospective modelling and deployable forecasting systems (Georgoulis et al., 2021). The recent ARCAFF programme extends this trajectory toward multimodal time-series forecasting with explicit forecast uncertainties. Thus, temporal evaluation, multimodality and uncertainty estimation are individually active areas of research rather than unexplored components.

A remaining practical challenge is how these components should interact when evidence about forecast trust is mixed. A model can be discriminative while poorly calibrated; an ensemble can become internally more consistent while rare-event coverage degrades; a modality can fail even when another remains available; and an input can move away from the physical development distribution without implying that the individual forecast is wrong. Uncertainty quantification alone does not resolve these cases. Conformal methods, for example, provide an important framework for set- or interval-valued uncertainty and have recently been explored for solar-flare regression (Hong et al., 2026), but their empirical reliability still depends on the relationship between calibration and deployment data. Under temporal dependence and distribution shift, coverage itself becomes an object that must be monitored rather than assumed to remain fixed.

This work addresses that systems-level problem for active-region M/X-class forecasting at a 72-hour horizon. We use the term **Gray-Box** in a deliberately restricted sense. The forecasting branches remain learned data-driven models; we do not solve magnetohydrodynamic equations or impose a physics-informed neural-network loss. The “gray” component is an explicit, interpretable magnetic-state layer derived from physically meaningful SHARP families, combined with statistical uncertainty, provenance and data-quality evidence. These signals are not collapsed into a single confidence score. Instead, they are evaluated as distinct trust dimensions around a frozen operational state machine.

The framework uses three operational states. **NORMAL** issues the multimodal forecast when the required branch and uncertainty conditions are satisfied. **DEGRADED** retains forecast availability through the SHARP branch when AIA information is unavailable or the multimodal trust conditions are not satisfied. **ABSTAIN** withholds the forecast when the SHARP fallback itself is unavailable. The final state semantics require an important chronology distinction. Earlier numerical thresholds were estimated only from designated development/calibration roles. Cycle-25 diagnostic evaluation of preceding policy versions showed that abstention disproportionately removed flare-producing cases, which motivated the fallback-first v3 redesign. No new numerical trust threshold was tuned on Cycle 25, and supplementary-2026 outcomes were not opened during the v3 freeze. Consequently, Cycle-25 evidence for v3 is post-hoc/diagnostic, whereas supplementary 2026 provides the clean future evaluation of the frozen v3 policy. The purpose is therefore not to demonstrate that fusion is the best predictor. It is to test whether a forecasting system can degrade gracefully and expose changes in reliability rather than silently treating every probability as equally trustworthy.

The principal contributions are:

1. **A frozen issue/degrade/withhold trust policy for 72-hour active-region flare forecasting.** Calibration, conformal uncertainty, statistical agreement and data availability are connected to explicit NORMAL, DEGRADED and ABSTAIN behaviour rather than reported only as separate diagnostics.
2. **A cross-regime audit of predictive reliability.** We evaluate discrimination, calibration and conformal coverage across earlier development support, later Cycle-25 data and a supplementary 2026 extension, while preserving the distinction between development, retrospective evaluation and post-hoc evidence.
3. **Realistic modality-failure experiments.** Empirical AIA acquisition incidents are reconstructed as temporally clustered outages, allowing direct evaluation of availability, fallback behaviour and the consequences of losing the image branch.
4. **An explicit physical-applicability monitor.** A nonredundant magnetic-state representation provides an interpretable development-domain reference. Its role is diagnostic: physical applicability is kept separate from case-level predictive uncertainty and does not automatically trigger abstention.
5. **A same-support baseline and uncertainty analysis.** Frozen SHARP, AIA and fusion forecasts are compared with training climatology and a simple 16-feature latest-state logistic model using paired active-region block bootstrap intervals, preventing the operational contribution from being confused with a claim of universal predictive superiority.
6. **A prospective-style chronological replay with explicit claim boundaries.** Past-only matured outcomes are used to replay the frozen framework through later time, while the study is explicitly distinguished from a live prospective deployment.

The resulting evidence leads to a deliberately narrower but operationally important conclusion: trustworthy flare forecasting cannot be inferred from discrimination or internal model agreement alone. Calibration, uncertainty coverage, modality availability, physical applicability and provenance can evolve differently, and a forecasting system should expose those differences in its behaviour.

---

## 2. Related Work

### 2.1 Magnetic predictors and temporal active-region modelling

Photospheric magnetic structure remains one of the dominant information sources for active-region flare prediction. Early statistical and knowledge-based approaches established the value of flare history and magnetic complexity, while the SDO era enabled standardized vector-magnetic measurements at scale. Bobra and Couvidat (2015) used HMI vector-magnetogram parameters with a support-vector machine and showed that M/X-class flare discrimination could be achieved with a relatively small number of magnetic predictors. Their work also reinforced the importance of extensive quantities related to current, free-energy proxies, magnetic flux and polarity-inversion-line structure.

A limitation of snapshot forecasting is that an active region is an evolving magnetic system. Liu et al. (2019) addressed this directly with an LSTM operating on magnetic and flare-history time series. The SWAN-SF benchmark subsequently provided a curated multivariate SHARP time-series resource with cross-checked flare labels and explicit discussion of partitioning, temporal slicing and sampling (Angryk et al., 2020). Our earlier evolution-aware work likewise showed that temporal magnetic complexity can support multi-horizon M/X-class forecasting (Akinwumi et al., 2026). The present study therefore treats temporal SHARP prediction as an existing component, not as the new contribution. Its role here is to provide a strong frozen magnetic branch whose reliability and operational behaviour can be audited under later conditions.

The baseline analysis in this manuscript is motivated by the same literature. A high-capacity temporal model should not be assumed useful merely because it is more complex. Wheatland (2005) explicitly argued for simple objective baselines, and Camporeale and Berger (2025) recently showed how strongly conclusions about operational flare forecasts can change when climatology, persistence and lightweight statistical models are included. We therefore compare the temporal SHARP branch against training climatology and a fixed 16-feature latest-state logistic model on exactly the same evaluation support.

### 2.2 Imaging, ensembles and multimodal forecasting

Magnetograms provide direct information about the photospheric magnetic field, but flare initiation and pre-flare activity can also be expressed through chromospheric and coronal emission. This motivates imaging and multimodal approaches. Francisco et al. (2024) examined full-disk magnetogram and UV/EUV inputs with deep learning and reported that coronal wavelengths can provide complementary discriminatory information. Riggi et al. (2026) compared foundation-model approaches across image, video and time-series modalities, illustrating the continuing expansion of flare forecasting beyond single-source tabular predictors.

The idea of combining forecasts is also well established. Guerra et al. (2015) linearly combined several probabilistic flare forecasts and found that ensemble performance could improve for selected thresholds and combinations, while also showing that gains depended on the properties of the constituent methods. FLARECAST later evaluated a large set of physical predictors and machine-learning approaches within a research-to-operations programme (Georgoulis et al., 2021). ARCAFF has more recently targeted point-in-time and time-series multimodal forecasting together with forecast uncertainties.

These studies motivate the inclusion of AIA and SHARP branches in the present work but also constrain our claim. The equal-weight SHARP–AIA probability used here is not presented as a novel fusion architecture. Its weight is fixed rather than optimized on later data, and our own results show that it is not universally superior to SHARP alone. Fusion is instead useful as an operational test case: it allows us to ask what happens when two modalities agree, disagree, drift differently, or become unavailable.

### 2.3 Forecast verification, calibration and operational benchmarks

Forecast quality is multidimensional. A categorical forecast can achieve useful discrimination while its probabilities remain poorly calibrated, and a model with a high skill score at one threshold can be unsuitable for users requiring a different trade-off between missed events and false alarms. Crown (2012) evaluated NOAA/SWPC flare probabilities using Brier skill and contingency-table measures, providing an early operational example of probabilistic verification. Guerra et al. (2015) likewise evaluated ensemble accuracy, reliability and resolution rather than relying on one categorical score.

Community benchmark studies made the comparison problem more explicit. Barnes et al. (2016) demonstrated that flare forecasting methods are difficult to compare when data, event definitions and metrics differ; on common support, no method clearly dominated. Leka et al. (2019a) extended this approach to operational systems using common benchmarks and multiple metrics, again finding no universal winner. Leka et al. (2019b) then examined systematic implementation behaviours, including the influence of prior flare activity, active-region evolution and operational choices. These findings are central to our evaluation philosophy: results are reported on fixed support with several categorical and probabilistic metrics, and claims are limited to the regimes and service conditions actually tested.

The importance of simple baselines has recently been reinforced by Camporeale and Berger (2025), who assessed SWPC forecasts across 1998–2024 and found major calibration and false-alarm limitations relative to inexpensive statistical comparators. Our same-support logistic experiment follows this principle. It is intentionally simple, uses only the latest available SHARP state, fits on the frozen training period, and selects its decision threshold only on the model-validation role. The objective is not to introduce another optimized competitor but to test whether the more elaborate forecast branches provide material evidence beyond a transparent baseline.

### 2.4 Uncertainty quantification and conformal reliability

Uncertainty in flare prediction has several meanings that should not be conflated. Probability calibration asks whether forecast probabilities correspond to empirical event frequencies. Ensemble spread measures sensitivity across fitted models or seeds. Cross-modal disagreement measures consistency between information sources. Conformal prediction asks whether prediction sets or intervals achieve a target coverage under the assumptions of the conformal procedure. None of these quantities is automatically equivalent to the probability that a specific forecast is wrong.

Uncertainty-aware solar-flare research is now emerging directly. Hong et al. (2026) compared conformal prediction, quantile regression and conformalized quantile regression for flare-regression intervals. ARCAFF also explicitly includes forecast uncertainties among its intended products. Our study therefore does not claim novelty for uncertainty quantification or conformal prediction itself. Instead, we examine how these signals behave when a frozen classification system is moved into later temporal regimes.

This distinction matters because conformal guarantees are tied to assumptions about the calibration and future data relationship. Solar active-region forecasts are temporally dependent, and the deployment population changes with the solar cycle and with observational conditions. We consequently treat conformal coverage empirically: later loss of coverage is not interpreted as a failure of conformal theory but as evidence that the calibration regime no longer represents the later rare-event population sufficiently well for the nominal behaviour to persist.

### 2.5 Temporal shift, simulated operations and live forecasting

Chronological evaluation is essential for a problem whose data-generating process changes over time. Nishizuka et al. (2021) deployed Deep Flare Net operationally and emphasized chronological splits and one-pass operational evaluation. Goodwin et al. (2024) performed simulated real-time experiments with stationary, rolling and expanding windows and showed that forecasting behaviour is affected by solar-cycle conditions. FLARECAST similarly emphasized rigorous pre-operational testing and research-to-operations practice.

Our rolling experiment belongs to this family but has a narrower evidential status. The predictor and policy are frozen, forecast issues are replayed chronologically, and only outcomes that would have matured by each replay time are admitted to the running evidence state. However, the forecasts were not generated in a live service at the historical issue times. We therefore use the term **prospective-style chronological replay** and do not describe the study as a genuine prospective deployment.

Temporal shift also motivates the physical-applicability layer introduced here. A model can encounter magnetic states farther from its development reference even when its own probability outputs become more decisive. We represent this with interpretable SHARP feature families and a training-reference distance, but deliberately separate applicability from forecast error. Our experiments show why: physically distant cases are not monotonically the cases with the largest prediction error. The applicability layer is thus closer to a domain/provenance monitor than to a direct selective-classification score.

### 2.6 Positioning of the present work

The literature already contains strong examples of magnetic machine learning, temporal neural networks, multimodal imaging, ensemble forecasting, operational verification, chronological evaluation, calibration and uncertainty quantification. The contribution of this paper is therefore **not** any one of these components in isolation.

Instead, we study how they behave together when the forecast system is required to make an operational decision under imperfect conditions. The proposed Gray-Box framework couples frozen learned predictors to four forms of evidence: probabilistic reliability, uncertainty/agreement, physical applicability, and data availability/provenance. These are then connected to a predeclared state machine that can issue normally, degrade to the magnetic fallback, or withhold when that fallback is unavailable. The experiments deliberately include negative findings: AIA-only fallback is not sufficiently stable for automatic use, equal-weight fusion is not universally better than SHARP, physical distance is not a monotonic error score, and increasing model agreement does not guarantee preserved rare-flare reliability.

This positioning is intentionally different from a state-of-the-art forecasting claim. The scientific question is whether operational trust can be made explicit, testable and reproducible around a frozen flare-prediction system. The answer is assessed through cross-regime calibration and coverage, realistic modality failures, active-region-level uncertainty intervals, physical/provenance diagnostics and a chronological replay of the resulting policy.

---

## 3. Data and Frozen Evaluation Design

### 3.1 Forecasting unit, horizon and issue cadence

The unit of analysis is an active-region forecast case defined by a HARP-linked solar active region at a specific issue time. The primary task is binary occurrence prediction: whether at least one region-associated M- or X-class flare begins in the interval (t, t + 72 h], where t is the forecast issue time. The native issue grid has a nominal cadence of 96 min. All retained SHARP and AIA histories precede the issue time and therefore contain no measurements from within the 72 h outcome window.

The Gray-Box candidate inventory spans 21 May 2010 to 17 August 2026 and contains 153,366 candidate forecast records. Of these, 113,433 are linked to the accepted three-history AIA-SHARP input package. The matched input population comprises 55,871 Cycle-24 cases, 45,403 primary Cycle-25 cases and 12,159 supplementary-2026 cases. The unmatched candidate records remain in the broader provenance inventory; they are not interpreted as operational outages because unmatched status can also reflect protocol exclusions or inputs that were never assembled.

GOES/NOAA event information is used to construct future outcomes, not as a continuous forecasting input. Event start time, rather than peak time, defines entry into the 72 h outcome window. Region association was reconciled against the available event/source lineage, with source-supported corrections versioned explicitly and unresolved associations preserved rather than silently converted to non-events. The primary target uses the reconciled primary-region scope. Cases whose negative status could not be supported because of unresolved region/event information are retained as unknown and excluded from supervised fitting and headline evaluation.

### 3.2 SHARP magnetic input

The magnetic branch uses 16 SDO/HMI SHARP quantities in the exact frozen order: MEANGBZ, MEANGAM, MEANGBT, MEANGBH, MEANJZD, TOTUSJZ, MEANALP, MEANJZH, ABSNJZH, SAVNCPP, MEANSHR, SHRGT45, R_VALUE, USFLUX, TOTPOT and TOTUSJH.

For each case, the aligned SHARP tensor contains three histories at issue time minus 288, 192 and 96 min, giving a frozen tensor of shape 113,433 × 3 × 16. The latest magnetic observation available to the forecast is therefore the t-96 min state. The accepted tensor contains no missing numerical values.

The frozen SHARP predictor is a one-layer GRU with 32 hidden units and a linear binary-output head. Three independently seeded models (17, 29 and 43) are retained, and the branch probability is their arithmetic mean. Seed selection used only the earlier model-validation block. The purpose of the present manuscript is not to re-establish the predictive value of temporal SHARP modelling; the trained branch is treated as a fixed forecasting component around which reliability and operational behaviour are evaluated.

### 3.3 AIA image input

The image branch is a separately trained temporal AIA model operating on three pre-issue image histories and six image channels. Images are resized to 256 × 256 for model input after a training-derived asinh/scale normalization. The branch uses a temporal CNN-GRU architecture trained from scratch for the 72 h task, again retaining three seeds (17, 29 and 43) and averaging their probabilities.

AIA and SHARP are kept as distinct branches because they fail differently. The image branch depends on acquisition, preprocessing and image availability, whereas the magnetic branch supplies the operational fallback. This separation is essential for the later outage experiments and for distinguishing cross-modal disagreement from branch unavailability.

### 3.4 Time-ordered role partition

All modeling roles follow a fixed chronological partition.

| Role | Date interval | Function |
|---|---|---|
| train | 2010-01-01 to 2013-12-31 | fit SHARP/AIA models and training-only transforms |
| model_validation | 2014-01-01 to 2014-06-30 | model/seed selection and alarm-threshold selection |
| probability_calibration | 2014-07-01 to 2014-12-31 | probability-calibration development |
| conformal_calibration | 2015-01-01 to 2015-06-30 | class-conditional conformal estimation and trust-cutoff reference |
| policy_validation | 2015-07-01 to 2019-12-31 | earlier policy evaluation |
| retrospective_cycle25 | 2021-01-01 to 2025-12-31 | later temporal-regime diagnostic/evaluation |
| supplementary_2026 | 2026-01-01 onward | frozen v3 future evaluation and later descriptive audits |

The final matched 72 h schedule contains 96,596 labeled forecast cases: 25,586 train; 3,905 model validation; 2,831 probability calibration; 4,168 conformal calibration; 13,142 policy validation; 35,846 retrospective Cycle 25; and 11,118 supplementary 2026. Cases outside these matched/labeled supports remain in the provenance inventory but do not enter model fitting or headline evaluation.

The 2020 gap is deliberate: candidate records exist for 2020, but no accepted input package is available in the three frozen matched cohorts, so the year is not silently interpolated into either development or evaluation.

### 3.5 Evidence chronology and leakage control

The chronological roles are necessary but not sufficient to describe the evidential status of the final policy. Model fitting, normalization, probability calibration, conformal estimation and numerical trust cutoffs are all restricted to earlier roles. Alarm thresholds for SHARP, AIA and equal-weight fusion are selected on model_validation using a deterministic rule: maximum TSS, then maximum HSS, then maximum recall, and finally the lower threshold in the event of a remaining tie.

The final fallback-first v3 routing semantics were not conceived before all Cycle-25 evidence was seen. Earlier v1/v2 policies used uncertainty more aggressively for abstention; Cycle-25 diagnostic results showed that this removed a disproportionate share of flare-producing cases. The v3 redesign therefore changed the decision semantics, not the numerical cutoffs: uncertainty that prevents NORMAL multimodal issuance routes to the SHARP fallback whenever SHARP remains technically available. Cycle 25 is consequently treated as development/diagnostic evidence for v3, not untouched confirmation. Supplementary-2026 outcomes were not read during the v3 freeze and form the clean future evaluation of that policy. Later Phase-H physical-layer analyses use Cycle-25 and 2026 outcomes post hoc and are labeled accordingly.

Active-region leakage was checked using the reconciled region_component_id. Mapped region-component overlap with the training set is zero for the evaluated chronological roles. Repeated 96-min forecasts from the same active region remain statistically dependent, so uncertainty intervals for headline model comparisons use active-region block resampling rather than treating forecast windows as independent observations.

### 3.6 Frozen provenance

The main 72 h artifacts are hash-pinned. The canonical 71,010-row non-training multimodal prediction table has SHA-256 17300093b88b7c61ecadf5eb28acba0d6d023b340d7740807852dd4352b22b46; the 113,433-row frozen SHARP prediction ledger has SHA-256 7368c002a978321cdc67221564f3b7901687badc7aba28ab9df615179d1e017b; and the exact recovered SHARP tensor has SHA-256 c9921be7b7fe381b43436a0e927d15d84de83e15376025ae0d8a79a6ef917639.

These hashes define the source objects used by the post-freeze reliability, physical-applicability and baseline analyses. Later diagnostic work does not overwrite the original A-G evidence chain.

---

## 4. Gray-Box Trust Framework

### 4.1 Design principle

The framework separates forecast generation from forecast trust. SHARP and AIA provide learned occurrence probabilities. Around those probabilities, the Gray-Box layer asks four different questions:

1. Is the branch probability calibrated on earlier data, and does that calibration remain plausible later?
2. Do conformal sets, seed disagreement, predictive entropy and cross-modal disagreement indicate ambiguity?
3. Does the magnetic state resemble the development-domain magnetic states represented in training?
4. Are the required inputs technically available and traceable?

These questions are deliberately not collapsed into one score because they can move in different directions under temporal shift.

### 4.2 Frozen forecast branches

Let p_S denote the frozen SHARP three-seed mean probability and p_A the corresponding AIA probability. The multimodal branch is a prespecified equal-probability average,

p_F = 0.5 p_S + 0.5 p_A.

The weights are fixed and are not selected from Cycle-25 or 2026 outcomes. This branch is therefore a controlled multimodal comparator rather than an optimized stacking model.

A separate same-support baseline analysis fits one L2-regularized logistic regression to the 16 latest-slot SHARP features using the training role only. That comparator does not enter the Gray-Box state machine; it exists solely to determine whether the frozen temporal branches demonstrate evidence beyond a transparent latest-state baseline.

### 4.3 Probability calibration

Three probability-calibration candidates were defined on the earlier probability-calibration role: identity/raw probability, Platt-style calibration and isotonic calibration. Candidate choice was based on an internal earlier-data selection procedure using Brier score, with log loss and declared method order used as tie-breakers. The selected calibration for SHARP, AIA and equal-weight fusion was the identity/raw map for all three branches.

This apparently simple outcome is important methodologically: later probabilities are not retroactively remapped to improve Cycle-25 or 2026 calibration. Calibration deterioration is therefore measured against a genuinely frozen earlier-data choice.

### 4.4 Class-conditional conformal uncertainty

Binary conformal uncertainty is represented using Mondrian class-conditional calibration at alpha = 0.10. For each branch, separate nonconformity quantiles are estimated for the no-flare and flare classes on the conformal-calibration role. Each probability then produces one of four set states: singleton no-flare, singleton flare, doubleton, or empty set.

The conformal state is interpreted as a coverage/ambiguity diagnostic rather than a probability-of-error estimate. In particular, later empirical loss of flare-class coverage is treated as evidence of calibration-regime mismatch, not as a contradiction of conformal theory under exchangeability.

For the final v3 policy, SHARP's conformal state remains advisory during fallback. It does not block a technically available SHARP forecast. This change is deliberate: earlier policy versions showed that aggressive uncertainty-based abstention could improve apparent retained-case quality by preferentially discarding flare-producing windows.

### 4.5 Statistical trust signals

Three continuous multimodal signals are retained:

- fusion seed spread: the standard deviation across the three seed-wise equal-weight fusion probabilities;
- SHARP-AIA probability gap: the absolute difference |p_S - p_A|;
- fusion entropy: -p_F log(p_F) - (1 - p_F) log(1 - p_F).

Candidate cutoffs were defined on earlier support using empirical percentiles. The primary v3 policy uses the pre-existing q90 cutoffs; q80 and q95 are retained only as sensitivity definitions. These signals are used to decide whether multimodal fusion is sufficiently well-behaved for NORMAL issuance. They are not assumed to be universal domain-shift detectors.

### 4.6 Explicit magnetic-state applicability layer

To make the term Gray-Box scientifically explicit, Phase H adds a transparent magnetic-state representation alongside the black-box forecast branches. The 16 SHARP quantities are grouped a priori into five physically interpretable families:

| Family | SHARP quantities |
|---|---|
| magnetic flux / PIL complexity | USFLUX, R_VALUE |
| free energy / shear | TOTPOT, MEANSHR, SHRGT45 |
| current / helicity / twist | TOTUSJZ, TOTUSJH, ABSNJZH, SAVNCPP, MEANJZD, MEANJZH, MEANALP |
| field gradients | MEANGBZ, MEANGBH, MEANGBT |
| inclination / geometry | MEANGAM |

Feature transforms and robust standardization parameters are fitted on the training role only. Magnitude-like quantities use log1p scaling; declared signed current/helicity/twist quantities are represented through magnitude for the family state. Each feature is centered by the training median and scaled by the training IQR. Within each family and history slot, the robust standardized feature values are summarized by their median so that large families do not dominate simply because they contain more variables.

After removal of an algebraic redundancy discovered in the first Phase-H representation, the final nonredundant Magnetic Physical State Vector (MPSV) contains three quantities for each family:

latest = x(-96),

net change = x(-96) - x(-288),

curvature = x(-288) - 2 x(-192) + x(-96).

This yields 15 interpretable state/evolution axes.

A Ledoit-Wolf covariance model is fitted to the training MPSV only. Physical applicability is summarized by squared Mahalanobis distance from that training reference, with a descriptive out-of-reference flag at the training 99th percentile. A low-capacity logistic model on the same MPSV provides an interpretable physical-state probability and an optional diagnostic log-odds gap relative to the fusion probability.

Crucially, Phase-H evidence did not justify turning this distance into a rejection rule. Physical distance was negatively associated with case-level squared error, and the 2026 elevation was dominated by the latest magnetic-gradient family rather than a broad shift across all families. Matched same-record JSOC queries also showed exact equality of the 16 SHARP keywords between the CCD and CEA series on the tested pairs, while the local solar-versus-pipeline origin of the April-2026 gradient shift remains unresolved. The physical layer therefore remains an independent applicability/provenance monitor and does not modify v3 routing.

### 4.7 Operational state machine

The primary policy is fallback_v3_q90. Its three states are mutually exclusive.

**NORMAL.** Issue the fixed SHARP-AIA fusion when:
1. SHARP and AIA are technically available;
2. the fusion Mondrian state is a singleton;
3. fusion seed spread is at or below the frozen q90 cutoff;
4. the SHARP-AIA probability gap is at or below the q90 cutoff; and
5. fusion entropy is at or below the q90 cutoff.

**DEGRADED.** If NORMAL is false but the SHARP branch is technically available, issue the frozen SHARP forecast and explicitly mark the state as DEGRADED. SHARP conformal state and seed disagreement remain recorded diagnostics but do not block issuance.

**ABSTAIN.** If SHARP itself is technically unavailable, withhold the forecast. Automatic AIA-only issuance is not enabled.

This asymmetry is evidence-driven. Post-hoc feasibility testing showed that AIA-only performance and flare-class conformal coverage deteriorated sharply in the later regimes, especially in 2026. The system therefore prefers graceful degradation to the stronger magnetic branch and reserves abstention for true fallback unavailability.

### 4.8 Availability and provenance layer

AIA acquisition failures are not modeled as independent Bernoulli masks. Upstream acquisition/recovery records are mapped back to Gray-Box cases and grouped into empirical failure episodes using their temporal and active-region structure. This enables replay of realistic clustered image failures while leaving SHARP availability unchanged.

The provenance layer also records source hashes, case IDs, source series, role assignments and input-status reason codes. Its purpose is not merely reproducibility: it prevents a change in upstream data representation from being silently interpreted as model uncertainty or solar physics.

### 4.9 Rolling monitoring without policy feedback

For the prospective-style replay, the frozen v3 policy is evaluated chronologically at the declared 96-min cadence. A case outcome becomes visible to the monitoring process only after the 72 h forecast horizon plus a 24 h reporting-delay allowance, giving a 96 h maturity lag. At monitoring time t, only cases satisfying issue_time + 96 h <= t can contribute to the running reliability summaries.

The rolling monitor records state mix, calibration/coverage summaries and uncertainty behaviour, but it never changes the forecast model, recalibrates probabilities, retunes thresholds or redesigns the policy. The exercise is therefore a prospective-style chronological replay, not a live prospective deployment.

### 4.10 Gray-Box interpretation

The final system is “Gray-Box” because it places an interpretable magnetic-state/applicability layer and explicit operational logic around learned predictors. It is not gray because the neural models are partially replaced by first-principles magnetohydrodynamics.

In compact form, the trust stack is:

learned forecasts -> {calibration, conformal UQ, agreement, physical applicability, availability/provenance} -> {NORMAL, DEGRADED, ABSTAIN}.

The framework's central methodological choice is to keep these trust dimensions separable. A forecast can be statistically decisive yet physically out-of-reference; physically unusual yet easy to classify; multimodally uncertain yet still issuable through SHARP; or unavailable because of an acquisition failure. The state machine encodes those distinctions rather than treating one confidence score as a complete measure of trust.

---

## 5. Experimental Protocol

### 5.1 Evaluation philosophy and metrics

The evaluation separates discrimination, probabilistic reliability, uncertainty coverage, service availability and selective routing. No single metric is treated as a complete measure of forecast quality.

Categorical discrimination is summarized primarily by the True Skill Statistic (TSS), defined as recall + specificity - 1. Heidke Skill Score (HSS), recall, specificity, precision and F1 are retained where relevant. Probabilistic discrimination is summarized by average precision (AP) and ROC-AUC. Probability quality is assessed with Brier score, Brier Skill Score (BSS), log loss, expected calibration error (ECE), reliability bins, and calibration slope/intercept. BSS uses the fixed training prevalence as the reference forecast rather than refitting climatology in each later regime.

Conformal behaviour is evaluated through overall set coverage, flare-class coverage, no-flare coverage, singleton rate and set composition. These are empirical transfer diagnostics; nominal conformal coverage is not assumed to remain guaranteed under non-exchangeable temporal shift.

Operational evaluation additionally records availability, state mix, positive retention, and performance of the actually issued forecast source. This prevents apparent improvements that are obtained only by rejecting difficult flare-producing cases.

### 5.2 Calibration, alarm thresholds and conformal freeze

Probability calibration is developed only on the probability-calibration role. Raw, Platt-style and isotonic candidates are compared using an internal earlier-data procedure, after which the selected map is frozen. The final selection is the identity/raw probability for SHARP, AIA and fusion.

Binary alarm thresholds for the three frozen branches are selected on model_validation only. Candidate thresholds consist of all distinct observed probabilities together with 0 and 1. The deterministic selection rule is maximum TSS, followed by maximum HSS, maximum recall and finally the lower threshold. The frozen thresholds are 0.0752000 for SHARP, 0.0251018 for AIA and 0.0905993 for equal-weight fusion.

Mondrian class-conditional conformal thresholds are estimated on the conformal-calibration role at alpha = 0.10. The uncertainty cutoffs used by the trust policy are also referenced to earlier calibration support. The primary q90 policy and q80/q95 sensitivity definitions are carried forward without later numerical retuning.

### 5.3 Policy chronology and fallback-first redesign

The policy-development history is retained rather than hidden. Earlier policy versions used uncertainty more directly to abstain. Diagnostic evaluation on Cycle 25 showed that those designs disproportionately removed flare-producing windows, creating an unacceptable availability/positive-retention trade-off.

The final v3 policy therefore changes the routing semantics: uncertainty can prevent NORMAL multimodal issuance; if SHARP remains technically available, the system degrades to SHARP rather than abstaining; and ABSTAIN is reserved for genuine SHARP unavailability.

Cycle 25 informed this semantic redesign and is therefore post-hoc/diagnostic evidence for v3. Importantly, Cycle 25 was not used to select new numerical uncertainty thresholds. Supplementary-2026 outcomes remained unopened during the v3 freeze and were used once for the clean future evaluation of the frozen fallback-first policy.

### 5.4 Cross-regime reliability audit

The frozen SHARP, AIA and equal-weight fusion branches are compared across three evidence regimes: earlier Cycle-24 development support, retrospective Cycle-25 support and supplementary 2026.

The audit is read-only: no model fitting, recalibration, threshold tuning or policy redesign is permitted. The objective is to determine whether discrimination, calibration and conformal reliability move together or diverge under later temporal conditions.

### 5.5 Empirical AIA acquisition failures

AIA failure experiments use recorded upstream acquisition/recovery incidents rather than independent random masking. The engineering inventory contains 6,461 incident samples across 2025-2026, of which 2,290 map exactly to Gray-Box forecast cases.

Failures are grouped descriptively into empirical episodes within the same year, HARP and recovery class; a new episode begins when the inter-failure gap exceeds 192 min. This rule is deterministic and not selected from forecast outcomes. The resulting replay asks what the frozen v3 policy would do if the AIA branch were unavailable for those mapped cases while SHARP remained available.

These incidents are engineering acquisition/recovery events, not verified historical live-service outages. They are used to construct realistic correlated modality failures, not to estimate the true uptime of a deployed forecasting service.

### 5.6 AIA-only fallback feasibility

Because ABSTAIN occurs when SHARP is unavailable, a separate post-hoc experiment asks whether the frozen AIA branch could safely serve as an automatic fallback in that condition. No new threshold is introduced. The decision is based jointly on branch TSS, recall, AP, conformal flare coverage and singleton behaviour across earlier, Cycle-25 and 2026 regimes.

This analysis is intentionally conservative: a branch is not considered suitable for automatic degraded issuance merely because it returns a confident probability.

### 5.7 Prospective-style chronological replay

The final v3 policy is replayed in chronological issue order at the nominal 96-min cadence. The model, calibration maps, conformal thresholds and routing rules remain frozen.

At monitoring time t, an outcome is visible only if the 72 h forecast window and an additional 24 h reporting-delay allowance have elapsed. Thus a case can enter the monitoring history only when issue_time + 96 h <= t. The rolling monitor does not feed back into decisions.

Two replay scenarios are evaluated: nominal frozen inputs and mapped real-AIA-incident replay. The procedure approximates the information flow of prospective monitoring but does not reconstruct a historical live service. We therefore refer to it as a prospective-style chronological replay.

### 5.8 Same-support baseline suite

To test whether the manuscript's findings depend on complex predictors, the exact frozen support is also evaluated against two simple references: training-prevalence climatology and one L2 logistic regression using the 16 latest-slot SHARP features.

The logistic model is fitted only on the 25,586 training rows. Median imputation and standard scaling are training-fitted. Its alarm threshold is selected on model_validation with the same deterministic rule used for the frozen branches. No hyperparameter search, tree model or neural retraining is performed.

Primary baseline reporting is restricted to policy_validation, retrospective Cycle 25 and supplementary 2026. For every model we report TSS at the appropriate frozen threshold, auxiliary TSS at 0.5, AP, Brier score and BSS relative to training climatology.

### 5.9 Active-region block bootstrap

Repeated forecasts from the same active region are not treated as independent. For the baseline comparisons, 2,000 deterministic bootstrap replicates resample region_component_id with replacement and retain all forecast windows belonging to each sampled region. The same block draw is used for every model within a role, enabling paired model-difference intervals.

Percentile 95% intervals are reported for TSS, AP, Brier and BSS. Paired deltas are reported for the frozen branches against climatology, the temporal SHARP model against the latest-state logistic comparator, and fusion against SHARP. These intervals quantify sampling uncertainty under clustered forecast windows; they do not create independence where none exists.

### 5.10 Physical-applicability diagnostics

Phase H is a hypothesis-driven post-freeze extension. The feature-family definitions, transform, covariance estimator and initial physical-state protocol were frozen before Phase-H execution, but Cycle-25 and 2026 outcomes had already been observed in A-G. Phase-H later-regime findings are therefore post-hoc.

The final nonredundant MPSV is used to examine cross-regime physical distance, class-conditional behaviour, active-region aggregation, feature-family sensitivity and source/provenance effects. A structural sensitivity check removes the initially redundant temporal slope; subsequent leave-family-out analyses test whether apparent OOD is broad or dominated by one physical family. No Phase-H result is allowed to alter the v3 routing policy.

### 5.11 Claim ledger and stopping rules

The experimental programme uses explicit stopping rules to prevent retrospective optimization. Once the 72 h A-G evidence chain was frozen, later analyses could add diagnostics but could not rewrite the original model, calibration or policy results. Negative findings are retained as results rather than converted into new tuning cycles.

The final evidence categories are earlier development/descriptive evidence for Cycle-24 roles, post-hoc/diagnostic evidence for v3 on Cycle 25, clean future evaluation for the one-shot supplementary-2026 v3 test, and post-hoc descriptive evidence for Phase-H analyses performed after 2026 had already been opened. This chronology is used throughout the Results and Discussion sections.

## 6. Results

### 6.1 Cross-regime discrimination degrades

The frozen forecast branches show substantial temporal degradation. On the common cross-regime support, SHARP remains the strongest individual branch, but its TSS falls from 0.689 on earlier Cycle-24 development support to 0.541 on Cycle 25 and 0.323 in 2026. AIA remains near 0.34 through Cycle 25 before falling sharply to 0.074 in 2026. Equal-weight fusion follows the same broad decline and does not consistently exceed SHARP.

| Regime | SHARP TSS | AIA TSS | Fusion TSS |
|---|---:|---:|---:|
| Earlier Cycle-24 development | 0.689 | 0.333 | 0.630 |
| Cycle 25 | 0.541 | 0.340 | 0.488 |
| Supplementary 2026 | 0.323 | 0.074 | 0.292 |

The careful interpretation is transfer degradation from an earlier development regime into later conditions. These values do not establish a causal effect of solar-cycle number itself.

### 6.2 Calibration and conformal reliability deteriorate

Probability calibration degrades alongside discrimination. SHARP Brier Skill Score decreases from approximately 0.336 to 0.215 and 0.160 across the three regimes. Fusion BSS similarly falls from approximately 0.247 to 0.176 and 0.097. Expected calibration error increases, and later calibration slopes/intercepts deviate more strongly from the ideal relationship.

The class-conditional conformal results reveal an even sharper rare-event problem. Flare-class empirical coverage for SHARP, AIA and fusion changes approximately as follows:

| Regime | SHARP flare coverage | AIA flare coverage | Fusion flare coverage |
|---|---:|---:|---:|
| Earlier Cycle 24 | 0.806 | 0.812 | 0.838 |
| Cycle 25 | 0.735 | 0.620 | 0.692 |
| 2026 | 0.435 | 0.064 | 0.346 |

Thus later predictions can remain highly decisive while failing to preserve the earlier flare-class coverage behaviour. The 2026 AIA branch is the clearest example: singleton rate is approximately 0.973 while flare-class coverage is only approximately 0.064.

### 6.3 Statistical agreement becomes more reassuring while reliability worsens

A central result is the divergence between model agreement and cross-regime reliability. On the frozen multimodal support, median fusion seed spread, fusion entropy and SHARP-AIA probability gap all decrease over time:

| Regime | Median fusion seed std | Median fusion entropy | Median SHARP-AIA gap |
|---|---:|---:|---:|
| Earlier Cycle-24 development | 0.01222 | 0.13841 | 0.02783 |
| Cycle 25 | 0.00643 | 0.11834 | 0.01923 |
| 2026 | 0.00163 | 0.04872 | 0.00404 |

The models therefore become more internally consistent and more decisive precisely while calibration and flare-class conformal reliability deteriorate. These agreement measures remain useful within a regime for ranking some forms of case-level risk, but they are poor standalone indicators of global temporal shift.

This is the principal reliability paradox of the study: **agreement is not equivalent to trustworthiness**.

### 6.4 Real AIA failures are temporally clustered

The upstream engineering inventory contains 6,461 AIA acquisition/recovery incidents, with 2,290 exact Gray-Box mappings. The failures are strongly clustered rather than independent.

In 2025, 359 empirical episodes are reconstructed; 332 are multi-sample clusters and 6,172 of 6,199 incident samples occur within clustered episodes. Median episode span is 19.2 h and the maximum is 124.8 h. Among mapped Gray-Box exposures, 2,085 cases occur in long episodes containing more than ten failed samples.

These results show why independent random masking is an incomplete robustness model for image-based operational forecasting.

### 6.5 SHARP fallback preserves issuance under mapped AIA failures

Under nominal Cycle-25 replay, the frozen v3 policy issues every forecast, with approximately 77.7% NORMAL and 22.3% DEGRADED states. Replaying mapped AIA incidents shifts cases from NORMAL to DEGRADED but preserves 100% issuance because SHARP remains available.

| Scenario | Regime | Availability | NORMAL | DEGRADED | TSS |
|---|---|---:|---:|---:|---:|
| nominal inputs | Cycle 25 | 1.000 | 0.777 | 0.223 | 0.507 |
| real AIA incident replay | Cycle 25 | 1.000 | 0.721 | 0.279 | 0.512 |
| nominal inputs | 2026 | 1.000 | 0.925 | 0.075 | 0.296 |

The small TSS difference between nominal and incident replay is not interpreted as an improvement caused by missing data. The operational result is preservation of service availability through a pre-specified stronger fallback.

### 6.6 AIA-only automatic fallback is not supported

The frozen AIA branch does not provide a stable basis for automatic degraded issuance when SHARP is unavailable.

| Regime | AIA TSS | Recall | AP | Flare conformal coverage | Singleton rate |
|---|---:|---:|---:|---:|---:|
| Earlier Cycle 24 | 0.343 | 0.911 | 0.122 | 0.837 | 0.635 |
| Cycle 25 | 0.340 | 0.743 | 0.184 | 0.620 | 0.727 |
| 2026 | 0.074 | 0.117 | 0.159 | 0.064 | 0.973 |

In 2026, the AIA branch becomes extremely decisive while detecting only about 11.7% of positive windows. This combination argues against interpreting low entropy or singleton conformal sets as sufficient evidence for standalone operational use. ABSTAIN therefore remains the defensible state when SHARP itself is unavailable.

### 6.7 Prospective-style replay confirms a confidence/reliability mismatch

The rolling replay uses only matured outcomes, with a 96 h issue-to-visibility delay. Monitoring never feeds back into the policy.

By the August-2026 endpoint, 46,964 past cases are matured, including 4,852 positives. Fusion flare conformal coverage is approximately 0.621 while singleton rate is approximately 0.934. Median fusion entropy is approximately 0.0912, seed spread 0.00395 and SHARP-AIA gap 0.0136.

Thus the running system would have appeared increasingly decisive under several internal uncertainty summaries even while its rare-event coverage had materially weakened. The replay therefore reproduces the cross-regime reliability paradox under a past-only information flow.

### 6.8 A simple latest-state logistic is a strong comparator

The same-support baseline suite shows that forecast complexity should not be confused with operational contribution.

On policy_validation, TSS is 0.597 for the latest-state logistic, 0.628 for frozen SHARP and 0.626 for fusion. On Cycle 25, the values are 0.501, 0.541 and 0.488 respectively. In 2026 they are 0.269, 0.323 and 0.292.

| Role | Logistic TSS | SHARP TSS | AIA TSS | Fusion TSS |
|---|---:|---:|---:|---:|
| policy_validation | 0.597 | 0.628 | 0.319 | 0.626 |
| Cycle 25 | 0.501 | 0.541 | 0.340 | 0.488 |
| 2026 | 0.269 | 0.323 | 0.074 | 0.292 |

The paired active-region bootstrap shows a small Cycle-25 TSS advantage for temporal SHARP over the latest-state logistic: median delta +0.039, 95% CI [0.0003, 0.0819]. The corresponding AP and Brier differences do not exclude zero. In 2026 the SHARP-logistic TSS point difference is larger (+0.052) but the 95% CI crosses zero [-0.004, 0.123].

The result supports only a modest temporal-model advantage, not a universal deep-learning superiority claim.

### 6.9 Equal-weight fusion is not universally superior to SHARP

The baseline suite provides direct paired evidence against a fusion-superiority narrative.

On Cycle 25, fusion minus SHARP is:
- TSS: -0.053, 95% CI [-0.091, -0.013];
- AP: -0.047, 95% CI [-0.084, -0.013].

Both intervals exclude zero in favour of SHARP. On policy_validation, fusion and SHARP have essentially identical TSS, but fusion has lower AP and worse Brier score. In 2026 the point estimates again favour SHARP, although active-region bootstrap intervals include zero.

The multimodal branch is therefore valuable primarily as an operational state whose trust and failure behaviour can be studied, not because it establishes superior predictive skill.

### 6.10 Physical applicability supplies a distinct monitoring signal

The Phase-H physical layer initially shows a 2026 increase in training-reference magnetic-state distance. After removal of a deterministic temporal redundancy, median squared Mahalanobis distance is approximately 10.86 on earlier Cycle-24 development support, 12.54 on Cycle 25 and 16.74 in 2026. The corresponding q99 OOD rates are approximately 1.30%, 1.44% and 2.82%.

However, feature-family sensitivity reveals that this elevation is not broad. Removing the complete gradient family reduces 2026 median distance to 9.38 and q99 OOD rate to 1.08%, both below the Cycle-25 values for the same reduced representation. Removing only gradients__latest is already sufficient to eliminate the aggregate 2026 elevation.

The shift is therefore **gradient-family dominated**.

Importantly, this physical signal moves in the opposite direction from the statistical agreement signals. On the common multimodal support, median physical distance increases from approximately 11.60 to 12.54 to 16.74 while seed spread, entropy and SHARP-AIA gap all decrease. Physical applicability and statistical agreement therefore capture different aspects of system state.

### 6.11 Physical distance is not a case-level error score

The physical monitor cannot be promoted directly into an abstention gate. Flare windows are systematically closer to the training physical-state cloud than non-flare windows:

- earlier Cycle 24: median 7.08 for flare windows versus 11.19 for non-flare windows;
- Cycle 25: 9.29 versus 13.07;
- 2026: 11.06 versus 17.23.

Accordingly, physical distance is negatively correlated with case-level fusion squared error. Magnetically unusual cases can be easy non-flare predictions, while difficult flare cases can lie closer to the development cloud.

The scientific role of the physical layer is therefore applicability/provenance monitoring, not direct selective classification.

### 6.12 Source-series parity narrows, but does not eliminate, the provenance question

The large 2026 gradient shift aligns temporally with an April-2026 transition in the local supplementary cohort from records sourced as hmi.sharp_720s to hmi.sharp_cea_720s. This initially creates a serious source-boundary confound.

A paired JSOC audit queries both series for 120 identical HARP/T_REC records. All 16 SHARP summary keywords are numerically identical on all 120 pairs at both 1e-12 and 1e-9 tolerances; maximum absolute and relative differences are zero. HARPNUM, T_REC, CODEVER7, CALVER64, CMASK and QUALITY also match exactly, and all 240 series queries succeed.

This rules out a keyword-definition difference within the queried JSOC pairs. It does not, by itself, prove that every locally stored CEA-era tensor row equals the corresponding JSOC record. The local solar-versus-pipeline origin of the April 2026 gradient shift therefore remains unresolved and is not attributed causally to Solar Cycle 25.

### 6.13 Summary of supported and unsupported claims

Taken together, the 72 h evidence supports the following:
- predictive discrimination, calibration and conformal flare coverage degrade under later temporal regimes;
- internal statistical agreement can become tighter while rare-event reliability worsens;
- real AIA acquisition failures are temporally clustered;
- SHARP is a defensible availability-preserving fallback when AIA fails;
- automatic AIA-only fallback is not supported;
- a simple latest-state magnetic baseline is competitive, and temporal SHARP provides only a modest later-regime advantage;
- equal-weight fusion is not universally superior to SHARP;
- physical applicability and statistical agreement are complementary monitoring dimensions;
- physical Mahalanobis distance is not a valid naive abstention score.

The evidence does not support universal predictive superiority, a causal Solar-Cycle-25 explanation of the 2026 gradient shift, a first-principles physical model, or a claim of true live prospective validation.

## 7. Discussion
Explain why confidence is not sufficient for trust, why physical applicability is monitoring rather than a gate, why fusion was not universally superior, and why graceful degradation is valuable even without a state-of-the-art predictor claim.

## 8. Limitations
State post-hoc elements, lack of true live prospective deployment, unresolved local-versus-JSOC origin of the April-2026 gradient shift, dependence among repeated forecast windows, and the absence of a first-principles physical model.

## 9. Conclusion
Reiterate the operational trust contribution and the evidence for separating predictive confidence, calibration, physical applicability, provenance, and availability.

---

## Provisional references used in Sections 1–2

- Angryk, R. A., Martens, P. C., Aydin, B., et al. (2020). Multivariate time series dataset for space weather data analytics. *Scientific Data*, 7, 227. https://doi.org/10.1038/s41597-020-0548-x
- Barnes, G., Leka, K. D., Schrijver, C. J., et al. (2016). A comparison of flare forecasting methods. I: Results from the “All-Clear” Workshop. *The Astrophysical Journal*, 829, 89.
- Bobra, M. G., & Couvidat, S. (2015). Solar flare prediction using SDO/HMI vector magnetic field data with a machine-learning algorithm. *The Astrophysical Journal*, 798, 135. https://doi.org/10.1088/0004-637X/798/2/135
- Camporeale, E., & Berger, T. E. (2025). Verification of the NOAA Space Weather Prediction Center solar flare forecast (1998–2024). *Space Weather*, 23, e2025SW004546. https://doi.org/10.1029/2025SW004546
- Crown, M. D. (2012). Validation of the NOAA Space Weather Prediction Center's solar flare forecasting look-up table and forecaster-issued probabilities. *Space Weather*. https://doi.org/10.1029/2011SW000760
- Francisco, G., Guastavino, S., Barata, T., Fernandes, J., & Del Moro, D. (2024). Multimodal Flare Forecasting with Deep Learning. arXiv:2410.16116.
- Georgoulis, M. K., Bloomfield, D. S., Piana, M., et al. (2021). The flare likelihood and region eruption forecasting (FLARECAST) project: flare forecasting in the big data & machine learning era. *Journal of Space Weather and Space Climate*, 11, 39. https://doi.org/10.1051/swsc/2021023
- Goodwin, G. T., Sadykov, V. M., & Martens, P. C. (2024). Investigating Performance Trends of Simulated Real-time Solar Flare Predictions: The Impacts of Training Windows, Data Volumes, and the Solar Cycle. *The Astrophysical Journal*, 964, 163. https://doi.org/10.3847/1538-4357/ad276c
- Guerra, J. A., Pulkkinen, A., & Uritsky, V. M. (2015). Ensemble forecasting of major solar flares: First results. *Space Weather*, 13, 626–642. https://doi.org/10.1002/2015SW001195
- Hong, J., Pandey, C., & Aydin, B. (2026). Uncertainty-Aware Solar Flare Regression. arXiv:2603.06712.
- Leka, K. D., Park, S.-H., Kusano, K., et al. (2019a). A Comparison of Flare Forecasting Methods. II. Benchmarks, Metrics, and Performance Results for Operational Solar Flare Forecasting Systems. *The Astrophysical Journal Supplement Series*, 243, 36. https://doi.org/10.3847/1538-4365/ab2e12
- Leka, K. D., Park, S.-H., Kusano, K., et al. (2019b). A Comparison of Flare Forecasting Methods. III. Systematic Behaviors of Operational Solar Flare Forecasting Systems. *The Astrophysical Journal*, 881, 101. https://doi.org/10.3847/1538-4357/ab2e11
- Liu, H., Liu, C., Wang, J. T. L., & Wang, H. (2019). Predicting Solar Flares Using a Long Short-Term Memory Network. *The Astrophysical Journal*, 877, 121. https://doi.org/10.3847/1538-4357/ab1b3c
- Nishizuka, N., Kubo, Y., Sugiura, K., Den, M., & Ishii, M. (2021). Operational solar flare prediction model using Deep Flare Net. *Earth, Planets and Space*, 73, 64. https://doi.org/10.1186/s40623-021-01381-9
- Riggi, S., Romano, P., Pilzer, A., & Becciani, U. (2026). Solar flare forecasting with foundational transformer models across image, video, and time-series modalities. *Astronomy and Computing*, 55, 101042. https://doi.org/10.1016/j.ascom.2025.101042
- Wheatland, M. S. (2005). A statistical solar flare forecast method. *Space Weather*, 3, S07003. https://doi.org/10.1029/2004SW000131
- Akinwumi, B. M., et al. (2026). Evolution-Aware Temporal Modelling of Magnetic Complexity for Multi-Horizon M/X-Class Solar Flare Prediction. *Advances in Space Research*. https://doi.org/10.1016/j.asr.2026.09.024
