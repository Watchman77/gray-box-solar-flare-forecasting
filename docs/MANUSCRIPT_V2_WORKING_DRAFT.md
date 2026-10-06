# Manuscript V2 — working draft

## Working title

**A Gray-Box Trust Framework for 72-Hour Solar Flare Forecasting Under Temporal Shift, Uncertainty Degradation, and Data Failure**

## Central contribution

This paper is not positioned as a fusion-superiority paper. Its central contribution is an experimentally validated operational trust framework for deciding when a solar-flare forecast should be issued normally, degraded to a robust fallback, or withheld under temporal shift, uncertainty degradation, and realistic input failures.

The framework combines learned SHARP and AIA predictors with calibration, conformal uncertainty, statistical agreement signals, an explicit interpretable magnetic-state applicability monitor, data-quality/provenance checks, and a frozen NORMAL / DEGRADED / ABSTAIN routing policy.

---

## Abstract — draft v1

Reliable solar-flare forecasting requires more than high discrimination under a fixed retrospective test set. Operational systems must remain useful when probability calibration drifts, uncertainty estimates degrade, input modalities fail, and the physical feature distribution moves away from the development regime. We present a Gray-Box trust framework for 72-hour M/X-class solar-flare forecasting that combines frozen SHARP and AIA predictors with calibration, conformal uncertainty, statistical agreement measures, an interpretable magnetic-state applicability layer, and an explicit NORMAL / DEGRADED / ABSTAIN routing policy. The framework was developed using time-ordered earlier data and evaluated across later Solar Cycle 25 support and a supplementary 2026 extension, with no policy reselection on later outcomes.

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

The framework uses three operational states. **NORMAL** issues the multimodal forecast when the required branch and uncertainty conditions are satisfied. **DEGRADED** retains forecast availability through the SHARP branch when AIA information is unavailable or the multimodal trust conditions are not satisfied. **ABSTAIN** withholds the forecast when the SHARP fallback itself is unavailable. The state definitions are frozen before the later evaluation regimes and are tested without reselection on Cycle-25 or supplementary-2026 outcomes. The purpose is therefore not to demonstrate that fusion is the best predictor. It is to test whether a forecasting system can degrade gracefully and expose changes in reliability rather than silently treating every probability as equally trustworthy.

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

## 3. Data and frozen evaluation design
Describe SHARP, AIA, GOES-derived labels, 72-hour target, 96-minute issue cadence, time-ordered role structure, AR leakage controls, frozen hashes, and the distinction between earlier development, Cycle-25 retrospective evaluation, and supplementary 2026 post-hoc evidence.

## 4. Gray-Box trust framework
Describe the frozen predictors, probability calibration, conformal label sets, agreement signals, explicit magnetic-state applicability layer, data-quality/provenance checks, and NORMAL / DEGRADED / ABSTAIN state machine.

## 5. Experimental protocol
Describe threshold freezing, policy freezing, outage replay, AIA-only fallback audit, rolling prospective-style replay, simple baseline suite, active-region block bootstrap, and claim boundaries.

## 6. Results
Organize around:
- cross-cycle performance and calibration drift;
- conformal reliability degradation;
- agreement-versus-reliability paradox;
- outage and fallback experiments;
- operational routing replay;
- physical applicability diagnostics;
- baseline comparison and uncertainty intervals.

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
