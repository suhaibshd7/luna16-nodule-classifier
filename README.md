# LUNA16 Lung Nodule Classifier

Binary classification of candidate lung nodules from CT scans, using a fine-tuned ResNet-18. Class-imbalance handling is chosen by a pre-registered, multi-seed ablation rather than assumed; AUC is reported with a bootstrap confidence interval and precision is additionally reported at true clinical prevalence, not only on the evaluation set; and a hand-rolled Grad-CAM is used to inspect failure modes, including a quantified limitation of the method itself.

---

## Clinical Context

Lung cancer is the leading cause of cancer death worldwide, largely because most cases are diagnosed too late for surgical resection. Low-dose CT screening detects lung nodules — small rounded opacities in the lung parenchyma — at a stage when they are still treatable. A radiologist reviewing a chest CT must identify and characterize every nodule, a time-consuming task subject to inter-reader variability. Automated nodule detection is one of the most studied applications of medical imaging AI for exactly this reason.

---

## The Task

Given a 64×64 CT patch centered on a candidate coordinate, predict whether it contains a lung nodule (1) or not (0). Candidate locations are pre-provided by LUNA16 annotations, so this model performs **candidate classification** — it does not locate candidates from scratch. Full nodule detection (finding candidates in an unannotated scan) is a harder, separate task.

---

## Dataset

LUNA16 (Lung Nodule Analysis 2016) is derived from LIDC-IDRI, with radiologist-consensus annotations across 888 CT volumes split into 10 patient-disjoint subsets, and 551,065 candidate locations in `candidates.csv`.

> Setio AAA, Traverso A, de Bel T, et al. Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images. *Medical Image Analysis* 42 (2017): 1–13. https://doi.org/10.1016/j.media.2017.06.015

Dataset: https://www.kaggle.com/datasets/avc0706/luna16

This project uses **subsets 0–4** (445 of 888 volumes), filtered to **275,358 candidate locations**:

| Split | Subsets | Patches | Positives |
|---|---|---|---|
| Train | 0–2 | 4,652 | 416 |
| Validation | 3 | 1,481 | 128 |
| Test | 4 | 3,048 | 144 |

True dataset-wide prevalence: **0.245% (~408:1 negative:positive)**.

---

## Method

- **Preprocessing:** CT volumes loaded with SimpleITK, resampled to 1mm isotropic spacing, world coordinates converted to voxel indices.
- **Patch extraction:** 64×64, three-channel patches (axial slices *z*−1, *z*, *z*+1), HU windowed to [−1000, 400], scaled to [0, 1].
- **Normalization:** a single scalar mean/std (0.3311 / 0.2977) fit on the training patches only, applied after augmentation. Scalar rather than per-channel, since the 3 channels are the same modality (HU) at different z-slices, not 3 different colors the way R/G/B are — reusing ImageNet's per-channel statistics here would apply color-channel corrections to a non-color signal.
- **Model:** ResNet-18, ImageNet-pretrained, final layer replaced with `nn.Linear(512, 1)`.
- **Split:** by LUNA16 subset (patient-disjoint by construction — no patient appears in more than one split).
- **Class-imbalance handling:** decided by ablation (below), not assumed.
- **Metrics:** AUC-ROC with bootstrap 95% CI (1,000 resamples); precision reported at true prevalence, not only on the evaluation set.

---

## Class-Imbalance Ablation

LUNA16's ~408:1 imbalance can be addressed by **oversampling** (a `WeightedRandomSampler` targeting ~1 positive per 4 negatives per batch) or by **loss reweighting** (`pos_weight` in `BCEWithLogitsLoss`). Stacking both without checking the interaction is a common mistake — the sampler already changes the effective ratio the network trains on, so a `pos_weight` computed from the *original* ratio applied on top risks over-correcting.

- **Config A** — sampler + pos_weight (stacked)
- **Config B** — sampler only

Both configs were trained across **7 seeds, fixed before looking at any result**. The same seed produces identical initialization and batch order for both configs, making the per-seed (B − A) differences a paired sample rather than independent draws, so the comparison uses a paired t-test rather than a win count.

| Config | Val AUC (mean ± SD) | Spec @ 90% sens (mean ± SD) |
|---|---|---|
| A (sampler + pos_weight) | 0.9774 ± 0.0045 | 0.9472 ± 0.0140 |
| B (sampler only) | 0.9686 ± 0.0064 | 0.9196 ± 0.0315 |

Paired t-test on spec@90%sens (B vs A): **t = −1.91, p = 0.104** — short of the pre-registered α = 0.05.

**Decision rule (fixed before this run):** a non-significant result defaults to selecting Config B, on parsimony grounds (one fewer hyperparameter to separately justify), rather than treating a non-significant p-value as evidence the configs are equivalent. That rule was followed, and **Config B is the configuration used for the results below.**

That said, the point estimate is stated plainly rather than only the decision: **Config A had the higher validation AUC in 6 of 7 seeds**, and its spec@90%sens was both higher on average and less than half as variable across seeds (SD 0.0140 vs 0.0315). A non-significant result at n=7 is a statement about the power of this comparison, not proof of equivalence — the direction consistently favored A, and whether that would firm up with more seeds is untested. Both facts are reported because a parsimony-based pick should not be misread as the ablation endorsing Config B.

---

## Results

Evaluated once on the held-out, patient-disjoint test set (subset 4), using the Config B checkpoint (seed 2, validation AUC 0.9788) and the operating threshold selected on validation — applied fixed to test, never re-tuned.

| Metric | Value |
|---|---|
| **Precision at true LUNA16-wide prevalence (~408:1)** | **5.02%** |
| Precision on the 20:1 subsampled test set actually evaluated | 51.63% |
| Test AUC-ROC | 0.9772 (95% CI [0.9621, 0.9893], bootstrap *n*=1000) |
| Sensitivity | 88.19% |
| Specificity | 95.90% |
| Operating threshold (from validation) | 0.0778 |
| TN / FP / FN / TP (on 20:1 test set) | 2785 / 119 / 17 / 127 |

The threshold targets ~90% sensitivity — the standard framing for screening, where a missed nodule is costlier than an unnecessary follow-up. The two precision figures answer different questions: 51.63% describes this exact classifier on the 20:1 subsampled set actually evaluated; **5.02% describes the same classifier's sensitivity/specificity applied at the true ~1-in-408 candidate prevalence** — roughly 95 of every 100 flagged candidates would be false alarms. This is not specific to this model; it is the expected shape of candidate-classification-stage precision before further pipeline stages (additional classifiers, radiologist review) filter further, which is why production detection systems are built as cascades rather than a single classifier.

---

## Grad-CAM Interpretability

Grad-CAM is implemented directly rather than via a library, so the raw pre-ReLU values are inspectable — a library's internal min-max normalization can silently turn a genuinely all-zero attribution map into a rescaled, ordinary-looking one, making it impossible to tell "no evidence" from "display artifact" without reaching into the internals.

For a single-logit binary head, standard Grad-CAM's final ReLU means one call can only surface evidence *for* whatever scalar is backpropagated. Asking "why did the model reject this" requires backpropagating from the *negated* logit, not the one used for "why did the model accept this" — both directions are computed here.

Before drawing any conclusion from a single image, the flat-map rate was checked numerically across up to 20 examples per outcome:

| Outcome | Flat-map rate (sign = evidence-for-nodule) |
|---|---|
| True Positive | 3/20 (15%) |
| True Negative | 20/20 (100%) |
| False Positive | 9/20 (45%) |
| False Negative | 17/17 (100%, all available) |

**True Negatives and False Negatives are flat 100% of the time** — this is a correct, mathematically expected result, not a failure of the method: a confidently-negative prediction means no spatial location increased the "nodule" logit, so there is nothing for a positive-evidence-only map to point to. Backpropagating from the negated logit on the same two illustrative examples confirms the network isn't simply ignoring the input — both show strong, non-degenerate "evidence against nodule" (pre-ReLU max of 2.33 and 1.65 respectively), meaning the rejection is driven by identifiable features that argue *against* a nodule, not an absence of signal.

**False Positives are flat 45% of the time, and even True Positives are flat 15% of the time** — this is the more consequential finding. Crossing the decision threshold only requires the *sum* of contributions across all 512 channels to be net positive; GAP-based Grad-CAM only visualizes locations whose *individual* contribution survives a ReLU. A prediction driven by many small, spatially-distributed contributions can cross the threshold while leaving no single location for Grad-CAM to highlight. This is a real, quantified limitation of the method on this model, not an implementation bug: a blank Grad-CAM map on a borderline call should be read as "no spatially-concentrated evidence," not "no evidence."

*(Figures are rendered inline in the notebook — see Sections 13–14.)*

---

## Reproducibility

Seeds are fixed for `numpy`, Python's `random`, and PyTorch (including CUDA), and `cudnn.deterministic` is enabled. This exact 7-seed ablation was executed twice — across an intervening code change that touched only the results-reporting logic, not the training loop — and both runs produced identical per-epoch training losses, validation AUCs, early-stopping epochs, and Grad-CAM diagnostic outputs to the reported precision. This is a stronger signal than a single run, though it's specific to this environment (Kaggle GPU) and PyTorch/cuDNN version; determinism settings do not guarantee bitwise reproducibility across different hardware or library versions.

---

## Scope Decisions and Limitations

Deliberate trade-offs for a portfolio-scope project, stated explicitly rather than left implicit:

1. **Candidate classification, not full detection.** Candidate coordinates are pre-provided by LUNA16; this model only classifies each one. A full pipeline would also need a candidate-proposal stage.
2. **A single train/val/test split on 5 of 10 subsets, not the official 10-fold cross-validation.** LUNA16's standard protocol trains on 9 subsets and tests on the 10th, rotating and averaging, typically reported as FROC/CPM rather than AUC. With ~144–147 test/val positives this is a noisier estimate than a 10-fold average, and not directly comparable to leaderboard figures.
3. **A 2.5D input (3 axial slices), not a full 3D CNN.** This trades the volumetric shape information a true 3D CNN could use to separate nodules from similar-looking vessel/airway cross-sections, for the ability to reuse a 2D ImageNet-pretrained backbone directly.
4. **Negative subsampling.** The true candidate ratio is ~408:1; this project subsamples to 10:1 (train/val) and 20:1 (test) for storage/compute reasons. AUC is largely unaffected (it's rank-based), but precision and the confusion matrix reflect the subsampled ratio unless explicitly corrected for prevalence (see Results).
5. **The class-imbalance ablation did not reach statistical significance at 7 seeds** (paired t-test p=0.104), despite Config A trending consistently better on both metrics. Config B was selected per a rule fixed before the run, not because the ablation demonstrated it superior — see the Ablation section for the full disclosure.
6. **GAP-based Grad-CAM has a quantified blind spot** on borderline predictions (flat on 45% of False Positives, 15% of True Positives) — see Interpretability.
7. **Labels are radiologist consensus** via the LIDC-IDRI multi-reader process — one of the higher-quality public annotation sets in medical imaging.

---

## Requirements

Run in a Kaggle notebook with GPU enabled. Dependencies: `torch`, `torchvision`, `SimpleITK`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib` — all present in the standard Kaggle Python GPU environment.

## How to Run

1. Add the [LUNA16 dataset](https://www.kaggle.com/datasets/avc0706/luna16) to a Kaggle notebook with GPU enabled.
2. Set `MINI_RUN = True` for a ~2-minute pipeline smoke-test, or `False` for the full run (patch extraction + 7-seed ablation, 14 training runs, plus Grad-CAM diagnostics).
3. Run all cells top to bottom. Section 9 trains both ablation configs across all seeds; Section 10 compares them and selects the final checkpoint; Section 11 evaluates once, on the held-out test set; Sections 12–14 run and diagnose Grad-CAM.
