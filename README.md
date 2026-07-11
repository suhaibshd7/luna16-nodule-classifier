# LUNA16 Lung Nodule Classifier

Binary classification of candidate lung nodules from CT scans, using a fine-tuned ResNet-18. Class-imbalance handling is chosen by an explicit ablation rather than assumed, results are reported with bootstrap confidence intervals, and Grad-CAM is used to inspect failure modes.

---

## Clinical Context

Lung cancer is the leading cause of cancer death worldwide, largely because most cases are diagnosed too late for surgical resection. Low-dose CT screening detects lung nodules — small rounded opacities in the lung parenchyma — at a stage when they are still treatable. A radiologist reviewing a chest CT must identify and characterise every nodule, a task that is time-consuming and subject to inter-reader variability. Automated nodule detection is one of the most studied applications of medical imaging AI for exactly this reason.

---

## The Task

Given a 64×64 CT patch centred on a candidate coordinate, predict whether it contains a lung nodule (1) or not (0). Candidate locations are pre-provided by LUNA16 annotations, so this model performs **candidate classification** — it does not locate candidates from scratch. Full nodule detection (finding candidates in an unannotated scan) is a harder, separate task.

---

## Dataset

LUNA16 (Lung Nodule Analysis 2016) is derived from LIDC-IDRI, with radiologist-consensus annotations across **888 CT volumes** split into 10 patient-disjoint subsets, and **551,065 total candidate locations** provided in `candidates.csv`.

> Setio AAA, Traverso A, de Bel T, et al. Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images. *Medical Image Analysis* 42 (2017): 1–13. https://doi.org/10.1016/j.media.2017.06.015

Dataset: https://www.kaggle.com/datasets/avc0706/luna16

This project uses **subsets 0–4** (445 of the 888 CT volumes), filtered down to **275,358 candidate locations**.

---

## Method

- **Preprocessing:** CT volumes loaded with SimpleITK, resampled to 1mm isotropic spacing, world coordinates converted to voxel indices.
- **Patch extraction:** 64×64 three-channel patches (axial slices *z*−1, *z*, *z*+1), HU windowed to [−1000, 400], scaled to [0, 1].
- **Model:** ResNet-18, pretrained on ImageNet, final layer replaced with `nn.Linear(512, 1)`.
- **Class-imbalance handling:** chosen by ablation (see below), not assumed.
- **Split:** by LUNA16 subset (0–2 train / 3 val / 4 test) — patient-disjoint by design, so no patient appears in more than one split.
- **Metric:** AUC-ROC, reported with a bootstrap 95% confidence interval (1,000 resamples).

---

## Scope Decisions and Limitations

These are deliberate trade-offs for a portfolio-scope project, stated explicitly rather than left implicit.

**1. Candidate classification, not full detection.** LUNA16 provides pre-annotated candidate coordinates; the model only classifies each one. A full pipeline would also need a candidate-proposal stage. Results here should not be read as performance on the full detection problem.

**2. A single train/val/test split, not the official 10-fold cross-validation.** LUNA16's standard protocol trains on 9 subsets and tests on the 10th, rotating and averaging across all 10, typically reported as FROC/CPM rather than AUC. This project uses a single fixed split on 5 of the 10 subsets instead, as a time/compute trade-off. Consequence, stated plainly: with only ~144 test positives, this gives a noisier AUC estimate than a 10-fold average would, and the result below is **not directly comparable** to papers reporting FROC/CPM over 10-fold CV on the full dataset. The AUC reported here is an internal metric only.

**3. A 2.5D input (3 axial slices), not a full 3D CNN.** Nodules are 3D structures, and a true 3D CNN — as used in most top LUNA16 leaderboard entries — can use volumetric shape to help separate real nodules from vessel/airway cross-sections that look similar in a single slice. This project uses 3 adjacent axial slices as pseudo-RGB channels for a standard 2D ResNet-18 instead, trading volumetric context for the ability to reuse an ImageNet-pretrained 2D backbone directly, with substantially lower training time and memory use. The Grad-CAM false-positive examples below are a direct, visible consequence of this choice.

**4. Negative subsampling.** LUNA16 has a 407:1 negative:positive ratio; extracting every candidate would require ~13GB of patch storage. This project subsamples negatives at 10:1 (train/val) and 20:1 (test). AUC is largely unaffected by this (it's rank-based), but the confusion matrix and specificity figures reflect the subsampled ratio, not the true clinical ratio.

**5. Labels are radiologist consensus.** Unlike datasets with NLP-derived labels, LUNA16 annotations come from multi-reader radiologist consensus through the LIDC-IDRI process — one of the higher-quality public annotation sets in medical imaging.

---

## Ablation: Class-Imbalance Strategy

LUNA16's 407:1 imbalance can be addressed two ways: **oversampling** (a `WeightedRandomSampler` that changes what the network actually sees during training, targeting roughly 1 positive per 4 negatives per batch) or **loss reweighting** (`pos_weight` in `BCEWithLogitsLoss`, penalising a missed positive more heavily). These solve the same problem, and stacking both without checking the interaction is a common mistake: the sampler already changes the effective ratio the network trains on to ~4:1, so a loss weight computed from the *original* 10:1 ratio applied on top of that risks over-correcting.

Both configurations were trained under identical initialisation and compared on the validation set before either was used on the held-out test set:

- **Config A** — sampler + pos_weight (both mechanisms stacked)
- **Config B** — sampler only

| Config | Val AUC | 95% CI | Specificity @ 90% sensitivity |
|---|---|---|---|
| A (sampler + pos_weight) | 0.9714 | [0.9572, 0.9834] | 91.28% |
| B (sampler only) | 0.9722 | [0.9574, 0.9857] | 93.20% |

AUC confidence intervals overlap almost completely — with only ~128 validation positives, AUC alone can't reliably distinguish the two. Specificity at matched sensitivity is the more informative comparison here, since it reflects the false-positive burden a screening workflow would actually see, and **Config B was consistently equal-or-better with no AUC cost**. This is consistent with the sampler already rebalancing the batches the network trains on, making the additional pos_weight correction an over-correction rather than a help.

**Config B (sampler only) was selected** and carried forward to the test evaluation below.

---

## Results

Evaluated once on the held-out, patient-disjoint test set (subset 4), using the operating threshold selected on the *validation* set above and applied fixed to test — the threshold was never tuned on the data it's reported on.

| Metric | Value |
|---|---|
| **Test AUC-ROC** | 0.9547 |
| **95% CI (bootstrap, n=1000)** | [0.9303, 0.9758] |
| **Applied threshold** (from validation) | 0.0571 |
| **Sensitivity** | 88.89% |
| **Specificity** | 93.11% |
| **Precision** | 39.02% |
| TN / FP | 2704 / 200 |
| FN / TP | 16 / 128 |

The threshold was selected on validation to target ~90% sensitivity — the standard clinical framing for screening, where a missed nodule (false negative) is costlier than an unnecessary follow-up (false positive). Sensitivity on test came out at 88.89%, slightly below the 90% validation target; this is expected, since the threshold was fixed in advance rather than re-tuned to hit exactly 90% on test data. Precision is low (39%) by design — the threshold prioritises catching nodules over avoiding false alarms, appropriate for a screening context but worth stating plainly rather than leaving implicit.

---

## Reproducibility

Global random seeds are set for numpy, Python's `random`, and PyTorch (including CUDA). Despite this, **running the full pipeline twice produced slightly different results**, because `torch.manual_seed` does not guarantee bitwise-identical GPU training — early stopping landed on a different epoch each run. This is worth stating directly rather than treating the reported numbers as exact:

| Run | Config A spec@90%sens | Config B spec@90%sens | Gap | Test AUC |
|---|---|---|---|---|
| Run 1 | 86.77% | 94.31% | 7.5 pts | 0.9633 |
| Run 2 (reported above) | 91.28% | 93.20% | 1.9 pts | 0.9547 |

The **direction** of the finding is stable across both runs — Config B is equal-or-better on both AUC and specificity every time — but the **magnitude** of the gap is not, and should not be quoted as a fixed number. Both runs' test AUC values fall within each other's bootstrap CI, so the two runs are consistent with a single underlying performance level rather than a real change in the model. This distinction — between resampling variance (captured by the bootstrap CI) and training-run variance (only visible by actually repeating the run) — is not something a single execution can show on its own.

---

## Grad-CAM Interpretability

Grad-CAM heatmaps are generated for one example per outcome category (TP/TN/FP/FN), using the terminal convolutional block (`model.layer4[-1]`) as the target layer. Each row shows the original patch, the raw heatmap, and the heatmap overlaid on the patch with the predicted probability.

- **True Positive (p=1.00):** a distinct nodule is visible in the patch, and activation is strongly localized — though in the bottom-left quadrant, adjacent to rather than precisely centred on the lesion itself.
- **True Negative (p=0.00):** normal vascular anatomy, no nodule present, and the heatmap shows no activation anywhere in the patch — a confident, correctly localized negative.
- **False Positive (p=0.13):** true label is 0, but the patch contains a bright, dense structure consistent with a rib or vertebral bone fragment. At 0.13, this sits just above the 0.0571 operating threshold and is misclassified as positive. The heatmap shows no localized activation at all — the above-threshold probability does not correspond to the model attending to the bony structure specifically, but to a diffuse, low-magnitude signal spread across the patch.
- **False Negative (p=0.02):** a clearly visible round nodule in the lung parenchyma is missed entirely. The heatmap shows no activation on the nodule, or anywhere else in the patch.

**Two things worth noting about this analysis, not just the observations themselves:**

- **Grad-CAM's spatial resolution is limited by patch size here.** Patches are 64×64, and ResNet-18 downsamples by 32× before its final convolutional block, so the feature map Grad-CAM is computed from is only ~2×2 pixels before upsampling. In practice, this means Grad-CAM can only localize to coarse quadrants, not precise boundaries — consistent with the TP heatmap landing adjacent to, rather than exactly on, the lesion. This is a property of the architecture and patch size, not a training failure.
- **This is an illustration from a single example per category, not a characterized failure mode.** Only one example per outcome is shown (see Scope Decisions), so the apparent pattern — that both error cases show diffuse, near-blank activation rather than confident-but-wrong localization onto a specific structure — is suggestive, not established. It would be a genuinely interesting finding if it held generally (the model's errors being driven by diffuse uncertainty rather than mistaking one anatomical structure for another), but confirming that would require running Grad-CAM across many FP/FN examples, which this project does not do.

---

## Requirements

Python 3.10

```text
torch==2.0.0
torchvision==0.15.0
SimpleITK==2.2.1
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.2.0
matplotlib==3.7.0
grad-cam==1.4.8
```

## How to Run

1. Add the [LUNA16 dataset](https://www.kaggle.com/datasets/avc0706/luna16) to your Kaggle notebook.
2. Set `MINI_RUN = True` in the Configuration section for a ~2-minute pipeline test, or `False` for the full run.
3. Run all cells top to bottom. Section 9 trains both ablation configurations; Section 10 reads the comparison table and selects one for the final test evaluation in Section 11.
