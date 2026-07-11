# Lung Nodule Classifier — LUNA16

Binary classification of lung nodule candidates from CT scans using a fine-tuned ResNet-18, with weighted sampling to address severe class imbalance and Grad-CAM interpretability.

---

## Clinical Context

Lung cancer is the leading cause of cancer death worldwide. The primary reason for poor outcomes is late detection — most cases are diagnosed at stage III or IV when surgical resection is no longer possible. Low-dose CT screening programmes detect lung nodules (small rounded opacities in the lung parenchyma) at a stage when they are still treatable.

A radiologist reviewing a chest CT must identify and characterise every nodule — a task that is time-consuming, subject to fatigue, and highly variable between readers. Automated nodule detection is one of the most studied applications of medical AI precisely because the clinical stakes and the potential for workflow improvement are both high.

---

## The Task

Given a 64×64 CT patch centred on a candidate coordinate, predict whether it contains a lung nodule (1) or not (0). This is **candidate classification** — the candidate locations are pre-provided by LUNA16 annotations, so the model only classifies, it does not detect. Full nodule detection (locating candidates from scratch) is a harder, separate task.

---

## Dataset

LUNA16 (Lung Nodule Analysis 2016) — 445 CT volumes derived from LIDC-IDRI with radiologist consensus annotations. Candidates are provided in `candidates.csv` with world-space coordinates and binary labels. 

The complete LUNA16 dataset contains **551,065 total candidate locations** across 10 patient subsets.

> Setio AAA, Traverso A, de Bel T, et al. Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images. *Medical Image Analysis* 42 (2017): 1–13.
> https://doi.org/10.1016/j.media.2017.06.015

Kaggle: https://www.kaggle.com/datasets/avc0706/luna16

---

## Dataset Limitations

### 1. Candidate classification is easier than full detection
LUNA16 provides pre-annotated candidate coordinates. The model never has to find where to look — it only decides whether each pre-identified location is a nodule. In clinical deployment, a full pipeline would also need a candidate proposal stage. Results on this task should not be interpreted as performance on the full detection problem.

### 2. Only 5 of 10 subsets used
The full LUNA16 dataset has 10 patient-disjoint subsets (~889 CT volumes). This project uses subsets 0–4 (445 volumes), filtering the original dataset down to **275,358 candidate locations** for our data pipeline. The full dataset would provide more training positives and is expected to improve performance, particularly for rare nodule morphologies.

### 3. Negative subsampling
LUNA16 has a 407:1 negative:positive ratio. Extracting all candidates would require ~13GB of patch storage and severe imbalance handling. This project subsamples negatives at 10:1 for train/val and 20:1 for test. AUC is largely unaffected by negative subsampling (it is rank-based), but the confusion matrix and specificity numbers reflect the subsampled ratio, not the true clinical ratio.

### 4. Labels are radiologist consensus — a strength, not a weakness
Unlike ChestX-ray14 (NLP-generated labels), LUNA16 annotations are based on multi-reader radiologist consensus through the LIDC-IDRI process. This is one of the highest-quality public annotation sets in medical imaging.

---

## Method

- **Preprocessing:** CT volumes loaded with SimpleITK, resampled to 1mm isotropic spacing, world coordinates converted to voxel indices
- **Patch extraction:** 64×64 three-channel patches (axial slices $z-1$, $z$, $z+1$), HU windowed to $[-1000, 400]$, scaled to $[0, 1]$
- **Model:** ResNet-18 pretrained on ImageNet, final layer replaced with `nn.Linear(512, 1)`
- **Imbalance handling:** WeightedRandomSampler (target ~1:4 pos:neg per batch) + BCEWithLogitsLoss with pos_weight $\approx$ 10.18 — both mechanisms required at this severity
- **Split:** By patient subset (subsets 0–2 train, 3 val, 4 test) — LUNA16 subsets are patient-disjoint by design
- **Metric:** AUC-ROC — correct metric for severely imbalanced binary detection tasks

---

## Results

Evaluated on a held-out patient-disjoint test set (subset 4, candidate classification task).

| Metric | Value |
| :--- | :--- |
| **AUC-ROC** | 0.9628 |
| **Sensitivity** | 90.28% |
| **Specificity** | 92.94% |
| **Decision threshold** | 0.5985 |
| **True Negatives (TN)** | 2699 |
| **False Positives (FP)** | 205 |
| **False Negatives (FN)** | 14 |
| **True Positives (TP)** | 130 |

**Threshold selection:** The threshold is chosen where sensitivity $\ge$ 90%. In lung cancer screening, a false negative (missed nodule) is clinically more harmful than a false positive (unnecessary follow-up). Prioritising sensitivity is the standard clinical framing for screening tasks.

**Comparison to published work:** Basic 2D CNN approaches on LUNA16 typically report AUC 0.85–0.95. This result (0.9628) is at the top of that range using ResNet-18 with 2D patches on 5 of 10 subsets. The honest qualifier: published papers use all 10 subsets, evaluate on all 549k candidates (not 20:1 subsampled), and often report FROC rather than AUC — a harder metric that penalises false positives per scan. Direct comparison is approximate.

---

## Grad-CAM Interpretability

Grad-CAM heatmaps generated on 4 examples per outcome category (TP, TN, FP, FN). Target layer: `model.layer4[-1]` (the terminal convolutional layer block).

True positives show focal activation centred on the bright circular opacity in the patch centre — the model attends to the nodule itself. False positives show diffuse activation on vessel cross-sections and airways, which share morphological features with small nodules at 64×64 resolution. This is the expected failure mode and aligns with published critiques of 2D patch-based approaches.

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
grad-cam==1.4.8'''

## How to Run

1. Add the [LUNA16 dataset](https://www.kaggle.com/datasets/avc0706/luna16) to your Kaggle notebook
2. Set `MINI_RUN = True` in Cell 2 for a 2-minute pipeline test
3. Set `MINI_RUN = False` for the full training run
4. Run all cells top to bottom
