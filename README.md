# Lung Nodule Classifier — LUNA16

Binary classification of lung nodule candidates from CT scans using a fine-tuned ResNet-18.

## Dataset

LUNA16 (Lung Nodule Analysis 2016) — 445 CT volumes with annotated candidate locations.  
Setio AAA et al. "Validation, comparison, and combination of algorithms for automatic 
detection of pulmonary nodules in computed tomography images." *Medical Image Analysis* 
42 (2017): 1–13. https://luna16.grand-challenge.org

## Method

- CT volumes loaded and resampled to 1 mm isotropic spacing with SimpleITK
- 64×64 three-channel patches extracted from axial slices centred on each candidate
- Class imbalance (407:1) handled with WeightedRandomSampler and BCEWithLogitsLoss pos_weight
- ResNet-18 pretrained on ImageNet, final layer replaced for binary output
- Grad-CAM visualisations generated on TP / TN / FP / FN test cases

## Results

Evaluated on a held-out patient-disjoint test set (LUNA16 candidate classification task):

| Metric | Value |
|---|---|
| AUC-ROC | 0.976 |
| Sensitivity | 90.3% |
| Specificity | 93.6% |
| Threshold | 0.51 |

## Requirements

Python 3.10
torch 2.0.0
torchvision 0.15.0
SimpleITK 2.2.1
numpy 1.24.0
pandas 2.0.0
scikit-learn 1.2.0
matplotlib 3.7.0
grad-cam 1.4.8

## How to Run

1. Add the [LUNA16 dataset](https://www.kaggle.com/datasets/avc0706/luna16) to your Kaggle notebook
2. Run all cells top to bottom — set `MINI_RUN = True` to test the pipeline first
