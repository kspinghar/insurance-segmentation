---
title: Insurance Risk Segmentation
emoji: 🚗
colorFrom: pink
colorTo: red
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Insurance Risk Segmentation: K-Means on French MTPL Policies

Unsupervised segmentation of 678,013 French motor third-party liability policies (the public `freMTPL2freq` dataset) into three actuarially distinct risk profiles. Pipeline: standardize five numeric features (`VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, log-`Density`) → `KMeans(K=3)`. Validated against observed claim frequency (`ClaimNb` / `Exposure`): clusters separate cleanly along the actuarial gradient.

## What this demo does

The interface is split into three tabs:

1. **Score a Policy:** enter five policy features (or pull a random policy from the bundled data), and see which of the three clusters it lands in, the cluster's observed claim frequency, and the position of the policy in the PCA(1,2) plane.
2. **Cluster Profiles:** at-a-glance reference card showing each cluster's mean feature profile and observed claim frequency.
3. **Sample Distribution:** draw N random policies from the bundled data and see how they split across the three clusters as a histogram + sample table.

## Pipeline

- **Features:** `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `Density` (log-transformed).
- **Scaler:** `StandardScaler` fit on the full 678k policies.
- **PCA:** 5-component PCA, only used for visualization (first two PCs).
- **Clustering:** `KMeans(n_clusters=3, random_state=42, n_init=10)` on the original 5D scaled features (this configuration had the highest silhouette in the comparison against PCA-3 / PCA-4 / hierarchical Ward).
- **Validation:** observed `ClaimNb` / `Exposure` per cluster as a supervised sanity check.

## Source code

https://github.com/kspinghar/insurance-segmentation
