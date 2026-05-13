"""
train_and_save.py
Mirror the SADM Insurance_Unsupervised_Portfolio notebook: load freMTPL2freq,
preprocess the five clustering features, fit StandardScaler + PCA + KMeans(K=3),
and dump artefacts into ./artifacts/ for the Gradio Space.

Run once. Re-run only if the source CSV or pipeline changes.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ───────────────────────── constants (match notebook) ─────────────────────────
SOURCE_CSV = Path(
    r"C:\Users\khali\iCloudDrive\Workspace\UNIVERSITIES\KRISTIANIA\Y1\Smart_Analysis_Decision_Making\Assignments\freMTPL2freq.csv"
)
FEATURE_COLS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
# Density is log-transformed before scaling — names in scaled space:
SCALED_COLS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density_log"]
N_CLUSTERS = 3
SEED = 42
PCA_COMPONENTS = 5
SCATTER_SUBSAMPLE = 5000  # rows for PCA scatter background in the demo

ART = Path(__file__).parent / "artifacts"
ART.mkdir(exist_ok=True)

# ───────────────────────── load ─────────────────────────
print(f"Loading {SOURCE_CSV.name}...")
df = pd.read_csv(SOURCE_CSV)
print(f"  rows: {len(df):,} | columns: {list(df.columns)}")

# Validate required cols
required = set(FEATURE_COLS) | {"ClaimNb", "Exposure"}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Source CSV missing columns: {missing}")

# ───────────────────────── preprocessing ─────────────────────────
df["Density_log"] = np.log(df["Density"])
df_scaled_input = df[SCALED_COLS].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_scaled_input)
X_scaled_df = pd.DataFrame(X_scaled, columns=SCALED_COLS, index=df.index)
print(f"  scaled feature shape: {X_scaled.shape}")

# ───────────────────────── PCA (for visualization) ─────────────────────────
pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
X_pca = pca.fit_transform(X_scaled_df)
ev = pca.explained_variance_ratio_
print(f"  PCA explained variance ratio: {[f'{v:.3f}' for v in ev]}  cum: {ev.cumsum()[-1]:.3f}")

# ───────────────────────── K-Means (K=3 on original scaled 5D) ─────────────────────────
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
clusters = kmeans.fit_predict(X_scaled_df)
print(f"  K-Means(K={N_CLUSTERS}) cluster sizes:")
unique, counts = np.unique(clusters, return_counts=True)
for u, c in zip(unique, counts):
    print(f"    cluster {u}: {c:,} policies ({c / len(clusters) * 100:.1f}%)")

# ───────────────────────── cluster profile (original-scale means + claim frequency) ─────────────────────────
df["cluster"] = clusters
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

profile = {}
for c in range(N_CLUSTERS):
    mask = df["cluster"] == c
    sub = df.loc[mask]
    total_exposure = float(sub["Exposure"].sum())
    total_claims = float(sub["ClaimNb"].sum())
    profile[str(c)] = {
        "count": int(mask.sum()),
        "share_pct": float(mask.mean() * 100),
        "mean_features": {col: float(sub[col].mean()) for col in FEATURE_COLS},
        "mean_claim_nb": float(sub["ClaimNb"].mean()),
        "mean_exposure": float(sub["Exposure"].mean()),
        "claim_frequency_per_year": total_claims / total_exposure if total_exposure > 0 else 0.0,
    }

# Population-level reference for the demo to compare against.
population = {
    "count": int(len(df)),
    "mean_features": {col: float(df[col].mean()) for col in FEATURE_COLS},
    "claim_frequency_per_year": float(df["ClaimNb"].sum() / df["Exposure"].sum()),
}

print(f"\nCluster claim frequencies (per year):")
for c, prof in profile.items():
    print(f"  cluster {c}: {prof['claim_frequency_per_year']:.4f}  (n={prof['count']:,})")
print(f"  overall:   {population['claim_frequency_per_year']:.4f}  (n={population['count']:,})")

# ───────────────────────── save artefacts ─────────────────────────
joblib.dump(scaler, ART / "scaler.joblib")
joblib.dump(pca, ART / "pca.joblib")
joblib.dump(kmeans, ART / "kmeans.joblib")

# Bundled enriched data — original columns + cluster + PC1 + PC2
df_out = df.copy()
df_out.to_csv(ART / "freMTPL2freq_with_clusters.csv", index=False)

# Cached PCA scatter subsample for the visualization tab (compact npz)
rng = np.random.RandomState(SEED)
idx = rng.choice(len(df), size=min(SCATTER_SUBSAMPLE, len(df)), replace=False)
np.savez_compressed(
    ART / "pca_scatter.npz",
    pc1=df["PC1"].iloc[idx].to_numpy(dtype=np.float32),
    pc2=df["PC2"].iloc[idx].to_numpy(dtype=np.float32),
    cluster=df["cluster"].iloc[idx].to_numpy(dtype=np.int8),
)

(ART / "cluster_profiles.json").write_text(
    json.dumps({"clusters": profile, "population": population}, indent=2)
)

meta = {
    "n_clusters": N_CLUSTERS,
    "random_state": SEED,
    "feature_cols": FEATURE_COLS,
    "scaled_cols": SCALED_COLS,
    "pca_components": PCA_COMPONENTS,
    "explained_variance_ratio": [float(v) for v in ev],
    "n_policies": int(len(df)),
    "source_csv": SOURCE_CSV.name,
}
(ART / "meta.json").write_text(json.dumps(meta, indent=2))

# Sanity check on disk sizes
print("\nArtefact sizes:")
for path in sorted(ART.iterdir()):
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  {path.name}: {size_mb:.2f} MB")

print("\nDone.")
