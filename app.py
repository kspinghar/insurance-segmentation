"""
app.py: Gradio demo for the Insurance Risk Segmentation Space.

Three tabs over a pre-fit StandardScaler + PCA + KMeans(K=3) pipeline:
1. Score a Policy: enter (or randomly draw) feature values, see the cluster.
2. Cluster Profiles: at-a-glance overview of the three risk groups.
3. Sample Distribution: draw N random policies and see the cluster split.
"""

import json
import tempfile
from pathlib import Path

import gradio as gr
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ART = Path(__file__).parent / "artifacts"

# ───────────────────────── load artefacts ─────────────────────────
print("Loading artefacts...")
scaler = joblib.load(ART / "scaler.joblib")
pca = joblib.load(ART / "pca.joblib")
kmeans = joblib.load(ART / "kmeans.joblib")
profiles = json.loads((ART / "cluster_profiles.json").read_text())
meta = json.loads((ART / "meta.json").read_text())
df = pd.read_csv(ART / "freMTPL2freq_with_clusters.csv")
scatter_npz = np.load(ART / "pca_scatter.npz")

FEATURE_COLS = meta["feature_cols"]
N_CLUSTERS = meta["n_clusters"]
EV = meta["explained_variance_ratio"]

CLUSTER_LABELS = {
    0: "Cluster 0: Mature urban / newer car",
    1: "Cluster 1: Rural / older car (low risk)",
    2: "Cluster 2: Young urban / bonus-malus penalty (high risk)",
}
# Distinct color per cluster, used consistently across all plots.
CLUSTER_COLORS = {0: "#6366f1", 1: "#10b981", 2: "#ef4444"}

POP_FREQ = profiles["population"]["claim_frequency_per_year"]
print(f"Loaded: {len(df):,} policies, K={N_CLUSTERS}, overall claim freq={POP_FREQ:.4f}")


# ───────────────────────── core scoring ─────────────────────────
def _score(VehPower, VehAge, DrivAge, BonusMalus, Density):
    """Transform raw policy features through the saved pipeline → (cluster, PC1, PC2)."""
    Density = max(float(Density), 1.0)  # log requires positive
    density_log = np.log(Density)
    X = np.array([[VehPower, VehAge, DrivAge, BonusMalus, density_log]], dtype=np.float64)
    X_scaled = scaler.transform(X)
    cluster = int(kmeans.predict(X_scaled)[0])
    pc = pca.transform(X_scaled)[0]
    return cluster, float(pc[0]), float(pc[1])


def _save_fig(fig):
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


# ───────────────────────── TAB 1: Score a Policy ─────────────────────────
def tab1_predict(VehPower, VehAge, DrivAge, BonusMalus, Density):
    try:
        cluster, pc1, pc2 = _score(VehPower, VehAge, DrivAge, BonusMalus, Density)
    except Exception as e:
        return f"### Error\n{e}", None

    prof = profiles["clusters"][str(cluster)]
    cluster_freq = prof["claim_frequency_per_year"]
    rel = (cluster_freq / POP_FREQ - 1) * 100  # % relative to population mean
    direction = "above" if rel >= 0 else "below"
    risk_word = "higher" if rel >= 0 else "lower"

    summary = (
        f"### {CLUSTER_LABELS[cluster]}\n\n"
        f"**Cluster claim frequency:** {cluster_freq * 100:.2f}% per policy-year &nbsp; · &nbsp; "
        f"**{abs(rel):.1f}% {direction}** the overall average ({POP_FREQ * 100:.2f}%).\n\n"
        f"This policy lands in a cluster with **{risk_word} expected claim frequency** than the overall book.\n\n"
        f"---\n"
        f"**Cluster centroid (mean of features):**\n\n"
        f"| Feature | This cluster | Overall |\n"
        f"|---|---:|---:|\n"
        + "\n".join(
            f"| {col} | {prof['mean_features'][col]:.2f} | {profiles['population']['mean_features'][col]:.2f} |"
            for col in FEATURE_COLS
        )
        + f"\n\n**Cluster size:** {prof['count']:,} of {profiles['population']['count']:,} policies "
        f"({prof['share_pct']:.1f}% of book)."
    )

    # PCA scatter with the input policy as a star marker.
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for c in range(N_CLUSTERS):
        mask = scatter_npz["cluster"] == c
        ax.scatter(
            scatter_npz["pc1"][mask], scatter_npz["pc2"][mask],
            s=10, alpha=0.35, color=CLUSTER_COLORS[c],
            label=f"Cluster {c}",
        )
    ax.scatter([pc1], [pc2], marker="*", s=420, color="#fbbf24",
               edgecolor="black", linewidth=1.5, zorder=10, label="This policy")
    ax.set_xlabel(f"PC1 ({EV[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({EV[1] * 100:.1f}% variance)")
    ax.set_title(f"PCA(1, 2): policy plotted against a 5 000-policy subsample")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return summary, _save_fig(fig)


def tab1_random():
    """Return a randomly-drawn real policy from the bundled data."""
    row = df.sample(n=1).iloc[0]
    return (
        float(row["VehPower"]),
        float(row["VehAge"]),
        float(row["DrivAge"]),
        float(row["BonusMalus"]),
        float(row["Density"]),
    )


# ───────────────────────── TAB 2: Cluster Profiles ─────────────────────────
def _tab2_render():
    """Build the cluster-profiles markdown + a normalized-feature bar chart."""
    pop = profiles["population"]
    md = "### Three risk profiles, ranked by observed claim frequency\n\n"
    # Sort clusters by claim frequency for the narrative
    ordered = sorted(
        profiles["clusters"].items(),
        key=lambda kv: kv[1]["claim_frequency_per_year"],
    )
    for cid, prof in ordered:
        md += (
            f"**{CLUSTER_LABELS[int(cid)]}** &nbsp; · &nbsp; "
            f"{prof['count']:,} policies ({prof['share_pct']:.1f}%) &nbsp; · &nbsp; "
            f"**Claim freq: {prof['claim_frequency_per_year'] * 100:.2f}%/year**\n\n"
        )
    md += (
        f"---\n**Overall:** {pop['count']:,} policies &nbsp; · &nbsp; "
        f"**Claim freq: {pop['claim_frequency_per_year'] * 100:.2f}%/year**\n\n"
        "The three clusters separate cleanly along the actuarial gradient. The rural / older-car group has "
        "claim rates well below the book average, the young urban / penalty group well above. The middle "
        "group (mature urban / newer car) is close to the overall mean. Below is the centroid of each cluster, "
        "z-scored so features are visually comparable on the same scale."
    )

    # Bar chart: z-scored centroid per feature per cluster
    pop_means = np.array([pop["mean_features"][c] for c in FEATURE_COLS])
    # Use std from the dataset's saved scaler.
    pop_stds = np.array(scaler.scale_)  # in scaled-feature order, Density is log
    # For visualization, also log-transform Density centroid to align with scaler space.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(FEATURE_COLS))
    width = 0.27
    for i, c in enumerate(range(N_CLUSTERS)):
        prof = profiles["clusters"][str(c)]
        means = []
        for j, col in enumerate(FEATURE_COLS):
            v = prof["mean_features"][col]
            if col == "Density":
                v = np.log(v)
                p = np.log(pop_means[j])
            else:
                p = pop_means[j]
            means.append((v - p) / pop_stds[j])
        ax.bar(x + (i - 1) * width, means, width, color=CLUSTER_COLORS[c],
               label=f"Cluster {c}", alpha=0.9)
    ax.axhline(0, color="#94a3b8", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_COLS, rotation=0)
    ax.set_ylabel("z-score vs population mean")
    ax.set_title("Cluster centroids vs population mean")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return md, _save_fig(fig)


TAB2_MD, TAB2_PLOT = _tab2_render()


# ───────────────────────── TAB 3: Sample Distribution ─────────────────────────
def tab3_draw(n_samples):
    n = int(n_samples)
    n = max(10, min(n, 5000))
    sample = df.sample(n=n).copy()
    counts = sample["cluster"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        [str(c) for c in counts.index],
        counts.values,
        color=[CLUSTER_COLORS[c] for c in counts.index],
    )
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Cluster")
    ax.set_ylabel(f"Count (out of {n})")
    ax.set_title(f"Cluster split across a random sample of {n} policies")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plot_path = _save_fig(fig)

    # Sample table (first 20 rows for readability) with the columns that matter.
    cols = FEATURE_COLS + ["cluster", "ClaimNb", "Exposure"]
    table = sample[cols].head(20).round(2)

    # Per-cluster summary in the sample
    summary_lines = ["**Sample composition:**\n"]
    for c in counts.index:
        c_mask = sample["cluster"] == c
        n_c = int(c_mask.sum())
        if n_c == 0:
            continue
        total_exp = float(sample.loc[c_mask, "Exposure"].sum())
        total_claims = float(sample.loc[c_mask, "ClaimNb"].sum())
        freq = total_claims / total_exp if total_exp > 0 else 0
        summary_lines.append(
            f"- Cluster {c}: **{n_c}** policies ({n_c / n * 100:.1f}%), "
            f"observed claim freq in sample: **{freq * 100:.2f}%/year**"
        )
    return plot_path, table, "\n".join(summary_lines)


# ───────────────────────── Gradio UI ─────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="Insurance Risk Segmentation") as demo:
    gr.Markdown(
        "# Insurance Risk Segmentation: K-Means on French MTPL\n"
        "Unsupervised segmentation of 678 013 French auto policies into three actuarially distinct "
        "risk groups. Switch between tabs to score a single policy, browse the cluster profiles, "
        "or draw random samples from the bundled dataset."
    )

    with gr.Tabs():
        # ─── Tab 1 ───
        with gr.Tab("Score a Policy"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Policy features**")
                    veh_power = gr.Number(value=6, label="VehPower (rated power, 4-15)", precision=0)
                    veh_age = gr.Number(value=7, label="VehAge (years)", precision=0)
                    driv_age = gr.Number(value=45, label="DrivAge (years)", precision=0)
                    bonus_malus = gr.Number(value=60, label="BonusMalus (50 = clean, 100+ = penalty)", precision=0)
                    density = gr.Number(value=1500, label="Density (population/km², log-transformed internally)")
                    with gr.Row():
                        rnd_btn = gr.Button("🎲 Random policy", variant="secondary")
                        pred_btn = gr.Button("Predict cluster", variant="primary")
                with gr.Column(scale=2):
                    t1_md = gr.Markdown()
                    t1_plot = gr.Image(label="PCA(1, 2): policy in cluster space", type="filepath")

            rnd_btn.click(
                tab1_random, inputs=[],
                outputs=[veh_power, veh_age, driv_age, bonus_malus, density],
            )
            pred_btn.click(
                tab1_predict,
                inputs=[veh_power, veh_age, driv_age, bonus_malus, density],
                outputs=[t1_md, t1_plot],
            )

        # ─── Tab 2 ───
        with gr.Tab("Cluster Profiles"):
            gr.Markdown(TAB2_MD)
            gr.Image(value=TAB2_PLOT, label="Centroid z-scores vs population mean", type="filepath",
                     interactive=False, show_download_button=False)

        # ─── Tab 3 ───
        with gr.Tab("Sample Distribution"):
            with gr.Row():
                with gr.Column(scale=1):
                    n_input = gr.Slider(minimum=50, maximum=2000, value=200, step=50,
                                        label="Number of policies to draw")
                    draw_btn = gr.Button("Draw random sample", variant="primary")
                with gr.Column(scale=2):
                    t3_plot = gr.Image(label="Cluster split", type="filepath")
                    t3_summary = gr.Markdown()
                    t3_table = gr.Dataframe(label="First 20 sampled policies",
                                            wrap=True, interactive=False)

            draw_btn.click(
                tab3_draw, inputs=[n_input],
                outputs=[t3_plot, t3_table, t3_summary],
            )

if __name__ == "__main__":
    demo.launch(show_api=False)
