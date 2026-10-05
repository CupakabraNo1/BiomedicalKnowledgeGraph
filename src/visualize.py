import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import config, link_pred

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, MUTED, GRID = "#898781", "#0b0b0b", "#52514e", "#e1e0d9"
MODELS = ("autoencoder", "autoencoder_head")

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.axisbelow": True, "font.size": 10,
})


def save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    print(f"[save]: {name}.png")


def plot_embeddings(embeddings, G):
    """Node embeddings squeezed to 2D with PCA, coloured by node kind."""
    points = PCA(n_components=2, random_state=config.SEED).fit_transform(embeddings)
    kinds = np.array([G.nodes[i]["kind"] for i in range(len(embeddings))])

    fig, ax = plt.subplots(figsize=(6, 4))
    for kind, color, size, alpha in (("Gene", BLUE, 3, 0.5), ("Disease", ORANGE, 18, 0.9)):
        mask = kinds == kind
        ax.scatter(points[mask, 0], points[mask, 1], s=size, color=color, alpha=alpha,
                   linewidths=0, label=f"{kind} ({mask.sum():,})")
    ax.legend(frameon=False, markerscale=2)
    save(fig, "embeddings_pca")


def plot_loss(history):
    """Train and validation loss per epoch."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, color, label in (("loss", BLUE, "train"), ("val_loss", ORANGE, "validation")):
        values = history[key]
        ax.plot(range(1, len(values) + 1), values, color=color, linewidth=2, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Autoencoder reconstruction loss", loc="left")
    ax.legend(frameon=False)
    save(fig, "loss")


def plot_errors(autoencoder, x_pos, x_neg):
    """Histogram of reconstruction error: real edges vs non-edges."""
    pos = autoencoder.reconstruction_error(x_pos)
    neg = autoencoder.reconstruction_error(x_neg)
    bins = np.linspace(0, np.percentile(neg, 99), 60)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(neg, bins=bins, density=True, histtype="step", linewidth=2, color=ORANGE, label="non-edges")
    ax.hist(pos, bins=bins, density=True, histtype="step", linewidth=2, color=BLUE, label="edges")
    ax.set_xlabel("reconstruction error (MSE)")
    ax.set_ylabel("density")
    ax.set_title("Autoencoder error: edges vs non-edges (validation)", loc="left")
    ax.legend(frameon=False)
    save(fig, "errors")


def plot_metric(results, metric, title, name):
    """One horizontal bar per method; the autoencoder rows in blue."""
    table = results.sort_values(metric)
    colors = [BLUE if method in MODELS else GRAY for method in table.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(table.index, table[metric], color=colors, height=0.6)
    for y, value in enumerate(table[metric]):
        ax.text(value, y, f" {value:.3f}", va="center", color=MUTED, fontsize=9)
    ax.set_xlim(0, table[metric].max() * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(metric.upper())
    ax.set_ylabel(title)
    save(fig, name)


def plot_warm_cold(results):
    """AUC on warm vs cold-start test edges, per method."""
    table = results.sort_values("auc_warm")
    y = np.arange(len(table))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(y + 0.2, table["auc_warm"], height=0.38, color=ORANGE, label="warm")
    ax.barh(y - 0.2, table["auc_cold"], height=0.38, color=BLUE, label="cold")
    ax.axvline(0.5, color=GRAY, linewidth=1, linestyle="--")
    ax.set_yticks(y, table.index)
    ax.set_xlim(0.4, 0.9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUC")
    ax.set_title("Warm vs cold-start test edges", loc="left")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "warm_cold")


def top_candidates(autoencoder, embeddings, disease, genes, known, node_attributes, k=10):
    """The k best-scoring genes for one disease, leaving out already known edges."""
    candidates = np.array([g for g in genes if (disease, g) not in known])
    pairs = np.column_stack([np.full(len(candidates), disease), candidates])
    scores = autoencoder.head_scores(link_pred.edge_features(embeddings, pairs))

    best = np.argsort(scores)[::-1][:k]
    return pd.DataFrame({
        "gene": [node_attributes[g]["name"] for g in candidates[best]],
        "idx": candidates[best],
        "score": scores[best],
    })


def plot_prediction_subgraph(G_train, disease, candidates, node_attributes, max_known=15):
    """The disease, its best-connected known genes and the predicted genes."""
    known = [n for n in G_train.neighbors(disease) if G_train.nodes[n]["kind"] == "Gene"]
    known = sorted(known, key=G_train.degree, reverse=True)[:max_known]
    predicted = list(candidates["idx"])

    H = G_train.subgraph([disease, *known, *predicted]).copy()
    for gene in predicted:
        H.add_edge(disease, gene, kind="predicted")

    pos = nx.spring_layout(H, seed=config.SEED, k=0.9)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axis("off")

    solid = [(u, v) for u, v, kind in H.edges(data="kind") if kind != "predicted"]
    dashed = [(u, v) for u, v, kind in H.edges(data="kind") if kind == "predicted"]
    nx.draw_networkx_edges(H, pos, edgelist=solid, edge_color=GRID, width=1, ax=ax)
    nx.draw_networkx_edges(H, pos, edgelist=dashed, edge_color=AQUA, width=2, style="dashed", ax=ax)

    for nodes, color, size, label in (([disease], ORANGE, 600, "disease"),
                                      (known, BLUE, 220, "known gene"),
                                      (predicted, AQUA, 220, "predicted gene")):
        nx.draw_networkx_nodes(H, pos, nodelist=nodes, node_color=color, node_size=size,
                               edgecolors="#fcfcfb", linewidths=2, label=label, ax=ax)

    label_pos = {n: (x, y + 0.06) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(H, label_pos, {n: node_attributes[n]["name"] for n in H},
                            font_size=8, font_color=MUTED, ax=ax)
    ax.set_title(f"Top predictions for {node_attributes[disease]['name']}", loc="left")
    ax.legend(frameon=False, loc="lower left", markerscale=0.5)
    save(fig, "prediction_subgraph")
