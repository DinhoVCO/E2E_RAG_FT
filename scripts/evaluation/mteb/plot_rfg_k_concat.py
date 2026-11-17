#!/usr/bin/env python3
"""Plot the RFG concat k ablation relative to conventional retrieval."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "reports" / "rfg_k_v1"

K_VALUES = np.array([1, 3, 5, 7, 10])
DELTAS = {
    "BioASQ-resplit": [-0.01050, -0.00339, 0.00476, 0.00335, -0.00111],
    "NarrativeQA": [-0.05549, 0.01632, 0.04036, 0.04989, np.nan],
    "QASPER": [-0.05087, 0.01727, 0.04607, 0.05244, 0.07096],
    "Telco-DPR": [-0.03691, 0.00240, 0.02221, 0.02369, 0.02017],
}
MACRO_AVERAGE = [-0.03844, 0.00815, 0.02835, 0.03234, np.nan]


def main() -> None:
    """Generate publication-ready PNG and PDF versions of the ablation plot."""
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(8.6, 5.2))

    scale = 100.0
    deltas = {name: [v * scale for v in values] for name, values in DELTAS.items()}
    macro_average = [v * scale for v in MACRO_AVERAGE]

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    markers = ["o", "s", "^", "D"]
    for (dataset, values), color, marker in zip(
        deltas.items(), colors, markers, strict=True
    ):
        axis.plot(
            K_VALUES,
            values,
            color=color,
            marker=marker,
            linewidth=1.7,
            markersize=5.5,
            label=dataset,
        )

    axis.plot(
        K_VALUES,
        macro_average,
        color="#222222",
        marker="o",
        linewidth=2.8,
        markersize=6,
        label="Média macro",
        zorder=5,
    )
    axis.axhline(0, color="#666666", linewidth=1.1, linestyle="--")
    axis.scatter([7], [macro_average[3]], color="#222222", s=70, zorder=6)
    axis.annotate(
        "Melhor média: $k=7$",
        xy=(7, macro_average[3]),
        xytext=(7.55, 4.0),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1},
        fontsize=9,
    )

    axis.set_xlabel("Documentos utilizados na expansão ($k$)")
    axis.set_ylabel(r"Variação do nDCG@10 ($\times 100$) em relação ao baseline")
    axis.set_xticks(K_VALUES)
    axis.set_ylim(-6.5, 8.0)
    axis.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        fontsize=8.5,
    )
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / "rfg_k_delta_ndcg10.png", dpi=300)
    figure.savefig(OUTPUT_DIR / "rfg_k_delta_ndcg10.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
