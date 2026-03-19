"""Generate architecture diagram using matplotlib."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


def create_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Architektura systemu - Predykcja Severity Wypadków", fontsize=16, fontweight="bold", pad=20)

    def draw_box(x, y, w, h, text, color, fontsize=9):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="black", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", wrap=True)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # === DATA LAYER ===
    draw_box(0.5, 10, 3, 1.2, "crash_data.csv\n172K rows x 43 cols", "#AED6F1")

    # === KEDRO PIPELINE ===
    draw_box(0.5, 7.5, 7, 2, "", "#D5F5E3")
    ax.text(4, 9.2, "Kedro Pipeline", ha="center", fontsize=11, fontweight="bold")

    draw_box(0.8, 7.8, 2.8, 1, "Data Preparation\n5 nodes: clean,\nengineer, encode", "#82E0AA")
    draw_box(4.2, 7.8, 2.8, 1, "Data Modeling\n3 nodes: split,\ntrain, evaluate", "#82E0AA")

    draw_arrow(2, 10, 2, 9.6)
    draw_arrow(3.6, 8.3, 4.2, 8.3)

    # === MODEL IMPROVEMENT ===
    draw_box(8.5, 7.5, 4.5, 2, "", "#FADBD8")
    ax.text(10.75, 9.2, "Model Improvement", ha="center", fontsize=11, fontweight="bold")

    draw_box(8.8, 8, 1.8, 0.7, "AutoML\n(TPOT)", "#F1948A")
    draw_box(11, 8, 1.8, 0.7, "Optuna\nBayesian", "#F1948A")
    draw_box(8.8, 7.7, 4, 0.3, "RF | GB | XGB | LGBM", "#F1948A")

    draw_arrow(7.5, 8.5, 8.5, 8.5)

    # === EXPERIMENT TRACKING ===
    draw_box(14, 7.5, 3.5, 2, "MLflow\nExperiment Tracking\n\nparams, metrics,\nmodel artifacts", "#D7BDE2")

    draw_arrow(13, 8.5, 14, 8.5)

    # === MODEL ARTIFACT ===
    draw_box(4, 5.5, 3, 1.2, "model.pkl\n(best model)", "#F9E79F")

    draw_arrow(5.5, 7.5, 5.5, 6.7)

    # === API ===
    draw_box(0.5, 3.5, 3.5, 1.5, "FastAPI\n\nPOST /predict\nGET /health", "#AED6F1")

    draw_arrow(4, 5.5, 2.25, 5.0)

    # === DOCKER ===
    draw_box(0.5, 1.5, 3.5, 1.5, "Docker\n\nKonteneryzacja\nuvicorn + model", "#85C1E9")

    draw_arrow(2.25, 3.5, 2.25, 3.0)

    # === MONITORING ===
    draw_box(5, 3.5, 3.5, 1.5, "Monitoring\n\npredictions.jsonl\nEvidently drift", "#ABEBC6")

    draw_arrow(4, 4.25, 5, 4.25)

    # === CI/CD ===
    draw_box(9.5, 3.5, 4, 3, "", "#FCF3CF")
    ax.text(11.5, 6.2, "CI/CD (GitHub Actions)", ha="center", fontsize=11, fontweight="bold")

    draw_box(9.8, 5, 1.5, 0.8, "CI\nlint + test", "#F7DC6F")
    draw_box(11.6, 5, 1.5, 0.8, "CD\nDocker\nbuild", "#F7DC6F")
    draw_box(9.8, 3.8, 3.3, 0.8, "CT\nContinuous Training\n(weekly retrain)", "#F7DC6F")

    # === CLIENT ===
    draw_box(14.5, 3.5, 3, 1.5, "Client\n\nSwagger UI\ncurl / app", "#D5F5E3")

    draw_arrow(4, 4.25, 14.5, 4.25)

    # === LEGEND ===
    legend_items = [
        mpatches.Patch(color="#AED6F1", label="Data / API"),
        mpatches.Patch(color="#D5F5E3", label="Pipeline"),
        mpatches.Patch(color="#FADBD8", label="Model Improvement"),
        mpatches.Patch(color="#D7BDE2", label="Tracking"),
        mpatches.Patch(color="#FCF3CF", label="CI/CD"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig("docs/architecture.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Diagram saved to docs/architecture.png")


if __name__ == "__main__":
    create_architecture_diagram()
