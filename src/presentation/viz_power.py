import numpy as np
import matplotlib.pyplot as plt


def plot_power_map(P: np.ndarray, title: str = "Power map", annotate: bool = False):
    """
    Малює power_map як heatmap. Можна вмикати annotate для маленьких N (типу 15..31).
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(P, origin="upper", interpolation="nearest")  # default colormap

    # Сітка
    n, m = P.shape
    ax.set_xticks(np.arange(-.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.3)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    if annotate:
        # Підходить тільки для малих матриць, бо інакше буде “каша”
        for i in range(n):
            for j in range(m):
                ax.text(j, i, f"{P[i,j]:.1f}", ha="center", va="center", fontsize=7, color="black")

    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Power per cell [W] (або ваші одиниці)")

    plt.tight_layout()
    plt.show()