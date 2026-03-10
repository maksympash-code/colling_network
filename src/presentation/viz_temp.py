import numpy as np
import matplotlib.pyplot as plt


def plot_temperature(
    T: np.ndarray,
    title: str = "Temperature field",
    annotate: bool = True,
    fmt: str = ".1f",
    show_contours: bool = True,
    alpha: float = 0.75,
):
    """
    Візуалізація температурного поля (10x10 ідеально).

    - annotate=True: пише значення температур у клітинках
    - show_contours=True: малює контур порогу hotspot: mu + alpha*sigma
    """
    T = np.asarray(T, dtype=float)
    n, m = T.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(T, origin="upper", interpolation="nearest")

    # Сітка
    ax.set_xticks(np.arange(-.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.6)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    # Підписи температур
    if annotate:
        for i in range(n):
            for j in range(m):
                ax.text(j, i, format(T[i, j], fmt),
                        ha="center", va="center",
                        fontsize=8, color="black")

    # Контур hotspot threshold: mu + alpha*sigma
    if show_contours:
        mu = float(np.mean(T))
        sigma = float(np.std(T))
        thr = mu + alpha * sigma
        # contour expects levels list
        ax.contour(T, levels=[thr], linewidths=2)

        ax.set_title(f"{title}\nHotspot thr = μ + {alpha}σ = {thr:.2f}")
    else:
        ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature [°C]")

    plt.tight_layout()
    plt.show()