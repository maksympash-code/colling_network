import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from src.cooling_network.types import CellType


def plot_network_grid(C: np.ndarray, title: str = "Cooling network", annotate: bool = True):
    """
    Малює матрицю CellType як у paper-картинках: різні кольори під -1/0/1/2/3 + числа.
    """
    # Порядок під BoundaryNorm: [-1,0,1,2,3]
    # Межі для класів: (-1.5,-0.5], (-0.5,0.5], (0.5,1.5], (1.5,2.5], (2.5,3.5]
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    cmap = ListedColormap([
        "#c97a2b",  # TSV (-1)  - orange-ish
        "#b22222",  # SILICON (0) - dark red
        "#bfefff",  # LIQUID (1) - light cyan
        "#7ec8ff",  # INLET (2) - blue-ish
        "#ff8080",  # OUTLET (3) - red-ish
    ])
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(C, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")

    # Grid lines (як сітка)
    n, m = C.shape
    ax.set_xticks(np.arange(-.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.8)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    # Annotate значеннями
    if annotate:
        for i in range(n):
            for j in range(m):
                v = int(C[i, j])
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9, color="black")

    ax.set_title(title)

    # Легенда
    legend_items = [
        ("TSV (-1)", "#c97a2b"),
        ("Silicon (0)", "#b22222"),
        ("Liquid (1)", "#bfefff"),
        ("Inlet (2)", "#7ec8ff"),
        ("Outlet (3)", "#ff8080"),
    ]
    handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=c, markersize=12)
               for _, c in legend_items]
    ax.legend(handles, [name for name, _ in legend_items], loc="upper right", framealpha=0.95)

    plt.tight_layout()
    plt.show()