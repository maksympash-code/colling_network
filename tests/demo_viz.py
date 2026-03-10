import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def make_demo_grid_10x10():
    n = 10
    C = np.ones((n, n), dtype=int)  # LIQUID = 1

    # TSV (-1) по шахматці
    for i in range(0, n, 2):
        for j in range(0, n, 2):
            C[i, j] = -1

    # Silicon (0) hotspot block
    C[4:7, 4:7] = 0

    # Inlet (2) зверху
    C[0, 1::2] = 2

    # Outlet (3) справа
    C[1::2, -1] = 3

    return C


def plot_grid(C, title="10x10 Demo Network"):
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]

    cmap = ListedColormap([
        "#f4a460",  # -1 TSV (orange)
        "#cc0000",  # 0 Silicon (red)
        "#bfefff",  # 1 Liquid (light blue)
        "#4169e1",  # 2 Inlet (blue)
        "#ff3030",  # 3 Outlet (bright red)
    ])

    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(C, cmap=cmap, norm=norm)

    # Grid
    n, m = C.shape
    ax.set_xticks(np.arange(-.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    # Annotate numbers
    for i in range(n):
        for j in range(m):
            ax.text(j, i, str(C[i, j]),
                    ha="center", va="center",
                    fontsize=9, color="black")

    ax.set_title(title)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    C = make_demo_grid_10x10()
    plot_grid(C)