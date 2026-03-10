import numpy as np
import matplotlib.pyplot as plt

def make_demo_power_10x10():
    n = 10
    P = np.ones((n, n)) * 10.0
    P[5, 5] = 200.0  # hotspot
    P[4, 5] = 150.0
    P[5, 4] = 150.0
    return P


def plot_power(P, title="10x10 Power Map"):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(P)

    n, m = P.shape
    ax.set_xticks(np.arange(-.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    for i in range(n):
        for j in range(m):
            ax.text(j, i, f"{P[i,j]:.0f}",
                    ha="center", va="center",
                    fontsize=8, color="black")

    plt.colorbar(im)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    P = make_demo_power_10x10()
    plot_power(P)