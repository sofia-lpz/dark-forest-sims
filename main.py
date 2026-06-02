"""
main.py – Dark Forest initial state viewer.
Usage
-----
    python main.py
    python main.py --cols 300 --rows 300 --planets 12 --civs 4
"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from galaxy import DarkGalaxy

CIV_COLORS = {
    "Red":    (0.91, 0.30, 0.24),
    "Blue":   (0.23, 0.51, 0.96),
    "Green":  (0.13, 0.77, 0.37),
    "Yellow": (0.92, 0.70, 0.03),
    "Purple": (0.66, 0.33, 0.97),
}
EMPTY_COLOR  = (0.04, 0.04, 0.06)
PLANET_COLOR = (0.15, 0.22, 0.18)


def build_colormap_and_norm(civilizations):
    colors = [EMPTY_COLOR, PLANET_COLOR]
    for civ in civilizations:
        colors.append(CIV_COLORS.get(civ.color, (1, 1, 1)))
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(range(len(colors) + 1), len(colors))
    return cmap, norm


def build_grid(galaxy: DarkGalaxy, civ_index: dict) -> np.ndarray:
    cols, rows = galaxy.grid_size
    grid = np.zeros((cols, rows), dtype=np.int32)
    for planet in galaxy.planets:
        x, y = planet.coordinates
        if planet.civilizations:
            grid[x, y] = 2 + civ_index[planet.civilizations[0].name]
        else:
            grid[x, y] = 1
    return grid


def print_galaxy_info(galaxy: DarkGalaxy):
    cols, rows = galaxy.grid_size
    print("=" * 56)
    print(f"  DARK FOREST  |  grid {cols}×{rows}")
    print("=" * 56)
    print("\n── PLANETS ──")
    for p in galaxy.planets:
        civs = ", ".join(c.name for c in p.civilizations) or "uninhabited"
        print(f"  {p.name:<12} {str(p.coordinates):<12} "
              f"size={p.size:<5} res={p.resources:<5} civs=[{civs}]")
    print("\n── CIVILIZATIONS ──")
    for c in galaxy.civilizations:
        print(f"  {c.name:<18} color={c.color:<8} "
              f"pop={c.population:<7.1f} sci={c.science:<5} "
              f"res={c.resources:<6.1f} @ {c.coordinates}")
    print()


def main():
    galaxy = DarkGalaxy()

    civ_index = {civ.name: i for i, civ in enumerate(galaxy.civilizations)}
    cmap, norm = build_colormap_and_norm(galaxy.civilizations)

    print_galaxy_info(galaxy)

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#050508")
    ax.set_facecolor("#050508")
    ax.axis("off")

    ax.imshow(
        build_grid(galaxy, civ_index).T,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="lower",
    )

    cols, rows = galaxy.grid_size
    ax.set_title(
        f"DARK FOREST  |  {cols}×{rows}  |  "
        f"{len(galaxy.planets)} planets  |  {len(galaxy.civilizations)} civs",
        color="#cccccc", fontsize=9, pad=6, fontfamily="monospace", loc="left",
    )

    handles = [
        mpatches.Patch(facecolor=CIV_COLORS.get(c.color, (1,1,1)),
                       edgecolor="none", label=c.name)
        for c in galaxy.civilizations
    ]
    handles.append(mpatches.Patch(facecolor=PLANET_COLOR,
                                  edgecolor="#3a5a6a", label="Uninhabited"))
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              facecolor="#111122", edgecolor="#333355",
              labelcolor="white", framealpha=0.6, handlelength=1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()