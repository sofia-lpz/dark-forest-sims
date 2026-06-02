"""
main.py – Dark Forest viewer + optional mock simulation.

Usage
-----
    python main.py                                        # static snapshot
    python main.py --cols 300 --rows 300 --planets 12 --civs 4
    python main.py --mock                                 # animate random-action sim
    python main.py --mock --steps 200 --interval 150     # tune speed / length
    python main.py --explore                              # exploration-only mock with visible radii
    python main.py --explore --steps 100 --interval 300  # tune speed / length
"""
import argparse
import random
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

from galaxy import DarkGalaxy
import science as sci

CIV_COLORS = {
    "Red":    (0.91, 0.30, 0.24),
    "Blue":   (0.23, 0.51, 0.96),
    "Green":  (0.13, 0.77, 0.37),
    "Yellow": (0.92, 0.70, 0.03),
    "Purple": (0.66, 0.33, 0.97),
}
EMPTY_COLOR  = (0.04, 0.04, 0.06)
PLANET_COLOR = (0.15, 0.22, 0.18)
DESTROYED_COLOR = (0.30, 0.05, 0.05)


# ── rendering helpers ──────────────────────────────────────────────────────

def build_colormap_and_norm(civilizations):
    colors = [EMPTY_COLOR, PLANET_COLOR, DESTROYED_COLOR]
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
        if planet.destroyed:
            grid[x, y] = 2                                      # destroyed slot
        elif planet.civilizations:
            name = planet.civilizations[0].name
            idx = civ_index.get(name)
            if idx is not None:
                grid[x, y] = 3 + idx                           # 3 = first civ offset
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
        status = "DESTROYED" if p.destroyed else civs
        print(f"  {p.name:<12} {str(p.coordinates):<12} "
              f"size={p.size:<5} res={p.resources:<5} civs=[{status}]")
    print("\n── CIVILIZATIONS ──")
    for c in galaxy.civilizations:
        print(f"  {c.name:<18} color={c.color:<8} "
              f"pop={c.population:<7.1f} sci={c.science:<5} "
              f"res={c.resources:<6.1f} radius={c.exploration_radius:<4} "
              f"tech={c.tech_level} @ {c.coordinates}")
    print()


# ── mock: random action picker ─────────────────────────────────────────────

# Each entry: (min_tech_required, weight, callable(civ, galaxy) -> str)
# Weight controls how often an action is chosen relative to others.

def _mock_explore(civ, galaxy):
    invest = random.randint(1, 3) * 10
    return civ.take_action("explore", resources=invest)

def _mock_pop_rate(civ, galaxy):
    return civ.take_action("increase_pop_rate")

def _mock_broadcast_exist(civ, galaxy):
    return civ.take_action("broadcast_i_exist", galaxy=galaxy)

def _mock_broadcast_here(civ, galaxy):
    return civ.take_action("broadcast_i_am_here", galaxy=galaxy)

def _mock_broadcast_he_exists(civ, galaxy):
    if not civ.known_civilizations:
        return None
    target = random.choice(civ.known_civilizations)
    return civ.take_action("broadcast_he_exists", galaxy=galaxy, target_civ=target)

def _mock_broadcast_he_here(civ, galaxy):
    candidates = [
        (c, coord)
        for c, coord in zip(civ.known_civilizations,
                            civ.known_civilization_coordinates)
        if coord is not None
    ]
    if not candidates:
        return None
    target, _ = random.choice(candidates)
    return civ.take_action("broadcast_he_is_here", galaxy=galaxy, target_civ=target)

def _mock_spy(civ, galaxy):
    if not civ.known_civilizations:
        return None
    target = random.choice(civ.known_civilizations)
    invest = random.randint(2, 6) * 10
    return civ.take_action("spy", target_civ=target, resources_invested=invest)

def _mock_colonize(civ, galaxy):
    candidates = [
        p for p in galaxy.planets
        if not p.destroyed
        and not p.civilizations
        and ((p.coordinates[0] - civ.coordinates[0]) ** 2 +
             (p.coordinates[1] - civ.coordinates[1]) ** 2) ** 0.5
            <= civ.exploration_radius
    ]
    if not candidates:
        return None
    planet = random.choice(candidates)
    pop_send = max(10, civ.population * 0.10)
    result = civ.take_action("colonize_empty", galaxy=galaxy,
                             planet=planet, population_sent=pop_send)
    if planet.civilizations:
        colony = planet.civilizations[-1]
        if colony not in galaxy.civilizations:
            galaxy.register_colony(colony)
    return result

def _mock_destroy(civ, galaxy):
    candidates = [
        p for p in galaxy.planets
        if not p.destroyed and p.civilizations
        and p.civilizations[0] is not civ
    ]
    if not candidates:
        return None
    planet = random.choice(candidates)
    return civ.take_action("destroy_planet", planet=planet)


# (min_tech, weight, fn)
_ACTION_TABLE = [
    (0,  6, _mock_explore),
    (1,  2, _mock_pop_rate),
    (2,  3, _mock_broadcast_exist),
    (3,  2, _mock_broadcast_here),
    (4,  2, _mock_broadcast_he_exists),
    (5,  1, _mock_broadcast_he_here),
    (6,  3, _mock_colonize),
    (7,  2, _mock_spy),
    (9,  1, _mock_destroy),
]


def mock_civ_action(civ, galaxy) -> str | None:
    """
    Pick and execute one random action for `civ`, filtered to those the
    civ's current tech_level allows.  Returns the action log string, or
    None if nothing was possible.
    """
    available = [(w, fn) for (min_t, w, fn) in _ACTION_TABLE
                 if civ.tech_level >= min_t]
    if not available:
        return None

    weights = [w for w, _ in available]
    fns     = [fn for _, fn in available]
    chosen  = random.choices(fns, weights=weights, k=1)[0]
    return chosen(civ, galaxy)


def run_mock(galaxy: DarkGalaxy, steps: int, interval_ms: int,
             verbose: bool = True):
    """
    Animate the galaxy for `steps` steps, each civ taking a random action
    every step.  Renders a live matplotlib window.
    """
    # snapshot initial roster for colormap (colonies added later get parent color)
    initial_civs = list(galaxy.civilizations)
    civ_index = {civ.name: i for i, civ in enumerate(initial_civs)}
    cmap, norm = build_colormap_and_norm(initial_civs)

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#050508")
    ax.set_facecolor("#050508")
    ax.axis("off")

    im = ax.imshow(
        build_grid(galaxy, civ_index).T,
        cmap=cmap, norm=norm,
        interpolation="nearest", origin="lower",
        animated=True,
    )

    cols, rows = galaxy.grid_size
    title = ax.set_title("", color="#cccccc", fontsize=9, pad=6,
                         fontfamily="monospace", loc="left")

    # legend
    handles = [
        mpatches.Patch(facecolor=CIV_COLORS.get(c.color, (1,1,1)),
                       edgecolor="none", label=c.name)
        for c in initial_civs
    ]
    handles.append(mpatches.Patch(facecolor=PLANET_COLOR,
                                  edgecolor="#3a5a6a", label="Uninhabited"))
    handles.append(mpatches.Patch(facecolor=DESTROYED_COLOR,
                                  edgecolor="none", label="Destroyed"))
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              facecolor="#111122", edgecolor="#333355",
              labelcolor="white", framealpha=0.6, handlelength=1)

    def _tick(frame):
        if not galaxy.step():
            ani.event_source.stop()
            return

        # each civ takes one random action per step
        for civ in list(galaxy.civilizations):
            log = mock_civ_action(civ, galaxy)
            if verbose and log:
                print(f"[step {galaxy.current_step:>4}] {log}")

        # update civ_index with any new colonies
        for civ in galaxy.civilizations:
            if civ.name not in civ_index:
                # inherit parent color slot (find by color)
                parent_idx = next(
                    (civ_index[c.name] for c in initial_civs
                     if c.color == civ.color), 0
                )
                civ_index[civ.name] = parent_idx

        im.set_data(build_grid(galaxy, civ_index).T)
        title.set_text(
            f"DARK FOREST  |  step {galaxy.current_step}/{steps}  |  "
            f"{sum(1 for p in galaxy.planets if not p.destroyed)} planets alive  |  "
            f"{len(galaxy.civilizations)} civs"
        )
        fig.canvas.draw_idle()

    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, _tick, frames=steps,
                        interval=interval_ms, repeat=False)

    plt.tight_layout()
    plt.show()



# ── explore mock: grow radii every step, draw them as circles ─────────────

def run_explore_mock(galaxy: DarkGalaxy, steps: int, interval_ms: int):
    """
    Exploration-only mock: every step each civ invests 10 resources into
    exploration (radius +1).  The explored area is shown as a filled,
    semi-transparent circle around each civ's planet that grows in real time.
    When two circles first overlap, a thin contact ring is drawn to mark the
    moment of mutual discovery.
    """
    initial_civs = list(galaxy.civilizations)
    civ_index    = {civ.name: i for i, civ in enumerate(initial_civs)}
    cmap, norm   = build_colormap_and_norm(initial_civs)

    # Give each civ a standing exploration reserve so they invest every step
    # from the very first frame (update_science would otherwise consume all res).
    for civ in initial_civs:
        civ.resources = max(civ.resources, 50)

    # Pre-compute display scale: grid units → data (pixel) coordinates.
    # imshow with origin="lower" maps grid[x,y] → pixel (x, y) directly,
    # so 1 grid cell = 1 display unit.  Circle radius is in those units.
    cols, rows = galaxy.grid_size

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#050508")
    ax.set_facecolor("#050508")
    ax.axis("off")
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)

    # Base grid (planets / empty space) as background image
    im = ax.imshow(
        build_grid(galaxy, civ_index).T,
        cmap=cmap, norm=norm,
        interpolation="nearest", origin="lower",
        extent=[-0.5, cols - 0.5, -0.5, rows - 0.5],
        animated=True,
        zorder=0,
    )

    # One filled circle per civ for the explored area
    explore_circles = {}
    for civ in initial_civs:
        cx, cy = civ.coordinates
        color  = CIV_COLORS.get(civ.color, (1, 1, 1))
        circle = plt.Circle(
            (cx, cy),
            radius=civ.exploration_radius,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.add_patch(circle)
        # Thin border ring so radius=0 civs are still visible as a dot
        border = plt.Circle(
            (cx, cy),
            radius=max(civ.exploration_radius, 0.6),
            color=color,
            fill=False,
            linewidth=0.8,
            alpha=0.55,
            zorder=2,
        )
        ax.add_patch(border)
        explore_circles[civ.name] = (circle, border)

    # Contact rings: shown when two radii first touch (stored so we only add once)
    already_contacted = set()
    contact_patches   = []

    title = ax.set_title("", color="#cccccc", fontsize=9, pad=6,
                         fontfamily="monospace", loc="left")

    # Legend
    handles = [
        mpatches.Patch(facecolor=CIV_COLORS.get(c.color, (1, 1, 1)),
                       edgecolor="none", label=c.name)
        for c in initial_civs
    ]
    handles.append(mpatches.Patch(facecolor=PLANET_COLOR,
                                  edgecolor="#3a5a6a", label="Uninhabited"))
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              facecolor="#111122", edgecolor="#333355",
              labelcolor="white", framealpha=0.6, handlelength=1)

    def _tick(frame):
        # Top-up exploration budget BEFORE galaxy.step() so update_science
        # can't drain it all.  Each civ gets exactly 10 res earmarked for
        # exploration; the top-up is additive so normal resource generation
        # still works — we just guarantee a floor.
        for civ in galaxy.civilizations:
            if civ.resources < 10:
                civ.resources = 10
            civ.take_action("explore", resources=10)

        galaxy.step()

        # Update exploration circle sizes
        for civ in initial_civs:
            if civ.name not in explore_circles:
                continue
            circle, border = explore_circles[civ.name]
            r = civ.exploration_radius
            circle.set_radius(r)
            border.set_radius(max(r, 0.6))

        # Check for new first-contacts and draw a flash ring
        civs = initial_civs
        for i in range(len(civs)):
            for j in range(i + 1, len(civs)):
                a, b = civs[i], civs[j]
                pair = (a.name, b.name)
                if pair in already_contacted:
                    continue
                dist = ((a.coordinates[0] - b.coordinates[0]) ** 2 +
                        (a.coordinates[1] - b.coordinates[1]) ** 2) ** 0.5
                if a.exploration_radius + b.exploration_radius >= dist:
                    already_contacted.add(pair)
                    # Draw a white ring at the midpoint to mark first contact
                    mx = (a.coordinates[0] + b.coordinates[0]) / 2
                    my = (a.coordinates[1] + b.coordinates[1]) / 2
                    ring = plt.Circle(
                        (mx, my), radius=2,
                        color="white", fill=False,
                        linewidth=1.2, alpha=0.7, zorder=3,
                    )
                    ax.add_patch(ring)
                    contact_patches.append(ring)
                    print(f"[step {galaxy.current_step:>4}] "
                          f"CONTACT: {a.name} ↔ {b.name}")

        # Fade out old contact rings gradually
        for ring in contact_patches:
            ring.set_alpha(max(0, ring.get_alpha() - 0.015))

        im.set_data(build_grid(galaxy, civ_index).T)
        title.set_text(
            f"DARK FOREST – EXPLORE  |  step {galaxy.current_step}/{steps}  |  "
            f"radii: " +
            "  ".join(f"{c.name.split('_')[1]}r={c.exploration_radius}"
                      for c in initial_civs)
        )
        fig.canvas.draw_idle()

    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, _tick, frames=steps,   # noqa: F841 (kept alive)
                        interval=interval_ms, repeat=False)

    plt.tight_layout()
    plt.show()


# ── entry point ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Dark Forest Simulation")
    p.add_argument("--cols",     type=int, default=300)
    p.add_argument("--rows",     type=int, default=300)
    p.add_argument("--planets",  type=int, default=100)
    p.add_argument("--civs",     type=int, default=10)
    p.add_argument("--mock",     action="store_true",
                   help="Run animated mock simulation with random civ actions")
    p.add_argument("--explore",  action="store_true",
                   help="Run exploration-only mock showing growing radius circles")
    p.add_argument("--steps",    type=int, default=500,
                   help="Number of simulation steps (mock mode only)")
    p.add_argument("--interval", type=int, default=200,
                   help="Milliseconds between frames (mock mode only)")
    p.add_argument("--quiet",    action="store_true",
                   help="Suppress per-step action log (mock mode only)")
    return p.parse_args()


def main():
    args = parse_args()

    galaxy = DarkGalaxy(
        num_planets=args.planets,
        num_civilizations=args.civs,
        grid_size=[args.cols, args.rows],
    )

    if args.mock:
        print(f"[mock] starting {args.steps}-step simulation "
              f"({args.civs} civs, {args.planets} planets, "
              f"{args.interval}ms/frame)")
        run_mock(galaxy, steps=args.steps,
                 interval_ms=args.interval,
                 verbose=not args.quiet)
    elif args.explore:
        print(f"[explore] starting {args.steps}-step exploration mock "
              f"({args.civs} civs, {args.planets} planets, "
              f"{args.interval}ms/frame)")
        run_explore_mock(galaxy, steps=args.steps, interval_ms=args.interval)
    else:
        civ_index = {civ.name: i for i, civ in enumerate(galaxy.civilizations)}
        cmap, norm = build_colormap_and_norm(galaxy.civilizations)
        print_galaxy_info(galaxy)

        fig, ax = plt.subplots(figsize=(9, 9))
        fig.patch.set_facecolor("#050508")
        ax.set_facecolor("#050508")
        ax.axis("off")

        ax.imshow(
            build_grid(galaxy, civ_index).T,
            cmap=cmap, norm=norm,
            interpolation="nearest", origin="lower",
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