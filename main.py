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
    python main.py --test                                 # scripted test of every action
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


# ── test mock: scripted walkthrough of every action ───────────────────────

def run_test_mock(galaxy: DarkGalaxy):
    """
    Scripted test that exercises every action via civ.take_action().
    Two civilizations are set up in a controlled environment:
      - Alpha  placed close to Beta so exploration contacts happen quickly
      - Both given enough resources / science to unlock each tier in sequence
    Prints a PASS / FAIL line per action and a final summary.
    No animation — runs headlessly and exits.
    """

    PASS = "\033[32mPASS\033[0m"
    FAIL = "\033[31mFAIL\033[0m"
    results = []

    def check(label: str, result: str, must_contain: str):
        ok = must_contain.lower() in result.lower()
        tag = PASS if ok else FAIL
        print(f"  [{tag}] {label}")
        print(f"         → {result}")
        results.append((label, ok))
        return ok

    def check_not(label: str, result: str, must_not_contain: str):
        ok = must_not_contain.lower() not in result.lower()
        tag = PASS if ok else FAIL
        print(f"  [{tag}] {label}")
        print(f"         → {result}")
        results.append((label, ok))
        return ok

    # ── Setup: two civs on adjacent planets in a small galaxy ─────────
    print("\n" + "═" * 60)
    print("  DARK FOREST – ACTION TEST MOCK")
    print("═" * 60)

    # Use first two civs; override positions so they are 12 cells apart
    # (exploration radius will reach each other after ~6 investments of 10 res)
    alpha = galaxy.civilizations[0]
    beta  = galaxy.civilizations[1]

    alpha.coordinates = (20, 50)
    beta.coordinates  = (32, 50)

    # Update their home planets to match
    for planet in galaxy.planets:
        if alpha in planet.civilizations:
            planet.coordinates = alpha.coordinates
        if beta in planet.civilizations:
            planet.coordinates = beta.coordinates

    def _fund(civ, resources=500, science=0):
        civ.resources = resources
        civ.science   = science
        civ._sync_tech_level()

    # ── Section 0: invalid action ──────────────────────────────────────
    print("\n── Section 0: invalid action name ──")
    try:
        alpha.take_action("nuke_the_sun")
        results.append(("ValueError on unknown action", False))
        print(f"  [{FAIL}] ValueError on unknown action → no exception raised")
    except ValueError as e:
        ok = "unknown action" in str(e).lower()
        print(f"  [{'PASS' if ok else 'FAIL'}] ValueError on unknown action")
        print(f"         → {e}")
        results.append(("ValueError on unknown action", ok))

    # ── Section 1: explore (no tech gate) ─────────────────────────────
    print("\n── Section 1: explore ──")
    _fund(alpha, resources=50)
    check("explore expands radius",
          alpha.take_action("explore", resources=30),
          "radius=3")
    check("explore: not enough resources",
          alpha.take_action("explore", resources=9999),
          "not enough resources")
    check("explore: too small to gain a cell",
          alpha.take_action("explore", resources=5),
          "need at least 10")

    # ── Section 2: tech-locked rejection ──────────────────────────────
    print("\n── Section 2: tech-lock guard ──")
    _fund(alpha, science=0)       # tech_level = 0, all gated actions locked
    check("locked action returns lock message",
          alpha.take_action("increase_pop_rate"),
          "locked")

    # ── Section 3: tier 1 – increase population rate ──────────────────
    print("\n── Section 3: increase_pop_rate (tier 1) ──")
    _fund(alpha, resources=500, science=100)   # tech_level = 1
    old_rate = alpha.birth_rate
    result = alpha.take_action("increase_pop_rate")
    check("birth_rate incremented",       result, "raised birth_rate")
    check_not("not a lock message",       result, "locked")
    assert alpha.birth_rate == old_rate + 1, "birth_rate did not change"

    # ── Section 4: make contact via exploration ────────────────────────
    print("\n── Section 4: exploration contact ──")
    _fund(alpha, resources=500, science=100)
    _fund(beta,  resources=500, science=100)
    # Invest until radii overlap (dist=12, need combined radius ≥ 12)
    for _ in range(7):
        alpha.take_action("explore", resources=10)
        beta.take_action("explore",  resources=10)
    galaxy._check_exploration_contacts()

    check("alpha knows beta after contact",
          "ok" if beta in alpha.known_civilizations else "not found",
          "ok")
    check("beta knows alpha after contact",
          "ok" if alpha in beta.known_civilizations else "not found",
          "ok")

    # ── Section 5: tier 2 – broadcast I exist ─────────────────────────
    print("\n── Section 5: broadcast_i_exist (tier 2) ──")
    _fund(beta, resources=500, science=200)   # tech_level = 2
    # Move beta close enough that alpha's radius reaches it
    result = beta.take_action("broadcast_i_exist", galaxy=galaxy)
    check("broadcast I exist fires",      result, "broadcast")
    check_not("not a lock message",       result, "locked")

    # ── Section 6: tier 3 – broadcast I am here ───────────────────────
    print("\n── Section 6: broadcast_i_am_here (tier 3) ──")
    _fund(beta, resources=500, science=300)
    result = beta.take_action("broadcast_i_am_here", galaxy=galaxy)
    check("broadcast I am here fires",    result, "broadcast")

    # ── Section 7: tier 4 – broadcast he exists ───────────────────────
    print("\n── Section 7: broadcast_he_exists (tier 4) ──")
    _fund(alpha, resources=500, science=400)
    # alpha must know beta for this
    if beta not in alpha.known_civilizations:
        alpha.known_civilizations.append(beta)
        alpha.known_civilization_coordinates.append(None)
    result = alpha.take_action("broadcast_he_exists", galaxy=galaxy, target_civ=beta)
    check("broadcast he exists fires",    result, "broadcast")
    # attempt with unknown civ should fail gracefully
    from civ import Civilization
    stranger = Civilization("Stranger", "Red")
    check("broadcast he exists: unknown target",
          alpha.take_action("broadcast_he_exists", galaxy=galaxy, target_civ=stranger),
          "cannot broadcast")

    # ── Section 8: tier 7 – spy ───────────────────────────────────────
    print("\n── Section 8: spy (tier 7) ──")
    _fund(alpha, resources=500, science=700)
    if beta not in alpha.known_civilizations:
        alpha.known_civilizations.append(beta)
        alpha.known_civilization_coordinates.append(None)
    result = alpha.take_action("spy", target_civ=beta, resources_invested=200)
    # Result contains "successfully spied" or "spy operation ... failed"
    check("spy fires (success or failed on chance)",
          result, "spied" if "successfully" in result else "failed")
    # Spy on unknown stranger must refuse
    check("spy: unknown target refused",
          alpha.take_action("spy", target_civ=stranger, resources_invested=50),
          "cannot spy")

    # ── Section 9: tier 5 – broadcast he is here ──────────────────────
    print("\n── Section 9: broadcast_he_is_here (tier 5) ──")
    _fund(alpha, resources=500, science=500)
    # Ensure alpha has beta's coords (force it after spy)
    idx = alpha.known_civilizations.index(beta)
    alpha.known_civilization_coordinates[idx] = beta.coordinates
    result = alpha.take_action("broadcast_he_is_here", galaxy=galaxy, target_civ=beta)
    check("broadcast he is here fires",   result, "broadcast")

    # ── Section 10: tier 6 – colonize empty planet ────────────────────
    print("\n── Section 10: colonize_empty (tier 6) ──")
    _fund(alpha, resources=500, science=600)
    alpha.population = 5000

    # Place an empty planet inside alpha's explored radius
    empty = next((p for p in galaxy.planets if not p.civilizations
                  and not p.destroyed), None)
    if empty:
        empty.coordinates = (alpha.coordinates[0] + 2, alpha.coordinates[1])
        result = alpha.take_action("colonize_empty", galaxy=galaxy,
                                   planet=empty, population_sent=200)
        check("colonize empty succeeds",  result, "colonized")
        if empty.civilizations:
            colony = empty.civilizations[-1]
            if colony not in galaxy.civilizations:
                galaxy.register_colony(colony)
        # Try again on same (now inhabited) planet → should refuse
        check("colonize already inhabited refusal",
              alpha.take_action("colonize_empty", galaxy=galaxy,
                                planet=empty, population_sent=50),
              "already inhabited")
    else:
        print(f"  [SKIP] no empty planet available for colonize test")

    # Out-of-range planet must refuse
    far_planet = next((p for p in galaxy.planets
                       if not p.civilizations and not p.destroyed), None)
    if far_planet:
        far_planet.coordinates = (alpha.coordinates[0] + 9999,
                                  alpha.coordinates[1])
        check("colonize out-of-range refused",
              alpha.take_action("colonize_empty", galaxy=galaxy,
                                planet=far_planet, population_sent=50),
              "outside explored radius")

    # ── Section 11: tier 9 – destroy planet ───────────────────────────
    print("\n── Section 11: destroy_planet (tier 9) ──")
    _fund(alpha, resources=2000, science=900)
    alpha.population = 10000
    victim = next((p for p in galaxy.planets
                   if p.civilizations and not p.destroyed
                   and p.civilizations[0] is not alpha), None)
    if victim:
        result = alpha.take_action("destroy_planet", planet=victim)
        check("destroy planet succeeds",  result, "destroyed")
        check("planet marked destroyed",
              "ok" if victim.destroyed else "not destroyed", "ok")
        # Second destroy on same planet must refuse
        check("destroy already-destroyed planet refused",
              alpha.take_action("destroy_planet", planet=victim),
              "already destroyed")
    else:
        print("  [SKIP] no eligible victim planet found")

    # ── Summary ───────────────────────────────────────────────────────
    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print("\n" + "═" * 60)
    print(f"  RESULTS  {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} failed):")
        for label, ok in results:
            if not ok:
                print(f"    ✗ {label}")
    else:
        print("  – all good!")
    print("═" * 60 + "\n")
    return failed == 0


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
    p.add_argument("--test",     action="store_true",
                   help="Run scripted test of every action and print pass/fail report")
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

    if args.test:
        galaxy = DarkGalaxy(
            num_planets=30,
            num_civilizations=3,
            grid_size=[100, 100],
        )
        success = run_test_mock(galaxy)
        raise SystemExit(0 if success else 1)
    elif args.mock:
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