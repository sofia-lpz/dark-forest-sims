
import functools
from typing import Iterable

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from Civilizations import Civilization
from Planets import Planet

SCIENCE_PER_RANGE = 50          # science needed to extend exploration radius by 1
BIRTH_RATE_STEP = 0.1           # how much increase_birth_rate() adds each call
COLONIZE_COST = 50              # resources to settle an empty planet
CONQUER_COST = 100              # resources to take an inhabited planet by force
DESTROY_COST = 150              # resources (science weapon) to destroy a planet
SCIENCE_PER_EXPLORE = 1         # science per newly revealed cell
SCIENCE_PER_BROADCAST = 5       # science per civ that newly hears your broadcast
CONQUER_SCIENCE_FRACTION = 0.5  # share of a conquered civ's science you absorb

MIN_PLANET_RESOURCES = 50       # smallest resource amount a planet can spawn with
MAX_PLANET_RESOURCES = 200      # largest resource amount a planet can spawn with

_MAP_CHANNELS = (
    "explored",        # 0: 1 where this civ has explored
    "empty_planet",    # 1: explored & planet alive & unowned
    "self_planet",     # 2: explored & owned by this civ
    "enemy_planet",    # 3: explored & owned by another civ
    "destroyed",       # 4: explored & planet destroyed
    "resources",       # 5: explored & planet alive -> resources / MAX
)
C = len(_MAP_CHANNELS)

# Non-targeted action ids.
A_EXPLORE, A_BIRTH, A_BROADCAST = 0, 1, 2
N_NONTARGETED = 3
N_TARGETED_TYPES = 3  # colonize_empty, destroy, colonize_inhabited

# The PettingZoo environment
class DarkForestParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "ansi"], "name": "dark_forest_v0"}

    def __init__(
        self,
        names: Iterable[str] = ("Santi", "earth", "aliens"),
        width: int = 20,
        height: int = 20,
        initial_planets: int = 10,
        max_steps: int = 200,
        initial_population: float = 10.0,
        initial_science: float = 0.0,
        initial_resources: float = 50.0,
        harvest_rate: float = 0.1,
        reward_weights: dict | None = None,
        render_mode: str | None = None,
    ):
        self.possible_agents = list(names)
        self.width = int(width)
        self.height = int(height)
        self.initial_planets = int(initial_planets)
        self.max_steps = int(max_steps)
        self.initial_population = float(initial_population)
        self.initial_science = float(initial_science)
        self.initial_resources = float(initial_resources)
        self.harvest_rate = float(harvest_rate)
        self.render_mode = render_mode

        if self.initial_planets < len(self.possible_agents):
            raise ValueError(
                "initial_planets must be >= number of civilizations "
                f"({self.initial_planets} < {len(self.possible_agents)}): "
                "every civilization must spawn on its own planet."
            )

        self.reward_weights = {
            "explore": 0.1,        # per newly explored cell
            "broadcast": 0.5,      # per civ that newly hears the broadcast
            "survive": 0.1,        # per step still alive
            "population": 0.01,    # per unit change in population (signed)
            "science": 0.01,       # per unit change in science (signed)
            "colonize": 1.0,       # successful empty colonization
            "conquer": 2.0,        # successful hostile takeover
            "destroyed": 10.0,     # subtracted if this civ is wiped this step
            "invalid": 0.0,        # subtracted if a (masked-out) action is a no-op
        }
        if reward_weights:
            self.reward_weights.update(reward_weights)

        self.n_cells = self.height * self.width
        self.action_dim = N_NONTARGETED + N_TARGETED_TYPES * self.n_cells

        obs_space = spaces.Dict({
            "map": spaces.Box(0.0, 1.0, shape=(C, self.height, self.width),
                              dtype=np.float32),
            "self": spaces.Box(0.0, np.inf, shape=(8,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(self.action_dim),
        })
        act_space = spaces.Discrete(self.action_dim)
        self._obs_spaces = {a: obs_space for a in self.possible_agents}
        self._act_spaces = {a: act_space for a in self.possible_agents}

        n = len(self.possible_agents)
        self._state_map_channels = 3 + n  # present, destroyed, resources, owner-onehot*n
        self._state_dim = (
            self._state_map_channels * self.n_cells + 9 * n  # 8 stats + alive flag
        )
        self.state_space = spaces.Box(0.0, np.inf, shape=(self._state_dim,),
                                      dtype=np.float32)

        self._colors = ["red", "blue", "green", "yellow", "purple", "cyan"]
        self.rng = np.random.default_rng()
        self.agents: list[str] = []

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._obs_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._act_spaces[agent]

    # --- grid helpers ------------------------------------------------------
    def neighborhood(self, coord, radius):
        r0, c0 = coord
        cells = set()
        H, W = self.height, self.width
        for dr in range(-radius, radius + 1):
            span = radius - abs(dr)
            for dc in range(-span, span + 1):
                cells.add(((r0 + dr) % H, (c0 + dc) % W))
        return cells

    def planet_at(self, coord):
        return self.planet_by_coord.get(coord)

    # --- reset -------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agents = list(self.possible_agents)
        self.steps = 0

        # place planets on distinct cells
        all_coords = [(r, c) for r in range(self.height) for c in range(self.width)]
        idx = self.rng.choice(len(all_coords), size=self.initial_planets,
                              replace=False)
        planet_coords = [all_coords[i] for i in idx]
        self.planets = [
            Planet(coord, int(self.rng.integers(MIN_PLANET_RESOURCES,
                                                MAX_PLANET_RESOURCES + 1)))
            for coord in planet_coords
        ]
        self.planet_by_coord = {p.coord: p for p in self.planets}

        # one civ per planet cell 
        home_idx = self.rng.choice(len(planet_coords),
                                   size=len(self.possible_agents), replace=False)
        self.civs = {}
        for k, name in enumerate(self.possible_agents):
            self.civs[name] = Civilization(
                env=self,
                name=name,
                color=self._colors[k % len(self._colors)],
                coord=planet_coords[home_idx[k]],
                population=self.initial_population,
                science=self.initial_science,
                resources=self.initial_resources,
                harvest_rate=self.harvest_rate,
            )

        observations = {a: self._observe(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    # --- step --------------------------------------------------------------
    def step(self, actions):
        acting = list(self.agents)          
        self.rng.shuffle(acting)

        rewards = {a: 0.0 for a in self.agents}
        before = {a: (self.civs[a].population, self.civs[a].science)
                  for a in self.agents}
        w = self.reward_weights

        for name in acting:
            civ = self.civs[name]
            if not civ.alive:
                continue
            self._apply_action(civ, int(actions[name]), rewards, w)

        for name in self.agents:
            civ = self.civs[name]
            if civ.alive:
                civ.update()

        alive_after = 0
        for name in self.agents:
            civ = self.civs[name]
            d_pop = civ.population - before[name][0]
            d_sci = civ.science - before[name][1]
            rewards[name] += w["population"] * d_pop + w["science"] * d_sci
            if civ.alive:
                rewards[name] += w["survive"]
                alive_after += 1
            else:
                rewards[name] -= w["destroyed"]

        self.steps += 1
        truncate = self.steps >= self.max_steps
        last_civ = alive_after <= 1   # dark-forest endgame: one (or none) left

        terminations = {}
        truncations = {}
        for name in self.agents:
            dead = not self.civs[name].alive
            terminations[name] = bool(dead or last_civ)
            truncations[name] = bool(truncate)

        observations = {a: self._observe(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        rewards = {a: float(rewards[a]) for a in self.agents}

        self.agents = [
            a for a in self.agents
            if not (terminations[a] or truncations[a])
        ]

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _apply_action(self, civ, action, rewards, w):
        if action == A_EXPLORE:
            rewards[civ.name] += w["explore"] * civ.explore()
            return
        if action == A_BIRTH:
            civ.increase_birth_rate()
            return
        if action == A_BROADCAST:
            rewards[civ.name] += w["broadcast"] * civ.broadcast_position()
            return

        # targeted action
        t = action - N_NONTARGETED
        ttype, cidx = divmod(t, self.n_cells)
        coord = (cidx // self.width, cidx % self.width)

        if ttype == 0:
            ok = civ.colonize_empty_planet(coord)
            rewards[civ.name] += w["colonize"] if ok else -w["invalid"]
        elif ttype == 1:
            ok = civ.destroy_planet(coord)
            if not ok:
                rewards[civ.name] -= w["invalid"]
        else:
            ok = civ.colonize_inhabited_planet(coord)
            rewards[civ.name] += w["conquer"] if ok else -w["invalid"]

    # --- observation -------------------------------------------------------
    def _observe(self, name):
        civ = self.civs[name]
        return {
            "map": self._map_view(civ),
            "self": self._self_vector(civ),
            "action_mask": self._action_mask(civ),
        }

    def _map_view(self, civ):
        m = np.zeros((C, self.height, self.width), dtype=np.float32)
        for (r, c) in civ.explored_cells:
            m[0, r, c] = 1.0  # explored
            p = self.planet_by_coord.get((r, c))
            if p is None:
                continue
            if p.destroyed:
                m[4, r, c] = 1.0
                continue
            if p.civilization is None:
                m[1, r, c] = 1.0
            elif p.civilization is civ:
                m[2, r, c] = 1.0
            else:
                m[3, r, c] = 1.0
            m[5, r, c] = p.resources / MAX_PLANET_RESOURCES
        return m

    def _self_vector(self, civ):
        n_owned = sum(1 for p in self.planets if p.civilization is civ)
        return np.array([
            max(civ.population, 0.0),
            max(civ.science, 0.0),
            max(civ.resources, 0.0),
            civ.birth_rate,
            civ.death_rate,
            float(n_owned),
            float(civ.exploration_radius),
            float(len(civ.known_civilizations)),
        ], dtype=np.float32)

    def _action_mask(self, civ):
        mask = np.zeros(self.action_dim, dtype=np.int8)
        # the three non-targeted actions are always available
        mask[A_EXPLORE] = mask[A_BIRTH] = mask[A_BROADCAST] = 1
        n = self.n_cells
        for (r, c) in civ.explored_cells:
            p = self.planet_by_coord.get((r, c))
            if p is None or p.destroyed:
                continue
            cidx = r * self.width + c
            if p.civilization is None and civ.resources >= COLONIZE_COST:
                mask[N_NONTARGETED + 0 * n + cidx] = 1
            if civ.resources >= DESTROY_COST:
                mask[N_NONTARGETED + 1 * n + cidx] = 1
            if (p.civilization is not None and p.civilization is not civ
                    and civ.resources >= CONQUER_COST):
                mask[N_NONTARGETED + 2 * n + cidx] = 1
        return mask

    # --- global state for a centralized critic -----------------------------
    def state(self):
        n = len(self.possible_agents)
        gmap = np.zeros((self._state_map_channels, self.height, self.width),
                        dtype=np.float32)
        owner_index = {name: i for i, name in enumerate(self.possible_agents)}
        for p in self.planets:
            r, c = p.coord
            if p.destroyed:
                gmap[1, r, c] = 1.0
                continue
            gmap[0, r, c] = 1.0
            gmap[2, r, c] = p.resources / MAX_PLANET_RESOURCES
            if p.civilization is not None:
                gmap[3 + owner_index[p.civilization.name], r, c] = 1.0

        stats = []
        for name in self.possible_agents:
            civ = self.civs[name]
            stats.extend(self._self_vector(civ).tolist())
            stats.append(1.0 if civ.alive else 0.0)
        return np.concatenate([gmap.ravel(), np.asarray(stats, dtype=np.float32)])

    # --- render ------------------------------------------------------------
    def render(self):
        symbols = {name: name[0].upper() for name in self.possible_agents}
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for p in self.planets:
            r, c = p.coord
            if p.destroyed:
                grid[r][c] = "x"
            elif p.civilization is None:
                grid[r][c] = "o"
            else:
                grid[r][c] = symbols.get(p.civilization.name, "?")
        lines = [" ".join(row) for row in grid]
        for name in self.possible_agents:
            civ = self.civs[name]
            lines.append(
                f"{name}: alive={civ.alive} pop={civ.population} "
                f"sci={civ.science:.0f} res={civ.resources:.0f} "
                f"radius={civ.exploration_radius}"
            )
        out = "\n".join(lines)
        if self.render_mode == "human":
            print(out)
        return out

    def close(self):
        pass


def parallel_env(**kwargs) -> DarkForestParallelEnv:
    """Factory for the parallel environment."""
    return DarkForestParallelEnv(**kwargs)


def env(**kwargs):
    """AEC version (wraps the parallel env) for tooling that needs it."""
    from pettingzoo.utils import parallel_to_aec
    return parallel_to_aec(DarkForestParallelEnv(**kwargs))

if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test

    test_env = parallel_env(max_steps=50, width=8, height=8, initial_planets=6)
    parallel_api_test(test_env, num_cycles=200)
    print("parallel_api_test passed.")

    e = parallel_env(max_steps=60)
    obs, infos = e.reset(seed=0)
    rng = np.random.default_rng(0)
    step = 0
    while e.agents:
        actions = {}
        for a in e.agents:
            legal = np.flatnonzero(obs[a]["action_mask"])  # only legal actions
            actions[a] = int(rng.choice(legal))
        obs, rewards, terms, truncs, infos = e.step(actions)
        step += 1
    print(f"random masked rollout finished after {step} steps; "
          f"state dim = {e.state().shape[0]}")