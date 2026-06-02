# civ.py – Civilization object
# Goals: grow population, explore, survive.

from science import get_tech_level, SCIENCE_THRESHOLD

class Civilization:

    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.coordinates = (0, 0)           # current home planet coordinates

        self.population = 0
        self.science = 0
        self.resources = 0

        self.birth_rate = 1
        self.death_rate = 0.5
        self.population_consumption = 1     # resources consumed per 100 pop per step

        self.known_civilizations = []              # list of Civilization objects
        self.known_civilization_coordinates = []   # parallel list; None = unknown coords

        self.tech_level = 0                 # increments every SCIENCE_THRESHOLD science
        self.exploration_radius = 0         # grid-cell radius of explored space

    # ------------------------------------------------------------------
    # Core per-step updates
    # ------------------------------------------------------------------

    def update_population(self):
        """
        Update population based on birth rate, death rate, and resource consumption.
        Science improves both rates.
        """
        self._sync_tech_level()

        adjusted_birth_rate = self.birth_rate + (self.science * 0.1)
        adjusted_death_rate = max(0.1, self.death_rate - (self.science * 0.05))

        births = self.population * (adjusted_birth_rate / 100)
        deaths = self.population * (adjusted_death_rate / 100)
        self.population += births - deaths

        consumption = self.population * (self.population_consumption / 100)
        self.resources -= consumption

        if self.resources < 0:
            starvation = min(self.population,
                             abs(self.resources) / self.population_consumption)
            self.population -= starvation
            self.resources = 0

    def update_science(self):
        """
        Convert resources into science: 10 resources → 1 science point.
        Also syncs tech_level after the gain.
        """
        if self.resources >= 10:
            science_gain = self.resources // 10
            self.science += science_gain
            self.resources %= 10
        self._sync_tech_level()

    def update_resources(self):
        """
        Resources regenerate every step.
        Base: 1 per step, bonus from science efficiency.
        TODO: make planet-dependent for multi-planet civs.
        """
        base_regen = 1
        efficiency_bonus = self.science * 0.5
        self.resources += base_regen + efficiency_bonus

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    def invest_in_exploration(self, resources_to_invest: float) -> str:
        """
        Spend resources to expand the exploration radius.
        Each 10 resources invested expands the radius by 1 cell.
        When two civs' radii overlap they automatically discover each other
        (handled in DarkGalaxy.step via check_exploration_contacts).
        """
        if resources_to_invest <= 0:
            return f"{self.name}: must invest a positive amount."
        if self.resources < resources_to_invest:
            return (f"{self.name}: not enough resources "
                    f"({self.resources:.1f} < {resources_to_invest:.1f}).")

        cells_gained = int(resources_to_invest // 10)
        if cells_gained == 0:
            return f"{self.name}: need at least 10 resources to expand radius by 1."

        cost = cells_gained * 10
        self.resources -= cost
        self.exploration_radius += cells_gained
        return (f"{self.name} expanded exploration radius by {cells_gained} "
                f"→ radius={self.exploration_radius} (cost {cost} resources).")

    # ------------------------------------------------------------------
    # Action dispatcher
    # ------------------------------------------------------------------

    def take_action(self, action: str, **kwargs) -> str:
        """
        Unified entry point for every civ action.  Callers never import
        science.py or call methods directly — everything goes through here.

        Actions and their required kwargs
        ----------------------------------
        Internal (no tech gate):
          "explore"                   resources=<int>

        Science-gated (tech_level must be >= tier):
          "increase_pop_rate"         (no extra args)           tier 1
          "broadcast_i_exist"         galaxy=<DarkGalaxy>       tier 2
          "broadcast_i_am_here"       galaxy=<DarkGalaxy>       tier 3
          "broadcast_he_exists"       galaxy, target_civ        tier 4
          "broadcast_he_is_here"      galaxy, target_civ        tier 5
          "colonize_empty"            galaxy, planet,           tier 6
                                      population_sent=<float>
          "spy"                       target_civ,               tier 7
                                      resources_invested=<float>
          "colonize_inhabited"        galaxy, planet,           tier 8
                                      population_sent=<float>
          "destroy_planet"            planet                    tier 9

        Returns a log string describing the outcome.
        Raises ValueError for unknown action names.
        """
        import science as sci  # local import avoids circular dependency

        # ── ungated actions ────────────────────────────────────────────
        if action == "explore":
            resources = kwargs.get("resources", 10)
            return self.invest_in_exploration(resources)

        # ── science-gated actions ──────────────────────────────────────
        _GATED = {
            "increase_pop_rate":   (1, lambda: sci.increase_population_rate(self)),
            "broadcast_i_exist":   (2, lambda: sci.broadcast_i_exist(
                                        self, kwargs["galaxy"])),
            "broadcast_i_am_here": (3, lambda: sci.broadcast_i_am_here(
                                        self, kwargs["galaxy"])),
            "broadcast_he_exists": (4, lambda: sci.broadcast_he_exists(
                                        self, kwargs["galaxy"], kwargs["target_civ"])),
            "broadcast_he_is_here":(5, lambda: sci.broadcast_he_is_here(
                                        self, kwargs["galaxy"], kwargs["target_civ"])),
            "colonize_empty":      (6, lambda: sci.colonize_empty_planet(
                                        self, kwargs["planet"],
                                        kwargs.get("population_sent", 10))),
            "spy":                 (7, lambda: sci.spy(
                                        self, kwargs["target_civ"],
                                        kwargs.get("resources_invested", 30))),
            "colonize_inhabited":  (8, lambda: sci.colonize_inhabited_planet(
                                        self, kwargs["planet"],
                                        kwargs.get("population_sent", 10))),
            "destroy_planet":      (9, lambda: sci.destroy_planet(
                                        self, kwargs["planet"])),
        }

        if action not in _GATED:
            raise ValueError(
                f"{self.name}: unknown action '{action}'. "
                f"Valid actions: explore, {', '.join(_GATED)}"
            )

        tier, fn = _GATED[action]
        if self.tech_level < tier:
            from science import science_dict
            needed = science_dict.get(tier, f"tier {tier}")
            return (
                f"{self.name}: '{action}' locked "
                f"(need tech level {tier} – '{needed}', "
                f"current level {self.tech_level})."
            )

        return fn()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_tech_level(self):
        """Keep tech_level in sync with accumulated science."""
        self.tech_level = get_tech_level(self.science)

    def __repr__(self):
        return (f"Civilization({self.name!r}, pop={self.population:.0f}, "
                f"sci={self.science:.0f}, tech={self.tech_level}, "
                f"radius={self.exploration_radius})")