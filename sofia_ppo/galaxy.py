"""
galaxy.py – DarkGalaxy simulation environment.
"""

import numpy as np
from planet import Planet
from civ import Civilization


def _distance(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class DarkGalaxy:
    """Simulation environment for dark-forest civilizations."""

    def __init__(self, num_planets=100, num_civilizations=10,
                 max_steps=1000, grid_size=None):
        self.num_planets = num_planets
        self.num_civilizations = num_civilizations
        self.max_steps = max_steps
        self.current_step = 0
        self.grid_size = grid_size or [300, 300]
        self.planets = []
        self.civilizations = []
        self.history = []
        self._initialize_environment()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _initialize_environment(self):
        """Initialize planets and civilizations."""
        occupied = set()
        for i in range(self.num_planets):
            while True:
                coords = (
                    np.random.randint(0, self.grid_size[0]),
                    np.random.randint(0, self.grid_size[1]),
                )
                if coords not in occupied:
                    occupied.add(coords)
                    break
            planet = Planet(
                name=f"Planet_{i}",
                size=np.random.randint(50, 200),
                resources=np.random.randint(100, 500),
                coordinates=coords,
            )
            self.planets.append(planet)

        colors = ["Red", "Blue", "Green", "Yellow", "Purple"]
        for i in range(self.num_civilizations):
            civ = Civilization(
                name=f"Civilization_{i}",
                color=colors[i % len(colors)],
            )
            civ.population = 100
            civ.science = 0
            civ.resources = 200
            civ.exploration_radius = 0

            planet_index = i % self.num_planets
            civ.coordinates = self.planets[planet_index].coordinates
            self.planets[planet_index].civilizations.append(civ)
            self.civilizations.append(civ)

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self, actions=None):
        """Advance simulation by one step. Returns False when finished."""
        if self.current_step >= self.max_steps:
            return False

        self.current_step += 1

        # Update all civilizations
        for civ in self.civilizations:
            civ.update_resources()
            civ.update_science()
            civ.update_population()

        # Regenerate planet resources
        for planet in self.planets:
            if not planet.destroyed:
                planet.resources += max(1, planet.size // 10)

        # Check exploration contacts (radii overlap → civs meet)
        self._check_exploration_contacts()

        self.history.append({
            "step": self.current_step,
            "civilizations": len(self.civilizations),
            "planets": len(self.planets),
        })

        return True

    # ------------------------------------------------------------------
    # Exploration contacts
    # ------------------------------------------------------------------

    def _check_exploration_contacts(self):
        """
        If the exploration radii of two civilizations overlap, they both
        learn of each other's existence (but NOT exact coordinates – for
        that they must spy or receive a broadcast).
        """
        civs = self.civilizations
        for i in range(len(civs)):
            for j in range(i + 1, len(civs)):
                a, b = civs[i], civs[j]
                dist = _distance(a.coordinates, b.coordinates)
                radii_reach = a.exploration_radius + b.exploration_radius
                if dist <= radii_reach:
                    self._introduce(a, b)
                    self._introduce(b, a)

    @staticmethod
    def _introduce(observer, discovered):
        """
        observer learns that discovered exists.
        Coordinates are stored as None until espionage or broadcast reveals them.
        """
        if discovered not in observer.known_civilizations:
            observer.known_civilizations.append(discovered)
            observer.known_civilization_coordinates.append(None)

    # ------------------------------------------------------------------
    # Helper: register a new colony (call after colonize_empty_planet)
    # ------------------------------------------------------------------

    def register_colony(self, colony: Civilization):
        """Add a freshly created colony to the galaxy roster."""
        self.civilizations.append(colony)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Reset simulation to initial state."""
        self.current_step = 0
        self.planets = []
        self.civilizations = []
        self.history = []
        self._initialize_environment()