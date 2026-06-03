"""
Galaxy model
"""

import math

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalVonNeumannGrid
from mesa.experimental.devs import ABMSimulator

from .Civilizations import Civilization
from .Planets import Planet


class DarkForest(Model):
    """Dark Forest Model.

    A model for simulating the dynamics of a dark forest ecosystem.
    """

    description = (
        "A model for simulating the dynamics of a dark forest ecosystem."
    )

    def __init__(
        self,
        width=20,
        height=20,
        initial_planets=10,
        initial_civilizations=3,
        seed=None,
        simulator: ABMSimulator = None,
    ):
        
        civ_names = ["Santi", "earth", "aliens"]

        super().__init__(seed=seed)
        self.simulator = simulator
        self.simulator.setup(self)

        # Initialize model parameters
        self.height = height
        self.width = width

        # Create grid using experimental cell space
        self.grid = OrthogonalVonNeumannGrid(
            [self.height, self.width],
            torus=True,
            capacity=math.inf,
            random=self.random,
        )

        # A civilization always needs its own planet to spawn on, so we cannot
        # have more civilizations than planets.
        if initial_civilizations > initial_planets:
            raise ValueError(
                "initial_civilizations cannot exceed initial_planets "
                f"({initial_civilizations} > {initial_planets}): "
                "every civilization must spawn on its own planet."
            )

        # Create planets FIRST, each on its own distinct cell. Resources are
        # left unspecified so each planet rolls a random amount (see Planet).
        planet_cells = self.random.sample(
            self.grid.all_cells.cells, k=initial_planets
        )
        Planet.create_agents(
            self,
            initial_planets,
            cell=planet_cells,
        )

        # Create civilizations: each spawns on one of the planet cells (one
        # civilization per planet, no two sharing a planet). The Civilization
        # constructor claims the planet sitting on its home cell.
        home_cells = self.random.sample(planet_cells, k=initial_civilizations)
        Civilization.create_agents(
            self,
            initial_civilizations,
            name=civ_names[:initial_civilizations],
            color=self.random.choice(["red", "blue", "green"]),
            cell=home_cells,
        )

        # Data collector — keys must match the lineplot_component in app.py
        self.datacollector = DataCollector(
            model_reporters={
                "Civilizations": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, Civilization) and getattr(a, "alive", True)
                ),
                "Planets": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, Planet) and not a.destroyed
                ),
                "Destroyed Planets": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, Planet) and a.destroyed
                ),
            }
        )
 
        # Collect initial state (step 0)
        self.datacollector.collect(self)

        
    def step(self):
        """Execute one step of the model."""
        self.agents_by_type[Civilization].shuffle_do("step")
        self.agents_by_type[Planet].shuffle_do("step")
        self.datacollector.collect(self)