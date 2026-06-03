"""
Galaxy model
"""

import math

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalVonNeumannGrid

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

        # Create civilizations:
        Civilization.create_agents(
            self,
            initial_civilizations,
            name=civ_names[:initial_civilizations],
            color=self.random.choice(["red", "blue", "green"]),
            cell=self.random.choices(self.grid.all_cells.cells, k=initial_civilizations),
        )

        # Create planets:
        Planet.create_agents(
            self,
            initial_planets,
            cell=self.random.choices(self.grid.all_cells.cells, k=initial_planets),
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
