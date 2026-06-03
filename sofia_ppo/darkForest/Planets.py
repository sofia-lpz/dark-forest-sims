from mesa.discrete_space import CellAgent, FixedAgent


class Planet(FixedAgent):
    """A planet that can be colonized by civilizations."""

    def __init__(self, model, cell, resources=0):
        """Create a new planet.

        Args:
            model: Model instance
            cell: Cell object for this planet
            resources: Initial amount of resources on the planet
        """
        super().__init__(model)
        self.cell = cell

        self.resources = resources
        self.civilization = None      # None = empty; otherwise the owning Civilization
        self.destroyed = False
