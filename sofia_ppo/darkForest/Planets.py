from mesa.discrete_space import CellAgent, FixedAgent


class Planet(FixedAgent):
    """A planet that can be colonized by civilizations."""

    def __init__(self, model, cell):
        """Create a new planet.

        Args:
            model: Model instance
            cell: Cell object for this planet
        """
        super().__init__(model)
        self.cell = cell

        self.resources = 0
        self.civilization = None      # None = empty; otherwise the owning Civilization
        self.destroyed = False
