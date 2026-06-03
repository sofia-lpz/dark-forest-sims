from mesa.discrete_space import FixedAgent

# --- Module-level tunables -------------------------------------------------
MIN_PLANET_RESOURCES = 50    # smallest resource amount a planet can spawn with
MAX_PLANET_RESOURCES = 200   # largest resource amount a planet can spawn with


class Planet(FixedAgent):
    """A planet that can be colonized by civilizations."""

    def __init__(self, model, cell, resources=None):
        """Create a new planet.

        Args:
            model: Model instance
            cell: Cell object for this planet
            resources: Initial amount of resources on the planet. If left as
                None, the planet spawns with a random amount in the range
                [MIN_PLANET_RESOURCES, MAX_PLANET_RESOURCES].
        """
        super().__init__(model)
        self.cell = cell

        if resources is None:
            resources = model.random.randint(
                MIN_PLANET_RESOURCES, MAX_PLANET_RESOURCES
            )
        self.resources = resources
        self.civilization = None      # None = empty; otherwise the owning Civilization
        self.destroyed = False