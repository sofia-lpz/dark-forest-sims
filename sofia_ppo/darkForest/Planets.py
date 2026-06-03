class Planet:
    """A planet that can be colonized by civilizations."""

    __slots__ = ("coord", "resources", "civilization", "destroyed")

    def __init__(self, coord, resources):
        self.coord = coord            # (row, col)
        self.resources = resources
        self.civilization = None      # None = empty; else the owning Civilization
        self.destroyed = False