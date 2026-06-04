class Planet:

    __slots__ = ("coord", "resources", "civilization", "destroyed")

    def __init__(self, coord, resources):
        self.coord = coord            
        self.resources = resources
        self.civilization = None      
        self.destroyed = False