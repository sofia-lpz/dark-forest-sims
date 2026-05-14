# planet object

class Planet:
    def __init__(self, name, size, resources, coordinates=(0, 0)):
        self.name = name
        self.size = size
        self.coordinates = coordinates


        self.resources = resources

        self.civilizations = []
        self.destroyed = False
