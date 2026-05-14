# planet object

class Planet:
    def __init__(self, name, size, resources):
        self.name = name
        self.size = size
        self.resources = resources

        self.civilizations = []
        self.destroyed = False