# civilization object
# has the ONLY goals of increasing population

class Civilization:

    def __init__(self, name, color):
        self.name = name
        self.color = color

        self.population = 0
        self.science = 0
        self.resources = 0

        self.is_on_planet = True
        self.is_expanding = False
