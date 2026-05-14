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

        self.birth_rate = 1
        self.death_rate = 0.5
        self.population_consumption = 1 

        self.known_civilizations = []
        self.known_civilization_coordinates = []

    def move():
        pass

    