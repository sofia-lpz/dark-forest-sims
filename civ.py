# civilization object
# has the ONLY goals of increasing population

class Civilization:

    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.coordinates = (0, 0)

        self.population = 0
        self.science = 0
        self.resources = 0

        self.birth_rate = 1
        self.death_rate = 0.5
        self.population_consumption = 1 # how much resources consumed by the population 

        self.known_civilizations = []
        self.known_civilization_coordinates = []
        
        self.tech_level = 0  # Determined by science threshold


    # science increases birth rate, decreases death rate, increases population consumption
    # science increases how much population can be moved to colonize new planets

    def move(self, target_planet, population_amount):
        pass

    def action(self, action_type, *args):
        pass

    def update_population(self):
        """
        Update population based on birth rate, death rate, and resource consumption
        Birth rate increases with science, death rate decreases with science
        """
        self.update_science()  # Ensure science is updated before calculating population changes
        
        # Modify rates based on science
        adjusted_birth_rate = self.birth_rate + (self.science * 0.1)
        adjusted_death_rate = max(0.1, self.death_rate - (self.science * 0.05))
        
        # Calculate births and deaths
        births = self.population * (adjusted_birth_rate / 100)
        deaths = self.population * (adjusted_death_rate / 100)
        
        # Apply population changes
        self.population += births - deaths
        
        # Consume resources based on population
        consumption = self.population * (self.population_consumption / 100)
        self.resources -= consumption
        
        # Starvation if resources run out
        if self.resources < 0:
            starvation = min(self.population, abs(self.resources) / self.population_consumption)
            self.population -= starvation
            self.resources = 0

    def update_science(self):
        """
        Increase science by investing resources
        10 resources = 1 science point

        increase science if exploration of new planets, colonization, or spying on other civilizations
        """
        if self.resources >= 10:
            science_gain = self.resources // 10
            self.science += science_gain
            self.resources %= 10


    def update_resources(self):
        """
        Resources regenerate slowly over time
        Base regeneration: 1 per step, increased by population efficiency

        TODO: change base regen to be planet dependant. take into account multi planet civilizartions
        """
        base_regen = 1
        efficiency_bonus = self.science * 0.5
        self.resources += base_regen + efficiency_bonus



    