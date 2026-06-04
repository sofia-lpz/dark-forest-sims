SCIENCE_PER_RANGE = 50          # science needed to extend exploration radius by 1
BIRTH_RATE_STEP = 0.1           # how much increase_birth_rate() adds each call
COLONIZE_COST = 50              # resources to settle an empty planet
CONQUER_COST = 100              # resources to take an inhabited planet by force
DESTROY_COST = 150              # resources (science weapon) to destroy a planet
SCIENCE_PER_EXPLORE = 1         # science per newly revealed cell
SCIENCE_PER_BROADCAST = 5       # science per civ that newly hears your broadcast
CONQUER_SCIENCE_FRACTION = 0.5  # share of a conquered civ's science you absorb

MIN_PLANET_RESOURCES = 50       # smallest resource amount a planet can spawn with
MAX_PLANET_RESOURCES = 200      # largest resource amount a planet can spawn with

class Civilization:
    """A civilization. The action methods mirror the original Mesa agent; the
    only structural change is that grid lookups go through the env and cells are
    plain (row, col) tuples rather than Mesa Cell objects."""

    def __init__(self, env, name, color, coord,
                 population, science, resources, harvest_rate):
        self.env = env
        self.name = name
        self.color = color
        self.coord = coord                 # home planet coordinate
        self.population = population
        self.science = science
        self.resources = resources
        self.birth_rate = 1.0
        self.death_rate = 0.5
        self.population_consumption = 1.0
        self.harvest_rate = harvest_rate
        self.known_civilizations = []      # other Civilization objects
        self.explored_cells = set()        # set of (row, col) coords
        self.alive = True

        # claim the home planet if one exists and is unowned
        home = self.env.planet_at(coord)
        if home is not None and not home.destroyed and home.civilization is None:
            home.civilization = self

    # --- helpers -----------------------------------------------------------
    def _all_civilizations(self):
        return [c for c in self.env.civs.values() if c is not self and c.alive]

    def _planet_on(self, coord):
        return self.env.planet_at(coord)

    def _planets_of(self, civ):
        return [p for p in self.env.planets if p.civilization is civ]

    def _wipe(self, civ):
        civ.population = 0
        civ.alive = False
        for p in self.env.planets:
            if p.civilization is civ:
                p.civilization = None

    @property
    def strength(self):
        return self.population + self.science + self.resources

    @property
    def exploration_radius(self):
        return 1 + int(self.science // SCIENCE_PER_RANGE)

    # --- per-step dynamics -------------------------------------------------
    def update(self):
        if self.harvest_rate:
            income = sum(
                self.harvest_rate * p.resources
                for p in self.env.planets
                if p.civilization is self and not p.destroyed
            )
            self.resources += income

        births = self.population * self.birth_rate
        deaths = self.population * self.death_rate
        self.population += births - deaths

        needed = self.population * self.population_consumption
        self.resources -= needed
        if self.resources < 0:
            starved = min(self.population, -self.resources)
            self.population -= starved
            self.resources = 0

        self.population = max(0, int(self.population))
        if self.population <= 0:
            self.alive = False

    # --- actions  ---------------------------
    def explore(self):
        """Reveal cells within the (science-scaled) radius around every owned
        planet. Returns the number of newly explored cells."""
        radius = self.exploration_radius
        origins = {self.coord}
        for p in self._planets_of(self):
            origins.add(p.coord)

        before = len(self.explored_cells)
        for origin in origins:
            self.explored_cells |= self.env.neighborhood(origin, radius)
        newly = len(self.explored_cells) - before
        self.science += newly * SCIENCE_PER_EXPLORE
        return newly

    def increase_birth_rate(self):
        self.birth_rate += BIRTH_RATE_STEP
        return self.birth_rate

    def broadcast_position(self):
        """Tell everyone where you live. Returns how many civs newly hear you."""
        newly_reached = 0
        for civ in self._all_civilizations():
            if self not in civ.known_civilizations:
                civ.known_civilizations.append(self)
                newly_reached += 1
            civ.explored_cells.add(self.coord)
            self.explored_cells.add(civ.coord)
            if civ not in self.known_civilizations:
                self.known_civilizations.append(civ)
        self.science += newly_reached * SCIENCE_PER_BROADCAST
        return newly_reached

    def colonize_empty_planet(self, coord):
        if coord not in self.explored_cells:
            return False
        planet = self._planet_on(coord)
        if planet is None or planet.destroyed:
            return False
        if planet.civilization is not None:
            return False
        if self.resources < COLONIZE_COST:
            return False
        self.resources -= COLONIZE_COST
        planet.civilization = self
        return True

    def destroy_planet(self, coord):
        if coord not in self.explored_cells:
            return False
        planet = self._planet_on(coord)
        if planet is None or planet.destroyed:
            return False
        if self.resources < DESTROY_COST:
            return False
        self.resources -= DESTROY_COST

        former_owner = planet.civilization
        planet.destroyed = True
        planet.civilization = None
        planet.resources = 0

        if former_owner is not None and former_owner is not self:
            if not self._planets_of(former_owner):
                self._wipe(former_owner)
        return True

    def colonize_inhabited_planet(self, coord):
        if coord not in self.explored_cells:
            return False
        planet = self._planet_on(coord)
        if planet is None or planet.destroyed:
            return False
        resident = planet.civilization
        if resident is None or resident is self:
            return False
        if self.resources < CONQUER_COST:
            return False
        self.resources -= CONQUER_COST

        if self.strength > resident.strength:
            # win: absorb part of their research, then seize the planet
            gained = resident.science * CONQUER_SCIENCE_FRACTION
            self.science += gained
            resident.science -= gained
            planet.civilization = self
            if not self._planets_of(resident):
                self._wipe(resident)
            return True
        else:
            # lose: pay a population price for the failed invasion
            self.population = max(0, int(self.population * 0.75))
            return False
