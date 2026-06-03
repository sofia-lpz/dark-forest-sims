from mesa.discrete_space import CellAgent, FixedAgent
from .Planets import Planet

# --- Module-level tunables -------------------------------------------------
SCIENCE_PER_RANGE = 50       # science needed to extend exploration radius by 1
BIRTH_RATE_STEP = 0.1        # how much increase_birth_rate() adds each call
COLONIZE_COST = 50           # resources to settle an empty planet
CONQUER_COST = 100           # resources to take an inhabited planet by force
DESTROY_COST = 150           # resources (science weapon) to destroy a planet
SCIENCE_PER_EXPLORE   = 1     # science per newly revealed cell
SCIENCE_PER_BROADCAST = 5     # science per civ that newly hears your broadcast
CONQUER_SCIENCE_FRACTION = 0.5  # share of a conquered civ's science you absorb

class Civilization(CellAgent):
    """
    actions
    explore, 
    increase pop birth rate,
    broadcast position,
    colonize empty planet,
    destroy planet,
    colonize inhabited planet,

    rewards are given for exploring, for broadcasting,
    for surviving, for having more population,
    and science

    punishments are given for losing population, for being destroyed, and for having less science
    """

    def __init__(self, model, name, color, cell, population=0, science=0, resources=50):
        super().__init__(model)
        self.name = name
        self.color = color
        self.cell = cell           # current home planet's cell (a Cell object)
        self.population = population
        self.science = science
        self.resources = resources
        self.birth_rate = 1
        self.death_rate = 0.5
        self.population_consumption = 1
        self.known_civilizations = []
        self.explored_cells = set()   # set of explored Cell objects
        self.alive = True             # set False when wiped out

        # claim the home planet if one exists on this cell and it's unowned
        home = self._planet_on(cell)
        if home is not None and not home.destroyed and home.civilization is None:
            home.civilization = self

    # --- helpers -----------------------------------------------------------
    def _all_civilizations(self):
        """Every other living Civilization in the model."""
        return [
            a for a in self.model.agents
            if isinstance(a, Civilization) and a is not self and getattr(a, "alive", True)
        ]

    def _planet_on(self, cell):
        """Return the Planet sitting on `cell`, or None."""
        # fast path: planet placed as a FixedAgent on the cell
        for a in cell.agents:
            if isinstance(a, Planet):
                return a
        # fallback: match by coordinate among all agents (works even if the
        # planet isn't registered in cell.agents)
        for a in self.model.agents:
            if isinstance(a, Planet) and a.coordinates == cell.coordinate:
                return a
        return None

    def _planets_of(self, civ):
        """All planets currently owned by `civ`."""
        return [
            a for a in self.model.agents
            if isinstance(a, Planet) and a.civilization is civ
        ]

    def _wipe(self, civ):
        """Remove a civilization that has lost everything."""
        civ.population = 0
        civ.alive = False
        civ.remove()

    @property
    def strength(self):
        """Combat/expansion power: people backed by science and resources."""
        return self.population + self.science + self.resources

    # --- per-step update ---------------------------------------------------
    def update(self):
        # update the population based on birth and death rates
        births = self.population * self.birth_rate
        deaths = self.population * self.death_rate
        self.population += births - deaths

        # consume resources based on population_consumption and population
        needed = self.population * self.population_consumption
        self.resources -= needed
        if self.resources < 0:
            # not enough to feed everyone -> the deficit starves part of the pop
            starved = min(self.population, -self.resources)
            self.population -= starved
            self.resources = 0

        # normalise population to a non-negative integer; mark extinction
        self.population = max(0, int(self.population))
        if self.population <= 0:
            self.alive = False

    # --- actions -----------------------------------------------------------
    def explore(self):
        # add explored cells to self.explored_cells
        # range grows with science: more advanced civs see further
        radius = 1 + int(self.science // SCIENCE_PER_RANGE)
        origins = {self.cell}
        for planet in self._planets_of(self):
            c = getattr(planet, "cell", None)
            if c is not None:
                origins.add(c)

        before = len(self.explored_cells)
        for origin in origins:
            for c in origin.get_neighborhood(radius=radius, include_center=True):
                self.explored_cells.add(c)

        newly_explored = len(self.explored_cells) - before
        self.science += newly_explored * SCIENCE_PER_EXPLORE
        return self.explored_cells

    def increase_birth_rate(self):
        # increase birth rate by some amount
        self.birth_rate += BIRTH_RATE_STEP
        return self.birth_rate

    def broadcast_position(self):   # NOTE: your stub spelled this "boradcast_position"
        # share position with all civilizations
        newly_reached = 0
        for civ in self._all_civilizations():
            if self not in civ.known_civilizations:
                civ.known_civilizations.append(self)
                newly_reached += 1
            # everyone who hears the broadcast now knows where this civ lives
            civ.explored_cells.add(self.cell)
            self.explored_cells.add(civ.cell)
            if civ not in self.known_civilizations:
                self.known_civilizations.append(civ)

        self.science += newly_reached * SCIENCE_PER_BROADCAST

    def colonize_empty_planet(self, cell):
        # colonize an empty planet (cell) if it's in explored_cells
        if cell not in self.explored_cells:
            return False
        planet = self._planet_on(cell)
        if planet is None or planet.destroyed:
            return False
        if planet.civilization is not None:          # not actually empty
            return False
        if self.resources < COLONIZE_COST:
            return False
        self.resources -= COLONIZE_COST
        planet.civilization = self
        return True

    def destroy_planet(self, cell):
        # destroy a planet (cell) if it's in explored_cells
        if cell not in self.explored_cells:
            return False
        planet = self._planet_on(cell)
        if planet is None or planet.destroyed:
            return False
        if self.resources < DESTROY_COST:
            return False
        self.resources -= DESTROY_COST

        former_owner = planet.civilization
        planet.destroyed = True
        planet.civilization = None
        planet.resources = 0

        # if the former owner has nothing left, they're gone
        if former_owner is not None and former_owner is not self:
            if not self._planets_of(former_owner):
                self._wipe(former_owner)
        return True

    def colonize_inhabited_planet(self, cell):
        # colonize an inhabited planet (cell) if it's in explored_cells
        # and has a civilization  --> hostile takeover
        if cell not in self.explored_cells:
            return False
        planet = self._planet_on(cell)
        if planet is None or planet.destroyed:
            return False
        resident = planet.civilization
        if resident is None or resident is self:     # nobody else to conquer
            return False
        if self.resources < CONQUER_COST:
            return False
        self.resources -= CONQUER_COST

        if self.strength > resident.strength:
                # win: absorb part of their research, then seize the planet
                gained = resident.science * CONQUER_SCIENCE_FRACTION
                self.science += gained
                resident.science -= gained          # transfer; drop this line for a pure gain
                planet.civilization = self
                if not self._planets_of(resident):
                    self._wipe(resident)
                return True
        else:
            # lose: pay a population price for the failed invasion
            self.population = max(0, int(self.population * 0.75))
            return False