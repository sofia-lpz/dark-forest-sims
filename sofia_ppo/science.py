# science.py – action registry and science-gated action functions

# Each time a civ's science crosses a multiple of SCIENCE_THRESHOLD,
# it unlocks the next tier of actions (in order of science_dict index).
SCIENCE_THRESHOLD = 100

# Unlock order (1-indexed). A civ with tech_level >= N can use action N.
science_dict = {
    1: "Increase population rate x1",          # +1 to birth_rate
    2: "Broadcast: I exist",
    3: "Broadcast: I am here",
    4: "Broadcast: He exists",
    5: "Broadcast: He is here",
    6: "Colonize empty planet",
    7: "Spy",
    8: "Colonize inhabited planet",
    9: "Destroy planet",
}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_tech_level(science: float) -> int:
    """Return how many tiers have been unlocked (floor(science / THRESHOLD))."""
    return int(science // SCIENCE_THRESHOLD)


def action_unlocked(civ, action_index: int) -> bool:
    """True when the civ's tech level is high enough to use this action."""
    return civ.tech_level >= action_index


# ---------------------------------------------------------------------------
# Tier 1 – Increase population rate
# ---------------------------------------------------------------------------

def increase_population_rate(civ) -> str:
    """
    Unlock index 1.
    Permanently raises birth_rate by 1 and costs 50 resources.
    Can be called multiple times.
    """
    COST = 50
    if not action_unlocked(civ, 1):
        return f"{civ.name}: 'Increase population rate' not yet unlocked."
    if civ.resources < COST:
        return f"{civ.name}: insufficient resources ({civ.resources:.1f} < {COST})."

    civ.resources -= COST
    civ.birth_rate += 1
    return (f"{civ.name} raised birth_rate to {civ.birth_rate} "
            f"(cost {COST} resources).")


# ---------------------------------------------------------------------------
# Tier 2-5 – Broadcasts
# Civs can only broadcast about known civilizations (for He exists / He is here).
# ---------------------------------------------------------------------------

def _do_broadcast(sender, galaxy, message: str) -> str:
    """
    Internal: deliver a broadcast to every civ whose explored radius
    overlaps the sender's home planet.
    """
    recipients = []
    for civ in galaxy.civilizations:
        if civ is sender:
            continue
        # A civ can hear if their exploration radius reaches the sender's coords.
        dist = _distance(civ.coordinates, sender.coordinates)
        if dist <= civ.exploration_radius:
            if sender not in civ.known_civilizations:
                civ.known_civilizations.append(sender)
                civ.known_civilization_coordinates.append(sender.coordinates)
            recipients.append(civ.name)

    if recipients:
        return (f"[BROADCAST] {sender.name} → '{message}' "
                f"heard by: {', '.join(recipients)}")
    return f"[BROADCAST] {sender.name} → '{message}' – no one in range."


def broadcast_i_exist(sender, galaxy) -> str:
    """Unlock index 2. Sender announces its own existence."""
    if not action_unlocked(sender, 2):
        return f"{sender.name}: 'Broadcast: I exist' not yet unlocked."
    return _do_broadcast(sender, galaxy, "I exist")


def broadcast_i_am_here(sender, galaxy) -> str:
    """Unlock index 3. Sender announces its own coordinates."""
    if not action_unlocked(sender, 3):
        return f"{sender.name}: 'Broadcast: I am here' not yet unlocked."
    # Reveal exact coordinates to hearers
    msg = f"I am here @ {sender.coordinates}"
    result = _do_broadcast(sender, galaxy, msg)
    # Ensure hearers also get the accurate coordinate
    for civ in galaxy.civilizations:
        if civ is sender:
            continue
        dist = _distance(civ.coordinates, sender.coordinates)
        if dist <= civ.exploration_radius:
            _update_known_coord(civ, sender, sender.coordinates)
    return result


def broadcast_he_exists(sender, galaxy, target_civ) -> str:
    """
    Unlock index 4.
    Sender reveals that target_civ exists. sender must know target_civ.
    """
    if not action_unlocked(sender, 4):
        return f"{sender.name}: 'Broadcast: He exists' not yet unlocked."
    if target_civ not in sender.known_civilizations:
        return (f"{sender.name}: cannot broadcast about unknown civ "
                f"'{target_civ.name}'.")
    msg = f"He exists: {target_civ.name}"
    result = _do_broadcast(sender, galaxy, msg)
    # Hearers now know the target exists (but not where)
    for civ in galaxy.civilizations:
        if civ is sender:
            continue
        dist = _distance(civ.coordinates, sender.coordinates)
        if dist <= civ.exploration_radius:
            if target_civ not in civ.known_civilizations:
                civ.known_civilizations.append(target_civ)
                civ.known_civilization_coordinates.append(None)  # location unknown
    return result


def broadcast_he_is_here(sender, galaxy, target_civ) -> str:
    """
    Unlock index 5.
    Sender reveals target_civ's coordinates. sender must know target_civ.
    """
    if not action_unlocked(sender, 5):
        return f"{sender.name}: 'Broadcast: He is here' not yet unlocked."
    if target_civ not in sender.known_civilizations:
        return (f"{sender.name}: cannot broadcast about unknown civ "
                f"'{target_civ.name}'.")
    idx = sender.known_civilizations.index(target_civ)
    known_coords = sender.known_civilization_coordinates[idx]
    if known_coords is None:
        return (f"{sender.name}: knows {target_civ.name} exists but "
                f"not its coordinates (spy first).")
    msg = f"He is here: {target_civ.name} @ {known_coords}"
    result = _do_broadcast(sender, galaxy, msg)
    for civ in galaxy.civilizations:
        if civ is sender:
            continue
        dist = _distance(civ.coordinates, sender.coordinates)
        if dist <= civ.exploration_radius:
            _update_known_coord(civ, target_civ, known_coords)
    return result


# ---------------------------------------------------------------------------
# Tier 6 – Colonize empty planet
# ---------------------------------------------------------------------------

def colonize_empty_planet(civ, planet, population_sent: float) -> str:
    """
    Unlock index 6.
    The planet must be within the civ's explored radius AND uninhabited.
    Sends `population_sent` people to settle the planet.
    Costs 100 resources.
    """
    COST = 100
    MIN_POP = 10
    if not action_unlocked(civ, 6):
        return f"{civ.name}: 'Colonize empty planet' not yet unlocked."
    if planet.destroyed:
        return f"{civ.name}: {planet.name} is destroyed."
    if planet.civilizations:
        return (f"{civ.name}: {planet.name} is already inhabited – "
                f"use 'Colonize inhabited planet' instead.")

    dist = _distance(civ.coordinates, planet.coordinates)
    if dist > civ.exploration_radius:
        return (f"{civ.name}: {planet.name} is outside explored radius "
                f"(dist={dist:.1f}, radius={civ.exploration_radius:.1f}).")
    if civ.resources < COST:
        return f"{civ.name}: insufficient resources for colonization."
    if civ.population < population_sent + MIN_POP:
        return f"{civ.name}: not enough population to send {population_sent}."

    civ.resources -= COST
    civ.population -= population_sent

    # Create a new branch civilization on the target planet
    from civ import Civilization
    colony = Civilization(name=f"{civ.name}_colony_{planet.name}",
                          color=civ.color)
    colony.population = population_sent
    colony.resources = 50
    colony.science = civ.science * 0.5   # colony starts with half the parent's science
    colony.tech_level = civ.tech_level
    colony.birth_rate = civ.birth_rate
    colony.death_rate = civ.death_rate
    colony.coordinates = planet.coordinates
    colony.exploration_radius = 0

    planet.civilizations.append(colony)

    # Register colony in the galaxy (caller must handle this if needed)
    return (f"{civ.name} colonized {planet.name} "
            f"(sent {population_sent} people, cost {COST} resources). "
            f"Colony '{colony.name}' established.")


# ---------------------------------------------------------------------------
# Tier 7 – Spy
# ---------------------------------------------------------------------------

def spy(civ, target_civ, resources_invested: float) -> str:
    """
    Unlock index 7.
    Invest resources to learn the target's exact coordinates.
    civ must already know target_civ.
    Success chance scales with resources invested (50 = ~50 %).
    """
    if not action_unlocked(civ, 7):
        return f"{civ.name}: 'Spy' not yet unlocked."
    if target_civ not in civ.known_civilizations:
        return (f"{civ.name}: cannot spy on unknown civ '{target_civ.name}'. "
                f"Learn of their existence first.")
    if civ.resources < resources_invested:
        return f"{civ.name}: insufficient resources to invest in espionage."

    import random
    civ.resources -= resources_invested
    # Success probability: 1 - e^(-invested/50), capped at 0.95
    success_chance = min(0.95, 1 - 2.718 ** (-resources_invested / 50))
    if random.random() < success_chance:
        _update_known_coord(civ, target_civ, target_civ.coordinates)
        return (f"{civ.name} successfully spied on {target_civ.name}! "
                f"Coordinates revealed: {target_civ.coordinates}.")
    else:
        return (f"{civ.name}'s spy operation on {target_civ.name} failed "
                f"(invested {resources_invested:.1f} resources).")


# ---------------------------------------------------------------------------
# Tier 8 – Colonize inhabited planet  (stub)
# ---------------------------------------------------------------------------

def colonize_inhabited_planet(civ, planet, population_sent: float) -> str:
    """
    Unlock index 8.
    Not yet implemented – placeholder for future combat/diplomacy logic.
    """
    if not action_unlocked(civ, 8):
        return f"{civ.name}: 'Colonize inhabited planet' not yet unlocked."
    # TODO: implement conflict / diplomacy resolution
    return (f"{civ.name}: colonize inhabited planet ({planet.name}) "
            f"– not yet implemented.")


# ---------------------------------------------------------------------------
# Tier 9 – Destroy planet
# ---------------------------------------------------------------------------

def destroy_planet(civ, planet) -> str:
    """
    Unlock index 9.
    Destroys the target planet (sets destroyed=True, clears civilizations).
    We assume the civ knows all surrounding planet coordinates.
    Costs 500 resources and 20 % of current population.
    """
    RESOURCE_COST = 500
    POP_COST_FRAC = 0.20
    if not action_unlocked(civ, 9):
        return f"{civ.name}: 'Destroy planet' not yet unlocked."
    if planet.destroyed:
        return f"{civ.name}: {planet.name} is already destroyed."
    if civ.resources < RESOURCE_COST:
        return (f"{civ.name}: insufficient resources to destroy "
                f"{planet.name} (need {RESOURCE_COST}).")

    pop_cost = civ.population * POP_COST_FRAC
    civ.resources -= RESOURCE_COST
    civ.population -= pop_cost

    casualties = sum(c.population for c in planet.civilizations)
    planet.civilizations.clear()
    planet.destroyed = True
    planet.resources = 0

    return (f"{civ.name} DESTROYED {planet.name}! "
            f"{casualties:.0f} killed. "
            f"Cost: {RESOURCE_COST} resources, {pop_cost:.0f} population.")


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _distance(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _update_known_coord(observer_civ, known_civ, coords):
    """Store or update the coordinates of a known civilization."""
    if known_civ in observer_civ.known_civilizations:
        idx = observer_civ.known_civilizations.index(known_civ)
        observer_civ.known_civilization_coordinates[idx] = coords
    else:
        observer_civ.known_civilizations.append(known_civ)
        observer_civ.known_civilization_coordinates.append(coords)