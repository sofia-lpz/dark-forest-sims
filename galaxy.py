
"""
Dark Forest Simulation - Main training loop for RL agents
"""

import numpy as np
from planet import Planet
from civ import Civilization

class DarkGalaxy:
    """Simulation environment for dark forest civilizations"""
    
    def __init__(self, num_planets=5, num_civilizations=2, max_steps=1000):
        self.num_planets = num_planets
        self.num_civilizations = num_civilizations
        self.max_steps = max_steps
        self.current_step = 0
        
        self.planets = []
        self.civilizations = []
        self.history = []
        
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize planets and civilizations"""
        # Create planets with varying resources
        for i in range(self.num_planets):
            planet = Planet(
                name=f"Planet_{i}",
                size=np.random.randint(50, 200),
                resources=np.random.randint(100, 500)
            )
            self.planets.append(planet)
        
        # Create civilizations
        colors = ["Red", "Blue", "Green", "Yellow", "Purple"]
        for i in range(self.num_civilizations):
            civ = Civilization(
                name=f"Civilization_{i}",
                color=colors[i % len(colors)]
            )
            civ.population = 100
            civ.science = 0
            civ.resources = 200
            
            # Place civilization on first planet
            self.planets[0].civilizations.append(civ)
            self.civilizations.append(civ)

    def step(self, actions=None):
        """
        Advance simulation by one step
        """
        if self.current_step >= self.max_steps:
            return False
            
        self.current_step += 1
        
        # Update each civilization
        for civ in self.civilizations:
            if hasattr(civ, 'update_population'):
                civ.update_population()
            if hasattr(civ, 'update_science'):
                civ.update_science()
            if hasattr(civ, 'update_resources'):
                civ.update_resources()
        
        # Regenerate planet resources
        for planet in self.planets:
            if not planet.destroyed:
                planet.resources += max(1, planet.size // 10)
        
        # Record history
        self.history.append({
            'step': self.current_step,
            'civilizations': len(self.civilizations),
            'planets': len(self.planets)
        })
        
        return True
    
    def reset(self):
        """Reset simulation to initial state"""
        self.current_step = 0
        self.planets = []
        self.civilizations = []
        self.history = []
        self._initialize_environment()
        