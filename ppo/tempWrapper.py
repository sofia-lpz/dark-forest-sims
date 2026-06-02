import numpy as np

class DarkForestRLWrapper:
    """
    Envuelve el DarkGalaxy de la simulación para que funcione como un Gym de RL
    sin modificar su código original.
    """
    def __init__(self, galaxy_instance):
        self.galaxy = galaxy_instance
        # Asignamos un ID único a cada civilización
        self.civ_names = [f"Civilization_{i}" for i in range(self.galaxy.num_civilizations)]

    def reset(self):
        self.galaxy.reset()
        return self._extract_states()

    def step(self, action_dict):
        # 1. Aplicar las acciones a los objetos de la simulación ANTES del step
        for civ_name, action in action_dict.items():
            civ = self._get_civ_by_name(civ_name)
            if civ:
                self._apply_action(civ, action)

        # 2. Avanzar el tiempo en la galaxia
        is_running = self.galaxy.step() # Devuelve False si se alcanzó el max_steps

        # 3. Calcular los resultados para el PPO
        next_states = self._extract_states()
        rewards = self._calculate_rewards()
        
        # Si is_running es False, el juego terminó (done = True)
        dones = {name: not is_running for name in self.civ_names}
        
        return next_states, rewards, dones, {}

    def _get_civ_by_name(self, name):
        for civ in self.galaxy.civilizations:
            if civ.name == name:
                return civ
        return None

    def _extract_states(self):
        """Convierte y NORMALIZA los objetos Civilization a tensores."""
        state_dict = {}
        for civ in self.galaxy.civilizations:
            # Usamos log1p para domar el crecimiento exponencial de estas variables
            pop_norm = np.log1p(max(0, civ.population))
            sci_norm = np.log1p(max(0, civ.science))
            res_norm = np.log1p(max(0, civ.resources))
            
            # Normalizamos el resto dividiendo por un máximo teórico razonable
            rad_norm = civ.exploration_radius / 100.0 
            x_norm = civ.coordinates[0] / self.galaxy.grid_size[0] # Normaliza de 0 a 1
            y_norm = civ.coordinates[1] / self.galaxy.grid_size[1] # Normaliza de 0 a 1
            civs_norm = len(civ.known_civilizations) / self.galaxy.num_civilizations
            
            state = np.array([
                pop_norm,
                sci_norm,
                res_norm,
                rad_norm,
                x_norm,
                y_norm,
                civs_norm
            ], dtype=np.float32)
            
            state_dict[civ.name] = state
        return state_dict

    def _apply_action(self, civ, action):
        """
        Traduce el número que escupió la red neuronal (0, 1, 2, 3) a una acción real.
        """
        if action == 0:
            pass # No hacer nada
        elif action == 1:
            civ.science += 10 # Investigar
        elif action == 2:
            civ.exploration_radius += 5 # Expandir radar
        elif action == 3:
            civ.resources -= 10 # Consumir recursos

    def _calculate_rewards(self):
        """Calcula y ESCALA las recompensas para evitar gradientes explosivos."""
        reward_dict = {}
        for civ in self.galaxy.civilizations:
            # Usamos logaritmo nuevamente porque la ciencia base crece muchísimo
            base_reward = np.log1p(max(0, civ.science)) * 0.1 
            exploration_bonus = len(civ.known_civilizations) * 0.5
            
            # La recompensa final debe ser pequeña y manejable
            reward_dict[civ.name] = base_reward + exploration_bonus
        return reward_dict