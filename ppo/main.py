# ppo/main.py
from curses import wrapper
import sys
import os
import torch


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from galaxy import DarkGalaxy

from agent import PPOAgent

from tempWrapper import DarkForestRLWrapper

def train():
    print("Iniciando simulación del Dark Forest...")

    raw_env = DarkGalaxy() 
    env = DarkForestRLWrapper(raw_env)
    
    # Configuración de espacios de observación y acción
    state_dim = 7
    action_dim = 4
    
    # Creación dinámica de agentes PPO
    agents = {}
    for civ_name in env.civ_names:
        agents[civ_name] = PPOAgent(state_dim, action_dim)
        
    print(f"Agentes creados exitosamente: {list(agents.keys())}")

    max_episodes = 500
    max_timesteps = 200 # Límite de turnos por partida
    update_timestep = 400 # Frecuencia de actualización de la red neuronal
    
    time_step = 0
    
    # Bucle de entrenamiento
    for episode in range(1, max_episodes + 1):
        # Reiniciar entorno
        state_dict = env.reset() 
        episode_rewards = {civ_id: 0 for civ_id in agents.keys()}
        
        for t in range(max_timesteps):
            time_step += 1
            action_dict = {}
            
            # Elección de acción por agente
            for civ_id, agent in agents.items():
                estado_actual = state_dict[civ_id]
                accion_elegida = agent.select_action(estado_actual)
                action_dict[civ_id] = accion_elegida
                
            # Ejecutar acciones simultáneas en el entorno
            next_state_dict, reward_dict, done_dict, _ = env.step(action_dict)
            
            # Guardar recompensas en buffer de cada agente
            for civ_id, agent in agents.items():
                agent.buffer.rewards.append(reward_dict[civ_id])
                agent.buffer.is_terminals.append(done_dict[civ_id])
                episode_rewards[civ_id] += reward_dict[civ_id]
                
            state_dict = next_state_dict
            
            # Actualizar redes neuronales periódicamente
            if time_step % update_timestep == 0:
                print(f"Actualizando redes neuronales en el paso {time_step}...")
                for agent in agents.values():
                    agent.update()
                    
            # Finalizar si todos los agentes terminaron
            if all(done_dict.values()):
                break
                
        print(f"Episodio {episode} completado. Recompensas: {episode_rewards}")

if __name__ == '__main__':
    train()