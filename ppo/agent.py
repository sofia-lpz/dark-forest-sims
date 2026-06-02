# ppo/agent.py
import torch
import torch.nn as nn
from ppo.model import ActorCritic
from ppo.buffer import RolloutBuffer

# Configuración agnóstica de hardware (CPU local / GPU servidor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        # Inicializamos la Política Actual
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        # Inicializamos la Política Vieja (necesaria para calcular el ratio r_t)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        """Selecciona una acción y guarda la experiencia en el buffer."""
        # Convertimos el estado (que viene del gym) a tensor y lo mandamos al device
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state_tensor)
        
        # Guardamos en el buffer (importante: guardamos como tensores o valores planos)
        self.buffer.states.append(state_tensor)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        
        return action.item()

    def update(self):
        """Actualiza las redes del Actor y el Crítico usando las trayectorias del buffer."""
        
        # --- 1. Calcular los Retornos Descontados (Monte Carlo) ---
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalización de recompensas (mejora mucho la estabilidad del entrenamiento)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convertir buffer a tensores
        old_states, old_actions, old_logprobs = self.buffer.to_tensors(device)

        # --- 2. Optimización PPO por K épocas ---
        for _ in range(self.K_epochs):
            
            # Evaluar las acciones viejas con la política ACTUAL
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Aplanar state_values para que coincida con la forma de rewards
            state_values = torch.squeeze(state_values)
            
            # Calcular la Ventaja: A_t = R_t - V(s_t)
            # Usamos detach() para que los gradientes no fluyan hacia el Crítico al actualizar el Actor
            advantages = rewards - state_values.detach()   
            
            # Calcular el Ratio: r_t = pi_theta / pi_theta_old
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Calcular las dos partes de la función de pérdida (Surrogate Loss)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # --- 3. Pérdida Final ---
            # Actor Loss = -min(surr1, surr2) -> Queremos MAXIMIZAR la ventaja, por lo que MINIMIZAMOS el negativo
            # Critic Loss = MSE(V(s_t), R_t) -> Queremos que el crítico prediga mejor el retorno real
            # Entropy = Fomenta la exploración
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Al final de la actualización, copiamos los pesos nuevos a la política vieja
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Limpiamos el buffer para el siguiente lote de episodios
        self.buffer.clear()