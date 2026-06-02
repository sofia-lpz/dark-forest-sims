# ppo/model.py
import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Red del Actor: Decide qué hacer en el Dark Forest
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1) # Genera una distribución de probabilidad válida
        )
        
        # Red del Crítico: Evalúa qué tan buena es la situación actual
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1) # Devuelve un único valor escalar V(s)
        )

    def forward(self):
        """Pytorch requiere sobreescribir forward, pero PPO separa la evaluación de la acción."""
        raise NotImplementedError
        
    def act(self, state):
        """El Actor toma una decisión basada en el estado actual."""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        """Evalúa acciones tomadas en el pasado para calcular pérdidas."""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy