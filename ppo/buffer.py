# ppo/buffer.py
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RolloutBuffer:
    def __init__(self):
        # Listas nativas para almacenar la experiencia del agente
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        """Limpia el buffer al final de cada ciclo de actualización."""
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        
    def to_tensors(self, device):
        """
        Convierte las listas acumuladas en tensores de PyTorch
        y los envía directamente al dispositivo configurado (CPU o GPU).
        """
        states_tensor = torch.stack(self.states).to(device).detach()
        actions_tensor = torch.stack(self.actions).to(device).detach()
        logprobs_tensor = torch.stack(self.logprobs).to(device).detach()
        
        return states_tensor, actions_tensor, logprobs_tensor