from abc import ABC, abstractmethod
from Score_Matching.sde import SDE_Wrapper
from Score_Matching.score_net import ScoreNet
import torch

class Sampler(ABC): 
    def __init__(self, 
                score_model: ScoreNet): 
        super().__init__()
        self.score_model = score_model
        
    @abstractmethod 
    def sample(self, 
               num_steps: int, 
               batch_size: int): 
        pass
        
class Predictor_Corrector_Sampler(Sampler): 
    def __init__(self, 
                score_model: ScoreNet,
                signal_to_noise_ratio, 
                device='cuda', 
                epsilon=1e-5): 
        super().__init__(score_model=score_model)
        self.signal_to_noise_ratio = signal_to_noise_ratio
        self.device = device
        self.epsilon = epsilon
        
    def sample(self, 
               num_steps: int,
               batch_size: int): 
        t = torch.linspace(0, self.score_model.sde.T(), num_steps).to(self.device)
        x_T = torch.randn(batch_size, 
                          self.score_model.sde.dim[0], self.score_model.sde.dim[1], self.score_model.sde.dim[2]).to(self.device)
        std = self.score_model.sde.marginal_probability(x_T, t)[1][:, None, None, None]
        x_T = x_T * std 
        
        time_steps = torch.linspace(1., self.epsilon, num_steps)
        step_size = time_steps[0] - time_steps[1]
        
        image_state = x_T 
        
        with torch.no_grad(): 
            for step in range(num_steps): 
                batch_time_step = torch.ones(batch_size, device=self.score_model.device) * step
                
                score = self.score_model.model(batch_time_step, image_state)
                score_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
                
                # Corrector step (Langevin MCMC)
                noise_norm = torch.sqrt(torch.prod(image_state.shape[1:]))
                langevin_step_size = 2 * (self.signal_to_noise_ratio * noise_norm / score_norm)**2
                image_state = image_state + langevin_step_size * score + torch.sqrt(2 * langevin_step_size) * torch.randn_like(image_state)
                
                # Predictor step (Euler-Maruyama)
                g = self.score_model.sde.sde(torch.zeros_like(image_state), batch_time_step)[1]
                x_mean = image_state + (g**2)[:, None, None, None] * self.score_model.model(image_state, batch_time_step) * step_size
                image_state = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(image_state)
                
        return x_mean