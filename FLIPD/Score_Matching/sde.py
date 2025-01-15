from abc import ABC, abstractmethod
import torch

def batch_mul(a, b):
    # PyTorch will automatically broadcast the shape of `a` to match `b`
    # so no need to manually unsqueeze `a`. Just ensure that `a` is 1D (or scalar) and `b` is 4D
    return a * b  # Efficient broadcasting

class SDE_Wrapper(ABC): 
    def __init__(self, discretization_steps): 
        super().__init__()
        self.discretization_steps = discretization_steps
        
    @abstractmethod
    def T(self):
        """
        End time of the SDE
        """ 
        pass 
    
    @abstractmethod
    def sde(self, x, t): 
        """
        SDE function

        Args:
            x (JAX Tensor): input data
            t (JAX Tensor): time vector 
            
        Returns:
            drift_term (JAX Tensor): drift term (f(x, t))
            diffusion_term (JAX Tensor): diffusion term (g(t))
        """
        pass
    
    @abstractmethod
    def marginal_probability(self, x, t): 
        """
        Compute the marginal probability of the SDE at time t

        Args:
            x (JAX Tensor): input data
            t (JAX Tensor): time vector 
            
        Returns:
            mean (JAX Tensor): mean of the marginal probability
            std (JAX Tensor): std of the marginal probability
        """
        pass 

    @abstractmethod
    def sampling_from_prior_distribution(self, x, t): 
        """
        Sample from the SDE at time t

        Args:
            x (JAX Tensor): input data
            t (JAX Tensor): time vector 
        """
        pass
    
    def get_forward_process_term_value(self, x, t): 
        """
        Add noise to the SDE at time t

        Args:
            x (JAX Tensor): input data
            t (JAX Tensor): time vector 
            
        Returns:
            F (JAX Tensor): drift term (f(x, t) * dt)
            G (JAX Tensor): diffusion term (g(t) * dW = g(t) * sqrt(dt))
        """
        
        dt = 1 / self.discretization_steps
        drift_term, diffusion_term = self.sde(x, t)
        
        F = drift_term * dt
        G = diffusion_term * torch.sqrt(dt)
        
        return F, G

class subVPSDE:
    def __init__(self,
                 beta_min: 0.1, 
                 beta_max: 20, 
                 discretization_steps: 1000):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = discretization_steps

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * batch_mul(beta_t, x)
        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_probability(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

        mean = batch_mul(torch.exp(log_mean_coeff), x)
        std = 1 - torch.exp(2. * log_mean_coeff)
        return mean, std

    def sampling_from_prior_distribution(self, shape):
        return torch.randn(shape)