from abc import ABC, abstractmethod
import torch

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
    
    @abstractmethod
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

class DDPM(SDE_Wrapper): 
    def __init__(self, 
                beta_min: 0.1, 
                beta_max: 20, 
                discretization_steps: 1000):
        super().__init__(discretization_steps)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.discretization_steps = discretization_steps
        self.discrete_betas = torch.linspace(beta_min / discretization_steps, 
                                           beta_max / discretization_steps, 
                                           discretization_steps)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
    def T(self): 
        return 1
    
    def sde(self, x, t): 
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        drift_term = -0.5 * torch.multiply(beta_t, x)
        diffusion_term = torch.sqrt(beta_t)
        return drift_term, diffusion_term
    
    def marginal_probability(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        log_mean_coeff = log_mean_coeff.unsqueeze(2).unsqueeze(3)  
        mean = (torch.exp(log_mean_coeff) * x)
        std = torch.sqrt(1 - torch.exp(2. * log_mean_coeff))
        return mean, std    
    
    def sampling_from_prior_distribution(self, x, t):
        shape = x.shape
        return torch.randn(shape)
    
    def get_forward_process_term_value(self, x, t):
        timestep = (t * (self.discretization_steps - 1) / self.T()).astype(torch.int32)
        
        beta = self.discrete_betas[timestep]
        alpha = self.alphas[timestep]
        sqrt_beta = torch.sqrt(beta)
        
        F = torch.multiply(torch.sqrt(alpha), x) - x
        G = sqrt_beta
        return F, G