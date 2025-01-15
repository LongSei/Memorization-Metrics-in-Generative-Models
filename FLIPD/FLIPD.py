from Score_Matching import * 
from math_utils import *
import torch

class FLIPD(): 
    """
    Class to implement the FLIPD algorithm to approximate Local Intrinsic Dimension
    """
    def __init__(self, 
                score_model: ScoreNet):
        self.score_net = score_model.model 
        self.sde = score_model.sde
        
    def get_mean_std(self, t_0, x): 
        """
        Compute the mean and std of the marginal probability of the SDE at time t_0 with input x

        Args:
            x (Tensor): input data
            t_0 (Tensor): time 
            
        Returns:
            mean (Tensor): mean of the marginal probability
            std (Tensor): std of the marginal probability
        """
        
        mean, std = self.sde.marginal_probability(t_0, x)
        return mean, std
    
    def get_total_dimension(self, x): 
        return x.shape[1] * x.shape[2] * x.shape[3]
    
    def flipd(self, t_0, x, y): 
        """ 
        Compute the FLIPD score for the input data x
        """
        
        mean, std = self.get_mean_std(t_0, x)
        d = self.get_total_dimension(x)
        
        score = self.score_net(t_0, mean, y)
        
        score_jacobian = compute_gradient(self.score_net, t_0, mean, y)
        score_trace = compute_trace_gradient(score_jacobian)
        score_norm = torch.sum(torch.square(score)) ** 0.5
        
        # Compute the FLIPD score
        flipd_score = d + std**2 * (score_trace + score_norm)
        return flipd_score

        