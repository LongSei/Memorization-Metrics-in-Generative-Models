import torch

class FLIPD(): 
    def __init__(self, 
                 pipeline,
                 device): 
        self.pipeline = pipeline
        self.device = device
        
    def compute_gradient(self, t, x, guidance_scale, prompt): 
        x = x.clone().detach().requires_grad_(True)
        score = self.pipeline.score_fn(t, x, guidance_scale, prompt)
        return torch.autograd.grad(outputs=score, inputs=x, grad_outputs=torch.ones_like(score), create_graph=True)[0]

    def compute_trace_gradient(self, grad_x, use_hutch=False, num_queries=100):
        """
        Compute the trace of the gradient of the score network.
        
        Args:
            grad_x (torch.Tensor): Gradient of the score network with respect to x, shape (batch_size, channels, H, W).
            use_hutch (bool): Whether to use the Hutch++ algorithm for trace estimation.
            num_queries (int): Number of matrix-vector products for Hutch++ if used.
            
        Returns:
            trace (torch.Tensor): Trace of the gradient of the score network for each batch and channel.
        """
        batch_size, channels, H, W = grad_x.shape

        # Initialize a tensor to store trace values
        trace = torch.zeros(batch_size, channels, device=grad_x.device)
        # Direct trace computation (sum of diagonals)
        for batch in range(batch_size):
            for channel in range(channels):
                trace[batch, channel] = grad_x[batch, channel].diagonal(offset=0, dim1=-2, dim2=-1).sum()

        return trace

    def flipd(self, t_0, x, guidance_scale, prompt): 
        """ 
        Compute the FLIPD score for the input data x
        """
        
        mean, std = self.pipeline.get_mean_and_std(t_0, x)
        
        score = self.pipeline.score_fn(t_0, mean, guidance_scale, prompt)
        
        score_jacobian = self.compute_gradient(t_0, x, guidance_scale=guidance_scale, prompt=prompt)
        score_trace = self.compute_trace_gradient(score_jacobian)
        score_norm = torch.sum(torch.square(score)) ** 0.5
        
        # Compute the FLIPD score
        flipd_score = std**2 * (score_trace + score_norm)
        return flipd_score