import torch 

def simple_hutchplusplus(A, num_queries=10):
    """
    Estimates the trace of a square matrix A with num_queries many matrix-vector products.
    Implements the Hutch++ algorithm using random sign vectors.
    
    Args:
        A (torch.Tensor): A square matrix (n x n) whose trace is to be estimated.
        num_queries (int): Number of matrix-vector products allowed.
        
    Returns:
        trace_est (float): Estimated trace of the matrix A.
    """
    n = A.shape[0]

    # Generate random sign matrices S and G
    S = 2 * torch.randint(2, (n, (num_queries + 2) // 3), dtype=torch.float32) - 1
    G = 2 * torch.randint(2, (n, num_queries // 3), dtype=torch.float32) - 1

    # Compute the Q matrix from the QR decomposition of A*S
    Q, _ = torch.linalg.qr(A @ S, mode='reduced')

    # Orthogonalize G with respect to Q
    G = G - Q @ (Q.T @ G)

    # Estimate the trace
    trace_Q = torch.trace(Q.T @ A @ Q)  # First term: trace(Q^T A Q)
    trace_G = torch.trace(G.T @ A @ G)  # Second term: trace(G^T A G)

    trace_est = trace_Q + (1 / G.shape[1]) * trace_G

    return trace_est.item()

def compute_gradient(score_net, t, x, y):
    """
    Compute the gradient of the score network with respect to the input data x.
    
    Args:
        score_net (torch.nn.Module): Score network model.
        t (torch.Tensor): Time vector.
        x (torch.Tensor): Input data.
        y (torch.Tensor): Conditional data.
        
    Returns:
        grad (torch.Tensor): Gradient of the score network with respect to x.
    """
    x = x.clone().detach().requires_grad_(True)
    score = score_net(t, x, y)
    grad_x = torch.autograd.grad(outputs=score, inputs=x, grad_outputs=torch.ones_like(score), create_graph=True)[0]
    
    return grad_x

def compute_trace_gradient(grad_x): 
    """
    Compute the trace of the gradient of the score network.
    
    Args:
        grad_x (torch.Tensor): Gradient of the score network with respect to x.
        
    Returns:
        trace (torch.Tensor): Trace of the gradient of the score network.
    """
    batch_size = grad_x.shape[0]
    channels = grad_x.shape[1]
    
    # each trace got (channel) value
    trace = torch.zeros(batch_size, channels)
    
    for batch in range(batch_size): 
        for channel in range(channels): 
            # trace[batch, channel] = simple_hutchplusplus(grad_x[batch, channel], num_queries=1000)
            trace[batch, channel] = torch.trace(grad_x[batch, channel])
    return trace