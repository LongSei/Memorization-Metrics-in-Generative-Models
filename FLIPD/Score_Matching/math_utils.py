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

def compute_gradient_with_jvp(score_net, t, x, y):
    """
    Compute the gradient of the score network with respect to the input data x using JVP.
    
    Args:
        score_net (torch.nn.Module): Score network model.
        t (torch.Tensor): Time vector.
        x (torch.Tensor): Input data.
        y (torch.Tensor): Conditional data.
        
    Returns:
        grad (torch.Tensor): Gradient of the score network with respect to x.
    """
    # Define the function to compute score_net output
    def fn(x_input):
        return score_net(t, x_input, y)

    # Initialize a tensor to store the gradients
    grad_x = torch.zeros_like(x)

    # Compute gradients for each basis vector (one-hot encoded)
    for i in range(x.numel()):
        # Create a basis vector for the JVP computation
        v = torch.zeros_like(x).view(-1)
        v[i] = 1
        v = v.view_as(x)
        
        # Perform JVP for the current basis vector
        _, jvp_result = torch.func.jvp(fn, (x,), (v,))
        
        # Accumulate the result in the gradient tensor
        grad_x.view(-1)[i] = jvp_result.view(-1).sum()

    return grad_x

def compute_trace_gradient(grad_x, use_hutch=False, num_queries=100):
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

    if use_hutch:
        # Hutch++ estimation (for large matrices)
        for batch in range(batch_size):
            for channel in range(channels):
                A = grad_x[batch, channel].view(H * W, H * W)  # Reshape to square matrix
                trace[batch, channel] = simple_hutchplusplus(A, num_queries=num_queries)
    else:
        # Direct trace computation (sum of diagonals)
        for batch in range(batch_size):
            for channel in range(channels):
                trace[batch, channel] = grad_x[batch, channel].diagonal(offset=0, dim1=-2, dim2=-1).sum()

    return trace
