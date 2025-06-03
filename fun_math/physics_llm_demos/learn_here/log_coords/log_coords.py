import torch
import numpy as np

def log_cartesian_distance(x1, x2, epsilon=1e-8):
    """
    Calculate distances in log-cylindrical space for more efficient attention.
    
    Args:
        x1 (torch.Tensor): First set of coordinates in log-cylindrical space (ln_r, theta, z)
                          Shape: [batch_size, seq_len_1, 3]
        x2 (torch.Tensor): Second set of coordinates in log-cylindrical space (ln_r, theta, z)
                          Shape: [batch_size, seq_len_2, 3]
        epsilon (float): Small value for numerical stability
        
    Returns:
        torch.Tensor: Pairwise distances in log-cylindrical space
                     Shape: [batch_size, seq_len_1, seq_len_2]
    """
    # Extract log-radial, angular, and z components
    # x[..., 0] = ln(r), x[..., 1] = theta, x[..., 2] = z
    ln_r1, theta1, z1 = x1[..., 0:1], x1[..., 1:2], x1[..., 2:3]  # Shape: [B, S1, 1]
    ln_r2, theta2, z2 = x2[..., 0:1], x2[..., 1:2], x2[..., 2:3]  # Shape: [B, S2, 1]
    
    # Calculate differences in each coordinate
    # For ln_r and z, we can just take the difference
    delta_ln_r = ln_r1.unsqueeze(-2) - ln_r2.unsqueeze(-3)  # Shape: [B, S1, S2, 1]
    delta_z = z1.unsqueeze(-2) - z2.unsqueeze(-3)  # Shape: [B, S1, S2, 1]
    
    # For theta (angle), we need to handle the circular nature
    # We want the smallest angle between the two points
    delta_theta = theta1.unsqueeze(-2) - theta2.unsqueeze(-3)  # Shape: [B, S1, S2, 1]
    delta_theta = torch.remainder(delta_theta + np.pi, 2 * np.pi) - np.pi  # Wrap to [-pi, pi]
    
    # Calculate squared distance in log-cylindrical space
    # Distance formula: sqrt((delta_ln_r)^2 + (delta_theta)^2 + (delta_z)^2)
    squared_dist = delta_ln_r**2 + delta_theta**2 + delta_z**2
    
    # Return euclidean distance
    return torch.sqrt(squared_dist + epsilon).squeeze(-1)  # Shape: [B, S1, S2]

def log_cylindrical_to_cartesian(ln_r, theta, z):
    """
    Convert log-cylindrical coordinates to Cartesian coordinates.
    
    Args:
        ln_r (torch.Tensor): Natural log of radial coordinate
        theta (torch.Tensor): Angular coordinate in radians
        z (torch.Tensor): Vertical coordinate
        
    Returns:
        tuple: (x, y, z) Cartesian coordinates
    """
    r = torch.exp(ln_r)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return x, y, z

def cartesian_to_log_cylindrical(x, y, z, epsilon=1e-8):
    """
    Convert Cartesian coordinates to log-cylindrical coordinates.
    
    Args:
        x (torch.Tensor): x-coordinate
        y (torch.Tensor): y-coordinate
        z (torch.Tensor): z-coordinate
        epsilon (float): Small value for numerical stability
        
    Returns:
        tuple: (ln_r, theta, z) log-cylindrical coordinates
    """
    r = torch.sqrt(x**2 + y**2 + epsilon)
    ln_r = torch.log(r)
    theta = torch.atan2(y, x)
    return ln_r, theta, z

def compute_attention_weights(q, k, scale_factor=1.0):
    """
    Compute attention weights based on log-cylindrical distances.
    
    Args:
        q (torch.Tensor): Query tensor in log-cylindrical space
                         Shape: [batch_size, seq_len_q, 3]
        k (torch.Tensor): Key tensor in log-cylindrical space
                         Shape: [batch_size, seq_len_k, 3]
        scale_factor (float): Scaling factor for attention weights
        
    Returns:
        torch.Tensor: Attention weights
                     Shape: [batch_size, seq_len_q, seq_len_k]
    """
    # Calculate log-cylindrical distances
    distances = log_cartesian_distance(q, k)
    
    # Convert distances to attention weights (smaller distance = higher attention)
    # Using exponential decay: exp(-distance * scale_factor)
    attention_weights = torch.exp(-distances * scale_factor)
    
    # Normalize attention weights
    attention_weights = attention_weights / (torch.sum(attention_weights, dim=-1, keepdim=True) + 1e-8)
    
    return attention_weights

def log_cylindrical_batch_distance(coords1, coords2, epsilon=1e-8):
    """
    Efficiently calculate distances between batches of log-cylindrical coordinates
    using einsum operations for better performance.
    
    Args:
        coords1 (torch.Tensor): First batch of coordinates (ln_r, theta, z)
                               Shape: [batch_size, seq_len_1, 3]
        coords2 (torch.Tensor): Second batch of coordinates (ln_r, theta, z)
                               Shape: [batch_size, seq_len_2, 3]
        epsilon (float): Small value for numerical stability
        
    Returns:
        torch.Tensor: Pairwise distances
                     Shape: [batch_size, seq_len_1, seq_len_2]
    """
    # Extract components from coords1
    ln_r1 = coords1[..., 0]  # Shape: [batch_size, seq_len_1]
    theta1 = coords1[..., 1]  # Shape: [batch_size, seq_len_1]
    z1 = coords1[..., 2]  # Shape: [batch_size, seq_len_1]
    
    # Extract components from coords2
    ln_r2 = coords2[..., 0]  # Shape: [batch_size, seq_len_2]
    theta2 = coords2[..., 1]  # Shape: [batch_size, seq_len_2]
    z2 = coords2[..., 2]  # Shape: [batch_size, seq_len_2]
    
    # Calculate squared differences using einsum for efficiency
    # For ln_r and z
    ln_r1_sq = torch.einsum('bi,bi->bi', ln_r1, ln_r1)  # [batch, seq_len_1]
    ln_r2_sq = torch.einsum('bj,bj->bj', ln_r2, ln_r2)  # [batch, seq_len_2]
    ln_r_cross = torch.einsum('bi,bj->bij', ln_r1, ln_r2)  # [batch, seq_len_1, seq_len_2]
    
    z1_sq = torch.einsum('bi,bi->bi', z1, z1)  # [batch, seq_len_1]
    z2_sq = torch.einsum('bj,bj->bj', z2, z2)  # [batch, seq_len_2]
    z_cross = torch.einsum('bi,bj->bij', z1, z2)  # [batch, seq_len_1, seq_len_2]
    
    # Calculate delta_ln_r^2 and delta_z^2 components
    delta_ln_r_sq = ln_r1_sq.unsqueeze(-1) + ln_r2_sq.unsqueeze(-2) - 2 * ln_r_cross
    delta_z_sq = z1_sq.unsqueeze(-1) + z2_sq.unsqueeze(-2) - 2 * z_cross
    
    # For theta, we need to handle circular distance
    # exp(i*theta) gives us a complex number on the unit circle
    # We can use the dot product to find the cosine of the angle difference
    exp_i_theta1 = torch.stack([torch.cos(theta1), torch.sin(theta1)], dim=-1)  # [batch, seq_len_1, 2]
    exp_i_theta2 = torch.stack([torch.cos(theta2), torch.sin(theta2)], dim=-1)  # [batch, seq_len_2, 2]
    
    # Calculate cos(theta1 - theta2) using dot product
    cos_delta_theta = torch.einsum('bik,bjk->bij', exp_i_theta1, exp_i_theta2)  # [batch, seq_len_1, seq_len_2]
    
    # delta_theta^2 = 2 - 2*cos(delta_theta)
    delta_theta_sq = 2.0 - 2.0 * torch.clamp(cos_delta_theta, min=-1.0, max=1.0)
    
    # Calculate total squared distance
    squared_dist = delta_ln_r_sq + delta_theta_sq + delta_z_sq
    
    # Return euclidean distance
    return torch.sqrt(squared_dist + epsilon)