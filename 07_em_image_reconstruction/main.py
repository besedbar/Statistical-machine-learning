#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import read_input, save_array_to_buffer, load_array_from_buffer, stable_softmax

parser = argparse.ArgumentParser()
parser.add_argument('input_file_path', type=str, help='Path to input .npz file')
parser.add_argument('output_file_path', nargs='?', default=None, type=str, help='Path to output file')
parser.add_argument('--brute', action='store_true', help='Evaluation in Brute')

def get_possible_shifts(max_shift):
    """
    Returns a list of tuples (dy, dx) for all possible shifts (k).
    Restricted to Horizontal shifts (0, dx) for this assignment.
    """
    return [(0, dx) for dx in range(-max_shift, max_shift+1)]

def e_step(X, template, shifts, sigma):
    """
    E-Step: Compute the optimal auxiliary variables alpha_x(k).
    
    alpha_x(k) = p(k | x; template, sigma)
    
    Args:
        X: (m, H, W) Dataset T_m containing m images.
        template: (H, W) Current estimate of the template (eta).
        shifts: List of k=(dy, dx) possible shifts.
        sigma: Current standard deviation of noise.
        
    Returns:
        alpha: (m, K) Matrix of posterior probabilities for each shift k given image x.
    """
    m = len(X)
    K = len(shifts)
    alpha = np.zeros((m, K))
    
    # Constant for Gaussian likelihood: 2 * sigma^2
    sigma_sq_2 = 2 * sigma**2
    # If it is tiny, do not square it
    if sigma_sq_2 < 1e-12: sigma_sq_2 = 1e-12
    
    for i in range(m):
        x_i = X[i]
        log_probs_unnorm = []
        
        for k, (dy, dx) in enumerate(shifts):
            # Hypothesis: Template shifted by k=(dy, dx)
            shifted_template = np.roll(template, shift=(dy, dx), axis=(0, 1))
            
            # TODO:
            log_prob =  - np.sum((x_i - shifted_template)**2) / sigma_sq_2
            log_probs_unnorm.append(log_prob)
            
        # TODO: 
        # Normalize with softmax to get alpha_x(k)
        # Subtract the maximum of the unnormalized log probs to get a stable normalization
        alpha[i, :] = stable_softmax(np.array(log_probs_unnorm))

    return alpha

def m_step_template(X, alpha, shifts, H, W):
    """
    M-Step (Part 1): Update the Template (eta).
    Maximizes the weighted log-likelihood with respect to the template.
    
    This effectively computes the ML estimate statistics, psi_k.
    """
    m = len(X)
    new_template = np.zeros((H, W))
    
    for i in range(m):
        x_i = X[i]
        alpha_i = alpha[i] # alpha_x(k) for current image x
        
        for k, (dy, dx) in enumerate(shifts):
            weight = alpha_i[k]
            # For speed, we skip the shifts with little to no effect
            if weight < 1e-8: continue
            
            # We align x_i back to the canonical frame
            # If the hypothesis was shift (dy, dx), we apply (-dy, -dx) to x_i
            unshifted_img = np.roll(x_i, shift=(-dy, -dx), axis=(0, 1))
            
            # TODO:
            new_template = new_template + weight * unshifted_img
      
    # TODO:      
    # Do not forget to normalize by dataset size m
    new_template =  new_template / m
    return new_template

def m_step_sigma(X, alpha, new_template, shifts):
    """
    M-Step (Part 2): Update the Noise Standard Deviation (sigma).
    """
    m, H, W = X.shape
    total_weighted_sq_error = 0.0
    
    for i in range(m):
        x_i = X[i]
        alpha_i = alpha[i]
        
        for k, (dy, dx) in enumerate(shifts):
            weight = alpha_i[k]
            # For speed, we skip the shifts with little to no effect
            if weight < 1e-8: continue
            
            # Compare x_i to the NEW template shifted by k
            shifted_template = np.roll(new_template, shift=(dy, dx), axis=(0, 1))
            
            # TODO:
            total_weighted_sq_error = total_weighted_sq_error + weight * np.sum((x_i - shifted_template)**2)
    
    
    # TODO:
    avg_variance = total_weighted_sq_error / (m * H * W)
    new_sigma = np.sqrt(avg_variance)
    return new_sigma 

def main(args):
    # Load Data (Dataset T_m)
    data = read_input(args.input_file_path)
    X = data['observations'] # Shape (m, H, W)
    max_shift = int(data['max_shift'])
    max_iter = int(data['max_iter'])
    tol = 1e-4

    m, H, W = X.shape
    shifts = get_possible_shifts(max_shift)
    
    # Initialize template with the naive average of samples and initialize sigma to 1
    template = np.mean(X, axis=0)
    sigma = 1.0 
    
    history_sigma = [sigma]
    
    # EM Loop
    print(f"Starting EM (m={m}, K={len(shifts)})...")
    
    for t in range(max_iter):
        # E-Step: Compute alpha_x(k)
        alpha = e_step(X, template, shifts, sigma)
        
        # M-Step: Update parameters (template, sigma)
        new_template = m_step_template(X, alpha, shifts, H, W)
        new_sigma = m_step_sigma(X, alpha, new_template, shifts)
        
        # Check Convergence
        t_diff = np.linalg.norm(new_template - template)
        s_diff = abs(new_sigma - sigma)
        
        template = new_template
        sigma = new_sigma
        history_sigma.append(sigma)
        
        if t_diff < tol and s_diff < tol:
            print(f"Converged at iteration {t+1}")
            break

    # Save or Visualize
    if args.brute:
        save_array_to_buffer(template, args.output_file_path)
    else:
        print("Comparing with reference solution...")
        try:
            sol_path = args.input_file_path.replace('instances', 'solutions').replace('.npz', '')
            reference = load_array_from_buffer(sol_path)
            
            if np.allclose(template, reference, atol=1e-4, rtol=1e-4):
                print("Test OK")
            else:
                print("Test Failed")
                print(f"Max Diff: {np.max(np.abs(template - reference)):.6f}")
        except FileNotFoundError:
            print("Reference solution not found. Skipping comparison.")

        print(f"Final Estimated Sigma: {sigma:.4f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Show one input observation x_i
        axes[0].imshow(X[0], cmap='gray', vmin=0, vmax=1)
        axes[0].set_title("Input Observation x_0")
        axes[0].axis('off')
        
        # Show Naive Average
        axes[1].imshow(np.mean(X, axis=0), cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Naive Average")
        axes[1].axis('off')
        
        # Show Reconstructed Template
        axes[2].imshow(template, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f"Reconstructed (EM)\nSigma={sigma:.3f}")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":            
    args = parser.parse_args()    
    main(args)