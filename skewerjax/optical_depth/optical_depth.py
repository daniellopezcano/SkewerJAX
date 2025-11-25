from __future__ import annotations

from skewerjax.optical_depth.voigt import log10_voigt

import logging
logger = logging.getLogger(__name__)

import jax
from jax import vmap, pmap
import jax.numpy as jnp
from functools import partial

def compute_line_profiles(nu, sigma, gamma, mode="ExoJAX", compute_mode="pmap", num_gpus=None):
    """
    Compute Voigt profiles for all species, spatial points, and wavelengths.
    
    Args:
        nu:      Shape (N_sp, N_sk, N_ppsk, N_lambda) — Frequency offsets (vary per point!)
        sigma:   Shape (N_sp, N_sk, N_ppsk)           — Thermal broadening
        gamma:   Shape (N_sp, N_sk, N_ppsk)           — Damping
        mode:    String, e.g., "ExoJAX"
        compute_mode: String, "vmap" or "pmap"
        num_gpus: Optional[int], number of GPUs for pmap. If None, uses all available.
    
    Returns:
        log10_phi: Shape (N_sp, N_sk, N_ppsk, N_lambda)
    """
    N_sp, N_sk, N_ppsk, N_lambda = nu.shape
    N_total = N_sp * N_sk * N_ppsk
    
    # Flatten EVERYTHING: each spatial point gets its own 1D frequency array
    nu_flat = nu.reshape(N_total, -1)      # (N_total, N_lambda)
    sigma_flat = sigma.reshape(N_total)          # (N_total,)
    gamma_flat = gamma.reshape(N_total)          # (N_total,)
    
    # ============================================================
    # VMODE: Single-device vectorization
    # ============================================================
    if compute_mode == "vmap":
        log10_phi_flat = vmap(lambda x, s, g: log10_voigt(x, s, g, mode=mode), in_axes=(0, 0, 0))(nu_flat, sigma_flat, gamma_flat)
        # Result: (N_total, N_lambda)
    
    # ============================================================
    # PMODE: Multi-device parallelization
    # ============================================================
    elif compute_mode == "pmap":
        # 1. Detect GPUs
        if num_gpus is None:
            num_gpus = jax.local_device_count()
            logger.info(f"Detected {num_gpus} GPU(s) for pmap.")
        
        # 2. Handle N_total not divisible by num_gpus
        if N_total % num_gpus != 0:
            pad_size = num_gpus - (N_total % num_gpus)
            logger.info(f"⚠️  Padding {pad_size} elements for pmap (N_total={N_total}, num_gpus={num_gpus})")
            
            nu_flat = jnp.pad(nu_flat, ((0, pad_size), (0, 0)), mode='constant')
            sigma_flat = jnp.pad(sigma_flat, (0, pad_size), mode='constant')
            gamma_flat = jnp.pad(gamma_flat, (0, pad_size), mode='constant')
            N_total_padded = N_total + pad_size
        else:
            N_total_padded = N_total
        
        N_per_gpu = N_total_padded // num_gpus
        
        # 3. Shard across devices: (num_gpus, N_per_gpu, ...)
        nu_sharded = nu_flat.reshape(num_gpus, N_per_gpu, N_lambda)
        sigma_sharded = sigma_flat.reshape(num_gpus, N_per_gpu)
        gamma_sharded = gamma_flat.reshape(num_gpus, N_per_gpu)
        
        # 4. Define per-device computation
        @partial(pmap, in_axes=(0, 0, 0))
        def compute_on_device(nu_shard, sigma_shard, gamma_shard):
            """
            Runs on ONE device. Inside, vmap handles the per-device batch.
            """
            return vmap(
                lambda x, s, g: log10_voigt(x, s, g, mode=mode),
                in_axes=(0, 0, 0)
            )(nu_shard, sigma_shard, gamma_shard)
            # Returns: (N_per_gpu, N_lambda)
        
        # 5. Execute across all devices
        log10_phi_padded = compute_on_device(nu_sharded, sigma_sharded, gamma_sharded)
        # Result: (num_gpus, N_per_gpu, N_lambda)
        
        # 6. Flatten back and remove padding
        log10_phi_flat = log10_phi_padded.reshape(N_total_padded, N_lambda)[:N_total, :]
    
    else:
        raise ValueError(f"compute_mode must be 'vmap' or 'pmap', got '{compute_mode}'")
    
    # ============================================================
    # Reshape to original structure (N_sp, N_sk, N_ppsk, N_lambda)
    # ============================================================
    return 10**log10_phi_flat.reshape(N_sp, N_sk, N_ppsk, -1)