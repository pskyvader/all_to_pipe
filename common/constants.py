"""
All-to-Pipe constants module.

Centralized constant definitions used by validators and nodes.
"""

from typing import List, Final

# Import samplers and schedulers directly from ComfyUI source
try:
    from comfy.samplers import SAMPLER_NAMES, SCHEDULER_NAMES
    SUPPORTED_SAMPLERS: Final[List[str]] = list(SAMPLER_NAMES)
    SUPPORTED_SCHEDULERS: Final[List[str]] = list(SCHEDULER_NAMES)
except ImportError:
    # Fallback to hardcoded lists if ComfyUI import fails
    # This ensures the module works even without ComfyUI in environment
    SUPPORTED_SAMPLERS: Final[List[str]] = [
        "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp",
        "heun", "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde",
        "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive",
        "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
        "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_cfg_pp",
        "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu",
        "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v",
        "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral",
        "res_multistep_ancestral_cfg_pp", "gradient_estimation", "gradient_estimation_cfg_pp",
        "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece",
        "ddim", "uni_pc", "uni_pc_bh2",
    ]
    
    SUPPORTED_SCHEDULERS: Final[List[str]] = [
        "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform",
        "beta", "normal", "linear_quadratic", "kl_optimal",
    ]

# Model subfolder paths
DEFAULT_MODEL_SUBFOLDER: Final[str] = ""
DEFAULT_LORA_SUBFOLDER: Final[str] = ""

# Validation ranges
MIN_STEPS: Final[int] = 1
MAX_STEPS: Final[int] = 100

MIN_CFG: Final[float] = 0.0
MAX_CFG: Final[float] = 20.0

MIN_SEED: Final[int] = 0
MAX_SEED: Final[int] = 0xFFFFFFFF

MIN_LORA_WEIGHT: Final[float] = -2.0
MAX_LORA_WEIGHT: Final[float] = 2.0

MIN_CLIP_WEIGHT: Final[float] = -2.0
MAX_CLIP_WEIGHT: Final[float] = 2.0

