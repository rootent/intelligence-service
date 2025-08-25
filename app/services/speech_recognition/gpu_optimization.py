"""
GPU Optimization Configuration
This file contains GPU optimization settings for maximum performance.
"""

import torch
import os
import logging

logger = logging.getLogger(__name__)


def optimize_gpu_settings():
    """Configure GPU settings for maximum performance."""
    if not torch.cuda.is_available():
        logger.info("No GPU detected. Using CPU optimization.")
        return False
    
    try:
        # Get GPU info
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {gpu_name}")
        logger.info(f"GPU memory: {gpu_memory:.1f} GB")
        
        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        # Enable cuDNN benchmarking for optimal performance
        torch.backends.cudnn.benchmark = True
        
        # Enable TensorFloat-32 for faster matrix operations on Ampere+ GPUs
        try:
            if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TensorFloat-32 for Ampere+ GPU")
        except Exception:
            logger.info("TensorFloat-32 not available on this GPU")
        
        # Set memory fraction to avoid OOM (only if supported)
        try:
            if gpu_memory < 8.0:  # Less than 8GB
                torch.cuda.set_per_process_memory_fraction(0.8)
                logger.info("Set GPU memory fraction to 0.8 for low memory GPU")
            elif gpu_memory < 16.0:  # Less than 16GB
                torch.cuda.set_per_process_memory_fraction(0.9)
                logger.info("Set GPU memory fraction to 0.9 for medium memory GPU")
            else:  # 16GB or more
                torch.cuda.set_per_process_memory_fraction(0.95)
                logger.info("Set GPU memory fraction to 0.95 for high memory GPU")
        except Exception:
            logger.info("Memory fraction setting not supported on this system")
        
        # Enable memory efficient attention if available
        try:
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("Enabled Flash Attention for memory efficiency")
        except Exception:
            logger.info("Flash Attention not available on this system")
        
        logger.info("GPU optimization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"GPU optimization failed: {e}")
        return False


def get_optimal_batch_size(gpu_memory_gb: float = None) -> int:
    """Get optimal batch size based on GPU memory."""
    if gpu_memory_gb is None:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            return 1
    
    if gpu_memory_gb < 4.0:
        return 1
    elif gpu_memory_gb < 8.0:
        return 2
    elif gpu_memory_gb < 16.0:
        return 4
    else:
        return 8


def get_optimal_compute_type(gpu_memory_gb: float = None) -> str:
    """Get optimal compute type based on GPU memory."""
    if gpu_memory_gb is None:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            return "int8"
    
    if gpu_memory_gb < 4.0:
        return "int8"  # Use int8 for very low memory GPUs
    else:
        return "float16"  # Use half precision for better speed on most GPUs


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")


def get_gpu_utilization() -> dict:
    """Get current GPU utilization stats."""
    if not torch.cuda.is_available():
        return {"error": "No GPU available"}
    
    try:
        return {
            "gpu_name": torch.cuda.get_device_name(),
            "memory_allocated": torch.cuda.memory_allocated() / 1e9,
            "memory_reserved": torch.cuda.memory_reserved() / 1e9,
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "memory_free": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Test GPU optimization
    logging.basicConfig(level=logging.INFO)
    success = optimize_gpu_settings()
    if success:
        print("GPU optimization successful!")
        print(f"Optimal batch size: {get_optimal_batch_size()}")
        print(f"Optimal compute type: {get_optimal_compute_type()}")
        print(f"GPU utilization: {get_gpu_utilization()}")
    else:
        print("GPU optimization failed or no GPU available.")
