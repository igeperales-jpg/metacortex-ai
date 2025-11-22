"""
🍎 Apple Silicon M4 + MPS (Metal Performance Shaders) Configuration
Configuración automática para usar GPU Metal en lugar de CPU
"""
import os
import torch

# ============================================================================
# CONFIGURACIÓN DE VARIABLES DE ENTORNO
# ============================================================================

# Forzar uso de MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Usar toda la memoria GPU
os.environ['PYTORCH_MPS_PREFER_METAL'] = '1'
os.environ['MPS_FORCE_ENABLE'] = '1'

# Optimizaciones para Apple Silicon
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['OMP_NUM_THREADS'] = '10'  # M4 tiene 10 performance cores
os.environ['MKL_NUM_THREADS'] = '10'
os.environ['OPENBLAS_NUM_THREADS'] = '10'

# Gestión de memoria
os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def get_device():
    """
    Obtiene el mejor device disponible (prioridad: MPS > CUDA > CPU)
    
    Returns:
        torch.device: Device óptimo para ejecución
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def verify_mps():
    """
    Verifica que MPS esté disponible y funcionando
    
    Returns:
        bool: True si MPS está operacional
    """
    if not torch.backends.mps.is_available():
        print("⚠️ MPS no disponible - usando CPU")
        return False
    
    try:
        device = torch.device("mps")
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = x @ y
        print(f"✅ MPS verificado - GPU Metal operacional")
        return True
    except Exception as e:
        print(f"❌ Error verificando MPS: {e}")
        return False


def get_system_info():
    """
    Obtiene información del sistema Apple Silicon
    
    Returns:
        dict: Información del sistema
    """
    import platform
    import subprocess
    
    info = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }
    
    # Obtener información del chip Apple
    try:
        chip_info = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True
        ).strip()
        info["chip"] = chip_info
    except:
        info["chip"] = "Unknown"
    
    return info


def print_config():
    """Imprime la configuración actual de MPS"""
    print("=" * 60)
    print("🍎 APPLE SILICON M4 + MPS CONFIGURATION")
    print("=" * 60)
    
    info = get_system_info()
    for key, value in info.items():
        print(f"{key:20s}: {value}")
    
    device = get_device()
    print(f"{'device':20s}: {device}")
    
    print("=" * 60)


# ============================================================================
# AUTO-CONFIGURACIÓN AL IMPORTAR
# ============================================================================

# Detectar y configurar automáticamente
DEVICE = get_device()

if __name__ == "__main__":
    print_config()
    verify_mps()
