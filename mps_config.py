"""
Configuración para Apple Silicon (MPS).
"""

def is_apple_silicon():
    """Verifica si el sistema es Apple Silicon."""
    return False

def configure_mps_system():
    """Configura el sistema para usar MPS."""
    print("WARN: mps_config.py es un stub. No se realizarán optimizaciones para Apple Silicon.")
    return {}
