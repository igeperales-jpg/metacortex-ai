"""
Fallback para el LLMIntegration.
"""
from typing import Any

def get_llm_integration(force_new: bool = False, **kwargs: Any) -> Any:
    """
    Firma idéntica a la real, devuelve None.
    """
    return None
