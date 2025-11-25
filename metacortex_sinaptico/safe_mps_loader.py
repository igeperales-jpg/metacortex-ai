#!/usr/bin/env python3
"""
Safe MPS Loader for SentenceTransformer v5.0
SOLUCIÓN DEFINITIVA: Thread-safe lock + cache
"""

import sys
import logging
import os
import threading

# Límite alto permanente para deep call stacks (consistente con daemon)
sys.setrecursionlimit(100000)

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Lock thread-safe + cache de modelos
_model_load_lock = threading.RLock()
_loaded_models = {}


def setup_mps_environment():
    """Configura variables de entorno óptimas para MPS"""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"
    torch.set_default_dtype(torch.float32)


def load_sentence_transformer_safe(
    model_name: str,
    device: str | None = None,
    cache_folder: str | None = None
) -> tuple[SentenceTransformer, str]:
    """
    Carga SentenceTransformer con lock thread-safe y cache
    
    SOLUCIÓN v5.0:
        pass  # TODO: Implementar
    - threading.RLock() previene carga concurrente
    - Cache global evita re-cargas del mismo modelo
    - Threads esperan si otro está cargando
    - Carga directa en MPS sin conversiones
    """
    
    cache_key = f"{model_name}:{device}"
    
    # Check cache sin lock (fast path)
    if cache_key in _loaded_models:
        logger.info(f"📦 Modelo en cache: {model_name}")
        return _loaded_models[cache_key]
    
    # Adquirir lock - otros threads ESPERAN aquí
    with _model_load_lock:
        # Double-check: puede haberse cargado mientras esperábamos
        if cache_key in _loaded_models:
            logger.info(f"📦 Modelo cargado por otro thread: {model_name}")
            return _loaded_models[cache_key]
        
        setup_mps_environment()
        
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        logger.info(f"🚀 Cargando modelo: {model_name} en {device}")
        
        # Carga directa
        model = SentenceTransformer(model_name, device=device, cache_folder=cache_folder)
        dimension = model.get_sentence_embedding_dimension()
        actual_device = verify_model_device(model)
        
        logger.info(f"✅ Modelo: {model_name}, Device: {actual_device}, Dim: {dimension}")
        
        # Guardar en cache
        result = (model, actual_device)
        _loaded_models[cache_key] = result
        
        return result


def verify_model_device(model: SentenceTransformer) -> str:
    """Verifica device real del modelo"""
    first_param = next(model.parameters())
    device = str(first_param.device)
    
    if "mps" in device:
        return "mps"
    elif "cuda" in device:
        return "cuda"
    return "cpu"