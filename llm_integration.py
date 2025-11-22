#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM Integration Module (Military-Grade)
=======================================

Módulo centralizado para interactuar con múltiples proveedores y modelos de LLM,
con selección inteligente, tolerancia a fallos y caching.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Callable, TYPE_CHECKING

# --- Configuración de Ruta para Importaciones ---
import sys
from pathlib import Path
try:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except IndexError:
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
# --- Fin Configuración de Ruta ---

# --- Importaciones robustas con fallbacks y TYPE_CHECKING ---

if TYPE_CHECKING:
    import ollama
    from unified_logging import get_logger
    from agent_modules.resilience import with_circuit_breaker, CircuitBreaker, CircuitBreakerConfig
    from agent_modules.distributed_cache import get_distributed_cache, cached, DistributedCacheSystem

try:
    import ollama
    from unified_logging import get_logger
    from agent_modules.resilience import with_circuit_breaker, CircuitBreaker, CircuitBreakerConfig
    from agent_modules.distributed_cache import get_distributed_cache, cached, DistributedCacheSystem
    imports_were_successful = True
    ollama_available = True
except ImportError:
    imports_were_successful = False
    ollama_available = False
    print("ADVERTENCIA: Módulos críticos no encontrados. LLMIntegration funcionará en modo degradado.")

    # --- Fallbacks ---
    def get_logger(name: str = "DefaultLogger") -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    class CircuitBreakerConfig:
        def __init__(self, *args: Any, **kwargs: Any): pass

    class CircuitBreaker:
        def __init__(self, name: str, config: Optional[Any] = None, *args: Any, **kwargs: Any): self.name = name
        def can_attempt(self) -> bool: return True
        def record_success(self): pass
        def record_failure(self): pass

    def with_circuit_breaker(circuit_breaker: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func
        return decorator

    class DistributedCacheSystem:
        pass

    def get_distributed_cache(**kwargs: Any) -> Optional[DistributedCacheSystem]:
        return None

    def cached(ttl_seconds: Optional[float] = None, key_prefix: str = "", use_args: bool = True) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func
        return decorator
    
    ollama = None # type: ignore


# Circuit Breaker específico para LLMs
if imports_were_successful:
    llm_cb_config = CircuitBreakerConfig(failure_threshold=3, timeout_seconds=60)
    LLM_CIRCUIT_BREAKER = CircuitBreaker(name="llm_service", config=llm_cb_config)
else:
    LLM_CIRCUIT_BREAKER = CircuitBreaker(name="llm_fallback")


class LLMIntegration:
    """
    🧠 Integración de LLM de Grado Militar
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.ollama_client: Optional[Any] = ollama.Client() if ollama_available and ollama else None
        self.available_models: List[str] = []
        self.preferred_model: Optional[str] = None
        self.cache = get_distributed_cache(l1_max_size=100) if imports_were_successful else None

        if ollama_available and self.ollama_client:
            self._check_ollama_models()
        else:
            self.logger.warning("Ollama no está instalado o no se pudo inicializar. La funcionalidad de LLM será limitada.")

    def _check_ollama_models(self):
        if not self.ollama_client:
            return
        try:
            models_info = self.ollama_client.list()
            self.available_models = [model['name'] for model in models_info.get('models', [])]
            
            model_preference = ['codellama', 'llama3', 'mistral']
            for preferred in model_preference:
                if self.preferred_model:
                    break
                for model_name in self.available_models:
                    if preferred in model_name:
                        self.preferred_model = model_name
                        break
            
            if not self.preferred_model and self.available_models:
                self.preferred_model = self.available_models[0]

            self.logger.info(f"Modelos de Ollama disponibles: {self.available_models}")
            if self.preferred_model:
                self.logger.info(f"Modelo de Ollama preferido seleccionado: {self.preferred_model}")

        except Exception as e:
            self.logger.error(f"No se pudo conectar con el servidor de Ollama: {e}", exc_info=True)
            global ollama_available
            ollama_available = False

    @cached(ttl_seconds=3600)
    @with_circuit_breaker(LLM_CIRCUIT_BREAKER)
    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        if not LLM_CIRCUIT_BREAKER.can_attempt():
            return {"success": False, "error": "Circuit breaker is open."}

        target_model = model or self.preferred_model
        if not ollama_available or not target_model or not self.ollama_client:
            self.logger.error("No hay modelos de LLM disponibles para generar una respuesta.")
            LLM_CIRCUIT_BREAKER.record_failure()
            return {"success": False, "error": "No LLM models available."}

        self.logger.info(f"Generando respuesta con el modelo: {target_model}")
        start_time = time.time()

        try:
            response = self.ollama_client.generate(
                model=target_model,
                prompt=prompt,
                system=system_prompt,
                options={"num_predict": max_tokens, "temperature": temperature}
            )
            elapsed_time = time.time() - start_time
            LLM_CIRCUIT_BREAKER.record_success()
            
            return {
                "success": True,
                "response": response.get('response', ''),
                "model": target_model,
                "latency_seconds": round(elapsed_time, 2),
                "done": response.get('done', False),
            }
        except Exception as e:
            self.logger.error(f"Error durante la generación con Ollama ({target_model}): {e}", exc_info=True)
            LLM_CIRCUIT_BREAKER.record_failure()
            return {"success": False, "error": str(e)}

# --- Singleton Factory ---
_llm_integration_instance: Optional[LLMIntegration] = None

def get_llm_integration(force_new: bool = False, **kwargs: Any) -> LLMIntegration:
    global _llm_integration_instance
    if _llm_integration_instance is None or force_new:
        _llm_integration_instance = LLMIntegration(**kwargs)
    return _llm_integration_instance

if __name__ == '__main__':
    print("Ejecutando LLMIntegration en modo de prueba...")
    logger = get_logger("LLMTest")
    llm_agent = get_llm_integration(force_new=True, logger=logger)

    if llm_agent.preferred_model:
        print(f"\n--- Probando el modelo: {llm_agent.preferred_model} ---")
        test_prompt = "Explica la importancia de la computación cuántica en 3 frases."
        result = llm_agent.generate(prompt=test_prompt)

        if result["success"]:
            print(f"Prompt: {test_prompt}")
            print(f"Respuesta: {result['response']}")
            print(f"Latencia: {result['latency_seconds']}s")
        else:
            print(f"Fallo en la prueba: {result['error']}")
    else:
        print("\nNo se encontraron modelos de LLM para realizar la prueba.")

    print("\nPrueba finalizada.")
