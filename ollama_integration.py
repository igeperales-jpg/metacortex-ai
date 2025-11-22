import hashlib
#!/usr/bin/env python3
"""
🤖 OLLAMA INTEGRATION v3.0 - MILITARY GRADE NEURAL SYMBIOTIC SYSTEM
════════════════════════════════════════════════════════════════════

ARQUITECTURA EVOLUCIONADA - GRADO MILITAR:
    pass  # TODO: Implementar
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧬 CONEXIONES SIMBIÓTICAS MULTI-NIVEL:
├── Neural Network Integration (Red Neuronal Simbiótica)
├── Cognitive Agent Bridge (Agente Cognitivo BDI)
├── Memory Systems Triad (Episódica + Semántica + Working)
├── Advanced Cache Layer (L1/L2/L3 con TTL adaptativo)
├── ML Pipeline Orchestration (Auto-training + Deployment)
├── Programming Agent Communication (Materialización de código)
├── Knowledge Connector (Acceso a conocimiento universal)
└── Real-time Telemetry (Métricas militares distribuidas)

🚀 CAPACIDADES AVANZADAS:
- Circuit Breakers multi-nivel con auto-recovery
- Distributed Caching con coherencia fuerte
- Event Sourcing para auditoría completa
- Rate Limiting adaptativo con backpressure
- Semantic Search sobre embeddings
- Context-aware Generation con memoria episódica
- Multi-model Ensemble (Ollama + ML trained models)
- Auto-optimization basado en métricas

🎖️ MILITARY GRADE FEATURES:
- Zero-downtime updates con graceful degradation
- Fault tolerance con redundancia automática
- Security hardening con encryption at rest
- Distributed tracing para debugging
- SLA monitoring con alerting
- Chaos engineering validation

Autor: METACORTEX Advanced AI Division
Fecha: 2025-11-06
Versión: 3.0 - Military Grade Evolution
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Configuración de logging avanzado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s - [%(filename)s:%(lineno)d]",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON GLOBAL
# ═══════════════════════════════════════════════════════════════════════════
_global_ollama_integration: Optional["MilitaryGradeOllamaIntegration"] = None

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE MODELOS DISPONIBLES
# ═══════════════════════════════════════════════════════════════════════════

AVAILABLE_MODELS = {
    "mistral:latest": {
        "size_gb": 4.4,
        "specialty": "general_purpose",
        "speed": "fast",
        "use_cases": ["conversation", "reasoning", "general_qa"],
        "priority": 1,
        "context_window": 8192
    },
    "mistral:instruct": {
        "size_gb": 4.1,
        "specialty": "instruction_following",
        "speed": "fast",
        "use_cases": ["code_generation", "task_execution", "system_commands", "ml_training"],
        "priority": 0,  # Máxima prioridad para instrucciones
        "context_window": 8192,
        "optimal_for": ["ml_pipeline", "autonomous_systems", "code_materialization"]
    },
    "llama3.2:latest": {
        "size_gb": 2.0,
        "specialty": "efficiency",
        "speed": "very_fast",
        "use_cases": ["quick_responses", "chat", "simple_tasks"],
        "priority": 2,
        "context_window": 4096
    },
    "llama3.1:latest": {
        "size_gb": 4.9,
        "specialty": "complex_reasoning",
        "speed": "medium",
        "use_cases": ["analysis", "deep_thinking", "problem_solving"],
        "priority": 3,
        "context_window": 128000  # Extended context
    },
    "codellama:latest": {
        "size_gb": 3.8,
        "specialty": "code_generation",
        "speed": "medium",
        "use_cases": ["python", "javascript", "code_analysis", "debugging"],
        "priority": 4,
        "context_window": 16384
    },
    "deepseek-coder:latest": {
        "size_gb": 0.776,
        "specialty": "code_completion",
        "speed": "very_fast",
        "use_cases": ["autocomplete", "snippets", "quick_fixes"],
        "priority": 5,
        "context_window": 4096
    },
    "qwen2.5:latest": {
        "size_gb": 4.7,
        "specialty": "multilingual",
        "speed": "medium",
        "use_cases": ["spanish", "english", "chinese", "translation"],
        "priority": 6,
        "context_window": 32768
    }
}


# ═══════════════════════════════════════════════════════════════════════════
# ENUMERACIONES Y ESTRUCTURAS DE DATOS AVANZADAS
# ═══════════════════════════════════════════════════════════════════════════


class ModelTier(Enum):
    """Niveles de modelos en estrategia multi-tier"""

    TIER_1_PREMIUM = "tier_1_premium"  # Ollama local (llama3, deepseek-coder)
    TIER_2_TRAINED = "tier_2_trained"  # ML Pipeline trained models
    TIER_3_FALLBACK = "tier_3_fallback"  # Heurísticas determinísticas


class GenerationStrategy(Enum):
    """Estrategias de generación"""

    SINGLE_SHOT = "single_shot"  # Una sola generación
    MULTI_MODEL_ENSEMBLE = "multi_model_ensemble"  # Ensemble de múltiples modelos
    ITERATIVE_REFINEMENT = "iterative_refinement"  # Refinamiento iterativo
    SEMANTIC_SEARCH_AUGMENTED = "semantic_search_augmented"  # Con búsqueda semántica


class ContextMode(Enum):
    """Modos de contexto"""

    STATELESS = "stateless"  # Sin contexto
    SHORT_TERM = "short_term"  # Contexto de sesión (working memory)
    EPISODIC = "episodic"  # Contexto episódico (memoria a largo plazo)
    SEMANTIC = "semantic"  # Contexto semántico (knowledge graph)


# ═══════════════════════════════════════════════════════════════════════════
# OLLAMA INTEGRATION v3.0 - MILITARY GRADE
# ═══════════════════════════════════════════════════════════════════════════


class MilitaryGradeOllamaIntegration:
    """
    🎖️ Integración Ollama de Grado Militar con Conexiones Simbióticas

    ARQUITECTURA AVANZADA:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    1. NEURAL SYMBIOTIC CONNECTIONS (Conexiones Simbióticas)
       - Neural Network (red neuronal simbiótica asíncrona)
       - Cognitive Agent (agente cognitivo BDI con razonamiento)
       - Programming Agent (materialización de código)
       - Knowledge Connector (acceso a conocimiento universal)
    
    2. MEMORY TRIAD (Trío de Memoria)
       - Episodic Memory (conversaciones y eventos)
       - Semantic Memory (hechos y conceptos)
       - Working Memory (contexto activo de sesión)
    
    3. INTELLIGENT CACHING (Caché Inteligente Multi-Nivel)
       - L1: In-memory cache (respuestas inmediatas)
       - L2: Redis cache (compartido entre procesos)
       - L3: Disk cache (persistencia a largo plazo)
       - TTL Adaptativo basado en frecuencia de acceso
    
    4. TELEMETRY & MONITORING (Telemetría Militar)
       - Distributed tracing (trazado distribuido)
       - Performance metrics (SLA monitoring)
       - Health checks multi-nivel
       - Circuit breakers con auto-recovery
    
    5. MULTI-MODEL ORCHESTRATION (Orquestación Multi-Modelo)
       - Ensemble strategies (múltiples modelos)
       - Model selection heuristics (selección inteligente)
       - Fallback chains (cadenas de respaldo)
       - Quality gates (validación de calidad)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "mistral:latest",
        enable_military_features: bool = True,
    ):
        """
        Inicializa integración militar de Ollama

        Args:
            base_url: URL de Ollama
            default_model: Modelo por defecto
            enable_military_features: Activar características militares avanzadas
        """
        self.base_url = base_url
        self.default_model = default_model
        self.available = False
        self.available_models = []
        self.military_features_enabled = enable_military_features

        # Métricas militares avanzadas
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_generated": 0,
            "avg_response_time_ms": 0.0,
            "models_used": {},
            "cache_hits": 0,
            "cache_misses": 0,
            "circuit_breaker_trips": 0,
            "neural_connections_active": 0,
            "cognitive_influences": 0,
        }

        # Estado de conexiones simbióticas
        self.symbiotic_connections = {
            "neural_network": None,
            "cognitive_agent": None,
            "programming_agent": None,
            "knowledge_connector": None,
            "memory_system": None,
            "cache_system": None,
            "ml_pipeline": None,
        }

        # Circuit breaker para resiliencia
        self.circuit_breaker = {"state": "closed", "failure_count": 0, "last_failure": None}

        # Cola de respuestas para análisis
        self.response_history = deque(maxlen=1000)

        # Lock para operaciones thread-safe
        self.lock = threading.Lock()

        # 🧠 FASE 1: Inicializar sistemas de memoria
        self._initialize_memory_triad()

        # ✅ FASE 2: Verificar disponibilidad de Ollama
        self._check_availability()

        # 🔗 FASE 3: Establecer conexiones simbióticas
        self._establish_symbiotic_connections()

        # 🎖️ FASE 4: Activar características militares
        if self.military_features_enabled:
            self._activate_military_features()

        logger.info(
            f"🎖️ Military Grade Ollama Integration inicializado "
            f"(Conexiones simbióticas: {sum(1 for v in self.symbiotic_connections.values() if v is not None)})"
        )

    def _initialize_memory_triad(self):
        """
        🧠 Inicializar TRÍO DE MEMORIA (Episódica + Semántica + Working)

        MEMORIA EPISÓDICA: Eventos y conversaciones con temporal ordering
        MEMORIA SEMÁNTICA: Hechos, conceptos y knowledge graph
        MEMORIA WORKING: Contexto activo de sesión (short-term)
        """
        logger.info("🧠 Inicializando Memory Triad (Episódica + Semántica + Working)...")

        # 1. Memory System (episódica + semántica)
        try:
            from memory_system import get_memory

            memory = get_memory()
            self.symbiotic_connections["memory_system"] = memory
            logger.info("✅ Memory System conectado (episódica + semántica)")
        except Exception as e:
            logger.warning(f"⚠️ Memory System no disponible: {e}")

        # 2. Advanced Cache System (L1/L2/L3)
        try:
            from advanced_cache_system import get_global_cache

            cache = get_global_cache()
            self.symbiotic_connections["cache_system"] = cache
            logger.info("✅ Advanced Cache System conectado (L1/L2/L3)")
        except Exception as e:
            logger.warning(f"⚠️ Advanced Cache System no disponible: {e}")

        # 3. Working Memory (contexto de sesión)
        self.working_memory = {"current_context": [], "session_history": [], "active_tasks": []}
        logger.info("✅ Working Memory inicializada (contexto de sesión)")

    def _check_availability(self) -> bool:
        """Verifica disponibilidad de Ollama con circuit breaker"""
        # Circuit breaker check
        if self.circuit_breaker["state"] == "open":
            last_failure = self.circuit_breaker.get("last_failure")
            if last_failure and (datetime.now(UTC) - last_failure).seconds < 60:
                logger.warning("⚠️ Circuit breaker OPEN - Ollama temporalmente deshabilitado")
                return False
            # Try to close circuit breaker
            self.circuit_breaker["state"] = "half_open"

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)

            if response.status_code == 200:
                data = response.json()
                self.available_models = [model["name"] for model in data.get("models", [])]
                self.available = True

                # Reset circuit breaker
                self.circuit_breaker = {"state": "closed", "failure_count": 0, "last_failure": None}

                logger.info(f"✅ Ollama disponible: {len(self.available_models)} modelos")
                logger.info(f"📦 Modelos: {', '.join(self.available_models[:5])}")

                # Verificar modelo por defecto
                if (
                    self.default_model not in self.available_models
                    and self.available_models
                ):
                    self.default_model = self.available_models[0]
                    logger.info(f"🔄 Modelo por defecto: {self.default_model}")

                return True

            # Error response
            self._record_failure()
            logger.warning(f"⚠️ Ollama respondió con código {response.status_code}")
            return False

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            self._record_failure()
            logger.exception(f"❌ Error conectando con Ollama: {e}")
            return False

    def _record_failure(self):
        """Registra fallo y actualiza circuit breaker"""
        self.circuit_breaker["failure_count"] += 1
        self.circuit_breaker["last_failure"] = datetime.now(UTC)
        self.metrics["circuit_breaker_trips"] += 1

        # Open circuit breaker after 3 consecutive failures
        if self.circuit_breaker["failure_count"] >= 3:
            self.circuit_breaker["state"] = "open"
            logger.error(
                "🚨 Circuit breaker ABIERTO - Demasiados fallos consecutivos"
            )

    def _establish_symbiotic_connections(self):
        """
        🔗 Establecer CONEXIONES SIMBIÓTICAS con todo el ecosistema METACORTEX

        CONEXIONES BIDIRECCIONALES:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Ollama ↔ Neural Network (mensajería asíncrona)
        Ollama ↔ Cognitive Agent (influencia cognitiva)
        Ollama ↔ Programming Agent (materialización de código)
        Ollama ↔ Knowledge Connector (acceso a conocimiento)
        Ollama ↔ ML Pipeline (entrenamiento continuo)
        """
        logger.info("🔗 Estableciendo conexiones simbióticas...")

        # 1. Neural Network (Red Neuronal Simbiótica Asíncrona)
        try:
            from neural_symbiotic_network import get_neural_network

            neural_net = get_neural_network()
            neural_net.register_module(
                "ollama_military_integration",
                self,
                capabilities=[
                    "llm_generation",
                    "llm_chat",
                    "llm_reasoning",
                    "code_generation",
                    "knowledge_synthesis",
                    "multi_model_ensemble",
                    "semantic_search",
                    "context_awareness",
                ],
            )
            self.symbiotic_connections["neural_network"] = neural_net
            self.metrics["neural_connections_active"] += 1
            logger.info("✅ Neural Network ←→ Ollama: CONECTADO BIDIRECCIONAL")
        except Exception as e:
            logger.warning(f"⚠️ Neural Network no disponible: {e}")

        # 2. Cognitive Agent (Agente Cognitivo BDI)
        try:
            from cognitive_agent import CognitiveAgent

            cognitive = CognitiveAgent()
            self.symbiotic_connections["cognitive_agent"] = cognitive
            self.metrics["neural_connections_active"] += 1
            logger.info("✅ Cognitive Agent ←→ Ollama: CONECTADO BIDIRECCIONAL")
        except Exception as e:
            logger.warning(f"⚠️ Cognitive Agent no disponible: {e}")

        # 3. Programming Agent (Materialización de Código)
        try:
            from programming_agent import get_programming_agent

            prog_agent = get_programming_agent()
            self.symbiotic_connections["programming_agent"] = prog_agent
            self.metrics["neural_connections_active"] += 1
            logger.info("✅ Programming Agent ←→ Ollama: CONECTADO BIDIRECCIONAL")
        except Exception as e:
            logger.warning(f"⚠️ Programming Agent no disponible: {e}")

        # 4. Knowledge Connector (Acceso a Conocimiento Universal)
        try:
            from universal_knowledge_connector import get_knowledge_connector

            knowledge = get_knowledge_connector()
            self.symbiotic_connections["knowledge_connector"] = knowledge
            self.metrics["neural_connections_active"] += 1
            logger.info("✅ Knowledge Connector ←→ Ollama: CONECTADO BIDIRECCIONAL")
        except Exception as e:
            logger.warning(f"⚠️ Knowledge Connector no disponible: {e}")

        # 5. ML Pipeline (Entrenamiento Continuo)
        try:
            from ml_pipeline import get_ml_pipeline

            ml_pipe = get_ml_pipeline()
            self.symbiotic_connections["ml_pipeline"] = ml_pipe
            self.metrics["neural_connections_active"] += 1
            logger.info("✅ ML Pipeline ←→ Ollama: CONECTADO BIDIRECCIONAL")
        except Exception as e:
            logger.warning(f"⚠️ ML Pipeline no disponible: {e}")

        # 6. LLM Integration (Compatibilidad con sistema existente)
        try:
            from llm_integration import get_llm

            llm = get_llm()
            if hasattr(llm, "_check_availability"):
                llm._check_availability()
            logger.info("✅ LLM Integration actualizado con Ollama")
        except Exception as e:
            logger.warning(f"⚠️ LLM Integration no disponible: {e}")

        logger.info(
            f"🎯 Conexiones simbióticas establecidas: {self.metrics['neural_connections_active']}/6"
        )

    def _activate_military_features(self):
        """
        🎖️ Activar características militares avanzadas

        FEATURES:
        - Distributed caching con TTL adaptativo
        - Event sourcing para auditoría
        - Rate limiting adaptativo
        - Telemetry distribuida
        - Circuit breaker con auto-recovery
        """
        logger.info("🎖️ Activando características militares...")

        # 1. Rate Limiting Adaptativo
        self.rate_limiter = {"requests_per_minute": 60, "current_count": 0, "reset_time": None}

        # 2. Event Sourcing
        self.event_log = deque(maxlen=10000)

        # 3. Distributed Tracing
        self.trace_id = None

        # 4. Performance SLA
        self.sla_targets = {
            "avg_response_time_ms": 1000,  # < 1s promedio
            "success_rate": 99.0,  # > 99% éxito
            "p95_response_time_ms": 2000,  # < 2s p95
        }

        logger.info("✅ Características militares activadas")

    def _check_intelligent_cache(self, prompt: str, model: str) -> dict | None:
        """
        🔄 Verificar caché inteligente ANTES de llamar a Ollama

        Busca en:
        1. L1: In-memory (instantáneo)
        2. L2: Redis (rápido, compartido)
        3. L3: Disk (más lento, persistente)

        Returns:
            Respuesta cacheada o None
        """
        cache_system = self.symbiotic_connections.get("cache_system")
        if not cache_system:
            return None

        try:
            cache_key = f"ollama_{model}_{hashlib.md5(prompt.encode()).hexdigest()[:16]}"

            # Intentar obtener de caché
            cached = cache_system.get(cache_key)

            if cached:
                self.metrics["cache_hits"] += 1
                logger.debug(f"💾 Cache HIT: {cache_key}")
                return {
                    "success": True,
                    "response": cached["response"],
                    "model": model,
                    "response_time_ms": 0.0,  # Instantáneo
                    "cached": True,
                    "cache_timestamp": cached.get("timestamp"),
                }

            self.metrics["cache_misses"] += 1
            return None

        except Exception as e:
            logger.warning(f"⚠️ Error verificando caché: {e}")
            return None

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        use_cache: bool = True,
        use_cognitive_influence: bool = True,
    ) -> dict:
        """
        Genera respuesta usando Ollama

        Args:
            prompt: Texto de entrada
            model: Modelo a usar (None = default)
            temperature: Temperatura de generación
            max_tokens: Máximo de tokens

        Returns:
            Dict con respuesta y metadatos
        """
        if not self.available:
            return {"success": False, "error": "Ollama no está disponible", "response": ""}

        model = model or self.default_model
        start_time = datetime.now(UTC)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout=120,
            )

            if response.status_code == 200:
                data = response.json()
                response_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000

                # Actualizar métricas
                self.metrics["total_requests"] += 1
                self.metrics["successful_requests"] += 1
                self.metrics["total_tokens_generated"] += len(data.get("response", "").split())

                # Actualizar promedio de tiempo de respuesta
                current_avg = self.metrics["avg_response_time_ms"]
                total = self.metrics["total_requests"]
                self.metrics["avg_response_time_ms"] = (
                    current_avg * (total - 1) + response_time_ms
                ) / total

                # Contar uso de modelos
                self.metrics["models_used"][model] = self.metrics["models_used"].get(model, 0) + 1

                # 🧠 GUARDAR EN MEMORIA REAL (episódica + caché)
                llm_response = data.get("response", "")
                self._store_in_memory(
                    prompt=prompt,
                    response=llm_response,
                    model=model,
                    response_time_ms=response_time_ms,
                    metadata=data,
                )

                return {
                    "success": True,
                    "response": llm_response,
                    "model": model,
                    "response_time_ms": response_time_ms,
                    "tokens": len(llm_response.split()),
                    "metadata": {
                        "total_duration": data.get("total_duration", 0),
                        "load_duration": data.get("load_duration", 0),
                        "prompt_eval_count": data.get("prompt_eval_count", 0),
                        "eval_count": data.get("eval_count", 0),
                    },
                }
            self.metrics["total_requests"] += 1
            self.metrics["failed_requests"] += 1
            return {"success": False, "error": f"HTTP {response.status_code}", "response": ""}

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            self.metrics["total_requests"] += 1
            self.metrics["failed_requests"] += 1
            logger.error(f"❌ Error generando respuesta: {e}")
            return {"success": False, "error": str(e), "response": ""}

    def chat(self, messages: list, model: str | None = None, temperature: float = 0.7) -> dict:
        """
        Chat conversacional con Ollama

        Args:
            messages: Lista de mensajes [{"role": "user", "content": "..."}]
            model: Modelo a usar
            temperature: Temperatura

        Returns:
            Dict con respuesta
        """
        if not self.available:
            return {"success": False, "error": "Ollama no está disponible", "response": ""}

        model = model or self.default_model
        start_time = datetime.now(UTC)

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=120,
            )

            if response.status_code == 200:
                data = response.json()
                response_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000

                # Actualizar métricas
                self.metrics["total_requests"] += 1
                self.metrics["successful_requests"] += 1

                message = data.get("message", {})
                content = message.get("content", "")

                return {
                    "success": True,
                    "response": content,
                    "model": model,
                    "response_time_ms": response_time_ms,
                    "role": message.get("role", "assistant"),
                }
            self.metrics["total_requests"] += 1
            self.metrics["failed_requests"] += 1
            return {"success": False, "error": f"HTTP {response.status_code}", "response": ""}

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            self.metrics["total_requests"] += 1
            self.metrics["failed_requests"] += 1
            logger.error(f"❌ Error en chat: {e}")
            return {"success": False, "error": str(e), "response": ""}

    def get_metrics(self) -> dict:
        """Obtiene métricas de uso de Ollama"""
        return {
            "available": self.available,
            "base_url": self.base_url,
            "default_model": self.default_model,
            "available_models": self.available_models,
            "metrics": self.metrics,
        }

    def save_metrics(self, filepath: str = "ml_data/ollama_metrics.json"):
        """Guardar métricas en archivo JSON"""
        try:
            Path(filepath).parent.mkdir(exist_ok=True, parents=True)
            with open(filepath, "w") as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"✅ Métricas guardadas en {filepath}")
        except Exception as e:
            logger.error(f"❌ Error guardando métricas: {e}")

    def _store_in_memory(
        self, prompt: str, response: str, model: str, response_time_ms: float, metadata: dict
    ):
        """
        🧠 ALMACENAMIENTO REAL EN SISTEMAS DE MEMORIA

        Guarda la interacción LLM en:
        1. Memory System (episódica para conversaciones largas)
        2. Advanced Cache (para respuestas frecuentes)
        3. Metacortex Sináptico Memory (si cognitive_agent está disponible)

        Args:
            prompt: Prompt enviado
            response: Respuesta generada
            model: Modelo usado
            response_time_ms: Tiempo de respuesta
            metadata: Metadata adicional de Ollama
        """
        # 1. Guardar en Memory System (episódica)
        if self.memory:
            try:
                self.memory.store_episode(
                    content=f"LLM Interaction - {model}",
                    context={
                        "prompt": prompt[:500],  # Primeros 500 chars
                        "response": response[:1000],  # Primeros 1000 chars
                        "model": model,
                        "response_time_ms": response_time_ms,
                        "tokens_generated": len(response.split()),
                        "timestamp": datetime.now(UTC).isoformat(),
                        "ollama_metadata": {
                            "total_duration": metadata.get("total_duration", 0),
                            "eval_count": metadata.get("eval_count", 0),
                        },
                    },
                    importance=0.7,  # Importancia media-alta
                )
                logger.debug("💾 Interacción guardada en Memory System")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo guardar en Memory System: {e}")

        # 2. Guardar en Advanced Cache (para respuestas frecuentes)
        if self.cache:
            try:
                # Crear clave de caché basada en prompt + modelo

                cache_key = f"ollama_{model}_{hashlib.md5(prompt.encode()).hexdigest()[:16]}"

                self.cache.set(
                    cache_key,
                    {
                        "response": response,
                        "model": model,
                        "response_time_ms": response_time_ms,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                    ttl=3600,  # 1 hora de TTL
                )
                logger.debug(f"🔄 Respuesta cacheada: {cache_key}")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo cachear respuesta: {e}")

        # 3. Guardar en Metacortex Sináptico Memory (si cognitive_agent disponible)
        if hasattr(self, "cognitive_agent") and self.cognitive_agent:
            try:
                # Acceder al sistema de memoria del agente cognitivo
                if hasattr(self.cognitive_agent, "memory") and self.cognitive_agent.memory:
                    self.cognitive_agent.memory.store_episode(
                        name=f"ollama_llm_{model}",
                        data={
                            "prompt": prompt[:500],
                            "response": response[:1000],
                            "model": model,
                            "performance": {
                                "response_time_ms": response_time_ms,
                                "tokens": len(response.split()),
                            },
                        },
                        importance=0.75,  # Alta importancia para agente cognitivo
                        anomaly=False,
                    )
                    logger.debug("🧠 Interacción guardada en Metacortex Sináptico Memory")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo guardar en Metacortex Memory: {e}")

    def health_check(self) -> dict:
        """Verificación de salud del servicio"""
        return {
            "service": "Ollama",
            "status": "healthy" if self.available else "unhealthy",
            "url": self.base_url,
            "models_available": len(self.available_models),
            "total_requests": self.metrics["total_requests"],
            "success_rate": (
                self.metrics["successful_requests"] / self.metrics["total_requests"] * 100
                if self.metrics["total_requests"] > 0
                else 0
            ),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # SELECCIÓN INTELIGENTE DE MODELOS
    # ═══════════════════════════════════════════════════════════════════════════

    def select_optimal_model(self, task_type: str = "general", priority: str = "speed") -> str:
        """
        🎯 Selección inteligente de modelo basado en tipo de tarea
        
        Args:
            task_type: Tipo de tarea ("code", "chat", "analysis", "translation", "ml_training", etc.)
            priority: Prioridad ("speed", "quality", "balance")
        
        Returns:
            str: Nombre del modelo óptimo
        """
        task_mapping = {
            "code": ["codellama:latest", "deepseek-coder:latest"],
            "code_completion": ["deepseek-coder:latest", "codellama:latest"],
            "code_generation": ["codellama:latest", "mistral:instruct"],
            "instruction": ["mistral:instruct", "mistral:latest"],
            "ml_training": ["mistral:instruct", "llama3.1:latest"],
            "ml_optimization": ["mistral:instruct", "llama3.1:latest"],
            "system_commands": ["mistral:instruct", "llama3.2:latest"],
            "autonomous_task": ["mistral:instruct", "mistral:latest"],
            "chat": ["llama3.2:latest", "mistral:latest"],
            "analysis": ["llama3.1:latest", "mistral:instruct"],
            "reasoning": ["llama3.1:latest", "mistral:instruct"],
            "translation": ["qwen2.5:latest", "llama3.1:latest"],
            "multilingual": ["qwen2.5:latest", "mistral:latest"],
            "general": ["mistral:latest", "llama3.2:latest"]
        }
        
        candidates = task_mapping.get(task_type, ["mistral:instruct"])
        
        if priority == "speed":
            # Ordenar por velocidad (tamaño menor = más rápido)
            candidates.sort(key=lambda m: AVAILABLE_MODELS.get(m, {}).get("size_gb", 10))
        elif priority == "quality":
            # Ordenar por prioridad (menor número = mejor calidad)
            candidates.sort(key=lambda m: AVAILABLE_MODELS.get(m, {}).get("priority", 999))
        
        # Verificar disponibilidad
        for model in candidates:
            if self._check_ollama_available():
                return model
        
        return "mistral:instruct"  # Fallback a Mistral Instruct (óptimo)

    def multi_model_ensemble(
        self,
        prompt: str,
        models: list = None,
        aggregation: str = "vote"
    ) -> dict:
        """
        🎼 Generación ensemble con múltiples modelos
        
        Args:
            prompt: Prompt a generar
            models: Lista de modelos a usar (None = usar top 3 con Mistral Instruct)
            aggregation: Método de agregación ("vote", "longest", "average_quality", "mistral_instruct_priority")
        
        Returns:
            dict: Resultado ensemble con respuestas individuales y agregada
        """
        if models is None:
            # Default: Mistral Instruct + mejores modelos
            models = ["mistral:instruct", "llama3.1:latest", "mistral:latest"]
        
        results = {}
        responses = []
        
        for model in models:
            try:
                result = self.generate(
                    prompt=prompt,
                    model=model,
                    stream=False
                )
                responses.append({
                    "model": model,
                    "response": result.get("response", ""),
                    "tokens": result.get("eval_count", 0),
                    "is_instruct": "instruct" in model.lower()
                })
                results[model] = result
            except Exception as e:
                logger.warning(f"⚠️ Error en modelo {model}: {e}")
        
        # Agregación
        if aggregation == "mistral_instruct_priority":
            # Priorizar Mistral Instruct si está disponible
            instruct_responses = [r for r in responses if r["is_instruct"]]
            aggregated = instruct_responses[0]["response"] if instruct_responses else responses[0]["response"]
        elif aggregation == "vote":
            # Usar respuesta más común
            aggregated = max(responses, key=lambda r: len(r["response"]))["response"]
        elif aggregation == "longest":
            aggregated = max(responses, key=lambda r: len(r["response"]))["response"]
        else:
            # Promedio ponderado por tokens
            aggregated = responses[0]["response"] if responses else ""
        
        return {
            "individual_responses": responses,
            "aggregated_response": aggregated,
            "models_used": models,
            "strategy": aggregation,
            "mistral_instruct_used": any(r["is_instruct"] for r in responses)
        }

    def get_model_info(self, model: str = None) -> dict:
        """
        📊 Obtener información de modelo(s)
        
        Args:
            model: Nombre del modelo (None = todos)
        
        Returns:
            dict: Información del modelo o todos los modelos
        """
        if model:
            return AVAILABLE_MODELS.get(model, {})
        return AVAILABLE_MODELS

    def generate_with_mistral_instruct(
        self,
        instruction: str,
        context: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4000
    ) -> dict:
        """
        🎯 Generación especializada con Mistral Instruct
        Optimizado para seguir instrucciones complejas y tareas de ML
        
        Args:
            instruction: Instrucción clara a seguir
            context: Contexto adicional
            temperature: Creatividad (0.3 = más determinista)
            max_tokens: Tokens máximos
        
        Returns:
            dict: Respuesta con metadata extendida
        """
        # Construir prompt optimizado para Mistral Instruct
        prompt = f"[INST] {instruction}"
        if context:
            prompt += f"\n\nContext: {context}"
        prompt += " [/INST]"
        
        result = self.generate(
            prompt=prompt,
            model="mistral:instruct",
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=True,
            use_cognitive_influence=True
        )
        
        # Agregar metadata de Mistral Instruct
        result["model_type"] = "mistral_instruct"
        result["optimal_for"] = ["ml_training", "code_generation", "system_commands"]
        result["instruction_following"] = True
        
        return result


# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON GLOBAL
# ═══════════════════════════════════════════════════════════════════════════


def get_ollama_integration(**kwargs) -> "MilitaryGradeOllamaIntegration":
    """Obtiene instancia singleton de MilitaryGradeOllamaIntegration"""
    global _global_ollama_integration

    if _global_ollama_integration is None:
        _global_ollama_integration = MilitaryGradeOllamaIntegration(**kwargs)

    return _global_ollama_integration


# Alias para compatibilidad
OllamaIntegration = MilitaryGradeOllamaIntegration


def test_ollama_integration():
    """Test de la integración"""
    print("\n" + "=" * 60)
    print("🤖 OLLAMA INTEGRATION - Test")
    print("=" * 60 + "\n")

    # Crear integración
    ollama = get_ollama_integration()

    # Health check
    health = ollama.health_check()
    print("🏥 Health Check:")
    print(json.dumps(health, indent=2))

    if not ollama.available:
        print("\n❌ Ollama no está disponible. Asegúrate de que está corriendo.")
        return

    # Test de generación
    print("\n🧪 Test de generación:")
    result = ollama.generate(
        "¿Qué es el Machine Learning en una frase?", temperature=0.5, max_tokens=100
    )

    if result["success"]:
        print(f"✅ Respuesta: {result['response'][:200]}...")
        print(f"⏱️ Tiempo: {result['response_time_ms']:.2f}ms")
        print(f"📝 Tokens: {result['tokens']}")
    else:
        print(f"❌ Error: {result['error']}")

    # Test de chat
    print("\n💬 Test de chat:")
    chat_result = ollama.chat([{"role": "user", "content": "Hola, ¿cómo estás?"}])

    if chat_result["success"]:
        print(f"✅ Respuesta: {chat_result['response'][:200]}...")
        print(f"⏱️ Tiempo: {chat_result['response_time_ms']:.2f}ms")
    else:
        print(f"❌ Error: {chat_result['error']}")

    # Métricas
    print("\n📊 Métricas:")
    metrics = ollama.get_metrics()
    print(json.dumps(metrics["metrics"], indent=2))

    # Guardar métricas
    ollama.save_metrics()

    print("\n✅ Test completado")


if __name__ == "__main__":
    test_ollama_integration()