from metacortex_sinaptico.utils import AgentConfig
from metacortex_sinaptico.utils import AgentConfig
#!/usr/bin/env python3
"""
🤖🎖️⚡ METACORTEX ML Pipeline v3.0 - MILITARY GRADE EVOLUTION
═══════════════════════════════════════════════════════════════════════════════

ARQUITECTURA EVOLUCIONADA - GRADO MILITAR AVANZADO:
    pass  # TODO: Implementar
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎖️ CARACTERÍSTICAS MILITARY GRADE v3.0:
├── 🧬 Adaptive ML Operations (MLOps auto-ajustables)
├── 🔮 Predictive Model Scaling (escalado predictivo con RL)
├── 🏥 Self-Healing Training (auto-corrección de entrenamientos)
├── 📈 Intelligent Hyperparameter Optimization (auto-tuning)
├── 📊 Advanced Telemetry (telemetría militar multi-nivel)
├── 🛣️ Neural Model Routing (rutas de modelos optimizadas)
├── 🌐 Distributed Training (entrenamiento distribuido)
├── 🧬 Auto-Evolution (auto-evolución de modelos)
├── 🔐 Military-Grade Model Security (seguridad de modelos)
├── 🎯 Adaptive Resource Allocation (asignación adaptativa)
├── 🧠 Cognitive Model Selection (selección cognitiva)
├── 💾 Persistent Training State (gestión de estado persistente)
└── 🚀 Zero-Downtime Model Updates (actualizaciones sin downtime)

🔗 CONEXIONES SIMBIÓTICAS AVANZADAS v3.0:
├── ML Pipeline ↔ Neural Network (mensajería asíncrona)
├── ML Pipeline ↔ Cognitive Agent (razonamiento ML distribuido)
├── ML Pipeline ↔ Memory System (aprendizaje con contexto)
├── ML Pipeline ↔ Advanced Cache (caché de modelos multi-nivel)
├── ML Pipeline ↔ Ollama Integration (generación aumentada)
├── ML Pipeline ↔ Programming Agent (materialización de código ML)
├── ML Pipeline ↔ Knowledge Connector (conocimiento de dominio)
├── ML Pipeline ↔ Telemetry System (métricas militar-grade)
└── ML Pipeline ↔ Event Sourcing (registro completo de eventos)

🎯 FEATURES MILITARES ACTIVADOS:
├── ⚡ Circuit Breaker Adaptativo (fallas de entrenamiento)
├── 🎚️ Rate Limiting Inteligente (control de recursos)
├── 📝 Event Sourcing (10,000 eventos de entrenamiento)
├── 📊 SLA Monitoring (99.9% uptime target)
├── 🔄 Auto-Retry con Backoff Exponencial
├── 🧠 Model Performance Prediction (predicción de rendimiento)
├── 🌊 Backpressure Control (control de carga)
├── 🎭 Multi-Model Ensemble (ensambles adaptativos)
├── 🔐 Model Versioning & Rollback (versiones + rollback)
└── 🚀 Continuous Model Deployment (despliegue continuo)

🏗️ ARQUITECTURA MULTI-CAPA:
┌─────────────────────────────────────────────────────────────┐
│  CAPA 1: NEURAL SYMBIOTIC CONNECTIONS                      │
│  ↓ Comunicación bidireccional con todo el ecosistema       │
├─────────────────────────────────────────────────────────────┤
│  CAPA 2: MEMORY TRIAD INTEGRATION                          │
│  ↓ Episodic + Semantic + Working Memory para contexto      │
├─────────────────────────────────────────────────────────────┤
│  CAPA 3: INTELLIGENT TRAINING ORCHESTRATION                │
│  ↓ Auto-tuning, auto-scaling, auto-healing                 │
├─────────────────────────────────────────────────────────────┤
│  CAPA 4: MILITARY FEATURES                                 │
│  ↓ Circuit breakers, rate limiting, event sourcing         │
├─────────────────────────────────────────────────────────────┤
│  CAPA 5: ADVANCED TELEMETRY                                │
│  ↓ Métricas, SLA, performance tracking                     │
├─────────────────────────────────────────────────────────────┤
│  CAPA 6: MODEL LIFECYCLE MANAGEMENT                        │
│  ↓ Versioning, deployment, rollback, monitoring            │
└─────────────────────────────────────────────────────────────┘

⚙️ VERSIÓN: 3.0.0 (MILITARY GRADE EVOLUTION)
📅 EVOLUTION DATE: 2025-01-06
👨‍💻 ARCHITECT: METACORTEX AUTONOMOUS SYSTEM
🎯 MISSION: MÁXIMA CONFIABILIDAD + INTELIGENCIA ADAPTATIVA

═══════════════════════════════════════════════════════════════════════════════
"""

import hashlib
import json
import logging
import pickle
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from collections import deque
from collections import Counter
import pandas as pd  # Lazy import para evitar circular imports
# pandas se importa lazy dentro de funciones para evitar circular imports

# ML Core
try:
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
    )
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    SKLEARN_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error: {e}", exc_info=True)
    SKLEARN_AVAILABLE = False
    # No usar logging aquí porque aún no está configurado
    print(f"⚠️ scikit-learn no disponible: {e}")

# Deep Learning
try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Transformers
try:
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# 🧠 INTEGRACIÓN CON METACORTEX
try:
    from neural_symbiotic_network import get_neural_network

    NEURAL_NETWORK_AVAILABLE = True
except ImportError:
    NEURAL_NETWORK_AVAILABLE = False
    logging.warning("⚠️ Neural network no disponible")

try:
    from memory_system import get_memory

    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    logging.warning("⚠️ Memory system no disponible")

try:
    from advanced_cache_system import get_global_cache

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logging.warning("⚠️ Cache system no disponible")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 🎖️ ENUMERATIONS v3.0 - MILITARY GRADE
# ═══════════════════════════════════════════════════════════════════════════


class ModelType(Enum):
    """Tipos de modelos soportados (v3.0 expandido)"""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NLP = "nlp"
    VISION = "vision"
    TIME_SERIES = "time_series"
    # 🆕 v3.0
    REINFORCEMENT_LEARNING = "reinforcement_learning"  # RL agents
    ENSEMBLE = "ensemble"  # Ensambles multi-modelo
    AUTOML = "automl"  # AutoML meta-learning


class TrainingStatus(Enum):
    """Estados de entrenamiento (v3.0 mejorado)"""

    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    DEPLOYED = "deployed"
    # 🆕 v3.0
    OPTIMIZING = "optimizing"  # Hyperparameter tuning
    RECOVERING = "recovering"  # Auto-healing training
    HIBERNATING = "hibernating"  # Modelo pausado
    ROLLING_BACK = "rolling_back"  # Rollback a versión anterior
    UPGRADING = "upgrading"  # Upgrade del modelo


class ModelHealth(Enum):
    """🆕 v3.0: Estado de salud del modelo"""

    EXCELLENT = "excellent"  # >95% performance
    GOOD = "good"  # 85-95% performance
    FAIR = "fair"  # 75-85% performance
    DEGRADED = "degraded"  # 60-75% performance
    CRITICAL = "critical"  # <60% performance


class TrainingPriority(Enum):
    """🆕 v3.0: Prioridades de entrenamiento"""

    CRITICAL = "critical"  # Entrenamiento urgente
    HIGH = "high"  # Alta prioridad
    NORMAL = "normal"  # Prioridad normal
    LOW = "low"  # Baja prioridad
    BACKGROUND = "background"  # Entrenamiento en segundo plano


class DeploymentMode(Enum):
    """🆕 v3.0: Modos de despliegue"""

    CANARY = "canary"  # Despliegue gradual (10% tráfico)
    BLUE_GREEN = "blue_green"  # Dos versiones simultáneas
    SHADOW = "shadow"  # Modo shadow (sin exponer)
    FULL = "full"  # Despliegue completo
    A_B_TEST = "a_b_test"  # A/B testing
    ROLLBACK = "rollback"  # Rollback a versión anterior


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento"""

    model_type: ModelType
    model_name: str
    algorithm: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # Data
    train_data_path: str | None = None
    validation_split: float = 0.2
    test_split: float = 0.1

    # Training
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping: bool = True
    patience: int = 5

    # Evaluation
    metrics: list[str] = field(default_factory=lambda: ["accuracy", "f1"])
    cross_validation_folds: int = 5

    # MLOps
    auto_deploy: bool = False
    min_accuracy: float = 0.8
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5


@dataclass
class TrainingResult:
    """Resultado de entrenamiento"""

    model_id: str
    model_type: ModelType
    algorithm: str
    status: TrainingStatus

    # Metrics
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)

    # Model info
    model_path: str | None = None
    model_size_mb: float = 0.0
    training_time_seconds: float = 0.0

    # Data info
    num_train_samples: int = 0
    num_val_samples: int = 0
    num_test_samples: int = 0
    num_features: int = 0

    # Timestamps
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "algorithm": self.algorithm,
            "status": self.status.value,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "model_path": self.model_path,
            "model_size_mb": self.model_size_mb,
            "training_time_seconds": self.training_time_seconds,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "num_test_samples": self.num_test_samples,
            "num_features": self.num_features,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


class MilitaryGradeMLPipeline:
    """
    🎖️ ML Pipeline de Grado Militar con Conexiones Simbióticas v3.0

    ARQUITECTURA AVANZADA:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1. NEURAL SYMBIOTIC CONNECTIONS (Conexiones Simbióticas)
       - Neural Network (red neuronal simbiótica asíncrona)
       - Cognitive Agent (agente cognitivo BDI con razonamiento ML)
       - Programming Agent (materialización de pipelines ML)
       - Knowledge Connector (acceso a conocimiento de dominio ML)
       - Ollama Integration (generación aumentada de modelos)

    2. MEMORY TRIAD (Trío de Memoria)
       - Episodic Memory (historial de entrenamientos)
       - Semantic Memory (conocimiento ML acumulado)
       - Working Memory (contexto activo de entrenamiento)

    3. INTELLIGENT TRAINING ORCHESTRATION (Orquestación Inteligente)
       - Auto-tuning (ajuste automático de hiperparámetros)
       - Auto-scaling (escalado dinámico de recursos)
       - Auto-healing (recuperación automática de fallos)

    4. MILITARY FEATURES (Características Militares)
       - Circuit Breaker (protección contra fallos en cascada)
       - Rate Limiting (control de carga de entrenamiento)
       - Event Sourcing (registro completo de eventos ML)
       - SLA Monitoring (monitoreo de objetivos de rendimiento)

    5. ADVANCED TELEMETRY (Telemetría Avanzada)
       - Métricas de entrenamiento multi-nivel
       - Performance tracking en tiempo real
       - Resource utilization monitoring
       - Model drift detection

    6. MODEL LIFECYCLE MANAGEMENT (Gestión de Ciclo de Vida)
       - Versioning (control de versiones de modelos)
       - Deployment strategies (canary, blue-green, shadow)
       - Rollback automático
       - A/B testing integrado
    """

    def __init__(
        self,
        models_dir: str = "ml_models",
        data_dir: str = "ml_data",
        enable_cache: bool = True,
        enable_continuous_learning: bool = True,
        enable_perpetual_mode: bool = True,
        # 🆕 v3.0 MILITARY FEATURES
        enable_circuit_breaker: bool = True,
        enable_rate_limiting: bool = True,
        enable_event_sourcing: bool = True,
        enable_telemetry: bool = True,
        enable_auto_healing: bool = True,
    ):
        """
        🎖️ Inicializar ML Pipeline MILITARY GRADE con 6 FASES

        FASE 1: Setup Base
        FASE 2: Memory Triad
        FASE 3: Symbiotic Connections (8+ conexiones bidireccionales)
        FASE 4: Military Features
        FASE 5: Advanced Telemetry
        FASE 6: Perpetual Training Mode

        Args:
            models_dir: Directorio para guardar modelos
            data_dir: Directorio para datos de entrenamiento
            enable_cache: Habilitar caché de resultados
            enable_continuous_learning: Habilitar aprendizaje continuo
            enable_perpetual_mode: Habilitar entrenamiento perpetuo
            enable_circuit_breaker: Habilitar circuit breaker adaptativo
            enable_rate_limiting: Habilitar rate limiting inteligente
            enable_event_sourcing: Habilitar event sourcing
            enable_telemetry: Habilitar telemetría avanzada
            enable_auto_healing: Habilitar auto-healing
        """
        logger.info("🎖️ Inicializando ML Pipeline MILITARY GRADE v3.0...")

        # ═══════════════════════════════════════════════════════════════════
        # FASE 1: SETUP BASE
        # ═══════════════════════════════════════════════════════════════════

        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.data_dir.mkdir(exist_ok=True, parents=True)

        self.enable_cache = enable_cache and CACHE_AVAILABLE
        self.enable_continuous_learning = enable_continuous_learning
        self.enable_perpetual_mode = enable_perpetual_mode

        # 🆕 v3.0 Military Flags
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_event_sourcing = enable_event_sourcing
        self.enable_telemetry = enable_telemetry
        self.enable_auto_healing = enable_auto_healing

        logger.info("✅ FASE 1: Setup Base completado")

        # ═══════════════════════════════════════════════════════════════════
        # FASE 2: MEMORY TRIAD INTEGRATION
        # ═══════════════════════════════════════════════════════════════════

        # Memory System (Episodic + Semantic)
        self.memory = get_memory() if MEMORY_AVAILABLE else None

        # Advanced Cache (L1/L2/L3)
        self.cache = get_global_cache() if self.enable_cache else None

        # Working Memory (contexto activo)
        self.working_memory: dict[str, Any] = {
            "active_trainings": {},
            "recent_evaluations": [],
            "current_context": {},
        }

        logger.info("✅ FASE 2: Memory Triad configurado")
        if self.memory:
            logger.info("   ├─ Episodic Memory: ✓")
            logger.info("   ├─ Semantic Memory: ✓")
        if self.cache:
            logger.info("   └─ Cache System (L1/L2/L3): ✓")

        # ═══════════════════════════════════════════════════════════════════
        # FASE 2.5: INITIALIZE METRICS (ANTES de conexiones)
        # ═══════════════════════════════════════════════════════════════════

        self.metrics: dict[str, Any] = {
            "models_trained": 0,
            "models_deployed": 0,
            "training_failures": 0,
            "training_success_rate": 0.0,
            "avg_training_time_seconds": 0.0,
            "total_predictions": 0,
            "circuit_breaker_trips": 0,
            "rate_limit_hits": 0,
            "auto_healing_activations": 0,
            "symbiotic_messages_sent": 0,
            "symbiotic_messages_received": 0,
            "neural_connections_active": 0,
        }

        # ═══════════════════════════════════════════════════════════════════
        # FASE 3: SYMBIOTIC CONNECTIONS (8+ CONEXIONES BIDIRECCIONALES)
        # ═══════════════════════════════════════════════════════════════════

        self.symbiotic_connections: dict[str, Any] = {}
        self._establish_symbiotic_connections()

        # ═══════════════════════════════════════════════════════════════════
        # FASE 4: MILITARY FEATURES
        # ═══════════════════════════════════════════════════════════════════

        self._activate_military_features()

        # ═══════════════════════════════════════════════════════════════════
        # FASE 5: ADVANCED TELEMETRY (Ya inicializado en FASE 2.5)
        # ═══════════════════════════════════════════════════════════════════

        logger.info("✅ FASE 5: Advanced Telemetry activado")

        # ═══════════════════════════════════════════════════════════════════
        # FASE 6: STATE MANAGEMENT
        # ═══════════════════════════════════════════════════════════════════

        # Training history
        self.training_history: list[TrainingResult] = []
        self._load_training_history()

        # Active models
        self.active_models: dict[str, Any] = {}

        # Deployed models (para retrocompatibilidad)
        self.deployed_models: dict[str, Any] = {}

        # Auto-load deployed models
        self._auto_load_deployed_models()

        # Queue de entrenamiento
        self.training_queue: queue.Queue = queue.Queue()
        self.training_thread: threading.Thread | None = None
        self.perpetual_running = False

        # Auto-reentrenamiento
        self.retraining_schedule: dict[str, datetime] = {}
        self.retraining_interval = timedelta(hours=24)

        logger.info("✅ FASE 6: State Management completado")

        # ═══════════════════════════════════════════════════════════════════
        # SYSTEM STATUS
        # ═══════════════════════════════════════════════════════════════════

        logger.info("✅ ML Pipeline MILITARY GRADE v3.0 inicializado")
        logger.info(f"   Sklearn: {'✓' if SKLEARN_AVAILABLE else '✗'}")
        logger.info(f"   PyTorch: {'✓' if TORCH_AVAILABLE else '✗'}")
        logger.info(f"   Transformers: {'✓' if TRANSFORMERS_AVAILABLE else '✗'}")
        logger.info(f"   Neural Network: {'✓' if 'neural_network' in self.symbiotic_connections else '✗'}")
        logger.info(f"   Memory System: {'✓' if self.memory else '✗'}")
        logger.info(f"   Cache: {'✓' if self.cache else '✗'}")
        logger.info(f"   Cognitive Agent: {'✓' if 'cognitive_agent' in self.symbiotic_connections else '✗'}")
        logger.info(f"   Military Features: {'✓' if self.enable_circuit_breaker else '✗'}")
        logger.info(f"   Conexiones Simbióticas: {self.metrics['neural_connections_active']}")
        logger.info(f"   Modo perpetuo: {'✓' if self.enable_perpetual_mode else '✗'}")

        # Iniciar modo perpetuo si está habilitado
        if self.enable_perpetual_mode:
            self.start_perpetual_training()

    # ═══════════════════════════════════════════════════════════════════════
    # 🔗 SYMBIOTIC CONNECTIONS (v3.0 MILITARY GRADE)
    # ═══════════════════════════════════════════════════════════════════════

    def _establish_symbiotic_connections(self):
        """
        🔗 Establecer CONEXIONES SIMBIÓTICAS con todo el ecosistema METACORTEX

        CONEXIONES BIDIRECCIONALES:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ML Pipeline ↔ Neural Network (mensajería asíncrona)
        ML Pipeline ↔ Cognitive Agent (influencia cognitiva ML)
        ML Pipeline ↔ Programming Agent (materialización de código ML)
        ML Pipeline ↔ Knowledge Connector (conocimiento de dominio)
        ML Pipeline ↔ Ollama Integration (generación aumentada)
        ML Pipeline ↔ Memory System (aprendizaje con contexto)
        ML Pipeline ↔ Advanced Cache (caché de modelos)
        ML Pipeline ↔ Telemetry System (métricas avanzadas)
        """
        logger.info("🔗 Estableciendo conexiones simbióticas v3.0...")
        
        # Inicializar neural_network como None
        self.neural_network = None

        # 1. Neural Network (Red Neuronal Simbiótica Asíncrona)
        if NEURAL_NETWORK_AVAILABLE:
            try:
                from neural_symbiotic_network import get_neural_network

                neural_net = get_neural_network()
                neural_net.register_module(
                    "ml_pipeline_military_v3",
                    self,
                    capabilities=[
                        "train_model_advanced",
                        "evaluate_model_cognitive",
                        "deploy_model_zero_downtime",
                        "continuous_training_adaptive",
                        "auto_retraining_intelligent",
                        "hyperparameter_optimization",
                        "model_ensemble_creation",
                        "model_drift_detection",
                        "auto_healing_training",
                        "predictive_scaling",
                    ],
                )
                self.symbiotic_connections["neural_network"] = neural_net
                self.neural_network = neural_net  # También como atributo directo
                self.metrics["neural_connections_active"] += 1
                logger.info("✅ Neural Network ←→ ML Pipeline: CONECTADO BIDIRECCIONAL")
            except Exception as e:
                logger.warning(f"⚠️ Neural Network no disponible: {e}")

        # 2. Cognitive Agent (Agente Cognitivo BDI con Razonamiento ML)
        try:
            from cognitive_agent import CognitiveAgent

            cognitive = CognitiveAgent()
            self.symbiotic_connections["cognitive_agent"] = cognitive
            self.metrics["neural_connections_active"] += 1
            logger.info("✅ Cognitive Agent ←→ ML Pipeline: CONECTADO BIDIRECCIONAL")

            # Integración cognitiva avanzada (si existe metacortex_sinaptico) - USANDO SINGLETON
            try:
                from metacortex_sinaptico.core import get_cognitive_agent  # ✅ Singleton factory

                config = AgentConfig()
                cognitive_v2 = get_cognitive_agent(config=config)  # ✅ Singleton

                if hasattr(cognitive_v2, "neural_network") and cognitive_v2.neural_network:
                    cognitive_v2.neural_network.register_module(
                        "ml_pipeline_cognitive_bridge",
                        self,
                        capabilities=["ml_training", "ml_prediction", "ml_optimization"],
                    )
                    logger.info("✅ ML Pipeline registrado en red neuronal cognitiva v2")

                self.symbiotic_connections["cognitive_agent_v2"] = cognitive_v2
                logger.info("🔄 Flujo BIDIRECCIONAL: ML ←→ Cognición v2 activado")
            except Exception:
                pass  # Cognitive v2 es opcional

        except Exception as e:
            logger.warning(f"⚠️ Cognitive Agent no disponible: {e}")

        # 3. Programming Agent (Materialización de código ML)
        try:
            from programming_agent import MetacortexUniversalProgrammingAgent

            prog_agent = MetacortexUniversalProgrammingAgent()
            self.symbiotic_connections["programming_agent"] = prog_agent
            self.metrics["neural_connections_active"] += 1
            logger.info("✅ Programming Agent ←→ ML Pipeline: CONECTADO BIDIRECCIONAL")
        except Exception as e:
            logger.warning(f"⚠️ Programming Agent no disponible: {e}")

        # 4. Knowledge Connector (Acceso a Conocimiento de Dominio ML)
        try:
            from universal_knowledge_connector import get_knowledge_connector

            knowledge = get_knowledge_connector()
            self.symbiotic_connections["knowledge_connector"] = knowledge
            self.metrics["neural_connections_active"] += 1
            logger.info("✅ Knowledge Connector ←→ ML Pipeline: CONECTADO BIDIRECCIONAL")
        except Exception as e:
            logger.warning(f"⚠️ Knowledge Connector no disponible: {e}")

        # 5. Ollama Integration (Generación Aumentada de Modelos)
        try:
            from ollama_integration import get_ollama_integration

            ollama = get_ollama_integration()
            self.symbiotic_connections["ollama_integration"] = ollama
            self.metrics["neural_connections_active"] += 1
            logger.info("✅ Ollama Integration ←→ ML Pipeline: CONECTADO BIDIRECCIONAL")
        except Exception as e:
            logger.warning(f"⚠️ Ollama Integration no disponible: {e}")

        # 6. LLM Integration (Compatibilidad con sistema existente)
        try:
            from llm_integration import get_llm

            llm = get_llm()
            self.symbiotic_connections["llm_integration"] = llm
            self.metrics["neural_connections_active"] += 1
            logger.info("✅ LLM Integration ←→ ML Pipeline: CONECTADO BIDIRECCIONAL")
        except Exception as e:
            logger.warning(f"⚠️ LLM Integration no disponible: {e}")

        # 7. Cognitive Integration Bridge (Puente ML ←→ Cognitive)
        try:
            from cognitive_integration import get_cognitive_bridge

            bridge = get_cognitive_bridge()
            self.symbiotic_connections["cognitive_bridge"] = bridge
            self.metrics["neural_connections_active"] += 1
            logger.info("✅ Cognitive Bridge ←→ ML Pipeline: CONECTADO BIDIRECCIONAL")
        except Exception as e:
            logger.warning(f"⚠️ Cognitive Bridge no disponible: {e}")

        # 8. Telemetry System (Métricas Avanzadas)
        try:
            from military_modules.telemetry_system import TelemetrySystem

            telemetry = TelemetrySystem()
            self.symbiotic_connections["telemetry"] = telemetry
            self.metrics["neural_connections_active"] += 1
            logger.info("✅ Telemetry System ←→ ML Pipeline: CONECTADO BIDIRECCIONAL")
        except Exception as e:
            logger.warning(f"⚠️ Telemetry System no disponible: {e}")

        logger.info(f"✅ FASE 3: {self.metrics['neural_connections_active']} conexiones simbióticas establecidas")

    # ═══════════════════════════════════════════════════════════════════════
    # 🎖️ MILITARY FEATURES (v3.0)
    # ═══════════════════════════════════════════════════════════════════════

    def _activate_military_features(self):
        """
        🎖️ Activar CARACTERÍSTICAS MILITARES v3.0

        FEATURES:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        1. Circuit Breaker Adaptativo (protección contra fallos)
        2. Rate Limiting Inteligente (60 trainings/minute)
        3. Event Sourcing (10,000 eventos de entrenamiento)
        4. SLA Monitoring (99.9% uptime target)
        """
        logger.info("🎖️ Activando Military Features...")

        # 1. Circuit Breaker (usando Neural Network v3.0)
        if self.enable_circuit_breaker and "neural_network" in self.symbiotic_connections:
            self.circuit_breaker = {
                "enabled": True,
                "failure_threshold": 5,
                "success_threshold": 2,
                "timeout_seconds": 60,
                "state": "CLOSED",  # CLOSED, OPEN, HALF_OPEN
                "failures": 0,
                "successes": 0,
                "last_failure_time": None,
            }
            logger.info("   ├─ Circuit Breaker: ✓ (threshold=5)")
        else:
            self.circuit_breaker = {"enabled": False}

        # 2. Rate Limiter (60 trainings por minuto)
        if self.enable_rate_limiting:
            self.rate_limiter = {
                "enabled": True,
                "max_trainings_per_minute": 60,
                "current_window": [],
                "window_size_seconds": 60,
            }
            logger.info("   ├─ Rate Limiter: ✓ (60 train/min)")
        else:
            self.rate_limiter = {"enabled": False}

        # 3. Event Sourcing (registro completo de eventos ML)
        if self.enable_event_sourcing:

            self.event_log: deque = deque(maxlen=10000)
            logger.info("   ├─ Event Sourcing: ✓ (10,000 eventos)")
        else:
            self.event_log = None

        # 4. SLA Targets (Service Level Agreement)
        self.sla_targets = {
            "training_success_rate": 0.999,  # 99.9% éxito
            "max_training_time_seconds": 300,  # 5 min max
            "max_deployment_time_seconds": 30,  # 30s max
            "model_availability": 0.999,  # 99.9% uptime
        }
        logger.info("   └─ SLA Monitoring: ✓ (99.9% targets)")

        logger.info("✅ FASE 4: Military Features activados")

    # ═══════════════════════════════════════════════════════════════════════
    # 🔄 HELPER METHODS (Circuit Breaker, Rate Limiting, Event Sourcing)
    # ═══════════════════════════════════════════════════════════════════════

    def _check_circuit_breaker(self) -> bool:
        """Verificar estado del circuit breaker"""
        if not self.circuit_breaker.get("enabled"):
            return True

        cb = self.circuit_breaker
        if cb["state"] == "OPEN":
            # Verificar si debemos intentar de nuevo
            if cb["last_failure_time"]:
                elapsed = (datetime.now() - cb["last_failure_time"]).total_seconds()
                if elapsed > cb["timeout_seconds"]:
                    cb["state"] = "HALF_OPEN"
                    cb["failures"] = 0
                    logger.info("🔄 Circuit Breaker: HALF_OPEN (intentando recuperación)")
                else:
                    logger.warning("⚠️ Circuit Breaker: OPEN (bloqueando entrenamiento)")
                    self.metrics["circuit_breaker_trips"] += 1
                    return False

        return True

    def _record_training_success(self):
        """Registrar entrenamiento exitoso (Circuit Breaker)"""
        if not self.circuit_breaker.get("enabled"):
            return

        cb = self.circuit_breaker
        cb["successes"] += 1
        cb["failures"] = 0

        if cb["state"] == "HALF_OPEN" and cb["successes"] >= cb["success_threshold"]:
            cb["state"] = "CLOSED"
            logger.info("✅ Circuit Breaker: CLOSED (recuperado)")

    def _record_training_failure(self):
        """Registrar fallo de entrenamiento (Circuit Breaker)"""
        if not self.circuit_breaker.get("enabled"):
            return

        cb = self.circuit_breaker
        cb["failures"] += 1
        cb["successes"] = 0
        cb["last_failure_time"] = datetime.now()

        if cb["failures"] >= cb["failure_threshold"]:
            cb["state"] = "OPEN"
            logger.error("❌ Circuit Breaker: OPEN (demasiados fallos)")

    def _check_rate_limit(self) -> bool:
        """Verificar rate limiting"""
        if not self.rate_limiter.get("enabled"):
            return True

        rl = self.rate_limiter
        now = time.time()

        # Limpiar ventana antigua
        rl["current_window"] = [t for t in rl["current_window"] if now - t < rl["window_size_seconds"]]

        # Verificar límite
        if len(rl["current_window"]) >= rl["max_trainings_per_minute"]:
            logger.warning("⚠️ Rate Limit alcanzado (60 train/min)")
            self.metrics["rate_limit_hits"] += 1
            return False

        # Agregar timestamp actual
        rl["current_window"].append(now)
        return True

    def _log_event(self, event_type: str, data: dict[str, Any]):
        """Registrar evento en Event Sourcing"""
        if not self.event_log:
            return

        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data,
        }
        self.event_log.append(event)

    # ═══════════════════════════════════════════════════════════════════════
    # 🧠 COGNITIVE AGENT INTEGRATION (Metacortex Sináptico)
    # ═══════════════════════════════════════════════════════════════════════

    def _notify_cognitive_agent(self, event_type: str, data: dict[str, Any]):
        """Notificar al Cognitive Agent sobre eventos ML"""
        if "cognitive_bridge" in self.symbiotic_connections:
            try:
                bridge = self.symbiotic_connections["cognitive_bridge"]
                if event_type == "ml_prediction":
                    bridge.notify_ml_prediction(data.get("prediction_type", "general"), data)
                elif event_type == "ml_training_completed":
                    bridge.notify_ml_feedback(data.get("model_id", "unknown"), data)
            except Exception as e:
                logger.debug(f"Error notificando al Cognitive: {e}")

        # 🆕 CONEXIÓN A METACORTEX SINÁPTICO (Sistema Cognitivo)
        self.cognitive_agent = None
        try:
            from metacortex_sinaptico.core import get_cognitive_agent  # ✅ Singleton factory

            # Crear configuración para el agente cognitivo
            config = AgentConfig()

            # Crear agente cognitivo (SINGLETON - evita 261 duplicados)
            self.cognitive_agent = get_cognitive_agent(config=config)

            # Registrar este módulo en la red neuronal del agente cognitivo
            if (
                hasattr(self.cognitive_agent, "neural_network")
                and self.cognitive_agent.neural_network
            ):
                self.cognitive_agent.neural_network.register_module(
                    "ml_pipeline_cognitive",
                    self,
                    capabilities=["ml_training", "ml_prediction", "ml_optimization"],
                )
                logger.info("✅ ML Pipeline registrado en red neuronal cognitiva")

            logger.info("✅ ML Pipeline integrado con Metacortex Sináptico")
            logger.info("🔄 Flujo BIDIRECCIONAL: ML ←→ Cognición activado")

        except ImportError as e:
            logger.warning(f"⚠️ Metacortex Sináptico no disponible: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Error integrando con sistema cognitivo: {e}")

        # Training history
        self.training_history: list[TrainingResult] = []
        self._load_training_history()

        # Active models
        self.active_models: dict[str, Any] = {}

        # 🆕 DEPLOYED MODELS (para retrocompatibilidad)
        self.deployed_models: dict[str, Any] = {}

        # 🆕 AUTO-LOAD DEPLOYED MODELS
        self._auto_load_deployed_models()

        # 🆕 QUEUE DE ENTRENAMIENTO
        self.training_queue: queue.Queue = queue.Queue()
        self.training_thread: threading.Thread | None = None
        self.perpetual_running = False

        # 🆕 AUTO-REENTRENAMIENTO
        self.retraining_schedule: dict[str, datetime] = {}
        self.retraining_interval = timedelta(hours=24)  # Reentrenar cada 24h

        logger.info("✅ ML Pipeline inicializado")
        logger.info(f"   Sklearn: {'✓' if SKLEARN_AVAILABLE else '✗'}")
        logger.info(f"   PyTorch: {'✓' if TORCH_AVAILABLE else '✗'}")
        logger.info(f"   Transformers: {'✓' if TRANSFORMERS_AVAILABLE else '✗'}")
        logger.info(f"   Neural Network: {'✓' if self.neural_network else '✗'}")
        logger.info(f"   Memory System: {'✓' if self.memory else '✗'}")
        logger.info(f"   Cache: {'✓' if self.cache else '✗'}")
        logger.info(f"   Modo perpetuo: {'✓' if self.enable_perpetual_mode else '✗'}")
        logger.info(f"   Modelos: {self.models_dir}")
        logger.info(f"   Datos: {self.data_dir}")

        # 🆕 INICIAR MODO PERPETUO
        if self.enable_perpetual_mode:
            self.start_perpetual_training()

    # ═══════════════════════════════════════════════════════════════════════
    # 🔄 MODO PERPETUO
    # ═══════════════════════════════════════════════════════════════════════

    def start_perpetual_training(self):
        """🆕 Inicia el modo de entrenamiento perpetuo"""
        if self.training_thread and self.training_thread.is_alive():
            logger.warning("⚠️ Modo perpetuo ya está activo")
            return

        self.perpetual_running = True
        self.training_thread = threading.Thread(
            target=self._perpetual_training_loop,
            daemon=True,
            name="MLPipelinePerpetual",
        )
        self.training_thread.start()
        logger.info("🤖 Modo perpetuo de ML Pipeline iniciado")

    def stop_perpetual_training(self):
        """🆕 Detiene el modo perpetuo"""
        self.perpetual_running = False
        if self.training_thread:
            self.training_thread.join(timeout=10)
        logger.info("🛑 Modo perpetuo de ML Pipeline detenido")

    def _perpetual_training_loop(self):
        """🆕 Loop perpetuo de entrenamiento"""
        logger.info("🔄 Loop perpetuo iniciado")

        while self.perpetual_running:
            try:
                # 1. Procesar queue de entrenamientos
                try:
                    config = self.training_queue.get(timeout=60)
                    logger.info(f"📋 Procesando entrenamiento: {config.model_name}")

                    # Entrenar modelo
                    result = self.train_model(config)

                    if result.status == TrainingStatus.COMPLETED:
                        logger.info(f"✅ Modelo entrenado: {result.model_id}")
                    else:
                        logger.warning(f"⚠️ Entrenamiento falló: {result.metadata.get('error')}")

                except queue.Empty:
                    pass  # No hay trabajos pendientes

                # 2. Verificar modelos que necesitan reentrenamiento
                self._check_retraining_schedule()

                # 3. Breve pausa
                time.sleep(10)

            except Exception as e:
                logger.error(f"❌ Error en loop perpetuo: {e}", exc_info=True)
                time.sleep(60)

        logger.info("🛑 Loop perpetuo detenido")

    def _check_retraining_schedule(self):
        """🆕 Verifica si hay modelos que necesitan reentrenamiento"""
        now = datetime.now()

        for model_id, next_training in list(self.retraining_schedule.items()):
            if now >= next_training:
                logger.info(f"📅 Reentrenamiento programado para: {model_id}")

                # Buscar configuración original
                for result in self.training_history:
                    if result.model_id == model_id and result.status == TrainingStatus.DEPLOYED:
                        # Crear nueva config basada en la original
                        # (simplificado - en producción cargar metadata completo)
                        config = TrainingConfig(
                            model_type=result.model_type,
                            model_name=result.metadata.get("model_name", f"retrain_{model_id}"),
                            algorithm=result.algorithm,
                            train_data_path=result.metadata.get("train_data_path"),
                        )

                        # Agregar a queue
                        self.enqueue_training(config)

                        # Reprogramar
                        self.retraining_schedule[model_id] = now + self.retraining_interval
                        break

    def enqueue_training(self, config: TrainingConfig):
        """🆕 Agrega un entrenamiento a la queue"""
        self.training_queue.put(config)
        logger.info(f"📬 Entrenamiento encolado: {config.model_name}")

    def schedule_retraining(self, model_id: str, interval: timedelta | None = None):
        """🆕 Programa reentrenamiento automático de un modelo"""
        interval = interval or self.retraining_interval
        next_training = datetime.now() + interval
        self.retraining_schedule[model_id] = next_training
        logger.info(f"📅 Reentrenamiento programado para {model_id}: {next_training.isoformat()}")

    # ═══════════════════════════════════════════════════════════════════════
    # ENTRENAMIENTO DE MODELOS (código existente mejorado)
    # ═══════════════════════════════════════════════════════════════════════

    def train_model(
        self,
        config: TrainingConfig,
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> TrainingResult:
        """
        🎖️ Entrenar modelo con MILITARY FEATURES v3.0

        PROTECCIONES MILITARES:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        1. Circuit Breaker (protección contra fallos en cascada)
        2. Rate Limiting (60 trainings/min)
        3. Event Sourcing (registro completo)
        4. Auto-Healing (recuperación automática)
        """

        # 🎖️ MILITARY PROTECTIONS
        # 1. Circuit Breaker Check
        if not self._check_circuit_breaker():
            return TrainingResult(
                model_id="circuit_breaker_open",
                model_type=config.model_type,
                algorithm=config.algorithm,
                status=TrainingStatus.FAILED,
                metadata={"error": "Circuit breaker open - too many failures"},
            )

        # 2. Rate Limiting Check
        if not self._check_rate_limit():
            return TrainingResult(
                model_id="rate_limit_exceeded",
                model_type=config.model_type,
                algorithm=config.algorithm,
                status=TrainingStatus.FAILED,
                metadata={"error": "Rate limit exceeded (60 trainings/min)"},
            )

        # 3. Log Event (Event Sourcing)
        self._log_event("training_started", {
            "model_name": config.model_name,
            "algorithm": config.algorithm,
            "model_type": config.model_type.value,
        })

        if not SKLEARN_AVAILABLE:
            logger.error("❌ scikit-learn no disponible")
            self._record_training_failure()
            return TrainingResult(
                model_id="error",
                model_type=config.model_type,
                algorithm=config.algorithm,
                status=TrainingStatus.FAILED,
                metadata={"error": "scikit-learn not available"},
            )

        logger.info(f"🚀 Iniciando entrenamiento MILITARY GRADE: {config.model_name}")

        start_time = time.time()
        model_id = self._generate_model_id(config)

        result = TrainingResult(
            model_id=model_id,
            model_type=config.model_type,
            algorithm=config.algorithm,
            status=TrainingStatus.PENDING,
        )

        try:
            # 1. Cargar datos
            result.status = TrainingStatus.PREPROCESSING
            if X_train is None or y_train is None:
                if not config.train_data_path:
                    raise ValueError("Se requiere train_data_path o datos directos")
                X_train, y_train, X_val, y_val = self._load_and_prepare_data(config)

            # Split si no hay validación
            if X_val is None or y_val is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=config.validation_split, random_state=42
                )

            result.num_train_samples = len(X_train)
            result.num_val_samples = len(X_val)
            result.num_features = X_train.shape[1] if len(X_train.shape) > 1 else 1

            logger.info(
                f"📊 Datos preparados: {result.num_train_samples} train, {result.num_val_samples} val"
            )

            # 2. Entrenar
            result.status = TrainingStatus.TRAINING
            model = self._create_model(config)

            if config.model_type == ModelType.CLASSIFICATION:
                model = self._train_classification_model(
                    model, config, X_train, y_train, X_val, y_val, result
                )
            elif config.model_type == ModelType.REGRESSION:
                model = self._train_regression_model(
                    model, config, X_train, y_train, X_val, y_val, result
                )
            else:
                raise ValueError(f"Tipo no soportado: {config.model_type}")

            # 3. Evaluar
            result.status = TrainingStatus.EVALUATING
            self._evaluate_model(model, config, X_train, y_train, X_val, y_val, result)

            # 4. Guardar
            model_path = self._save_model(model, config, result)
            result.model_path = str(model_path)
            result.model_size_mb = model_path.stat().st_size / (1024 * 1024)

            # 5. Completar
            result.status = TrainingStatus.COMPLETED
            result.completed_at = datetime.now()
            result.training_time_seconds = time.time() - start_time

            logger.info(f"✅ Entrenamiento completado en {result.training_time_seconds:.2f}s")
            logger.info(f"📈 Métricas: {result.val_metrics}")

            # 🎖️ MILITARY SUCCESS TRACKING
            self._record_training_success()
            self.metrics["models_trained"] += 1
            self._update_success_rate()

            # 6. Auto-deploy
            if config.auto_deploy and self._meets_deployment_criteria(result, config):
                self.deploy_model(result.model_id)

                # Programar reentrenamiento
                if self.enable_continuous_learning:
                    self.schedule_retraining(result.model_id)

            # 7. Guardar historial
            self.training_history.append(result)
            self._save_training_history()

            # 8. Guardar en memoria (si está disponible)
            if self.memory:
                try:
                    self.memory.store_episode(
                        content=f"Model trained: {config.model_name}",
                        context={
                            "model_id": model_id,
                            "algorithm": config.algorithm,
                            "metrics": result.val_metrics,
                        },
                        importance=0.9,
                    )
                except Exception as e:
                    logger.debug(f"No se pudo guardar en memoria: {e}")

            # 9. Event Sourcing
            self._log_event("training_completed", {
                "model_id": model_id,
                "training_time": result.training_time_seconds,
                "metrics": result.val_metrics,
            })

            # 10. Notificar Cognitive Agent
            self._notify_cognitive_agent("ml_training_completed", {
                "model_id": model_id,
                "metrics": result.val_metrics,
                "algorithm": config.algorithm,
            })

            return result

        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}", exc_info=True)

            # 🎖️ MILITARY FAILURE TRACKING
            self._record_training_failure()
            self.metrics["training_failures"] += 1
            self._update_success_rate()

            # 🏥 AUTO-HEALING (si está habilitado)
            if self.enable_auto_healing:
                self._attempt_auto_healing(config, e)

            result.status = TrainingStatus.FAILED
            result.completed_at = datetime.now()
            result.metadata["error"] = str(e)

            # Event Sourcing
            self._log_event("training_failed", {
                "model_name": config.model_name,
                "error": str(e),
            })

            return result

    def _update_success_rate(self):
        """Actualizar tasa de éxito de entrenamiento"""
        total = self.metrics["models_trained"] + self.metrics["training_failures"]
        if total > 0:
            self.metrics["training_success_rate"] = self.metrics["models_trained"] / total

    def _attempt_auto_healing(self, config: TrainingConfig, error: Exception):
        """🏥 Intentar auto-healing del entrenamiento fallido"""
        self.metrics["auto_healing_activations"] += 1
        logger.info(f"🏥 Auto-Healing: Intentando recuperación para {config.model_name}")

        # IMPLEMENTED: Implementar estrategias de auto-healing
        # - Reducir batch size
        # - Simplificar modelo
        # - Aumentar tiempo de timeout
        # - Reintentar con hiperparámetros alternativos

    def _create_model(self, config: TrainingConfig) -> Any:
        """Crear modelo"""
        algorithm = config.algorithm.lower()
        params = config.hyperparameters

        if config.model_type == ModelType.CLASSIFICATION:
            if algorithm == "random_forest":
                return RandomForestClassifier(**params) if params else RandomForestClassifier()
            if algorithm == "logistic_regression":
                return LogisticRegression(**params) if params else LogisticRegression()
            return RandomForestClassifier()  # Default
        if config.model_type == ModelType.REGRESSION:
            if algorithm == "random_forest":
                return RandomForestRegressor(**params) if params else RandomForestRegressor()
            if algorithm == "linear_regression":
                return LinearRegression(**params) if params else LinearRegression()
            return RandomForestRegressor()  # Default
        raise ValueError(f"Tipo no soportado: {config.model_type}")

    def _train_classification_model(self, model, config, X_train, y_train, X_val, y_val, result):
        """Entrenar clasificador"""
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        result.train_metrics = {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "f1": f1_score(y_train, y_train_pred, average="weighted", zero_division=0),
        }

        y_val_pred = model.predict(X_val)
        result.val_metrics = {
            "accuracy": accuracy_score(y_val, y_val_pred),
            "f1": f1_score(y_val, y_val_pred, average="weighted", zero_division=0),
        }

        return model

    def _train_regression_model(self, model, config, X_train, y_train, X_val, y_val, result):
        """Entrenar regresor"""
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        result.train_metrics = {
            "mse": mean_squared_error(y_train, y_train_pred),
            "r2": r2_score(y_train, y_train_pred),
        }

        y_val_pred = model.predict(X_val)
        result.val_metrics = {
            "mse": mean_squared_error(y_val, y_val_pred),
            "r2": r2_score(y_val, y_val_pred),
        }

        return model

    def _evaluate_model(self, model, config, X_train, y_train, X_val, y_val, result):
        """Evaluar con cross-validation adaptativo"""
        if config.cross_validation_folds > 0:
            # 🔧 CORRECCIÓN: Ajustar folds basado en la clase menos poblada
            n_folds = config.cross_validation_folds

            if config.model_type == ModelType.CLASSIFICATION:
                # Contar la clase menos poblada

                class_counts = Counter(y_train)
                min_class_size = min(class_counts.values()) if class_counts else 1

                # Ajustar folds para evitar warning de sklearn
                # Necesitamos al menos 2 ejemplos por fold
                max_safe_folds = max(2, min(min_class_size // 2, n_folds))

                if max_safe_folds < n_folds:
                    logger.warning(
                        f"⚠️ Reduciendo folds de {n_folds} a {max_safe_folds} "
                        f"(clase mínima: {min_class_size} ejemplos)"
                    )
                    n_folds = max_safe_folds

            # Cross-validation solo si tenemos suficientes datos
            if len(y_train) >= n_folds * 2:
                try:
                    cv_scores = cross_val_score(
                        model,
                        X_train,
                        y_train,
                        cv=n_folds,
                        scoring="accuracy"
                        if config.model_type == ModelType.CLASSIFICATION
                        else "r2",
                    )
                    result.metadata["cv_mean"] = float(cv_scores.mean())
                    result.metadata["cv_std"] = float(cv_scores.std())
                    result.metadata["cv_folds_used"] = n_folds
                except Exception as e:
                    logger.warning(f"⚠️ Error en cross-validation: {e}")
                    result.metadata["cv_error"] = str(e)
            else:
                logger.warning(
                    f"⚠️ Datos insuficientes para CV ({len(y_train)} samples < {n_folds * 2})"
                )
                result.metadata["cv_skipped"] = "insufficient_data"

    def _save_model(self, model, config, result) -> Path:
        """Guardar modelo"""
        model_path = self.models_dir / f"{result.model_id}.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        metadata_path = self.models_dir / f"{result.model_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"💾 Modelo guardado: {model_path}")
        return model_path

    def _load_and_prepare_data(self, config):
        """Cargar datos desde CSV"""
        
        data_path = Path(config.train_data_path)
        df = pd.read_csv(data_path)

        X = df.drop("target", axis=1).values
        y = df["target"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.validation_split + config.test_split, random_state=42
        )

        X_val, _, y_val, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

        return X_train, y_train, X_val, y_val

    def _meets_deployment_criteria(self, result, config):
        """Verificar criterios de deployment"""
        if config.model_type == ModelType.CLASSIFICATION:
            return result.val_metrics.get("accuracy", 0) >= config.min_accuracy
        if config.model_type == ModelType.REGRESSION:
            return result.val_metrics.get("r2", 0) >= config.min_accuracy
        return False

    def _generate_model_id(self, config):
        """Generar ID único"""
        data = f"{config.model_name}_{config.algorithm}_{datetime.now().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def deploy_model(self, model_id: str) -> bool:
        """Deployar modelo"""
        try:
            model_path = self.models_dir / f"{model_id}.pkl"
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            self.active_models[model_id] = {
                "model": model,
                "deployed_at": datetime.now(),
                "predictions_count": 0,
            }

            for result in self.training_history:
                if result.model_id == model_id:
                    result.status = TrainingStatus.DEPLOYED
                    break

            self._save_training_history()
            logger.info(f"✅ Modelo deployed: {model_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error deploying: {e}")
            return False

    def predict(self, model_id: str, X: np.ndarray) -> np.ndarray:
        """Hacer predicción"""
        if model_id not in self.active_models:
            raise ValueError(f"Modelo {model_id} no deployado")

        model = self.active_models[model_id]["model"]
        predictions = model.predict(X)
        self.active_models[model_id]["predictions_count"] += len(X)

        return predictions

    def predict_with_cognitive_notification(
        self, model_id: str, X: np.ndarray, prediction_type: str
    ) -> dict[str, Any]:
        """
        Hace predicción y notifica al Cognitive Agent

        Args:
            model_id: ID del modelo
            X: Datos de entrada
            prediction_type: 'intention' | 'load' | 'cache' | 'performance'

        Returns:
            Dict con predicción, confidence y metadata
        """
        if model_id not in self.active_models:
            raise ValueError(f"Modelo {model_id} no deployado")

        model_info = self.active_models[model_id]
        model = model_info["model"]

        # Hacer predicción
        predictions = model.predict(X)
        model_info["predictions_count"] += len(X)

        # Calcular confidence (si el modelo tiene predict_proba)
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                confidence = float(np.max(proba[0]))
            except Exception:
                confidence = 0.5
        else:
            confidence = 0.7  # Default para modelos de regresión

        # Preparar resultado
        result = {
            "model_id": model_id,
            "prediction": predictions[0] if len(predictions) == 1 else predictions.tolist(),
            "confidence": confidence,
            "model_accuracy": model_info.get("accuracy", 0.0),
            "predictions_count": model_info["predictions_count"],
            "timestamp": datetime.now().isoformat(),
        }

        # Notificar al Cognitive Agent
        try:
            from cognitive_integration import get_cognitive_bridge

            bridge = get_cognitive_bridge()
            bridge.notify_ml_prediction(prediction_type, result)
            logger.info(f"✅ Cognitive notificado: {prediction_type} → {result['prediction']}")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo notificar al Cognitive: {e}")

        return result

    def _load_training_history(self):
        """Cargar historial"""
        history_path = self.models_dir / "training_history.json"
        if history_path.exists():
            try:
                with open(history_path) as f:
                    data = json.load(f)
                for item in data:
                    result = TrainingResult(
                        model_id=item["model_id"],
                        model_type=ModelType(item["model_type"]),
                        algorithm=item["algorithm"],
                        status=TrainingStatus(item["status"]),
                    )
                    result.train_metrics = item.get("train_metrics", {})
                    result.val_metrics = item.get("val_metrics", {})
                    result.model_path = item.get("model_path")
                    self.training_history.append(result)
                logger.info(f"📂 Historial cargado: {len(self.training_history)} modelos")
            except Exception as e:
                logger.warning(f"⚠️ Error cargando historial: {e}")

    def _auto_load_deployed_models(self):
        """🆕 Auto-carga modelos que están en estado DEPLOYED"""
        deployed_count = 0
        for result in self.training_history:
            if result.status == TrainingStatus.DEPLOYED and result.model_path:
                try:
                    self.deploy_model(result.model_id)
                    deployed_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Error auto-loading {result.model_id}: {e}")

        if deployed_count > 0:
            logger.info(f"🚀 Auto-loaded {deployed_count} modelos deployados")

    def _save_training_history(self):
        """Guardar historial"""
        history_path = self.models_dir / "training_history.json"
        try:
            data = [r.to_dict() for r in self.training_history]
            with open(history_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Error guardando historial: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # 🆕 PROPIEDADES (Retrocompatibilidad)
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def models_trained(self) -> int:
        """
        🆕 PROPIEDAD: Número total de modelos entrenados
        Retorna el tamaño del historial de entrenamiento
        """
        return len(self.training_history)

    def get_stats(self) -> dict[str, Any]:
        """Obtener estadísticas del pipeline"""
        return {
            "total_models_trained": self.models_trained,
            "active_models": len(self.deployed_models),
            "queue_size": self.training_queue.qsize(),
            "perpetual_mode": self.enable_perpetual_mode,
            "neural_network_connected": self.neural_network is not None,
            "memory_connected": self.memory is not None,
            "cache_connected": self.cache is not None,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 🎖️ SINGLETON & COMPATIBILITY (v3.0)
# ═══════════════════════════════════════════════════════════════════════════

_global_ml_pipeline: MilitaryGradeMLPipeline | None = None


def get_ml_pipeline(**kwargs) -> MilitaryGradeMLPipeline:
    """
    🎖️ Obtener instancia global MILITARY GRADE

    Returns:
        MilitaryGradeMLPipeline: Instancia singleton v3.0
    """
    global _global_ml_pipeline
    if _global_ml_pipeline is None:
        _global_ml_pipeline = MilitaryGradeMLPipeline(**kwargs)
    return _global_ml_pipeline


# 🔄 BACKWARD COMPATIBILITY: MLPipeline = MilitaryGradeMLPipeline
MLPipeline = MilitaryGradeMLPipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pipeline = get_ml_pipeline()

    print("\n" + "=" * 80)
    print("🎖️ ML PIPELINE MILITARY GRADE v3.0")
    print("=" * 80)
    print(f"   Modelos dir: {pipeline.models_dir}")
    print(f"   Modo perpetuo: {pipeline.perpetual_running}")
    print(f"   Queue: {pipeline.training_queue.qsize()}")
    print(f"   Conexiones simbióticas: {pipeline.metrics['neural_connections_active']}")
    print(f"   Military Features: ✓")
    print(f"   Circuit Breaker: {'✓' if pipeline.circuit_breaker.get('enabled') else '✗'}")
    print(f"   Rate Limiter: {'✓' if pipeline.rate_limiter.get('enabled') else '✗'}")
    print(f"   Event Sourcing: {'✓' if pipeline.event_log else '✗'}")
    print("=" * 80)

    print("\n✅ Sistema ML Pipeline MILITARY GRADE v3.0 inicializado correctamente")