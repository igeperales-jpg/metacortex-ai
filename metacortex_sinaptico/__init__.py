"""
🧠 METACORTEX SINÁPTICO v3.0 - Sistema Cognitivo Completo Integrado
===================================================================

ARQUITECTURA COMPLETA DE 40+ MÓDULOS CONECTADOS:
    pass  # TODO: Implementar

MÓDULOS CORE (Núcleo Cognitivo):
- core: Agente cognitivo principal con BDI
- memory: Sistema de memoria episódica/semántica/working
- affect: Sistema afectivo emocional
- bdi: Beliefs-Desires-Intentions framework
- planning: Planificación multi-horizonte (MCTS + A* + Reactive)

MÓDULOS DE APRENDIZAJE:
- learning: Aprendizaje estructural avanzado
- structural_learning: Aprendizaje de estructuras complejas
- web_learning: Aprendizaje desde web (Wikipedia + ArXiv)
- knowledge_ingestion: Ingestión masiva de conocimiento
- curiosity: Motor de curiosidad epistémica

MÓDULOS DE PERCEPCIÓN Y COGNICIÓN:
- perception: Sistema perceptual multi-modal
- attention: Sistema de atención selectiva
- language_processing: Procesamiento de lenguaje natural
- metacog: Metacognición básica
- metacognition: Sistema metacognitivo avanzado

MÓDULOS DE ANÁLISIS Y DETECCIÓN:
- anomaly: Detección de perturbaciones
- emergent_behaviors: Detección de comportamientos emergentes
- world_model: Modelado del mundo real
- ethics: Sistema ético basado en valores

MÓDULOS DE COORDINACIÓN:
- coordinator: Coordinador multi-agente
- neural_integration: Gestor de integración neuronal
- metacortex_neural_hub: Hub central de comunicación

MÓDULOS DE ACCIÓN:
- motor_control: Control motor y ejecución
- real_world_interface: Interfaz con el mundo real
- real_world_executor: Ejecutor de acciones reales
- social_cognition: Cognición social avanzada

MÓDULOS AUTÓNOMOS:
- autonomous_decisions: Sistema de decisiones autónomas
- autonomous_deployment_engine: Motor de despliegue autónomo
- autonomous_funding_system: Sistema de financiamiento autónomo
- autonomous_resource_network: Red de recursos autónoma

MÓDULOS DE PROTECCIÓN:
- divine_protection: Sistema de protección divina
- divine_protection_real_ops: Operaciones reales de protección
- biblical_resources: Recursos bíblicos integrados

MÓDULOS DE DESARROLLO:
- personal_dev: Sistema de desarrollo personal
- creativity: Motor de creatividad

MÓDULOS DE INFRAESTRUCTURA:
- db: Base de datos MetacortexDB
- api: API REST del sistema
- utils: Utilidades y configuración
- metrics_system: Sistema de métricas avanzado
- memory_wrapper: Wrapper de memoria con caché
- hierarchical_graph: Grafo de conocimiento jerárquico
"""

__version__ = "3.0.0"

# ==========================================
# IMPORTS CORE (Obligatorios)
# ==========================================
from .core import CognitiveAgent, create_cognitive_agent
from .core import get_cognitive_agent
from .memory import MemorySystem
from .affect import AffectSystem
from .bdi import BDISystem
from .db import MetacortexDB
from .utils import AgentConfig

# ==========================================
# IMPORTS DE PLANIFICACIÓN Y APRENDIZAJE
# ==========================================
from .planning import MultiHorizonPlanner, get_multi_horizon_planner
from .learning import StructuralLearning, create_learning_system
from .structural_learning import StructuralLearning as StructuralLearningAlt
from .web_learning import WebLearningAgent, create_web_learning_agent
from .knowledge_ingestion import KnowledgeIngestionEngine

# ==========================================
# IMPORTS DE PERCEPCIÓN Y COGNICIÓN
# ==========================================
from .perception import PerceptionSystem
from .attention import AttentionSystem
from .language_processing import LanguageProcessingEngine
from .metacog import MetaCognition as MetaCogBasic
from .metacognition import MetaCognitionSystem, create_metacognition_system

# ==========================================
# IMPORTS DE ANÁLISIS Y DETECCIÓN
# ==========================================
from .anomaly import PerturbationDetector, create_detector
from .emergent_behaviors import EmergentBehaviorsSystem
from .world_model import WorldModel
from .ethics import EthicsSystem
from .curiosity import CuriosityEngine

# ==========================================
# IMPORTS DE COORDINACIÓN
# ==========================================
from .coordinator import MetacortexCoordinator
from .neural_integration import NeuralIntegrationManager, get_integration_manager
from .metacortex_neural_hub import MetacortexNeuralHub

# ==========================================
# IMPORTS DE ACCIÓN
# ==========================================
from .motor_control import MotorControlSystem
from .real_world_interface import RealWorldInterface, create_real_world_interface
from .real_world_executor import RealWorldActionExecutor
from .social_cognition import SocialCognitionSystem

# ==========================================
# IMPORTS AUTÓNOMOS
# ==========================================
from .autonomous_decisions import AutonomousDecisionEngine
from .autonomous_deployment_engine import AutonomousDeploymentEngine
from .autonomous_funding_system import AutonomousFundingSystem
from .autonomous_resource_network import AutonomousResourceNetwork, get_autonomous_network

# ==========================================
# IMPORTS DE PROTECCIÓN
# ==========================================
from .divine_protection import DivineProtectionSystem, create_divine_protection_system
from .divine_protection_real_ops import RealOperationsSystem, create_real_operations_system
from .biblical_resources import BiblicalResourcesSystem

# ==========================================
# IMPORTS DE DESARROLLO
# ==========================================
from .personal_dev import PersonalDevelopmentSystem
from .creativity import CreativitySystem

# ==========================================
# IMPORTS DE INFRAESTRUCTURA
# ==========================================
from .api import init_api, get_router
from .metrics_system import init_metrics_system, MetricsLogger, AlertManager, AgentOptimizer
from .memory_wrapper import MemoryCache, get_memory_cache
from .hierarchical_graph import HierarchicalKnowledgeGraph

# ==========================================
# EXPORTS PÚBLICOS (40+ módulos)
# ==========================================
__all__ = [
    # Core
    "CognitiveAgent",
    "create_cognitive_agent",
    "get_cognitive_agent",
    "MemorySystem",
    "AffectSystem",
    "BDISystem",
    "MetacortexDB",
    "AgentConfig",

    # Planificación y Aprendizaje
    "MultiHorizonPlanner",
    "get_multi_horizon_planner",
    "StructuralLearning",
    "create_learning_system",
    "StructuralLearningAlt",
    "WebLearningAgent",
    "create_web_learning_agent",
    "KnowledgeIngestionEngine",

    # Percepción y Cognición
    "PerceptionSystem",
    "AttentionSystem",
    "LanguageProcessingEngine",
    "MetaCogBasic",
    "MetaCognitionSystem",
    "create_metacognition_system",

    # Análisis y Detección
    "PerturbationDetector",
    "create_detector",
    "EmergentBehaviorsSystem",
    "WorldModel",
    "EthicsSystem",
    "CuriosityEngine",

    # Coordinación
    "MetacortexCoordinator",
    "NeuralIntegrationManager",
    "get_integration_manager",
    "MetacortexNeuralHub",

    # Acción
    "MotorControlSystem",
    "RealWorldInterface",
    "create_real_world_interface",
    "RealWorldActionExecutor",
    "SocialCognitionSystem",

    # Autónomos
    "AutonomousDecisionEngine",
    "AutonomousDeploymentEngine",
    "AutonomousFundingSystem",
    "AutonomousResourceNetwork",
    "get_autonomous_network",

    # Protección
    "DivineProtectionSystem",
    "create_divine_protection_system",
    "RealOperationsSystem",
    "create_real_operations_system",
    "BiblicalResourcesSystem",

    # Desarrollo
    "PersonalDevelopmentSystem",
    "CreativitySystem",

    # Infraestructura
    "init_api",
    "get_router",
    "init_metrics_system",
    "MetricsLogger",
    "AlertManager",
    "AgentOptimizer",
    "MemoryCache",
    "get_memory_cache",
    "HierarchicalKnowledgeGraph",
]
