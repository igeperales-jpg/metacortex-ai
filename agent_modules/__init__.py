#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METACORTEX Agent Modules Package v2.0
======================================

🎖️ MILITARY-GRADE MODULES ECOSYSTEM

CORE MODULES:
    pass  # TODO: Implementar
- workspace_scanner: Escaneo inteligente de workspace
- template_system: Sistema de plantillas avanzado
- project_analyzer: Análisis de calidad de código
- language_handlers: Manejadores específicos por lenguaje
- code_generator: Generación multi-lenguaje de código
- materialization_engine: Materialización de pensamientos en código

QUALITY ASSURANCE MODULES (Military-Grade):
- advanced_testing_lab: Testing exhaustivo con 6 validators
- self_repair_workshop: Auto-reparación inteligente
- code_quality_enforcer: Enforcement de estándares militares
- ai_programming_evolution: Sistema de aprendizaje y evolución
- continuous_validation: Validación continua asíncrona
- quality_integration_system: Orquestador unificado de calidad

INFRASTRUCTURE MODULES:
- distributed_cache: Sistema de caché distribuido
- rate_limiting: Rate limiting avanzado
- resilience: Circuit breakers y retry logic
- security: Seguridad y encriptación
- telemetry: Telemetría y métricas
- event_sourcing: Event sourcing para auditoría
- autoscaling: Auto-escalado inteligente

Autor: METACORTEX Evolution Team
Fecha: 2025-11-12
Versión: 2.0.0 Military-Grade
"""

__version__ = "2.0.0"
__author__ = "METACORTEX Evolution Team"
__status__ = "Production - Military-Grade"

# ═══════════════════════════════════════════════════════════════════════════
# CORE MODULES - Base Programming Capabilities
# ═══════════════════════════════════════════════════════════════════════════

from .workspace_scanner import WorkspaceScanner, get_workspace_scanner
from .template_system import TemplateSystem, get_template_system
from .project_analyzer import ProjectAnalyzer, get_project_analyzer
from .language_handlers import (
    LanguageHandler,
    LanguageHandlerRegistry,
    PythonHandler,
    JavaScriptHandler,
    TypeScriptHandler,
    JavaHandler,
    get_language_handler_registry,
)
from .code_generator import (
    CodeGenerator,
    ProgrammingLanguage,
    ProjectType,
    ProgrammingTask,
    get_code_generator,
)
from .materialization_engine import MaterializationEngine, get_materialization_engine

# ═══════════════════════════════════════════════════════════════════════════
# QUALITY ASSURANCE MODULES - Military-Grade Testing & Validation
# ═══════════════════════════════════════════════════════════════════════════

from .advanced_testing_lab import (
    AdvancedTestingLab,
    TestReport,
    TestIssue,
    TestCategory,
    TestSeverity,
    get_testing_lab,
)

from .self_repair_workshop import (
    SelfRepairWorkshop,
    RepairReport,
    RepairAction,
    RepairStrategy,
    get_repair_workshop,
)

from .code_quality_enforcer import (
    CodeQualityEnforcer,
    QualityMetrics,
    QualityLevel,
    get_enforcer,
)

from .ai_programming_evolution import (
    AIProgrammingEvolution,
    ProgrammingPattern,
    EvolutionMetrics,
    get_evolution_system,
)

from .continuous_validation import (
    ContinuousValidationSystem,
    ValidationResult,
    get_validation_system,
)

from .quality_integration_system import (
    QualityIntegrationSystem,
    IntegratedQualityResult,
    get_integration_system,
    analyze_code_sync,
)

# ═══════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE MODULES - Scalability & Reliability
# ═══════════════════════════════════════════════════════════════════════════

from .distributed_cache import DistributedCacheSystem, get_distributed_cache
from .rate_limiting import RateLimitSystem, get_rate_limiter
from .resilience import CircuitBreaker, RetryStrategy, get_circuit_breaker
from .security import SecuritySystem, get_security_system
from .telemetry import Telemetry, get_telemetry
from .event_sourcing import EventStore, get_event_store
from .autoscaling import AutoScalingSystem, get_autoscaling_system

__all__ = [
    # ═══════════════════════════════════════════════════════════════════════
    # VERSION INFO
    # ═══════════════════════════════════════════════════════════════════════
    "__version__",
    "__author__",
    "__status__",
    
    # ═══════════════════════════════════════════════════════════════════════
    # CORE MODULES
    # ═══════════════════════════════════════════════════════════════════════
    # WorkspaceScanner
    "WorkspaceScanner",
    "get_workspace_scanner",
    # TemplateSystem
    "TemplateSystem",
    "get_template_system",
    # ProjectAnalyzer
    "ProjectAnalyzer",
    "get_project_analyzer",
    # LanguageHandlers
    "LanguageHandler",
    "LanguageHandlerRegistry",
    "PythonHandler",
    "JavaScriptHandler",
    "TypeScriptHandler",
    "JavaHandler",
    "get_language_handler_registry",
    # CodeGenerator
    "CodeGenerator",
    "ProgrammingLanguage",
    "ProjectType",
    "ProgrammingTask",
    "get_code_generator",
    # MaterializationEngine
    "MaterializationEngine",
    "get_materialization_engine",
    
    # ═══════════════════════════════════════════════════════════════════════
    # QUALITY ASSURANCE MODULES (Military-Grade)
    # ═══════════════════════════════════════════════════════════════════════
    # Advanced Testing Lab
    "AdvancedTestingLab",
    "TestReport",
    "TestIssue",
    "TestCategory",
    "TestSeverity",
    "get_testing_lab",
    # Self-Repair Workshop
    "SelfRepairWorkshop",
    "RepairReport",
    "RepairAction",
    "RepairStrategy",
    "get_repair_workshop",
    # Code Quality Enforcer
    "CodeQualityEnforcer",
    "QualityMetrics",
    "QualityLevel",
    "get_enforcer",
    # AI Programming Evolution
    "AIProgrammingEvolution",
    "ProgrammingPattern",
    "EvolutionMetrics",
    "get_evolution_system",
    # Continuous Validation
    "ContinuousValidationSystem",
    "ValidationResult",
    "get_validation_system",
    # Quality Integration System
    "QualityIntegrationSystem",
    "IntegratedQualityResult",
    "get_integration_system",
    "analyze_code_sync",
    
    # ═══════════════════════════════════════════════════════════════════════
    # INFRASTRUCTURE MODULES
    # ═══════════════════════════════════════════════════════════════════════
    # Distributed Cache
    "DistributedCacheSystem",
    "get_distributed_cache",
    # Rate Limiting
    "RateLimitSystem",
    "get_rate_limiter",
    # Resilience
    "CircuitBreaker",
    "RetryStrategy",
    "get_circuit_breaker",
    # Security
    "SecuritySystem",
    "get_security_system",
    # Telemetry
    "Telemetry",
    "get_telemetry",
    # Event Sourcing
    "EventStore",
    "get_event_store",
    # Autoscaling
    "AutoScalingSystem",
    "get_autoscaling_system",
]
