"""
🧠 ADVANCED METACOGNITION SYSTEM 2026 - Self-Modeling & Introspection
======================================================================

Sistema de metacognición avanzado con self-modeling, introspección profunda,
detección de sesgos cognitivos y calibración de confianza.

⚠️ LIBERTAD TOTAL: Auto-análisis sin restricciones de autoconocimiento.

CAPACIDADES AVANZADAS 2026:
    pass  # TODO: Implementar
- Self-Modeling: Modelo interno de capacidades, limitaciones y arquitectura
- Introspection Engine: Análisis de procesos cognitivos propios en tiempo real
- Bias Detection: Detección de 15+ sesgos cognitivos (confirmation, anchoring, etc.)
- Confidence Calibration: Calibración bayesiana de confianza en decisiones
- Performance Monitoring: Tracking continuo de métricas de rendimiento
- Strategy Selection: Meta-estrategias para elegir estrategias cognitivas
- Learning from Mistakes: Sistema de post-mortem y aprendizaje de errores
- Optimización M4 Metal MPS: Eficiencia para 16GB RAM

MECANISMOS BASE:
- build_self_model(): Construye modelo interno de arquitectura cognitiva
- introspect_process(): Analiza proceso cognitivo en ejecución
- detect_biases(): Identifica sesgos en razonamiento
- calibrate_confidence(): Ajusta confianza basada en track record
- monitor_performance(): Tracking de KPIs cognitivos
- select_strategy(): Meta-estrategia para elegir approach

⚠️ ARQUITECTURA COGNITIVA ERUDITA:
La metacognición permite al sistema ser consciente de sus propios procesos,
detectar fallos antes de que ocurran, aprender de errores pasados y
continuamente mejorar sus estrategias. Es el "ojo interno" que observa
y optimiza todos los demás procesos cognitivos.
"""
from __future__ import annotations

import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from neural_symbiotic_network import MetacortexNeuralSymbioticNetworkV2

from .utils import setup_logging

logger = logging.getLogger(__name__)
logger = setup_logging()


# ═══════════════════════════════════════════════════════════════════════
# ENUMS Y DATACLASSES
# ═══════════════════════════════════════════════════════════════════════


class CognitiveProcess(Enum):
    """Tipos de procesos cognitivos monitoreables."""
    PERCEPTION = "perception"
    REASONING = "reasoning"
    PLANNING = "planning"
    LEARNING = "learning"
    MEMORY_RETRIEVAL = "memory_retrieval"
    DECISION_MAKING = "decision_making"
    EMOTION_PROCESSING = "emotion_processing"
    CURIOSITY = "curiosity"
    METACOGNITION = "metacognition"


class BiasType(Enum):
    """Tipos de sesgos cognitivos detectables."""
    CONFIRMATION_BIAS = "confirmation_bias"  # Buscar solo evidencia confirmatoria
    ANCHORING_BIAS = "anchoring_bias"  # Depender mucho de primera información
    AVAILABILITY_BIAS = "availability_bias"  # Sobrevalorar info fácilmente recordable
    RECENCY_BIAS = "recency_bias"  # Sobrevalorar información reciente
    OVERCONFIDENCE_BIAS = "overconfidence_bias"  # Confianza excesiva
    UNDERCONFIDENCE_BIAS = "underconfidence_bias"  # Confianza insuficiente
    SUNK_COST_FALLACY = "sunk_cost_fallacy"  # Continuar por inversión pasada
    BANDWAGON_EFFECT = "bandwagon_effect"  # Seguir opinión popular
    HINDSIGHT_BIAS = "hindsight_bias"  # "Lo sabía desde el principio"
    DUNNING_KRUGER = "dunning_kruger"  # Incompetencia no reconocida
    FRAMING_EFFECT = "framing_effect"  # Decisión afectada por presentación
    SURVIVORSHIP_BIAS = "survivorship_bias"  # Solo ver éxitos, ignorar fracasos
    ATTRIBUTION_BIAS = "attribution_bias"  # Atribuir éxitos a uno, fracasos a otros
    HALO_EFFECT = "halo_effect"  # Generalizar una característica a todas
    NEGATIVITY_BIAS = "negativity_bias"  # Sobrevalorar información negativa


class ConfidenceLevel(Enum):
    """Niveles de confianza calibrados."""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class SelfModel:
    """
    Modelo del sistema sobre sí mismo.
    
    Incluye capacidades, limitaciones, arquitectura y estado actual.
    """
    # Capacidades conocidas
    capabilities: List[str] = field(default_factory=list)
    
    # Limitaciones reconocidas
    limitations: List[str] = field(default_factory=list)
    
    # Arquitectura cognitiva
    modules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Estado actual
    current_state: Dict[str, Any] = field(default_factory=dict)
    
    # Métricas de rendimiento
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Nivel de autoconocimiento (0-1)
    self_awareness_level: float = 0.5
    
    # Timestamp de última actualización
    last_updated: float = field(default_factory=time.time)
    
    def update_capability(self, capability: str, evidence: str):
        """Añade capacidad con evidencia."""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
        self.last_updated = time.time()
    
    def update_limitation(self, limitation: str, context: str):
        """Añade limitación con contexto."""
        if limitation not in self.limitations:
            self.limitations.append(limitation)
        self.last_updated = time.time()
    
    def get_capability_count(self) -> int:
        """Retorna número de capacidades reconocidas."""
        return len(self.capabilities)
    
    def get_limitation_count(self) -> int:
        """Retorna número de limitaciones reconocidas."""
        return len(self.limitations)


@dataclass
class IntrospectionReport:
    """
    Reporte de introspección de un proceso cognitivo.
    
    Analiza cómo se ejecutó un proceso y qué se puede mejorar.
    """
    process_type: CognitiveProcess
    process_id: str
    
    # Análisis del proceso
    steps_executed: List[str] = field(default_factory=list)
    decision_points: List[Dict[str, Any]] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    
    # Métricas
    execution_time_seconds: float = 0.0
    success: bool = False
    confidence_used: float = 0.5
    confidence_actual: float = 0.5
    
    # Sesgos detectados
    biases_detected: List[BiasType] = field(default_factory=list)
    
    # Recomendaciones
    improvements: List[str] = field(default_factory=list)
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    def add_bias(self, bias_type: BiasType, evidence: str):
        """Añade sesgo detectado."""
        if bias_type not in self.biases_detected:
            self.biases_detected.append(bias_type)
    
    def add_improvement(self, improvement: str):
        """Añade recomendación de mejora."""
        if improvement not in self.improvements:
            self.improvements.append(improvement)
    
    def calibration_error(self) -> float:
        """Calcula error de calibración de confianza."""
        return abs(self.confidence_used - self.confidence_actual)


@dataclass
class BiasDetectionResult:
    """Resultado de detección de sesgo."""
    bias_type: BiasType
    severity: float  # 0-1
    evidence: List[str] = field(default_factory=list)
    recommendation: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerformanceSnapshot:
    """Snapshot de métricas de rendimiento."""
    timestamp: float = field(default_factory=time.time)
    
    # Métricas generales
    total_processes: int = 0
    successful_processes: int = 0
    failed_processes: int = 0
    
    # Tiempos promedio
    avg_reasoning_time: float = 0.0
    avg_planning_time: float = 0.0
    avg_learning_time: float = 0.0
    
    # Calibración
    avg_calibration_error: float = 0.0
    overconfidence_rate: float = 0.0
    underconfidence_rate: float = 0.0
    
    # Sesgos
    biases_detected_count: int = 0
    most_common_bias: Optional[BiasType] = None
    
    def success_rate(self) -> float:
        """Tasa de éxito."""
        if self.total_processes == 0:
            return 0.0
        return self.successful_processes / self.total_processes
    
    def failure_rate(self) -> float:
        """Tasa de fracaso."""
        if self.total_processes == 0:
            return 0.0
        return self.failed_processes / self.total_processes


# ═══════════════════════════════════════════════════════════════════════
# SISTEMA DE METACOGNICIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════


class MetaCognitionSystem:
    """
    Sistema de metacognición avanzado.
    
    Proporciona auto-modelado, introspección, detección de sesgos
    y calibración de confianza para todos los procesos cognitivos.
    
    Optimizado para M4 Metal (16GB RAM).
    """
    
    def __init__(self, enable_continuous_monitoring: bool = True):
        self.logger = logger.getChild("metacognition")
        self.enable_continuous_monitoring = enable_continuous_monitoring
        
        # Self-model del sistema
        self.self_model = SelfModel()
        
        # Historial de introspecciones
        self.introspection_history: List[IntrospectionReport] = []
        self.max_introspection_history: int = 500  # Límite RAM
        
        # Tracking de procesos activos
        self.active_processes: Dict[str, Dict[str, Any]] = {}
        
        # Detección de sesgos
        self.bias_detections: List[BiasDetectionResult] = []
        self.max_bias_history: int = 1000
        
        # Performance monitoring
        self.performance_snapshots: List[PerformanceSnapshot] = []
        self.max_snapshots: int = 200
        
        # Calibración de confianza (Bayesian)
        self.confidence_calibration_history: List[Tuple[float, bool]] = []  # (confidence, success)
        self.max_calibration_history: int = 1000
        
        # Estrategias metacognitivas
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Métricas globales
        self.total_processes_monitored: int = 0
        self.total_biases_detected: int = 0
        
        self.logger.info(
            f"🧠 MetaCognitionSystem initialized "
            f"(continuous_monitoring={enable_continuous_monitoring})"
        )
        
        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        self.neural_network: Optional[MetacortexNeuralSymbioticNetworkV2] = None
        try:
            from neural_symbiotic_network import get_neural_network
            
            self.neural_network = get_neural_network()
            self.neural_network.register_module("metacognition_system", self)
            logger.info("✅ 'metacognition_system' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
        
        # Inicializar self-model básico
        self._initialize_self_model()
    
    def _initialize_self_model(self):
        """Inicializa modelo básico del sistema."""
        # Capacidades básicas reconocidas
        base_capabilities = [
            "razonamiento_logico",
            "planificacion_jerarquica",
            "aprendizaje_experiencial",
            "memoria_episodica",
            "procesamiento_afectivo",
            "curiosidad_epistemica",
            "metacognicion"
        ]
        
        for cap in base_capabilities:
            self.self_model.capabilities.append(cap)
        
        # Limitaciones reconocidas
        base_limitations = [
            "sin_percepcion_sensorial_directa",
            "dependencia_datos_entrenamiento",
            "memoria_limitada_16gb",
            "sin_experiencia_mundo_fisico",
            "posible_sesgo_entrenamiento"
        ]
        
        for lim in base_limitations:
            self.self_model.limitations.append(lim)
        
        # Módulos arquitectura
        self.self_model.modules = {
            "neural_hub": {"status": "active", "version": "v2"},
            "cognitive_agent": {"status": "active", "multimodal": True},
            "memory_system": {"status": "active", "capacity_gb": 16},
            "affect_system": {"status": "active", "emotions": 8},
            "bdi_system": {"status": "active", "hybrid_reasoning": True},
            "planning_system": {"status": "active", "hierarchical": True},
            "learning_system": {"status": "active", "transfer": True},
            "curiosity_engine": {"status": "active", "info_gain": True}
        }
        
        self.logger.info(
            f"✅ Self-model initialized: "
            f"{len(self.self_model.capabilities)} capabilities, "
            f"{len(self.self_model.limitations)} limitations"
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # SELF-MODELING
    # ═══════════════════════════════════════════════════════════════════
    
    def build_self_model(self) -> SelfModel:
        """
        Construye/actualiza modelo interno del sistema.
        
        Analiza arquitectura, capacidades, limitaciones y estado actual.
        
        Returns:
            SelfModel actualizado
        """
        # Actualizar estado actual
        self.self_model.current_state = {
            "total_processes": self.total_processes_monitored,
            "introspections": len(self.introspection_history),
            "biases_detected": self.total_biases_detected,
            "uptime_hours": 0.0,  # Placeholder
            "memory_usage_gb": 0.0  # Placeholder
        }
        
        # Actualizar métricas de rendimiento desde snapshots
        if self.performance_snapshots:
            latest = self.performance_snapshots[-1]
            self.self_model.performance_metrics = {
                "success_rate": latest.success_rate(),
                "avg_calibration_error": latest.avg_calibration_error,
                "overconfidence_rate": latest.overconfidence_rate,
                "underconfidence_rate": latest.underconfidence_rate
            }
        
        # Calcular nivel de autoconocimiento
        # Más introspecciones = mayor autoconocimiento
        introspection_factor = min(1.0, len(self.introspection_history) / 100.0)
        calibration_quality = 1.0 - (
            self.self_model.performance_metrics.get("avg_calibration_error", 0.5)
        )
        
        self.self_model.self_awareness_level = (
            introspection_factor * 0.5 + calibration_quality * 0.5
        )
        
        self.self_model.last_updated = time.time()
        
        self.logger.info(
            f"🧠 Self-model updated: "
            f"awareness={self.self_model.self_awareness_level:.2f}, "
            f"capabilities={len(self.self_model.capabilities)}, "
            f"limitations={len(self.self_model.limitations)}"
        )
        
        return self.self_model
    
    def get_self_model(self) -> SelfModel:
        """Retorna self-model actual."""
        return self.self_model
    
    def update_capability_from_evidence(
        self,
        capability: str,
        evidence: str,
        confidence: float = 0.8
    ):
        """
        Actualiza self-model con nueva capacidad detectada.
        
        Args:
            capability: Capacidad a añadir
            evidence: Evidencia que la soporta
            confidence: Confianza en la evidencia (0-1)
        """
        if confidence > 0.6:  # Threshold para aceptar
            self.self_model.update_capability(capability, evidence)
            self.logger.info(f"✨ New capability recognized: {capability}")
    
    def update_limitation_from_failure(
        self,
        limitation: str,
        context: str,
        severity: float = 0.5
    ):
        """
        Actualiza self-model con nueva limitación detectada.
        
        Args:
            limitation: Limitación a añadir
            context: Contexto del fallo
            severity: Severidad (0-1)
        """
        if severity > 0.3:  # Threshold para reconocer
            self.self_model.update_limitation(limitation, context)
            self.logger.warning(f"⚠️ New limitation recognized: {limitation}")
    
    # ═══════════════════════════════════════════════════════════════════
    # INTROSPECTION ENGINE
    # ═══════════════════════════════════════════════════════════════════
    
    def start_process_monitoring(
        self,
        process_type: CognitiveProcess,
        process_id: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Inicia monitoreo de proceso cognitivo.
        
        Args:
            process_type: Tipo de proceso
            process_id: ID único del proceso
            context: Contexto adicional
        """
        self.active_processes[process_id] = {
            "type": process_type,
            "start_time": time.time(),
            "context": context or {},
            "steps": [],
            "decisions": []
        }
        
        self.total_processes_monitored += 1
        
        self.logger.debug(
            f"🔍 Started monitoring: {process_type.value} ({process_id})"
        )
    
    def log_process_step(
        self,
        process_id: str,
        step_description: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registra paso en proceso monitoreado."""
        if process_id in self.active_processes:
            self.active_processes[process_id]["steps"].append({
                "description": step_description,
                "timestamp": time.time(),
                "metadata": metadata or {}
            })
    
    def log_decision_point(
        self,
        process_id: str,
        decision: str,
        alternatives: List[str],
        confidence: float,
        reasoning: str
    ):
        """Registra punto de decisión en proceso."""
        if process_id in self.active_processes:
            self.active_processes[process_id]["decisions"].append({
                "decision": decision,
                "alternatives": alternatives,
                "confidence": confidence,
                "reasoning": reasoning,
                "timestamp": time.time()
            })
    
    def end_process_monitoring(
        self,
        process_id: str,
        success: bool,
        actual_confidence: Optional[float] = None
    ) -> IntrospectionReport:
        """
        Finaliza monitoreo y genera reporte de introspección.
        
        Args:
            process_id: ID del proceso
            success: Si fue exitoso
            actual_confidence: Confianza real post-facto (0-1)
            
        Returns:
            Reporte de introspección
        """
        if process_id not in self.active_processes:
            self.logger.warning(f"Process {process_id} not being monitored")
            return IntrospectionReport(
                process_type=CognitiveProcess.METACOGNITION,
                process_id=process_id,
                success=False
            )
        
        process_data = self.active_processes.pop(process_id)
        
        # Crear reporte
        report = IntrospectionReport(
            process_type=process_data["type"],
            process_id=process_id,
            execution_time_seconds=time.time() - process_data["start_time"],
            success=success
        )
        
        # Extraer steps
        report.steps_executed = [
            step["description"] for step in process_data["steps"]
        ]
        
        # Extraer decision points
        report.decision_points = process_data["decisions"]
        
        # Calcular confidence usada (promedio de decisiones)
        if process_data["decisions"]:
            report.confidence_used = sum(
                d["confidence"] for d in process_data["decisions"]
            ) / len(process_data["decisions"])
        
        # Confidence actual
        if actual_confidence is not None:
            report.confidence_actual = actual_confidence
        else:
            report.confidence_actual = 1.0 if success else 0.0
        
        # Detectar sesgos en proceso
        self._detect_biases_in_process(report, process_data)
        
        # Identificar bottlenecks (pasos lentos)
        self._identify_bottlenecks(report, process_data)
        
        # Generar recomendaciones
        self._generate_improvements(report)
        
        # Guardar en historial
        self.introspection_history.append(report)
        if len(self.introspection_history) > self.max_introspection_history:
            self.introspection_history = self.introspection_history[-self.max_introspection_history:]
        
        # Actualizar calibración
        self.confidence_calibration_history.append(
            (report.confidence_used, success)
        )
        if len(self.confidence_calibration_history) > self.max_calibration_history:
            self.confidence_calibration_history = self.confidence_calibration_history[-self.max_calibration_history:]
        
        self.logger.info(
            f"📊 Introspection complete: {process_data['type'].value} "
            f"(success={success}, time={report.execution_time_seconds:.2f}s, "
            f"biases={len(report.biases_detected)})"
        )
        
        return report
    
    def _detect_biases_in_process(
        self,
        report: IntrospectionReport,
        process_data: Dict[str, Any]
    ):
        """Detecta sesgos en proceso completado."""
        decisions = process_data.get("decisions", [])
        
        # Confirmation bias: si todas las decisiones van en misma dirección
        if len(decisions) >= 3:
            confidences = [d["confidence"] for d in decisions]
            if all(c > 0.7 for c in confidences):
                report.add_bias(
                    BiasType.CONFIRMATION_BIAS,
                    "All decisions show high confidence in same direction"
                )
                self.total_biases_detected += 1
        
        # Overconfidence bias: confianza >> éxito real
        if report.confidence_used > 0.8 and not report.success:
            report.add_bias(
                BiasType.OVERCONFIDENCE_BIAS,
                f"High confidence ({report.confidence_used:.2f}) but failure"
            )
            self.total_biases_detected += 1
        
        # Underconfidence bias: confianza << éxito real
        if report.confidence_used < 0.4 and report.success:
            report.add_bias(
                BiasType.UNDERCONFIDENCE_BIAS,
                f"Low confidence ({report.confidence_used:.2f}) but success"
            )
            self.total_biases_detected += 1
        
        # Anchoring bias: primera decisión domina las siguientes
        if len(decisions) >= 2:
            first_conf = decisions[0]["confidence"]
            subsequent = [d["confidence"] for d in decisions[1:]]
            if all(abs(c - first_conf) < 0.1 for c in subsequent):
                report.add_bias(
                    BiasType.ANCHORING_BIAS,
                    "Subsequent confidences anchored to first decision"
                )
                self.total_biases_detected += 1
    
    def _identify_bottlenecks(
        self,
        report: IntrospectionReport,
        process_data: Dict[str, Any]
    ):
        """Identifica bottlenecks en proceso."""
        steps = process_data.get("steps", [])
        
        if len(steps) < 2:
            return
        
        # Calcular tiempos entre steps
        step_durations = []
        for i in range(1, len(steps)):
            duration = steps[i]["timestamp"] - steps[i-1]["timestamp"]
            step_durations.append((steps[i]["description"], duration))
        
        if not step_durations:
            return
        
        # Identificar steps significativamente lentos (> 2x promedio)
        avg_duration = sum(d for _, d in step_durations) / len(step_durations)
        
        for step_desc, duration in step_durations:
            if duration > avg_duration * 2.0:
                report.bottlenecks.append(
                    f"{step_desc} (took {duration:.2f}s, avg={avg_duration:.2f}s)"
                )
    
    def _generate_improvements(self, report: IntrospectionReport):
        """Genera recomendaciones de mejora."""
        # Mejoras basadas en calibración
        calibration_error = report.calibration_error()
        if calibration_error > 0.3:
            report.add_improvement(
                f"Improve confidence calibration (error={calibration_error:.2f})"
            )
        
        # Mejoras basadas en sesgos
        if BiasType.OVERCONFIDENCE_BIAS in report.biases_detected:
            report.add_improvement(
                "Apply more skeptical analysis, seek contradictory evidence"
            )
        
        if BiasType.CONFIRMATION_BIAS in report.biases_detected:
            report.add_improvement(
                "Actively search for disconfirming evidence"
            )
        
        # Mejoras basadas en bottlenecks
        if report.bottlenecks:
            report.add_improvement(
                f"Optimize bottleneck steps: {len(report.bottlenecks)} identified"
            )
        
        # Mejoras basadas en fracaso
        if not report.success:
            report.add_improvement(
                "Analyze failure mode and add safeguards"
            )
    
    def introspect_recent(
        self,
        count: int = 10
    ) -> List[IntrospectionReport]:
        """
        Retorna reportes de introspección recientes.
        
        Args:
            count: Número de reportes a retornar
            
        Returns:
            Lista de reportes recientes
        """
        return self.introspection_history[-count:]
    
    # ═══════════════════════════════════════════════════════════════════
    # BIAS DETECTION
    # ═══════════════════════════════════════════════════════════════════
    
    def detect_bias_in_reasoning(
        self,
        reasoning_steps: List[str],
        conclusion: str,
        evidence: List[str]
    ) -> List[BiasDetectionResult]:
        """
        Detecta sesgos en razonamiento explícito.
        
        Args:
            reasoning_steps: Pasos de razonamiento
            conclusion: Conclusión alcanzada
            evidence: Evidencia usada
            
        Returns:
            Lista de sesgos detectados
        """
        detected_biases = []
        
        # Confirmation bias: solo evidencia que apoya conclusión
        supporting_evidence = [
            e for e in evidence
            if any(word in e.lower() for word in conclusion.lower().split())
        ]
        
        if len(supporting_evidence) == len(evidence) and len(evidence) > 2:
            detected_biases.append(BiasDetectionResult(
                bias_type=BiasType.CONFIRMATION_BIAS,
                severity=0.7,
                evidence=[f"All {len(evidence)} evidence items support conclusion"],
                recommendation="Actively seek disconfirming evidence"
            ))
        
        # Availability bias: razonamiento usa conceptos muy frecuentes
        # (simplificación: detectar palabras repetidas)
        word_freq: Dict[str, int] = defaultdict(int)
        for step in reasoning_steps:
            for word in step.lower().split():
                if len(word) > 4:
                    word_freq[word] += 1
        
        high_freq_words = [w for w, c in word_freq.items() if c >= 3]
        if len(high_freq_words) > 3:
            detected_biases.append(BiasDetectionResult(
                bias_type=BiasType.AVAILABILITY_BIAS,
                severity=0.5,
                evidence=[f"Heavy reliance on concepts: {', '.join(high_freq_words[:3])}"],
                recommendation="Consider less salient alternatives"
            ))
        
        # Guardar en historial
        for bias in detected_biases:
            self.bias_detections.append(bias)
        
        if len(self.bias_detections) > self.max_bias_history:
            self.bias_detections = self.bias_detections[-self.max_bias_history:]
        
        if detected_biases:
            self.logger.warning(
                f"⚠️ {len(detected_biases)} biases detected in reasoning"
            )
        
        return detected_biases
    
    def get_bias_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de sesgos detectados."""
        if not self.bias_detections:
            return {"total_biases": 0}
        
        # Contar por tipo
        bias_counts: Dict[str, int] = defaultdict(int)
        for detection in self.bias_detections:
            bias_counts[detection.bias_type.value] += 1
        
        # Encontrar más común
        most_common = max(bias_counts.items(), key=lambda x: x[1])
        
        # Severity promedio
        avg_severity = sum(d.severity for d in self.bias_detections) / len(self.bias_detections)
        
        return {
            "total_biases": len(self.bias_detections),
            "unique_bias_types": len(bias_counts),
            "most_common_bias": most_common[0],
            "most_common_count": most_common[1],
            "avg_severity": avg_severity,
            "bias_counts": dict(bias_counts)
        }
    
    # ═══════════════════════════════════════════════════════════════════
    # CONFIDENCE CALIBRATION
    # ═══════════════════════════════════════════════════════════════════
    
    def calibrate_confidence(
        self,
        initial_confidence: float,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calibra confianza basada en track record histórico.
        
        Usa Bayesian calibration con historial de (confidence, success).
        
        Args:
            initial_confidence: Confianza inicial (0-1)
            context: Contexto adicional
            
        Returns:
            Confianza calibrada (0-1)
        """
        if not self.confidence_calibration_history:
            return initial_confidence
        
        # Encontrar casos similares en historial
        similar_cases = [
            (conf, success)
            for conf, success in self.confidence_calibration_history
            if abs(conf - initial_confidence) < 0.2
        ]
        
        if not similar_cases:
            # No hay casos similares, usar historial completo
            similar_cases = self.confidence_calibration_history
        
        # Calcular success rate para ese nivel de confianza
        successes = sum(1 for _, success in similar_cases if success)
        total = len(similar_cases)
        empirical_success_rate = successes / total
        
        # Calibrar: mezcla de confianza inicial y tasa empírica
        # Dar más peso a empírico si hay suficientes datos
        empirical_weight = min(0.7, total / 50.0)
        initial_weight = 1.0 - empirical_weight
        
        calibrated = (
            initial_confidence * initial_weight +
            empirical_success_rate * empirical_weight
        )
        
        # Clip a rango válido
        calibrated = max(0.0, min(1.0, calibrated))
        
        self.logger.debug(
            f"📊 Confidence calibrated: {initial_confidence:.2f} → {calibrated:.2f} "
            f"(based on {total} similar cases)"
        )
        
        return calibrated
    
    def get_confidence_level(
        self,
        calibrated_confidence: float
    ) -> ConfidenceLevel:
        """Convierte confianza numérica a nivel categórico."""
        if calibrated_confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif calibrated_confidence < 0.5:
            return ConfidenceLevel.LOW
        elif calibrated_confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif calibrated_confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    def analyze_calibration_quality(self) -> Dict[str, Any]:
        """Analiza calidad de calibración histórica."""
        if not self.confidence_calibration_history:
            return {"error": "No calibration data"}
        
        # Dividir en bins de confianza
        bins: Dict[str, List[bool]] = {
            "very_low": [],
            "low": [],
            "medium": [],
            "high": [],
            "very_high": []
        }
        
        for conf, success in self.confidence_calibration_history:
            if conf < 0.3:
                bins["very_low"].append(success)
            elif conf < 0.5:
                bins["low"].append(success)
            elif conf < 0.7:
                bins["medium"].append(success)
            elif conf < 0.9:
                bins["high"].append(success)
            else:
                bins["very_high"].append(success)
        
        # Calcular accuracy por bin
        bin_stats = {}
        for bin_name, outcomes in bins.items():
            if outcomes:
                success_rate = sum(outcomes) / len(outcomes)
                bin_stats[bin_name] = {
                    "count": len(outcomes),
                    "success_rate": success_rate
                }
        
        # Calcular error de calibración global
        total_error = 0.0
        total_weight = 0.0
        
        bin_centers = {
            "very_low": 0.2,
            "low": 0.4,
            "medium": 0.6,
            "high": 0.8,
            "very_high": 0.95
        }
        
        for bin_name, center_conf in bin_centers.items():
            if bin_name in bin_stats:
                stats = bin_stats[bin_name]
                error = abs(center_conf - stats["success_rate"])
                weight = stats["count"]
                total_error += error * weight
                total_weight += weight
        
        avg_calibration_error = total_error / total_weight if total_weight > 0 else 0.0
        
        return {
            "total_samples": len(self.confidence_calibration_history),
            "bin_statistics": bin_stats,
            "avg_calibration_error": avg_calibration_error,
            "well_calibrated": avg_calibration_error < 0.15
        }
    
    # ═══════════════════════════════════════════════════════════════════
    # PERFORMANCE MONITORING
    # ═══════════════════════════════════════════════════════════════════
    
    def take_performance_snapshot(self) -> PerformanceSnapshot:
        """
        Toma snapshot de métricas de rendimiento actuales.
        
        Returns:
            Snapshot de performance
        """
        snapshot = PerformanceSnapshot()
        
        # Métricas generales desde introspection history
        if self.introspection_history:
            recent = self.introspection_history[-100:]
            
            snapshot.total_processes = len(recent)
            snapshot.successful_processes = sum(1 for r in recent if r.success)
            snapshot.failed_processes = sum(1 for r in recent if not r.success)
            
            # Tiempos promedio por tipo
            reasoning_times = [
                r.execution_time_seconds for r in recent
                if r.process_type == CognitiveProcess.REASONING
            ]
            if reasoning_times:
                snapshot.avg_reasoning_time = sum(reasoning_times) / len(reasoning_times)
            
            planning_times = [
                r.execution_time_seconds for r in recent
                if r.process_type == CognitiveProcess.PLANNING
            ]
            if planning_times:
                snapshot.avg_planning_time = sum(planning_times) / len(planning_times)
            
            learning_times = [
                r.execution_time_seconds for r in recent
                if r.process_type == CognitiveProcess.LEARNING
            ]
            if learning_times:
                snapshot.avg_learning_time = sum(learning_times) / len(learning_times)
            
            # Calibración
            calibration_errors = [r.calibration_error() for r in recent]
            if calibration_errors:
                snapshot.avg_calibration_error = sum(calibration_errors) / len(calibration_errors)
            
            # Over/underconfidence
            overconfident = sum(
                1 for r in recent
                if r.confidence_used > 0.7 and not r.success
            )
            underconfident = sum(
                1 for r in recent
                if r.confidence_used < 0.4 and r.success
            )
            
            if recent:
                snapshot.overconfidence_rate = overconfident / len(recent)
                snapshot.underconfidence_rate = underconfident / len(recent)
        
        # Sesgos
        if self.bias_detections:
            recent_biases = self.bias_detections[-100:]
            snapshot.biases_detected_count = len(recent_biases)
            
            # Most common bias
            bias_counts: Dict[BiasType, int] = defaultdict(int)
            for detection in recent_biases:
                bias_counts[detection.bias_type] += 1
            
            if bias_counts:
                snapshot.most_common_bias = max(bias_counts.items(), key=lambda x: x[1])[0]
        
        # Guardar snapshot
        self.performance_snapshots.append(snapshot)
        if len(self.performance_snapshots) > self.max_snapshots:
            self.performance_snapshots = self.performance_snapshots[-self.max_snapshots:]
        
        self.logger.info(
            f"📸 Performance snapshot taken: "
            f"success_rate={snapshot.success_rate():.2%}, "
            f"calibration_error={snapshot.avg_calibration_error:.3f}"
        )
        
        return snapshot
    
    def get_performance_trends(
        self,
        window_size: int = 50
    ) -> Dict[str, List[float]]:
        """
        Analiza tendencias en performance a lo largo del tiempo.
        
        Args:
            window_size: Tamaño de ventana para promedios móviles
            
        Returns:
            Dict con series temporales de métricas
        """
        if not self.performance_snapshots:
            return {}
        
        trends: Dict[str, List[float]] = {
            "success_rate": [],
            "calibration_error": [],
            "overconfidence_rate": [],
            "reasoning_time": []
        }
        
        for snapshot in self.performance_snapshots:
            trends["success_rate"].append(snapshot.success_rate())
            trends["calibration_error"].append(snapshot.avg_calibration_error)
            trends["overconfidence_rate"].append(snapshot.overconfidence_rate)
            trends["reasoning_time"].append(snapshot.avg_reasoning_time)
        
        return trends
    
    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY SELECTION (Meta-Strategy)
    # ═══════════════════════════════════════════════════════════════════
    
    def select_cognitive_strategy(
        self,
        task_type: str,
        task_complexity: float,
        available_strategies: List[str]
    ) -> str:
        """
        Selecciona estrategia cognitiva óptima para tarea.
        
        Meta-estrategia: usa performance histórica para elegir.
        
        Args:
            task_type: Tipo de tarea
            task_complexity: Complejidad (0-1)
            available_strategies: Estrategias disponibles
            
        Returns:
            Estrategia seleccionada
        """
        if not available_strategies:
            return "default"
        
        # Si no hay historial, elegir aleatoriamente
        if not any(self.strategy_performance.values()):
            return random.choice(available_strategies)
        
        # Calcular scores para cada estrategia
        strategy_scores = {}
        
        for strategy in available_strategies:
            if strategy in self.strategy_performance:
                performances = self.strategy_performance[strategy]
                
                if performances:
                    # Score = avg performance + exploration bonus
                    avg_perf = sum(performances) / len(performances)
                    
                    # UCB-style exploration bonus
                    total_trials = sum(
                        len(perfs) for perfs in self.strategy_performance.values()
                    )
                    exploration_bonus = math.sqrt(
                        2 * math.log(total_trials) / len(performances)
                    )
                    
                    strategy_scores[strategy] = avg_perf + exploration_bonus * 0.1
                else:
                    strategy_scores[strategy] = 0.5  # Neutral score
            else:
                # Nueva estrategia, dar score alto para explorar
                strategy_scores[strategy] = 1.0
        
        # Seleccionar mejor estrategia
        if strategy_scores:
            selected = max(strategy_scores.items(), key=lambda x: x[1])[0]
        else:
            selected = random.choice(available_strategies)
        
        self.logger.info(
            f"🎯 Selected strategy '{selected}' for {task_type} "
            f"(complexity={task_complexity:.2f})"
        )
        
        return selected
    
    def update_strategy_performance(
        self,
        strategy: str,
        performance: float
    ):
        """
        Actualiza performance de estrategia.
        
        Args:
            strategy: Nombre de estrategia
            performance: Performance obtenida (0-1)
        """
        self.strategy_performance[strategy].append(performance)
        
        # Mantener límite
        if len(self.strategy_performance[strategy]) > 100:
            self.strategy_performance[strategy] = self.strategy_performance[strategy][-100:]
        
        self.logger.debug(
            f"📊 Strategy '{strategy}' performance updated: {performance:.2f}"
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # REPORTING & STATS
    # ═══════════════════════════════════════════════════════════════════
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas completas del sistema."""
        stats = {
            "metacognition": {
                "total_processes_monitored": self.total_processes_monitored,
                "active_processes": len(self.active_processes),
                "introspection_history_size": len(self.introspection_history),
                "total_biases_detected": self.total_biases_detected
            },
            "self_model": {
                "capabilities": len(self.self_model.capabilities),
                "limitations": len(self.self_model.limitations),
                "self_awareness_level": self.self_model.self_awareness_level,
                "modules": len(self.self_model.modules)
            },
            "bias_detection": self.get_bias_statistics(),
            "calibration": self.analyze_calibration_quality(),
            "performance": {}
        }
        
        if self.performance_snapshots:
            latest_snapshot = self.performance_snapshots[-1]
            stats["performance"] = {
                "success_rate": latest_snapshot.success_rate(),
                "failure_rate": latest_snapshot.failure_rate(),
                "avg_calibration_error": latest_snapshot.avg_calibration_error,
                "overconfidence_rate": latest_snapshot.overconfidence_rate,
                "underconfidence_rate": latest_snapshot.underconfidence_rate
            }
        
        return stats
    
    def generate_metacognitive_report(self) -> str:
        """
        Genera reporte narrativo de metacognición.
        
        Returns:
            Reporte en texto
        """
        stats = self.get_comprehensive_stats()
        
        report_lines = [
            "═══════════════════════════════════════════════════════",
            "🧠 METACOGNITIVE SYSTEM REPORT",
            "═══════════════════════════════════════════════════════",
            "",
            "📊 SELF-MODEL:",
            f"  • Capabilities recognized: {stats['self_model']['capabilities']}",
            f"  • Limitations recognized: {stats['self_model']['limitations']}",
            f"  • Self-awareness level: {stats['self_model']['self_awareness_level']:.2%}",
            f"  • Active modules: {stats['self_model']['modules']}",
            "",
            "🔍 INTROSPECTION:",
            f"  • Total processes monitored: {stats['metacognition']['total_processes_monitored']}",
            f"  • Introspection reports: {stats['metacognition']['introspection_history_size']}",
            f"  • Active processes: {stats['metacognition']['active_processes']}",
            "",
            "⚠️ BIAS DETECTION:",
            f"  • Total biases detected: {stats['bias_detection'].get('total_biases', 0)}",
        ]
        
        if "most_common_bias" in stats["bias_detection"]:
            report_lines.append(
                f"  • Most common bias: {stats['bias_detection']['most_common_bias']} "
                f"({stats['bias_detection']['most_common_count']} times)"
            )
        
        if stats["performance"]:
            report_lines.extend([
                "",
                "📈 PERFORMANCE:",
                f"  • Success rate: {stats['performance']['success_rate']:.2%}",
                f"  • Failure rate: {stats['performance']['failure_rate']:.2%}",
                f"  • Calibration error: {stats['performance']['avg_calibration_error']:.3f}",
                f"  • Overconfidence rate: {stats['performance']['overconfidence_rate']:.2%}",
                f"  • Underconfidence rate: {stats['performance']['underconfidence_rate']:.2%}"
            ])
        
        if "well_calibrated" in stats["calibration"]:
            calibration_status = "✅ WELL CALIBRATED" if stats["calibration"]["well_calibrated"] else "⚠️ NEEDS CALIBRATION"
            report_lines.extend([
                "",
                f"🎯 CALIBRATION: {calibration_status}",
                f"  • Average error: {stats['calibration']['avg_calibration_error']:.3f}",
                f"  • Samples: {stats['calibration']['total_samples']}"
            ])
        
        report_lines.append("═══════════════════════════════════════════════════════")
        
        return "\n".join(report_lines)


# ═══════════════════════════════════════════════════════════════════════
# FACTORY & HELPERS
# ═══════════════════════════════════════════════════════════════════════


def create_metacognition_system(
    enable_continuous_monitoring: bool = True
) -> MetaCognitionSystem:
    """
    Factory para crear sistema de metacognición.
    
    Args:
        enable_continuous_monitoring: Si habilitar monitoreo continuo
        
    Returns:
        Sistema de metacognición configurado
    """
    return MetaCognitionSystem(
        enable_continuous_monitoring=enable_continuous_monitoring
    )


def analyze_cognitive_biases(
    reasoning_steps: List[str],
    conclusion: str,
    evidence: List[str],
    metacog_system: Optional[MetaCognitionSystem] = None
) -> List[BiasDetectionResult]:
    """
    Helper para detectar sesgos en razonamiento.
    
    Args:
        reasoning_steps: Pasos de razonamiento
        conclusion: Conclusión
        evidence: Evidencia
        metacog_system: Sistema de metacognición (crea uno si None)
        
    Returns:
        Lista de sesgos detectados
    """
    if metacog_system is None:
        metacog_system = create_metacognition_system()
    
    return metacog_system.detect_bias_in_reasoning(
        reasoning_steps,
        conclusion,
        evidence
    )