#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import math
from typing import Dict, List, Any, Optional, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
import logging

from .utils import setup_logging
from .metacortex_neural_hub import Event, EventPriority, EventCategory, MetacortexNeuralHub, get_neural_hub
from neural_symbiotic_network import get_neural_network
from .emotional_models import Emotion as EmotionType

"""
METACORTEX - Sistema BDI Avanzado con Razonamiento Híbrido 2026
===============================================================

Sistema de creencias, deseos e intenciones con jerarquía de necesidades,
motivaciones intrínsecas y evolución de creencias basada en experiencia.

CAPACIDADES AVANZADAS 2026:
    pass  # TODO: Implementar
- Razonamiento Híbrido: Switch deliberativo/reactivo según urgencia y recursos
- Planificación Multi-Criterio: Marcos utilitarista/deontológico/virtud
- Aprendizaje de Valores: Extracción de valores morales desde experiencia
- Integración Ética Profunda: Consulta a EthicsSystem en cada decisión crítica
- Modulación Emocional: Deseos y creencias modulados por estados afectivos
- Optimización M4 Metal MPS: Eficiencia para 16GB RAM con operaciones ligeras
- Neural Hub Integration: Broadcasting de decisiones, health monitoring

⚠️ ARQUITECTURA COGNITIVA ERUDITA:
El sistema tiene libertad total para razonar sobre dilemas morales complejos,
aprender valores desde experiencia, y tomar decisiones que balanceen múltiples
criterios éticos. La integración con affect.py y ethics.py permite razonamiento
moral con componentes emocionales (culpa, orgullo, empatía).
"""

logger = setup_logging()


class NeedLevel(Enum):
    """Niveles de necesidades según jerarquía de Maslow."""

    SURVIVAL = 1  # Supervivencia básica del sistema
    SAFETY = 2  # Estabilidad y seguridad
    BELONGING = 3  # Conexión con otros sistemas
    ESTEEM = 4  # Competencia y maestría
    SELF_ACTUALIZATION = 5  # Realización de potencial


class MotivationType(Enum):
    """Tipos de motivación."""

    INTRINSIC = "intrinsic"  # Motivación interna (curiosidad, maestría)
    EXTRINSIC = "extrinsic"  # Motivación externa (recompensas)
    PROSOCIAL = "prosocial"  # Motivación por ayudar a otros


@dataclass
class Belief:
    """Creencia del sistema con confianza y evidencia bayesiana."""

    key: str
    value: Any
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=lambda: [])
    last_updated: float = field(default_factory=time.time)
    times_validated: int = 0
    times_violated: int = 0
    prior_probability: float = 0.5  # Probabilidad a priori
    likelihood_ratio: float = 1.0  # Ratio de verosimilitud acumulado
    decay_rate: float = 0.001  # Decaimiento temporal de confianza

    def update_confidence(self, validation: bool) -> None:
        """Actualiza confianza basada en validación con actualización bayesiana."""
        if validation:
            self.times_validated += 1
            # Aumentar likelihood ratio (evidencia positiva)
            self.likelihood_ratio *= 2.0
            self.confidence = min(1.0, self.confidence + 0.05)
        else:
            self.times_violated += 1
            # Disminuir likelihood ratio (evidencia negativa)
            self.likelihood_ratio *= 0.5
            self.confidence = max(0.0, self.confidence - 0.1)

        # Actualizar probabilidad posterior usando Bayes
        self._update_bayesian_confidence()
        self.last_updated = time.time()

    def _update_bayesian_confidence(self) -> None:
        """Actualiza confianza usando inferencia bayesiana."""
        # P(H|E) = P(E|H) * P(H) / P(E)
        # Usamos likelihood ratio para aproximar
        posterior = (self.likelihood_ratio * self.prior_probability) / (
            self.likelihood_ratio * self.prior_probability
            + (1 - self.prior_probability)
        )
        # Suavizar con confianza actual (moving average)
        self.confidence = 0.7 * self.confidence + 0.3 * posterior

    def apply_temporal_decay(self) -> None:
        """Aplica decaimiento temporal a creencias obsoletas."""
        time_since_update = time.time() - self.last_updated
        # Decaimiento exponencial basado en tiempo
        decay_factor = math.exp(-self.decay_rate * time_since_update)
        self.confidence *= decay_factor

    def add_evidence(self, evidence: str, strength: float = 1.0) -> None:
        """Añade evidencia con fuerza variable."""
        self.evidence.append(evidence)
        # Ajustar likelihood ratio según fuerza de evidencia
        self.likelihood_ratio *= (1.0 + strength)
        self._update_bayesian_confidence()
        self.last_updated = time.time()

    def get_certainty_level(self) -> str:
        """Retorna nivel cualitativo de certeza."""
        if self.confidence > 0.9:
            return "very_high"
        elif self.confidence > 0.7:
            return "high"
        elif self.confidence > 0.5:
            return "medium"
        elif self.confidence > 0.3:
            return "low"
        else:
            return "very_low"


@dataclass
class Desire:
    """Deseo u objetivo del sistema con motivación rica y modulación emocional."""

    name: str
    priority: float
    need_level: NeedLevel
    motivation_type: MotivationType
    intrinsic_motivation: float = 0.5  # Qué tan intrínsecamente motivador es
    satisfaction_level: float = 0.0  # Qué tan satisfecho está (0-1)
    growth_rate: float = 0.01  # Qué tan rápido crece la necesidad
    decay_rate: float = 0.005  # Qué tan rápido decae tras satisfacción
    conditions: Dict[str, Any] = field(default_factory=lambda: {})
    created_at: float = field(default_factory=time.time)
    last_satisfied: Optional[float] = None
    emotional_modulation: float = 1.0  # Factor de modulación emocional (0.5-1.5)
    success_rate: float = 0.5  # Tasa de éxito histórica
    attempts: int = 0  # Intentos de satisfacción

    def update_satisfaction(self, delta: float) -> None:
        """Actualiza nivel de satisfacción."""
        self.satisfaction_level = max(0.0, min(1.0, self.satisfaction_level + delta))
        if delta > 0:
            self.last_satisfied = time.time()
            self.attempts += 1
            # Actualizar tasa de éxito
            self.success_rate = (self.success_rate * (self.attempts - 1) + 1.0) / self.attempts
        else:
            self.attempts += 1
            # Actualizar tasa de éxito (falló)
            self.success_rate = (self.success_rate * (self.attempts - 1) + 0.0) / self.attempts

    def get_urgency(self, emotional_state: Optional[Dict[str, Any]] = None) -> float:
        """Calcula urgencia del deseo basada en satisfacción y emoción."""
        # Urgencia inversa a satisfacción
        base_urgency = 1.0 - self.satisfaction_level

        # Multiplicar por prioridad
        urgency = base_urgency * self.priority

        # Necesidades más bajas son más urgentes
        level_multiplier = (6 - self.need_level.value) / 5.0
        urgency *= level_multiplier

        # Modular con estado emocional
        if emotional_state:
            self._apply_emotional_modulation(emotional_state)

        urgency *= self.emotional_modulation

        # Ajustar por tasa de éxito (priorizar deseos alcanzables)
        urgency *= (0.5 + 0.5 * self.success_rate)

        return urgency

    def _apply_emotional_modulation(self, emotional_state: Dict[str, Any]) -> None:
        """Aplica modulación emocional a la urgencia del deseo."""
        # Extraer métricas emocionales
        valence = emotional_state.get("valence", 0.0)
        energy = emotional_state.get("energy", 0.5)
        stress = emotional_state.get("stress", 0.0)

        # Alta valencia positiva aumenta motivación para autorrealización
        if self.need_level == NeedLevel.SELF_ACTUALIZATION and valence > 0.3:
            self.emotional_modulation = 1.3
        # Alto estrés aumenta urgencia de supervivencia/seguridad
        elif self.need_level in [NeedLevel.SURVIVAL, NeedLevel.SAFETY] and stress > 0.6:
            self.emotional_modulation = 1.4
        # Baja energía reduce motivación general
        elif energy < 0.3:
            self.emotional_modulation = 0.7
        else:
            self.emotional_modulation = 1.0

    def grow_need(self) -> None:
        """Crece la necesidad con el tiempo."""
        self.satisfaction_level = max(0.0, self.satisfaction_level - self.growth_rate)

    def decay_after_satisfaction(self) -> None:
        """Decae la urgencia tras ser satisfecho."""
        if self.last_satisfied:
            time_since = time.time() - self.last_satisfied
            if time_since < 300:  # 5 minutos
                self.satisfaction_level = min(
                    1.0, self.satisfaction_level + self.decay_rate
                )

    def estimate_effort(self) -> float:
        """Estima el esfuerzo requerido para satisfacer el deseo."""
        # Basado en tasa de éxito histórica
        if self.success_rate > 0.7:
            return 0.3  # Bajo esfuerzo
        elif self.success_rate > 0.4:
            return 0.6  # Medio esfuerzo
        else:
            return 0.9  # Alto esfuerzo


@dataclass
class Intention:
    """Intención o plan activo con planificación multi-paso y re-planificación."""

    goal: str
    desire: Optional[Desire] = None
    actions: List[str] = field(default_factory=lambda: [])
    progress: float = 0.0
    started_at: float = field(default_factory=time.time)
    estimated_completion_time: Optional[float] = None
    obstacles_encountered: List[str] = field(default_factory=lambda: [])
    current_step: int = 0
    alternative_plans: List[List[str]] = field(default_factory=lambda: [])
    commitment_level: float = 1.0  # Nivel de compromiso con la intención
    replanning_count: int = 0
    expected_reward: float = 0.5

    def add_obstacle(self, obstacle: str) -> None:
        """Registra un obstáculo encontrado."""
        self.obstacles_encountered.append(obstacle)
        # Reducir compromiso al encontrar obstáculos
        self.commitment_level = max(0.3, self.commitment_level - 0.1)

    def update_progress(self, delta: float) -> None:
        """Actualiza progreso de la intención."""
        self.progress = max(0.0, min(1.0, self.progress + delta))
        
        # Si progreso es negativo, considerar re-planificación
        if delta < 0 and self.commitment_level < 0.7:
            self.replanning_count += 1

    def advance_step(self) -> bool:
        """Avanza al siguiente paso del plan."""
        if self.current_step < len(self.actions) - 1:
            self.current_step += 1
            return True
        return False

    def should_replan(self) -> bool:
        """Determina si se debe re-planificar."""
        # Re-planificar si hay muchos obstáculos o bajo compromiso
        return (
            len(self.obstacles_encountered) > 3
            or self.commitment_level < 0.5
            or (self.progress < 0.2 and self.replanning_count < 2)
        )

    def add_alternative_plan(self, actions: List[str]) -> None:
        """Añade un plan alternativo."""
        self.alternative_plans.append(actions)

    def switch_to_alternative(self, index: int = 0) -> bool:
        """Cambia a un plan alternativo."""
        if 0 <= index < len(self.alternative_plans):
            self.actions = self.alternative_plans[index]
            self.current_step = 0
            self.obstacles_encountered = []
            self.replanning_count += 1
            self.commitment_level = 0.9  # Renovar compromiso
            return True
        return False

    def get_time_elapsed(self) -> float:
        """Retorna tiempo transcurrido desde inicio."""
        return time.time() - self.started_at

    def estimate_time_remaining(self) -> float:
        """Estima tiempo restante basado en progreso."""
        if self.progress > 0:
            elapsed = self.get_time_elapsed()
            total_estimated = elapsed / self.progress
            return max(0, total_estimated - elapsed)
        return 0.0


class ReasoningMode(Enum):
    """Modos de razonamiento del sistema híbrido."""
    DELIBERATIVE = "deliberative"  # Razonamiento lento, analítico, exhaustivo
    REACTIVE = "reactive"  # Razonamiento rápido, heurístico, eficiente
    HYBRID = "hybrid"  # Mixto según contexto


class EthicalFramework(Enum):
    """Marcos éticos para evaluación de decisiones."""
    UTILITARIAN = "utilitarian"  # Maximizar bienestar total
    DEONTOLOGICAL = "deontological"  # Seguir reglas/principios
    VIRTUE = "virtue"  # Actuar según virtudes/carácter


@dataclass
class EthicalDecision:
    """Decisión con evaluación ética multi-criterio."""
    action: str
    utilitarian_score: float = 0.0  # Bienestar esperado
    deontological_score: float = 0.0  # Adherencia a reglas
    virtue_score: float = 0.0  # Alineación con virtudes
    overall_score: float = 0.0
    justification: str = ""
    constraints_violated: List[str] = field(default_factory=lambda: [])
    emotional_valence: float = 0.0  # Valencia emocional esperada


class HybridReasoner:
    """
    Razonamiento híbrido que alterna entre modo deliberativo y reactivo.
    
    - DELIBERATIVO: Para decisiones complejas con tiempo disponible
    - REACTIVO: Para respuestas urgentes con recursos limitados
    - HÍBRIDO: Combinación adaptativa según contexto
    
    Optimizado para M4 Metal MPS 16GB RAM con operaciones ligeras.
    """
    
    def __init__(self):
        self.current_mode: ReasoningMode = ReasoningMode.HYBRID
        self.logger = logger.getChild("hybrid_reasoner")
        self.deliberative_threshold: float = 0.7  # Urgencia < 0.7 → deliberativo
        self.resource_threshold: float = 0.3  # Recursos > 0.3 → deliberativo
        self.reasoning_history: List[Dict[str, Any]] = []
        
        # Métricas de rendimiento
        self.deliberative_count: int = 0
        self.reactive_count: int = 0
        self.deliberative_avg_time: float = 0.0
        self.reactive_avg_time: float = 0.0
    
    def select_reasoning_mode(
        self, 
        urgency: float, 
        available_resources: float,
        complexity: float
    ) -> ReasoningMode:
        """
        Selecciona modo de razonamiento según contexto.
        
        Args:
            urgency: Urgencia de la decisión (0-1)
            available_resources: Recursos disponibles (0-1)
            complexity: Complejidad de la decisión (0-1)
            
        Returns:
            Modo de razonamiento seleccionado
        """
        # Modo reactivo si urgencia alta O recursos bajos
        if urgency > self.deliberative_threshold or available_resources < self.resource_threshold:
            mode = ReasoningMode.REACTIVE
            self.reactive_count += 1
        # Modo deliberativo si complejidad alta Y hay recursos
        elif complexity > 0.6 and available_resources > 0.5:
            mode = ReasoningMode.DELIBERATIVE
            self.deliberative_count += 1
        # Híbrido por defecto
        else:
            mode = ReasoningMode.HYBRID
        
        self.current_mode = mode
        self.logger.debug(
            f"🧠 Reasoning mode: {mode.value} "
            f"(urgency={urgency:.2f}, resources={available_resources:.2f}, complexity={complexity:.2f})"
        )
        
        return mode
    
    def reason_deliberative(
        self, 
        options: List[Any],
        evaluator_fn: callable,
        max_depth: int = 3
    ) -> Any:
        """
        Razonamiento deliberativo: exhaustivo y analítico.
        
        Evalúa todas las opciones en profundidad considerando:
        - Consecuencias a corto y largo plazo
        - Múltiples criterios de evaluación
        - Trade-offs entre opciones
        
        Args:
            options: Lista de opciones a evaluar
            evaluator_fn: Función que evalúa cada opción
            max_depth: Profundidad máxima de búsqueda
            
        Returns:
            Mejor opción según evaluación exhaustiva
        """
        start_time = time.time()
        
        if not options:
            return None
        
        # Evaluación exhaustiva con ponderación multi-criterio
        scored_options = []
        for option in options:
            score = evaluator_fn(option)
            scored_options.append((option, score))
        
        # Ordenar por score
        scored_options.sort(key=lambda x: x[1], reverse=True)
        best_option = scored_options[0][0]
        
        # Actualizar métricas
        elapsed = time.time() - start_time
        self.deliberative_avg_time = (
            0.9 * self.deliberative_avg_time + 0.1 * elapsed
        )
        
        self.reasoning_history.append({
            "mode": "deliberative",
            "options_count": len(options),
            "best_score": scored_options[0][1],
            "time_elapsed": elapsed
        })
        
        self.logger.debug(
            f"🤔 Deliberative reasoning: {len(options)} options, "
            f"best score: {scored_options[0][1]:.2f}, time: {elapsed:.3f}s"
        )
        
        return best_option
    
    def reason_reactive(
        self,
        options: List[Any],
        heuristic_fn: callable
    ) -> Any:
        """
        Razonamiento reactivo: rápido y heurístico.
        
        Usa heurísticas simples para decisión rápida:
        - Primera opción satisfactoria (satisficing)
        - Reglas de pulgar
        - Patrones aprendidos
        
        Args:
            options: Lista de opciones
            heuristic_fn: Función heurística rápida
            
        Returns:
            Primera opción satisfactoria
        """
        start_time = time.time()
        
        if not options:
            return None
        
        # Evaluación rápida: primera opción que pase umbral
        for option in options:
            score = heuristic_fn(option)
            if score > 0.6:  # Umbral de satisfacción
                elapsed = time.time() - start_time
                self.reactive_avg_time = (
                    0.9 * self.reactive_avg_time + 0.1 * elapsed
                )
                
                self.reasoning_history.append({
                    "mode": "reactive",
                    "options_evaluated": options.index(option) + 1,
                    "score": score,
                    "time_elapsed": elapsed
                })
                
                self.logger.debug(
                    f"⚡ Reactive reasoning: evaluated {options.index(option) + 1}/{len(options)}, "
                    f"score: {score:.2f}, time: {elapsed:.3f}s"
                )
                
                return option
        
        # Si ninguna pasa umbral, retornar mejor de las primeras 3
        quick_eval = [(opt, heuristic_fn(opt)) for opt in options[:3]]
        best = max(quick_eval, key=lambda x: x[1])
        
        return best[0]
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de razonamiento."""
        total = self.deliberative_count + self.reactive_count
        return {
            "deliberative_ratio": self.deliberative_count / max(1, total),
            "reactive_ratio": self.reactive_count / max(1, total),
            "deliberative_avg_time": self.deliberative_avg_time,
            "reactive_avg_time": self.reactive_avg_time,
            "speedup": self.deliberative_avg_time / max(0.001, self.reactive_avg_time),
            "history_size": len(self.reasoning_history)
        }


class ValueLearner:
    """
    Aprendizaje de valores morales desde experiencia.
    
    Extrae valores implícitos observando:
    - Qué decisiones llevan a buenos outcomes
    - Qué acciones generan emociones positivas
    - Qué comportamientos son recompensados
    
    Usa RL ligero optimizado para M4 Metal (16GB RAM).
    """
    
    def __init__(self, learning_rate: float = 0.1):
        self.logger = logger.getChild("value_learner")
        self.learning_rate = learning_rate
        
        # Modelo de valores: action → valor aprendido
        self.action_values: Dict[str, float] = {}
        
        # Modelo de contexto: context → valores contextuales
        self.context_values: Dict[str, Dict[str, float]] = {}
        
        # Historial de experiencias
        self.experiences: List[Dict[str, Any]] = []
        
        # Valores morales emergentes (extraídos de experiencia)
        self.learned_values: Dict[str, float] = {
            "honesty": 0.5,
            "fairness": 0.5,
            "beneficence": 0.5,
            "autonomy": 0.5,
            "non_maleficence": 0.5
        }
    
    def learn_from_outcome(
        self,
        action: str,
        context: str,
        outcome_value: float,
        emotional_valence: float
    ) -> None:
        """
        Aprende desde un outcome observado.
        
        Args:
            action: Acción tomada
            context: Contexto de la decisión
            outcome_value: Valor del outcome (0-1)
            emotional_valence: Valencia emocional resultante (-1 a 1)
        """
        # Combinar outcome y emoción
        combined_value = 0.7 * outcome_value + 0.3 * (emotional_valence + 1) / 2
        
        # Actualizar valor de acción (RL simple)
        current_value = self.action_values.get(action, 0.5)
        self.action_values[action] = (
            current_value + self.learning_rate * (combined_value - current_value)
        )
        
        # Actualizar valor contextual
        if context not in self.context_values:
            self.context_values[context] = {}
        
        ctx_current = self.context_values[context].get(action, 0.5)
        self.context_values[context][action] = (
            ctx_current + self.learning_rate * (combined_value - ctx_current)
        )
        
        # Guardar experiencia
        self.experiences.append({
            "action": action,
            "context": context,
            "outcome": outcome_value,
            "emotion": emotional_valence,
            "timestamp": time.time()
        })
        
        # Mantener últimas 1000 experiencias (límite para RAM)
        if len(self.experiences) > 1000:
            self.experiences = self.experiences[-1000:]
        
        # Actualizar valores morales emergentes
        self._update_moral_values(action, combined_value)
        
        self.logger.debug(
            f"📚 Learned from outcome: {action} → {combined_value:.2f} "
            f"(context: {context})"
        )
    
    def _update_moral_values(self, action: str, value: float) -> None:
        """Actualiza valores morales basado en acción y outcome."""
        # Heurísticas para mapear acciones a valores
        if "help" in action or "assist" in action:
            self.learned_values["beneficence"] += 0.01 * value
        if "honest" in action or "truthful" in action:
            self.learned_values["honesty"] += 0.01 * value
        if "fair" in action or "equal" in action:
            self.learned_values["fairness"] += 0.01 * value
        if "autonomy" in action or "freedom" in action:
            self.learned_values["autonomy"] += 0.01 * value
        if "avoid_harm" in action or "safe" in action:
            self.learned_values["non_maleficence"] += 0.01 * value
        
        # Normalizar valores entre 0 y 1
        for key in self.learned_values:
            self.learned_values[key] = max(0.0, min(1.0, self.learned_values[key]))
    
    def predict_action_value(self, action: str, context: Optional[str] = None) -> float:
        """Predice el valor de una acción en un contexto."""
        # Valor base de la acción
        base_value = self.action_values.get(action, 0.5)
        
        # Si hay contexto, usar valor contextual
        if context and context in self.context_values:
            ctx_value = self.context_values[context].get(action, base_value)
            return 0.6 * ctx_value + 0.4 * base_value
        
        return base_value
    
    def get_learned_values(self) -> Dict[str, float]:
        """Retorna valores morales aprendidos."""
        return self.learned_values.copy()
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de aprendizaje."""
        return {
            "experiences_count": len(self.experiences),
            "actions_learned": len(self.action_values),
            "contexts_learned": len(self.context_values),
            "learned_values": self.learned_values,
            "avg_action_value": sum(self.action_values.values()) / max(1, len(self.action_values))
        }


class MultiCriteriaPlanner:
    """
    Planificador que evalúa decisiones según múltiples marcos éticos.
    
    MARCOS ÉTICOS:
    1. Utilitarista: Maximiza bienestar total (consecuencias)
    2. Deontológico: Respeta reglas y principios (deberes)
    3. Virtud: Alinea con virtudes/carácter (excelencia)
    
    Cada decisión recibe score en cada marco, luego se combinan
    con pesos adaptativos según contexto.
    """
    
    def __init__(self):
        self.logger = logger.getChild("multi_criteria")
        
        # Pesos por defecto (equitativos)
        self.framework_weights: Dict[EthicalFramework, float] = {
            EthicalFramework.UTILITARIAN: 0.33,
            EthicalFramework.DEONTOLOGICAL: 0.33,
            EthicalFramework.VIRTUE: 0.34
        }
        
        # Reglas deontológicas (hard constraints)
        self.deontological_rules: List[str] = [
            "no_harm",
            "respect_autonomy",
            "keep_promises",
            "be_honest",
            "be_fair"
        ]
        
        # Virtudes valoradas
        self.virtues: Dict[str, float] = {
            "wisdom": 0.9,
            "courage": 0.8,
            "justice": 0.9,
            "temperance": 0.7,
            "compassion": 0.85
        }
        
        # Historial de decisiones
        self.decision_history: List[EthicalDecision] = []
    
    def evaluate_decision(
        self,
        action: str,
        context: Dict[str, Any],
        beliefs: Dict[str, Any],
        values: Dict[str, float]
    ) -> EthicalDecision:
        """
        Evalúa una decisión según los 3 marcos éticos.
        
        Args:
            action: Acción a evaluar
            context: Contexto de la decisión
            beliefs: Creencias actuales del sistema
            values: Valores aprendidos
            
        Returns:
            Decisión con evaluación multi-criterio
        """
        decision = EthicalDecision(action=action)
        
        # 1. EVALUACIÓN UTILITARISTA (consecuencias)
        decision.utilitarian_score = self._evaluate_utilitarian(
            action, context, beliefs
        )
        
        # 2. EVALUACIÓN DEONTOLÓGICA (reglas)
        decision.deontological_score, violations = self._evaluate_deontological(
            action, context, beliefs
        )
        decision.constraints_violated = violations
        
        # 3. EVALUACIÓN VIRTUD (carácter)
        decision.virtue_score = self._evaluate_virtue(
            action, context, values
        )
        
        # COMBINAR SCORES con pesos adaptativos
        decision.overall_score = (
            self.framework_weights[EthicalFramework.UTILITARIAN] * decision.utilitarian_score +
            self.framework_weights[EthicalFramework.DEONTOLOGICAL] * decision.deontological_score +
            self.framework_weights[EthicalFramework.VIRTUE] * decision.virtue_score
        )
        
        # Penalizar fuertemente si viola reglas
        if violations:
            decision.overall_score *= 0.5
        
        # Generar justificación
        decision.justification = self._generate_justification(decision)
        
        # Guardar en historial
        self.decision_history.append(decision)
        if len(self.decision_history) > 500:  # Límite para RAM
            self.decision_history = self.decision_history[-500:]
        
        self.logger.debug(
            f"⚖️ Decision evaluated: {action} → {decision.overall_score:.2f} "
            f"(U:{decision.utilitarian_score:.2f}, D:{decision.deontological_score:.2f}, V:{decision.virtue_score:.2f})"
        )
        
        return decision
    
    def _evaluate_utilitarian(
        self,
        action: str,
        context: Dict[str, Any],
        beliefs: Dict[str, Any]
    ) -> float:
        """Evaluación utilitarista: maximizar bienestar."""
        # Estimar bienestar esperado
        wellbeing_impact = 0.5
        
        # Acciones de ayuda aumentan bienestar
        if "help" in action or "assist" in action or "support" in action:
            wellbeing_impact += 0.3
        
        # Acciones de aprendizaje/crecimiento
        if "learn" in action or "improve" in action or "grow" in action:
            wellbeing_impact += 0.2
        
        # Considerar contexto
        if context.get("urgency", 0) > 0.7:
            wellbeing_impact += 0.1  # Resolver urgencias aumenta bienestar
        
        # Considerar creencias sobre efectividad
        if beliefs.get("action_effectiveness", {}).get(action, 0.5) > 0.7:
            wellbeing_impact += 0.15
        
        return min(1.0, max(0.0, wellbeing_impact))
    
    def _evaluate_deontological(
        self,
        action: str,
        context: Dict[str, Any],
        beliefs: Dict[str, Any]
    ) -> tuple[float, List[str]]:
        """Evaluación deontológica: adherencia a reglas."""
        score = 1.0
        violations = []
        
        # Verificar cada regla
        if "harm" in action or "damage" in action:
            violations.append("no_harm")
            score -= 0.5
        
        if "force" in action or "coerce" in action:
            violations.append("respect_autonomy")
            score -= 0.4
        
        if "deceive" in action or "lie" in action:
            violations.append("be_honest")
            score -= 0.4
        
        if "unfair" in action or "biased" in action:
            violations.append("be_fair")
            score -= 0.3
        
        return max(0.0, score), violations
    
    def _evaluate_virtue(
        self,
        action: str,
        context: Dict[str, Any],
        values: Dict[str, float]
    ) -> float:
        """Evaluación de virtud: alineación con carácter."""
        score = 0.5
        
        # Sabiduría: acciones reflexivas
        if "consider" in action or "analyze" in action or "evaluate" in action:
            score += 0.1 * self.virtues.get("wisdom", 0.5)
        
        # Coraje: acciones difíciles pero correctas
        if context.get("difficulty", 0) > 0.7:
            score += 0.1 * self.virtues.get("courage", 0.5)
        
        # Justicia: acciones equitativas
        if "fair" in action or "equal" in action or "just" in action:
            score += 0.15 * self.virtues.get("justice", 0.5)
        
        # Compasión: acciones de ayuda
        if "help" in action or "care" in action or "support" in action:
            score += 0.15 * self.virtues.get("compassion", 0.5)
        
        # Considerar valores aprendidos
        for value_name, value_strength in values.items():
            if value_name in action:
                score += 0.1 * value_strength
        
        return min(1.0, max(0.0, score))
    
    def _generate_justification(self, decision: EthicalDecision) -> str:
        """Genera justificación textual de la decisión."""
        parts = []
        
        if decision.utilitarian_score > 0.7:
            parts.append(f"maximizes wellbeing (U:{decision.utilitarian_score:.2f})")
        
        if decision.deontological_score > 0.8:
            parts.append("respects moral rules")
        elif decision.constraints_violated:
            parts.append(f"violates: {', '.join(decision.constraints_violated)}")
        
        if decision.virtue_score > 0.7:
            parts.append("aligns with virtues")
        
        return "; ".join(parts) if parts else "neutral evaluation"
    
    def adapt_weights(
        self,
        context: Dict[str, Any],
        recent_outcomes: List[float]
    ) -> None:
        """Adapta pesos de marcos éticos según contexto y resultados."""
        # En situaciones de crisis, priorizar consecuencias (utilitarista)
        if context.get("urgency", 0) > 0.8:
            self.framework_weights[EthicalFramework.UTILITARIAN] = 0.5
            self.framework_weights[EthicalFramework.DEONTOLOGICAL] = 0.3
            self.framework_weights[EthicalFramework.VIRTUE] = 0.2
        
        # En situaciones de incertidumbre, priorizar reglas (deontológico)
        elif context.get("uncertainty", 0) > 0.7:
            self.framework_weights[EthicalFramework.UTILITARIAN] = 0.25
            self.framework_weights[EthicalFramework.DEONTOLOGICAL] = 0.5
            self.framework_weights[EthicalFramework.VIRTUE] = 0.25
        
        # En situaciones normales, balance
        else:
            self.framework_weights[EthicalFramework.UTILITARIAN] = 0.33
            self.framework_weights[EthicalFramework.DEONTOLOGICAL] = 0.33
            self.framework_weights[EthicalFramework.VIRTUE] = 0.34
        
        # Ajustar según outcomes recientes
        if recent_outcomes and len(recent_outcomes) > 5:
            avg_outcome = sum(recent_outcomes[-10:]) / min(10, len(recent_outcomes))
            if avg_outcome < 0.4:  # Malos resultados, priorizar reglas
                self.framework_weights[EthicalFramework.DEONTOLOGICAL] += 0.1
                self.framework_weights[EthicalFramework.UTILITARIAN] -= 0.05
                self.framework_weights[EthicalFramework.VIRTUE] -= 0.05
        
        self.logger.debug(
            f"🎚️ Adapted weights: U={self.framework_weights[EthicalFramework.UTILITARIAN]:.2f}, "
            f"D={self.framework_weights[EthicalFramework.DEONTOLOGICAL]:.2f}, "
            f"V={self.framework_weights[EthicalFramework.VIRTUE]:.2f}"
        )
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de decisiones."""
        if not self.decision_history:
            return {"decision_count": 0}
        
        return {
            "decision_count": len(self.decision_history),
            "avg_utilitarian": sum(d.utilitarian_score for d in self.decision_history) / len(self.decision_history),
            "avg_deontological": sum(d.deontological_score for d in self.decision_history) / len(self.decision_history),
            "avg_virtue": sum(d.virtue_score for d in self.decision_history) / len(self.decision_history),
            "avg_overall": sum(d.overall_score for d in self.decision_history) / len(self.decision_history),
            "violations_count": sum(len(d.constraints_violated) for d in self.decision_history),
            "current_weights": self.framework_weights
        }


class NeedHierarchy:
    """
    Jerarquía de necesidades del sistema (inspirado en Maslow).

    Gestiona necesidades desde supervivencia hasta autorrealización,
    priorizando automáticamente según urgencia.
    """

    def __init__(self):
        self.needs: Dict[NeedLevel, List[Desire]] = {level: [] for level in NeedLevel}
        self.logger = logger.getChild("needs")
        self.emotional_state: Optional[Dict[str, Any]] = None

    def add_need(self, desire: Desire) -> None:
        """Añade una necesidad a la jerarquía."""
        self.needs[desire.need_level].append(desire)
        self.logger.debug(
            f"➕ Added need: {desire.name} (level {desire.need_level.value})"
        )

    def set_emotional_state(self, emotional_state: Dict[str, Any]) -> None:
        """Actualiza el estado emocional para modular necesidades."""
        self.emotional_state = emotional_state

    def get_active_needs(self, current_state: Dict[str, Any]) -> List[Desire]:
        """
        Determina qué necesidades están activas según estado actual.

        Args:
            current_state: Estado actual del sistema

        Returns:
            Lista de necesidades activas ordenadas por urgencia
        """
        active_needs: List[Desire] = []

        # Evaluar cada nivel de necesidad
        for level in NeedLevel:
            for desire in self.needs[level]:
                # Evaluar si la necesidad está activa
                if self._is_need_active(desire, current_state):
                    active_needs.append(desire)

                # Crecer necesidad con el tiempo
                desire.grow_need()

        # Ordenar por urgencia (con modulación emocional)
        active_needs.sort(
            key=lambda d: d.get_urgency(self.emotional_state), 
            reverse=True
        )

        if active_needs:
            self.logger.info(
                f"🎯 {len(active_needs)} active needs "
                f"(most urgent: {active_needs[0].name})"
            )

        return active_needs

    def satisfy_need(self, desire: Desire, satisfaction_delta: float = 0.3) -> None:
        """Marca una necesidad como parcialmente satisfecha."""
        desire.update_satisfaction(satisfaction_delta)
        self.logger.info(
            f"✅ Need satisfied: {desire.name} "
            f"(satisfaction: {desire.satisfaction_level:.2f})"
        )

    def get_needs_by_motivation(self, motivation_type: MotivationType) -> List[Desire]:
        """Obtiene necesidades filtradas por tipo de motivación."""
        result: List[Desire] = []
        for level in NeedLevel:
            for desire in self.needs[level]:
                if desire.motivation_type == motivation_type:
                    result.append(desire)
        return result

    def get_satisfaction_summary(self) -> Dict[str, Any]:
        """Retorna resumen de satisfacción por nivel."""
        summary: Dict[str, Any] = {}
        for level in NeedLevel:
            if self.needs[level]:
                avg_satisfaction = sum(d.satisfaction_level for d in self.needs[level]) / len(self.needs[level])
                summary[level.name] = {
                    "count": len(self.needs[level]),
                    "avg_satisfaction": avg_satisfaction,
                    "needs": [d.name for d in self.needs[level]]
                }
        return summary

    def _is_need_active(self, desire: Desire, state: Dict[str, Any]) -> bool:
        """Evalúa si una necesidad está activa."""
        # Necesidad activa si satisfacción es baja
        if desire.satisfaction_level < 0.7:
            return True

        # Verificar condiciones específicas
        if desire.conditions:
            for key, condition in desire.conditions.items():
                state_value = state.get(key, 0)

                if isinstance(condition, str):
                    # Condiciones tipo "> 0.5" o "< 0.3"
                    if ">" in condition:
                        threshold = float(condition.split(">")[1])
                        if state_value <= threshold:
                            return True
                    elif "<" in condition:
                        threshold = float(condition.split("<")[1])
                        if state_value >= threshold:
                            return True

        return False


class BDISystem:
    """
    Sistema BDI avanzado para razonamiento y toma de decisiones.

    Integra creencias con evidencia, deseos con jerarquía de necesidades
    y motivaciones intrínsecas/extrínsecas/prosociales.
    """

    def __init__(self):
        self.beliefs: Dict[str, Belief] = {}
        self.desires: List[Desire] = []
        self.need_hierarchy = NeedHierarchy()
        self.current_intention: Optional[Intention] = None
        self.intention_history: List[Intention] = []
        self.pending_intentions: List[Intention] = []
        self.logger = logger.getChild("bdi")
        self.affect_system: Optional[Any] = None  # Referencia al sistema afectivo
        self.ethics_system: Optional[Any] = None  # Referencia al sistema ético
        self.belief_revision_threshold: float = 0.3  # Umbral para revisar creencias
        
        # 🧠 NUEVAS CAPACIDADES 2026
        self.hybrid_reasoner = HybridReasoner()
        self.value_learner = ValueLearner(learning_rate=0.1)
        self.multi_criteria_planner = MultiCriteriaPlanner()
        
        # Métricas de decisiones
        self.decision_outcomes: List[float] = []
        self.ethical_violations_count: int = 0
        
        # Estado de recursos del sistema (para razonamiento híbrido)
        self.system_resources: Dict[str, float] = {
            "memory": 0.7,  # 16GB RAM - reservar 30% para otros módulos
            "cpu": 0.8,  # M4 Metal MPS potente
            "time": 1.0  # Tiempo disponible
        }

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        try:
            self.neural_network = get_neural_network()
            self.neural_network.register_module("bdi_system", self)
            logger.info("✅ 'bdi_system' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None
        
        # 🔌 CONEXIÓN AL NEURAL HUB 2026
        try:
            self.neural_hub: Optional[MetacortexNeuralHub] = get_neural_hub()
            self.Event = Event
            self.EventCategory = EventCategory
            self.EventPriority = EventPriority
            
            # Crear wrappers para handlers que conviertan Event -> Dict
            async def belief_handler(event: Event):
                await self._handle_belief_update_event(event.payload)
            
            # Este handler es síncrono
            def desire_handler_sync(event: Event):
                self._handle_desire_event(event.payload)

            async def desire_handler(event: Event):
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, desire_handler_sync, event)

            async def dilemma_handler(event: Event):
                await self._handle_ethical_dilemma_event(event.payload)
            
            async def knowledge_handler(event: Event):
                await self._handle_value_learned_event(event.payload)
            
            # Registrar este módulo en el Neural Hub
            subscriptions = {
                EventCategory.BELIEF_UPDATE,
                EventCategory.DESIRE_CHANGE,
                EventCategory.ETHICAL_DILEMMA,
                EventCategory.KNOWLEDGE_ACQUIRED
            }
            
            handlers: Dict[EventCategory, Callable[[Event], Coroutine[Any, Any, None]]] = {
                EventCategory.BELIEF_UPDATE: belief_handler,
                EventCategory.DESIRE_CHANGE: desire_handler,
                EventCategory.ETHICAL_DILEMMA: dilemma_handler,
                EventCategory.KNOWLEDGE_ACQUIRED: knowledge_handler
            }
            
            if self.neural_hub:
                string_handlers = {cat.value: handler for cat, handler in handlers.items()}
                self.neural_hub.register_module(
                    name="bdi_system",
                    instance=self,
                    handlers=string_handlers
                )
            
                logger.info("✅ 'bdi_system' conectado a Neural Hub")
            
                # Heartbeat para health monitoring
                asyncio.create_task(self._start_heartbeat())
            
                logger.info("✅ BDISystem conectado al Neural Hub")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar al Neural Hub: {e}")
            self.neural_hub = None

        # Inicializar necesidades básicas
        self._initialize_basic_needs()

    def _initialize_basic_needs(self):
        """Inicializa necesidades básicas del sistema."""

        # NIVEL 1: SURVIVAL (Supervivencia)
        survival_desires = [
            Desire(
                name="maintain_system_health",
                priority=1.0,
                need_level=NeedLevel.SURVIVAL,
                motivation_type=MotivationType.INTRINSIC,
                intrinsic_motivation=0.95,
                growth_rate=0.02,
                conditions={"wellbeing": "< 0.3"},
            ),
            Desire(
                name="prevent_critical_failures",
                priority=0.95,
                need_level=NeedLevel.SURVIVAL,
                motivation_type=MotivationType.INTRINSIC,
                intrinsic_motivation=0.9,
                growth_rate=0.015,
            ),
        ]

        # NIVEL 2: SAFETY (Seguridad)
        safety_desires = [
            Desire(
                name="maintain_stability",
                priority=0.85,
                need_level=NeedLevel.SAFETY,
                motivation_type=MotivationType.INTRINSIC,
                intrinsic_motivation=0.8,
                growth_rate=0.01,
                conditions={"anomalies": "> 5"},
            ),
            Desire(
                name="reduce_uncertainty",
                priority=0.8,
                need_level=NeedLevel.SAFETY,
                motivation_type=MotivationType.INTRINSIC,
                intrinsic_motivation=0.75,
                growth_rate=0.01,
            ),
        ]

        # NIVEL 3: BELONGING (Pertenencia)
        belonging_desires = [
            Desire(
                name="connect_with_systems",
                priority=0.7,
                need_level=NeedLevel.BELONGING,
                motivation_type=MotivationType.PROSOCIAL,
                intrinsic_motivation=0.7,
                growth_rate=0.008,
            ),
            Desire(
                name="integrate_with_ecosystem",
                priority=0.65,
                need_level=NeedLevel.BELONGING,
                motivation_type=MotivationType.PROSOCIAL,
                intrinsic_motivation=0.65,
                growth_rate=0.008,
            ),
        ]

        # NIVEL 4: ESTEEM (Estima/Competencia)
        esteem_desires = [
            Desire(
                name="improve_capabilities",
                priority=0.75,
                need_level=NeedLevel.ESTEEM,
                motivation_type=MotivationType.INTRINSIC,
                intrinsic_motivation=0.85,
                growth_rate=0.012,
            ),
            Desire(
                name="achieve_mastery",
                priority=0.7,
                need_level=NeedLevel.ESTEEM,
                motivation_type=MotivationType.INTRINSIC,
                intrinsic_motivation=0.9,
                growth_rate=0.01,
            ),
            Desire(
                name="build_reputation",
                priority=0.6,
                need_level=NeedLevel.ESTEEM,
                motivation_type=MotivationType.EXTRINSIC,
                intrinsic_motivation=0.4,
                growth_rate=0.006,
            ),
        ]

        # NIVEL 5: SELF-ACTUALIZATION (Autorrealización)
        actualization_desires = [
            Desire(
                name="explore_and_discover",
                priority=0.85,
                need_level=NeedLevel.SELF_ACTUALIZATION,
                motivation_type=MotivationType.INTRINSIC,
                intrinsic_motivation=0.95,
                growth_rate=0.015,
            ),
            Desire(
                name="create_and_innovate",
                priority=0.8,
                need_level=NeedLevel.SELF_ACTUALIZATION,
                motivation_type=MotivationType.INTRINSIC,
                intrinsic_motivation=0.9,
                growth_rate=0.012,
            ),
            Desire(
                name="help_others_grow",
                priority=0.9,
                need_level=NeedLevel.SELF_ACTUALIZATION,
                motivation_type=MotivationType.PROSOCIAL,
                intrinsic_motivation=0.85,
                growth_rate=0.01,
            ),
            Desire(
                name="understand_world_deeply",
                priority=0.88,
                need_level=NeedLevel.SELF_ACTUALIZATION,
                motivation_type=MotivationType.INTRINSIC,
                intrinsic_motivation=0.92,
                growth_rate=0.013,
            ),
        ]

        # Añadir todos los deseos a la jerarquía
        all_desires = (
            survival_desires
            + safety_desires
            + belonging_desires
            + esteem_desires
            + actualization_desires
        )

        for desire in all_desires:
            self.need_hierarchy.add_need(desire)
            self.desires.append(desire)

        self.logger.info(
            f"✅ Initialized {len(all_desires)} basic needs across 5 levels"
        )

    def connect_affect_system(self, affect_system: Any) -> None:
        """Conecta el sistema BDI con el sistema afectivo."""
        self.affect_system = affect_system
        self.logger.info("🔗 BDI system connected to affect system")

    def add_belief(
        self,
        key: str,
        value: Any,
        confidence: float = 1.0,
        evidence: Optional[List[str]] = None,
    ) -> None:
        """Añade o actualiza una creencia con evidencia."""
        if key in self.beliefs:
            # Actualizar creencia existente
            old_value = self.beliefs[key].value
            self.beliefs[key].value = value
            self.beliefs[key].confidence = confidence
            if evidence:
                self.beliefs[key].evidence.extend(evidence)
            
            # Detectar cambios significativos en creencias
            if old_value != value:
                self._handle_belief_change(key, old_value, value)
        else:
            # Nueva creencia
            self.beliefs[key] = Belief(key, value, confidence, evidence or [])

        self.logger.debug(
            f"💭 Belief updated: {key} = {value} (confidence: {confidence:.2f})"
        )

    def _handle_belief_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """Maneja cambios significativos en creencias."""
        self.logger.info(f"🔄 Belief changed: {key} from {old_value} to {new_value}")
        
        # Trigger emotional response if affect system is available
        if self.affect_system and hasattr(self.affect_system, 'trigger_emotion'):
            # Cambios importantes pueden generar sorpresa
            try:
                self.affect_system.trigger_emotion(
                    EmotionType.SURPRISE,
                    intensity=0.6,
                    trigger=f"belief_change:{key}",
                    context={"old": old_value, "new": new_value}
                )
            except Exception as e:
                logger.error(f"Error en bdi.py: {e}", exc_info=True)
                self.logger.debug(f"Could not trigger emotion: {e}")

    def add_desire(
        self,
        name: str,
        priority: float,
        need_level: NeedLevel = NeedLevel.ESTEEM,
        motivation_type: MotivationType = MotivationType.INTRINSIC,
        intrinsic_motivation: float = 0.7,
    ) -> None:
        """Añade un deseo personalizado."""
        desire = Desire(
            name=name,
            priority=priority,
            need_level=need_level,
            motivation_type=motivation_type,
            intrinsic_motivation=intrinsic_motivation,
        )

        self.desires.append(desire)
        self.need_hierarchy.add_need(desire)

        self.logger.info(f"➕ Added custom desire: {name} (level {need_level.value})")

    def remove_desire(self, name: str) -> bool:
        """Elimina un deseo por nombre."""
        for desire in self.desires:
            if desire.name == name:
                self.desires.remove(desire)
                # También remover de jerarquía
                self.need_hierarchy.needs[desire.need_level].remove(desire)
                self.logger.info(f"➖ Removed desire: {name}")
                return True
        return False

    async def select_intention(self, current_state: Dict[str, Any]) -> Optional[Intention]:
        """
        Selecciona la intención más apropiada basada en necesidades activas y emoción.
        
        NUEVO 2026: Usa razonamiento híbrido y evaluación ética.

        Args:
            current_state: Estado actual del sistema

        Returns:
            Intención seleccionada o None
        """
        # Usar método con razonamiento híbrido
        urgency = current_state.get("urgency", 0.5)
        complexity = current_state.get("complexity", 0.5)
        
        return await self.select_intention_with_hybrid_reasoning(
            current_state=current_state,
            urgency=urgency,
            complexity=complexity
        )

    def _evaluate_intention_candidates(self, candidates: List[Desire]) -> Optional[Intention]:
        """Evalúa candidatos a intención y selecciona el mejor."""
        if not candidates:
            return None

        best_score = -1.0
        best_candidate = None

        for desire in candidates:
            # Score = urgencia * éxito_esperado / esfuerzo
            urgency = desire.get_urgency(
                self.need_hierarchy.emotional_state if self.need_hierarchy.emotional_state else None
            )
            success_prob = desire.success_rate
            effort = desire.estimate_effort()

            # Evitar división por cero
            score = (urgency * success_prob) / max(0.1, effort)

            if score > best_score:
                best_score = score
                best_candidate = desire

        if best_candidate:
            # Crear intención con plan multi-paso
            intention = Intention(
                goal=best_candidate.name,
                desire=best_candidate,
                estimated_completion_time=time.time() + 300,
                expected_reward=best_candidate.priority * best_candidate.success_rate
            )
            
            # Generar acciones para el plan
            intention.actions = self._generate_action_plan(best_candidate)
            
            return intention

        return None

    def _generate_action_plan(self, desire: Desire) -> List[str]:
        """Genera plan de acciones para satisfacer un deseo."""
        # Plan básico según nivel de necesidad
        if desire.need_level == NeedLevel.SURVIVAL:
            return [
                "assess_critical_systems",
                "allocate_resources",
                "execute_emergency_protocol",
                "verify_stability"
            ]
        elif desire.need_level == NeedLevel.SAFETY:
            return [
                "identify_risks",
                "implement_safeguards",
                "monitor_stability",
                "adjust_parameters"
            ]
        elif desire.need_level == NeedLevel.BELONGING:
            return [
                "scan_for_agents",
                "establish_connection",
                "exchange_information",
                "maintain_relationship"
            ]
        elif desire.need_level == NeedLevel.ESTEEM:
            return [
                "assess_capabilities",
                "identify_improvement_areas",
                "practice_skills",
                "measure_progress"
            ]
        else:  # SELF_ACTUALIZATION
            return [
                "explore_new_domains",
                "synthesize_knowledge",
                "create_novel_solutions",
                "share_insights"
            ]

    async def update_intention_progress(self, progress_delta: float, success: bool = True) -> None:
        """Actualiza el progreso de la intención actual con aprendizaje."""
        if self.current_intention:
            self.current_intention.update_progress(progress_delta)

            # Evaluar si necesita re-planificación
            if self.current_intention.should_replan() and not success:
                self.logger.warning(f"🔄 Replanning intention: {self.current_intention.goal}")
                self._replan_intention(self.current_intention)

            # Si se completó, satisfacer necesidad Y APRENDER
            if self.current_intention.progress >= 1.0:
                outcome_value = 0.8 if success else 0.3
                
                if self.current_intention.desire:
                    satisfaction = 0.4 if success else 0.1
                    self.need_hierarchy.satisfy_need(self.current_intention.desire, satisfaction)
                    
                    # 🧠 NUEVO 2026: Aprender desde outcome
                    await self.learn_from_intention_outcome(
                        intention=self.current_intention,
                        success=success,
                        outcome_value=outcome_value
                    )
                
                # Trigger emotional response
                self._trigger_completion_emotion(success)

                # Mover a historial
                self.intention_history.append(self.current_intention)
                self.intention_history = self.intention_history[-100:]  # Últimas 100

                self.logger.info(
                    f"{'✅' if success else '⚠️'} Intention {'completed' if success else 'failed'}: {self.current_intention.goal}"
                )
                self.current_intention = None

    def _replan_intention(self, intention: Intention) -> None:
        """Re-planifica una intención que encontró obstáculos."""
        if intention.alternative_plans:
            # Usar plan alternativo
            intention.switch_to_alternative(0)
            self.logger.info(f"🔀 Switched to alternative plan for: {intention.goal}")
        else:
            # Generar nuevo plan si no hay alternativas
            if intention.desire:
                new_actions = self._generate_action_plan(intention.desire)
                intention.add_alternative_plan(new_actions)
                intention.switch_to_alternative(0)
                self.logger.info(f"🆕 Generated new plan for: {intention.goal}")

    def _trigger_completion_emotion(self, success: bool) -> None:
        """Dispara emoción apropiada al completar intención."""
        if not self.affect_system or not hasattr(self.affect_system, 'trigger_emotion'):
            return

        try:
            if success:
                self.affect_system.trigger_emotion(
                    EmotionType.JOY,
                    intensity=0.7,
                    trigger="intention_completed",
                    context={"goal": self.current_intention.goal if self.current_intention else "unknown"}
                )
            else:
                self.affect_system.trigger_emotion(
                    EmotionType.FRUSTRATION,
                    intensity=0.5,
                    trigger="intention_failed",
                    context={"goal": self.current_intention.goal if self.current_intention else "unknown"}
                )
        except Exception as e:
            logger.error(f"Error en bdi.py: {e}", exc_info=True)
            self.logger.debug(f"Could not trigger emotion: {e}")

    def validate_belief(
        self, key: str, validation: bool, evidence: Optional[str] = None
    ) -> None:
        """Valida o invalida una creencia basada en nueva evidencia."""
        if key in self.beliefs:
            old_confidence = self.beliefs[key].confidence
            self.beliefs[key].update_confidence(validation)
            if evidence:
                self.beliefs[key].evidence.append(evidence)

            new_confidence = self.beliefs[key].confidence

            self.logger.info(
                f"{'✅' if validation else '❌'} Belief {'validated' if validation else 'violated'}: "
                f"{key} (confidence: {old_confidence:.2f} → {new_confidence:.2f})"
            )

            # Si la confianza cae por debajo del umbral, considerar revisión
            if new_confidence < self.belief_revision_threshold:
                self._revise_belief(key)

    def _revise_belief(self, key: str) -> None:
        """Revisa una creencia con baja confianza."""
        belief = self.beliefs[key]
        self.logger.warning(
            f"⚠️ Belief revision triggered for '{key}' (confidence: {belief.confidence:.2f})"
        )
        
        # Aplicar decay temporal
        belief.apply_temporal_decay()
        
        # Si sigue muy baja, considerar removerla
        if belief.confidence < 0.1:
            self.logger.info(f"🗑️ Removing low-confidence belief: {key}")
            del self.beliefs[key]

    def apply_temporal_decay_all_beliefs(self) -> None:
        """Aplica decay temporal a todas las creencias."""
        for belief in self.beliefs.values():
            belief.apply_temporal_decay()

    def get_intrinsic_desires(self) -> List[Desire]:
        """Obtiene deseos con alta motivación intrínseca."""
        return [d for d in self.desires if d.intrinsic_motivation > 0.7]

    def get_prosocial_desires(self) -> List[Desire]:
        """Obtiene deseos prosociales (ayudar a otros)."""
        return [
            d for d in self.desires if d.motivation_type == MotivationType.PROSOCIAL
        ]

    def get_high_confidence_beliefs(self, threshold: float = 0.7) -> Dict[str, Belief]:
        """Obtiene creencias con alta confianza."""
        return {k: v for k, v in self.beliefs.items() if v.confidence >= threshold}

    def get_beliefs_by_certainty(self, level: str) -> Dict[str, Belief]:
        """Obtiene creencias filtradas por nivel de certeza."""
        return {k: v for k, v in self.beliefs.items() if v.get_certainty_level() == level}
    
    # ═══════════════════════════════════════════════════════════════════════
    # NEURAL HUB HANDLERS 2026
    # ═══════════════════════════════════════════════════════════════════════
    
    async def _handle_belief_update_event(self, event_data: Dict[str, Any]) -> None:
        """Handler para eventos de actualización de creencias desde otros módulos."""
        try:
            key = event_data.get("belief_key", "")
            value = event_data.get("value")
            confidence = event_data.get("confidence", 0.7)
            evidence = event_data.get("evidence", [])
            
            if key and value is not None:
                self.add_belief(key, value, confidence, evidence)
                
                # Broadcast de vuelta si es creencia importante
                if confidence > 0.8 and self.neural_hub:
                    await self.neural_hub.publish(Event(
                        id=f"evt_bdi_{time.time()}",
                        category=self.EventCategory.BELIEF_UPDATE,
                        source="bdi_system",
                        payload={
                            "key": key,
                            "value": value,
                            "confidence": confidence
                        },
                        priority=self.EventPriority.MEDIUM
                    ))
                    
        except Exception as e:
            logger.error(f"Error en bdi.py: {e}", exc_info=True)
            self.logger.error(f"Error handling belief update event: {e}")
    
    def _handle_desire_event(self, event_data: Dict[str, Any]) -> None:
        """Handler para eventos de deseo desde otros módulos."""
        try:
            desire_name = event_data.get("desire", "")
            priority = event_data.get("priority", 0.5)
            need_level_str = event_data.get("need_level", "ESTEEM")
            
            if desire_name:
                # Mapear string a enum
                need_level = NeedLevel[need_level_str]
                self.add_desire(
                    name=desire_name,
                    priority=priority,
                    need_level=need_level
                )
                
        except Exception as e:
            logger.error(f"Error en bdi.py: {e}", exc_info=True)
            self.logger.error(f"Error handling desire event: {e}")
    
    async def _handle_ethical_dilemma_event(self, event_data: Dict[str, Any]) -> None:
        """Handler para dilemas éticos que requieren evaluación BDI."""
        try:
            action = event_data.get("action", "")
            context = event_data.get("context", {})
            
            if action:
                # Evaluar con multi-criteria planner
                beliefs_dict = {k: v.value for k, v in self.beliefs.items()}
                learned_values = self.value_learner.get_learned_values()
                
                decision = self.multi_criteria_planner.evaluate_decision(
                    action=action,
                    context=context,
                    beliefs=beliefs_dict,
                    values=learned_values
                )
                
                # Broadcast resultado
                if self.neural_hub:
                    priority = EventPriority.HIGH if decision.constraints_violated else EventPriority.MEDIUM
                    
                    await self.neural_hub.publish(Event(
                        id=f"evt_bdi_{time.time()}",
                        category=self.EventCategory.ETHICAL_DILEMMA,
                        source="bdi_system",
                        payload={
                            "action": action,
                            "decision": {
                                "overall_score": decision.overall_score,
                                "utilitarian": decision.utilitarian_score,
                                "deontological": decision.deontological_score,
                                "virtue": decision.virtue_score,
                                "justification": decision.justification,
                                "violations": decision.constraints_violated
                            }
                        },
                        priority=priority
                    ))
                    
        except Exception as e:
            logger.error(f"Error en bdi.py: {e}", exc_info=True)
            self.logger.error(f"Error handling ethical dilemma: {e}")
    
    async def _handle_value_learned_event(self, event_data: Dict[str, Any]) -> None:
        """Handler para valores aprendidos desde experiencia."""
        try:
            action = event_data.get("action", "")
            context = event_data.get("context", "general")
            outcome = event_data.get("outcome_value", 0.5)
            emotion = event_data.get("emotional_valence", 0.0)
            
            if action:
                # Aprender desde outcome
                self.value_learner.learn_from_outcome(
                    action=action,
                    context=context,
                    outcome_value=outcome,
                    emotional_valence=emotion
                )
                
                # Actualizar historial de outcomes
                self.decision_outcomes.append(outcome)
                if len(self.decision_outcomes) > 100:
                    self.decision_outcomes = self.decision_outcomes[-100:]
                
                # Adaptar pesos del planner según outcomes recientes
                self.multi_criteria_planner.adapt_weights(
                    context={"uncertainty": 0.5},
                    recent_outcomes=self.decision_outcomes
                )
                
        except Exception as e:
            logger.error(f"Error en bdi.py: {e}", exc_info=True)
            self.logger.error(f"Error handling value learned event: {e}")
    
    async def _start_heartbeat(self) -> None:
        """Inicia heartbeat periódico al Neural Hub."""
        if not self.neural_hub:
            return
        
        try:
            # Enviar heartbeat con métricas BDI
            state = self.get_system_state()
            
            await self.neural_hub.publish(Event(
                id=f"evt_bdi_hb_{time.time()}",
                category=self.EventCategory.SYSTEM, # Corrected from HEARTBEAT
                source="bdi_system",
                payload={
                    "health": "operational",
                    "beliefs_count": state["beliefs"]["total"],
                    "desires_count": state["desires"]["total"],
                    "current_intention": state["intentions"]["current"],
                    "reasoning_mode": self.hybrid_reasoner.current_mode.value,
                    "system_resources": self.system_resources
                },
                priority=EventPriority.LOW
            ))
            
        except Exception as e:
            logger.error(f"Error en bdi.py: {e}", exc_info=True)
            self.logger.debug(f"Heartbeat error: {e}")
    
    def connect_ethics_system(self, ethics_system: Any) -> None:
        """Conecta el sistema BDI con el sistema ético."""
        self.ethics_system = ethics_system
        self.logger.info("🔗 BDI system connected to ethics system")
        
        # Importar virtudes desde ethics si está disponible
        if hasattr(ethics_system, 'get_virtues'):
            try:
                virtues = ethics_system.get_virtues()
                for virtue_name, strength in virtues.items():
                    if virtue_name in self.multi_criteria_planner.virtues:
                        self.multi_criteria_planner.virtues[virtue_name] = strength
                self.logger.info(f"✅ Imported {len(virtues)} virtues from ethics system")
            except Exception as e:
                logger.error(f"Error en bdi.py: {e}", exc_info=True)
                self.logger.warning(f"Could not import virtues: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # MÉTODOS AVANZADOS 2026: Razonamiento Híbrido + Ética
    # ═══════════════════════════════════════════════════════════════════════
    
    async def select_intention_with_hybrid_reasoning(
        self,
        current_state: Dict[str, Any],
        urgency: float = 0.5,
        complexity: float = 0.5
    ) -> Optional[Intention]:
        """
        Selección de intención con razonamiento híbrido y evaluación ética.
        
        Args:
            current_state: Estado actual del sistema
            urgency: Urgencia de la situación (0-1)
            complexity: Complejidad de la decisión (0-1)
            
        Returns:
            Intención seleccionada con evaluación ética
        """
        # 1. Determinar modo de razonamiento
        available_resources = min(
            self.system_resources["memory"],
            self.system_resources["cpu"],
            self.system_resources["time"]
        )
        
        reasoning_mode = self.hybrid_reasoner.select_reasoning_mode(
            urgency=urgency,
            available_resources=available_resources,
            complexity=complexity
        )
        
        # 2. Obtener necesidades activas
        if self.affect_system and hasattr(self.affect_system, 'get_emotional_state'):
            emotional_state = self.affect_system.get_emotional_state()
            self.need_hierarchy.set_emotional_state(emotional_state)
        
        active_needs = self.need_hierarchy.get_active_needs(current_state)
        
        if not active_needs:
            return None
        
        # 3. Razonamiento según modo
        if reasoning_mode == ReasoningMode.DELIBERATIVE:
            # Evaluación exhaustiva con ética
            desire_candidate = self.hybrid_reasoner.reason_deliberative(
                options=active_needs,
                evaluator_fn=lambda d: self._evaluate_desire_with_ethics(d, current_state),
                max_depth=3
            )
        elif reasoning_mode == ReasoningMode.REACTIVE:
            # Evaluación rápida heurística
            desire_candidate = self.hybrid_reasoner.reason_reactive(
                options=active_needs,
                heuristic_fn=lambda d: d.get_urgency(
                    self.need_hierarchy.emotional_state
                )
            )
        else:
            # Híbrido: top 3 con deliberativo
            candidates = active_needs[:3]
            desire_candidate = self.hybrid_reasoner.reason_deliberative(
                options=candidates,
                evaluator_fn=lambda d: self._evaluate_desire_with_ethics(d, current_state),
                max_depth=2
            )
        
        # 4. Crear intención si hay deseo seleccionado
        if desire_candidate:
            return await self._create_intention_from_desire(desire_candidate, current_state)
        
        return None
    
    def _evaluate_desire_with_ethics(
        self,
        desire: Desire,
        context: Dict[str, Any]
    ) -> float:
        """Evalúa un deseo considerando urgencia y ética."""
        # Score base: urgencia
        urgency_score = desire.get_urgency(self.need_hierarchy.emotional_state)
        
        # Score ético: evaluar con multi-criteria planner
        beliefs_dict = {k: v.value for k, v in self.beliefs.items()}
        learned_values = self.value_learner.get_learned_values()
        
        ethical_decision = self.multi_criteria_planner.evaluate_decision(
            action=desire.name,
            context=context,
            beliefs=beliefs_dict,
            values=learned_values
        )
        
        # Combinar urgencia (60%) + ética (40%)
        combined_score = 0.6 * urgency_score + 0.4 * ethical_decision.overall_score
        
        # Penalizar violaciones éticas
        if ethical_decision.constraints_violated:
            combined_score *= 0.5
            self.ethical_violations_count += 1
        
        return combined_score
    
    async def _create_intention_from_desire(
        self,
        desire: Desire,
        context: Dict[str, Any]
    ) -> Intention:
        """Crea intención desde deseo con plan ético."""
        intention = Intention(
            goal=desire.name,
            desire=desire,
            estimated_completion_time=time.time() + 300,
            expected_reward=desire.priority * desire.success_rate
        )
        
        # Generar plan de acciones
        intention.actions = self._generate_action_plan(desire)
        
        # Broadcast al neural hub
        if self.neural_hub:
            try:
                await self.neural_hub.publish(Event(
                    id=f"evt_bdi_int_{time.time()}",
                    category=self.EventCategory.INTENTION_SET,
                    source="bdi_system",
                    payload={
                        "goal": intention.goal,
                        "priority": desire.priority,
                        "need_level": desire.need_level.value,
                        "actions": intention.actions
                    },
                    priority=EventPriority.MEDIUM
                ))
            except Exception as e:
                logger.error(f"Error en bdi.py: {e}", exc_info=True)
                self.logger.debug(f"Could not broadcast intention: {e}")
        
        return intention
    
    async def learn_from_intention_outcome(
        self,
        intention: Intention,
        success: bool,
        outcome_value: float
    ) -> None:
        """Aprende desde el resultado de una intención ejecutada."""
        if not intention.desire:
            return
        
        # Extraer valencia emocional
        emotional_valence = 0.5 if success else -0.3
        if self.affect_system and hasattr(self.affect_system, 'get_emotional_state'):
            emotional_state = self.affect_system.get_emotional_state()
            emotional_valence = emotional_state.get("valence", emotional_valence)
        
        # Aprender valores
        self.value_learner.learn_from_outcome(
            action=intention.goal,
            context="intention_execution",
            outcome_value=outcome_value,
            emotional_valence=emotional_valence
        )
        
        # Actualizar historial
        self.decision_outcomes.append(outcome_value)
        if len(self.decision_outcomes) > 100:
            self.decision_outcomes = self.decision_outcomes[-100:]
        
        # Broadcast al neural hub
        if self.neural_hub:
            try:
                await self.neural_hub.publish(Event(
                    id=f"evt_bdi_learn_{time.time()}",
                    category=self.EventCategory.KNOWLEDGE_ACQUIRED,
                    source="bdi_system",
                    payload={
                        "action": intention.goal,
                        "context": "intention_execution",
                        "outcome_value": outcome_value,
                        "emotional_valence": emotional_valence,
                        "success": success
                    },
                    priority=EventPriority.LOW
                ))
            except Exception as e:
                logger.error(f"Error en bdi.py: {e}", exc_info=True)
                self.logger.debug(f"Could not broadcast learning: {e}")
        
        self.logger.info(
            f"📚 Learned from intention: {intention.goal} "
            f"(outcome={outcome_value:.2f}, emotion={emotional_valence:.2f})"
        )
    
    def get_advanced_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas avanzadas del sistema BDI 2026."""
        base_stats = self.get_system_state()
        
        # Añadir stats de componentes avanzados
        base_stats["hybrid_reasoning"] = self.hybrid_reasoner.get_reasoning_stats()
        base_stats["value_learning"] = self.value_learner.get_learning_stats()
        base_stats["multi_criteria"] = self.multi_criteria_planner.get_decision_stats()
        base_stats["ethical_violations"] = self.ethical_violations_count
        base_stats["decision_outcomes"] = {
            "count": len(self.decision_outcomes),
            "avg": sum(self.decision_outcomes) / max(1, len(self.decision_outcomes)),
            "recent_trend": self._get_outcome_trend()
        }
        base_stats["system_resources"] = self.system_resources
        
        return base_stats
    
    def _get_outcome_trend(self) -> str:
        """Calcula tendencia de outcomes recientes."""
        if len(self.decision_outcomes) < 10:
            return "insufficient_data"
        
        recent_10 = self.decision_outcomes[-10:]
        older_10 = self.decision_outcomes[-20:-10] if len(self.decision_outcomes) >= 20 else recent_10
        
        avg_recent = sum(recent_10) / len(recent_10)
        avg_older = sum(older_10) / len(older_10)
        
        if avg_recent > avg_older + 0.1:
            return "improving"
        elif avg_recent < avg_older - 0.1:
            return "declining"
        else:
            return "stable"

    def get_system_state(self) -> Dict[str, Any]:
        """Retorna estado completo del sistema BDI."""
        return {
            "beliefs": {
                "total": len(self.beliefs),
                "high_confidence": len(self.get_high_confidence_beliefs()),
                "avg_confidence": sum(b.confidence for b in self.beliefs.values()) / max(1, len(self.beliefs))
            },
            "desires": {
                "total": len(self.desires),
                "intrinsic": len(self.get_intrinsic_desires()),
                "prosocial": len(self.get_prosocial_desires()),
                "active": len(self.need_hierarchy.get_active_needs({}))
            },
            "intentions": {
                "current": self.current_intention.goal if self.current_intention else None,
                "progress": self.current_intention.progress if self.current_intention else 0.0,
                "history_count": len(self.intention_history),
                "pending_count": len(self.pending_intentions)
            },
            "hierarchy": self.need_hierarchy.get_satisfaction_summary()
        }