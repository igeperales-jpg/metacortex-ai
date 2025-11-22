#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METACORTEX - Metacognición
===========================

Sistema de metacognición que monitorea, evalúa y controla los procesos cognitivos.
Integrado con motor de curiosidad epistémica para exploración activa.
"""

from __future__ import annotations

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .curiosity import CuriosityEngine, KnowledgeGap
from .utils import setup_logging

logger = setup_logging()


@dataclass
class MetaCognitiveState:
    """Estado metacognitivo del sistema con métricas avanzadas."""

    wellbeing: float = 0.5
    confidence: float = 0.5
    learning_rate: float = 0.1
    attention_focus: str = "general"
    cognitive_load: float = 0.5  # Carga cognitiva actual (0-1)
    meta_awareness: float = 0.7  # Nivel de auto-conciencia (0-1)
    bias_detection_score: float = 0.0  # Score de sesgo detectado
    strategy_effectiveness: float = 0.5  # Efectividad de estrategia actual
    timestamp: float = field(default_factory=time.time)


class MetaCognition:
    """
    Sistema de metacognición: monitor-evaluar-control con curiosidad activa.

    Monitorea rendimiento cognitivo, evalúa procesos, genera controles
    y dirige exploración activa mediante curiosidad epistémica.
    """

    def __init__(self, curiosity_intensity: float = 0.8):
        self.state = MetaCognitiveState()
        self.evaluation_history: List[float] = []
        self.curiosity_engine = CuriosityEngine(curiosity_intensity)
        self.reflection_depth = 0.7  # Qué tan profunda es la reflexión
        self.reflections: List[Dict[str, Any]] = []
        self.logger = logger.getChild("metacog")
        
        # Referencias a otros subsistemas
        self.affect_system: Optional[Any] = None
        self.bdi_system: Optional[Any] = None
        self.learning_system: Optional[Any] = None
        
        # Métricas avanzadas
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.bias_patterns: List[Dict[str, Any]] = []
        self.cognitive_efficiency_history: List[float] = []
        self.meta_learning_rate: float = 0.01  # Tasa de meta-aprendizaje

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            self.neural_network.register_module("metacognition", self)
            logger.info("✅ 'metacognition' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None

    def connect_subsystems(
        self, 
        affect_system: Optional[Any] = None,
        bdi_system: Optional[Any] = None,
        learning_system: Optional[Any] = None
    ) -> None:
        """Conecta la metacognición con otros subsistemas."""
        if affect_system:
            self.affect_system = affect_system
            self.logger.info("🔗 Metacognition connected to affect system")
        if bdi_system:
            self.bdi_system = bdi_system
            self.logger.info("🔗 Metacognition connected to BDI system")
        if learning_system:
            self.learning_system = learning_system
            self.logger.info("🔗 Metacognition connected to learning system")

    def monitor(self, cognitive_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitorea el estado cognitivo con métricas avanzadas.

        Args:
            cognitive_state: Estado actual del sistema cognitivo

        Returns:
            Datos de monitoreo expandidos
        """
        monitoring_data: Dict[str, Any] = {
            "performance": cognitive_state.get("success_rate", 0.5),
            "resources": cognitive_state.get("resource_usage", 0.5),
            "wellbeing": cognitive_state.get("wellbeing", 0.5),
            "anomalies": cognitive_state.get("recent_anomalies", 0),
            "learning_activity": cognitive_state.get("learning_activity", 0.5),
            "curiosity_level": self.curiosity_engine.get_curiosity_level(),
            "active_knowledge_gaps": len(self.curiosity_engine.knowledge_gaps),
            "active_questions": len(self.curiosity_engine.active_questions),
            "most_curious_about": self.curiosity_engine.get_most_curious_about(3),
            "cognitive_load": self.state.cognitive_load,
            "meta_awareness": self.state.meta_awareness,
            "bias_score": self.state.bias_detection_score,
            "strategy_effectiveness": self.state.strategy_effectiveness
        }
        
        # Añadir métricas de subsistemas si están conectados
        if self.affect_system and hasattr(self.affect_system, 'get_emotional_state'):
            monitoring_data["emotional_state"] = self.affect_system.get_emotional_state()
        
        if self.bdi_system and hasattr(self.bdi_system, 'get_system_state'):
            monitoring_data["bdi_state"] = self.bdi_system.get_system_state()

        return monitoring_data

    def evaluate(self, monitoring_data: Dict[str, Any]) -> float:
        """
        Evalúa el rendimiento cognitivo.

        Args:
            monitoring_data: Datos del monitoreo

        Returns:
            Score de evaluación (0.0-1.0)
        """
        performance = monitoring_data.get("performance", 0.5)
        resources = monitoring_data.get("resources", 0.5)
        curiosity = monitoring_data.get("curiosity_level", 0.0)
        learning = monitoring_data.get("learning_activity", 0.5)

        # Evaluación expandida: rendimiento, recursos, curiosidad, aprendizaje
        evaluation = (
            performance * 0.4 + (1 - resources) * 0.2 + curiosity * 0.2 + learning * 0.2
        )

        self.evaluation_history.append(evaluation)

        # Mantener historial limitado
        if len(self.evaluation_history) > 100:
            self.evaluation_history = self.evaluation_history[-100:]

        return evaluation

    def control(self, evaluation_score: float, strategy: str = "balanced") -> Dict[str, Any]:
        """
        Genera acciones de control basadas en evaluación con estrategias adaptativas.

        Args:
            evaluation_score: Score de la evaluación
            strategy: Estrategia cognitiva actual

        Returns:
            Dict con acciones de control
        """
        control_actions: Dict[str, Any] = {}

        # Ajustar learning rate con meta-aprendizaje
        if evaluation_score < 0.4:
            adjustment = 1.2 + (self.meta_learning_rate * (0.4 - evaluation_score))
            self.state.learning_rate = min(0.5, self.state.learning_rate * adjustment)
            control_actions["adjust_learning_rate"] = self.state.learning_rate
            control_actions["reason"] = "low_performance"
        elif evaluation_score > 0.8:
            adjustment = 0.9 - (self.meta_learning_rate * (evaluation_score - 0.8))
            self.state.learning_rate = max(0.01, self.state.learning_rate * adjustment)
            control_actions["adjust_learning_rate"] = self.state.learning_rate
            control_actions["reason"] = "high_performance"

        # Ajustar wellbeing
        self.state.wellbeing = (self.state.wellbeing * 0.7) + (evaluation_score * 0.3)

        # Ajustar curiosidad basada en rendimiento
        if evaluation_score > 0.7:
            # Alto rendimiento -> más curiosidad exploratoria
            self.curiosity_engine.increase_curiosity(0.1)
            control_actions["curiosity_adjustment"] = "increase"
        elif evaluation_score < 0.3:
            # Bajo rendimiento -> consolidar conocimiento
            self.curiosity_engine.decrease_curiosity(0.1)
            control_actions["curiosity_adjustment"] = "decrease"

        # Generar pregunta activa si hay gaps de conocimiento
        if len(self.curiosity_engine.knowledge_gaps) > 0:
            question = self.curiosity_engine.generate_active_question()
            if question:
                control_actions["active_question"] = question
        
        # Actualizar eficiencia de estrategia
        self.strategy_performance[strategy].append(evaluation_score)
        self.state.strategy_effectiveness = evaluation_score
        
        # Sugerir cambio de estrategia si es inefectiva
        if len(self.strategy_performance[strategy]) >= 5:
            avg_performance = sum(self.strategy_performance[strategy][-5:]) / 5
            if avg_performance < 0.4:
                control_actions["suggest_strategy_change"] = self._suggest_better_strategy(strategy)
        
        # Actualizar carga cognitiva
        self._update_cognitive_load(control_actions)

        return control_actions

    def reflect_deeply(self, experience: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Reflexiona profundamente sobre una experiencia.

        Args:
            experience: Experiencia a analizar (situación, acción, resultado)

        Returns:
            Lista de reflexiones generadas
        """
        reflections: List[Dict[str, Any]] = []

        # 1. Análisis de éxito/fracaso
        outcome = experience.get("outcome", {})
        success_level = outcome.get("success_level", 0.5)

        if success_level > 0.7:
            # Analizar por qué tuvo éxito
            success_factors = self._identify_success_factors(experience)
            reflections.append(
                {
                    "type": "success_analysis",
                    "question": f"Why did {experience.get('action')} succeed?",
                    "factors": success_factors,
                    "lesson": f"Success pattern: {success_factors[0] if success_factors else 'unknown'}",
                    "generalizability": 0.8,
                    "timestamp": time.time(),
                }
            )

        elif success_level < 0.3:
            # Analizar por qué falló
            failure_factors = self._identify_failure_factors(experience)
            reflections.append(
                {
                    "type": "failure_analysis",
                    "question": f"Why did {experience.get('action')} fail?",
                    "factors": failure_factors,
                    "lesson": f"Avoid: {failure_factors[0] if failure_factors else 'unknown'}",
                    "improvement_opportunity": True,
                    "timestamp": time.time(),
                }
            )

        # 2. Identificar knowledge gaps
        if "encountered_unknowns" in experience:
            for unknown in experience["encountered_unknowns"]:
                reflections.append(
                    {
                        "type": "knowledge_gap",
                        "question": f"What is {unknown}?",
                        "gap_concept": unknown,
                        "importance": 0.7,
                        "timestamp": time.time(),
                    }
                )

        # 3. Analizar sorpresas (aprendizaje de expectativas violadas)
        if "expected_outcome" in experience:
            expected = experience["expected_outcome"]
            actual = experience.get("outcome", {})

            discrepancy = self._measure_discrepancy(expected, actual)
            if discrepancy > 0.5:
                reflections.append(
                    {
                        "type": "expectation_violation",
                        "question": "Why was the outcome different than expected?",
                        "expected": expected,
                        "actual": actual,
                        "surprise_level": discrepancy,
                        "lesson": "Update predictive model",
                        "timestamp": time.time(),
                    }
                )

        # 4. Identificar oportunidades de mejora
        if self.reflection_depth > 0.5:
            improvements = self._identify_improvement_opportunities(experience)
            for improvement in improvements:
                reflections.append(
                    {
                        "type": "improvement",
                        "question": f"How can I improve {improvement['skill']}?",
                        "skill": improvement["skill"],
                        "potential_impact": improvement["impact"],
                        "actionable": True,
                        "timestamp": time.time(),
                    }
                )

        # Almacenar reflexiones
        self.reflections.extend(reflections)
        if len(self.reflections) > 200:
            self.reflections = self.reflections[-200:]

        if reflections:
            self.logger.info(
                f"🤔 Deep reflection generated {len(reflections)} insights"
            )

        return reflections

    def get_wellbeing_score(self) -> float:
        """Calcula puntuación de bienestar."""
        if len(self.evaluation_history) == 0:
            return 0.5

        recent_evals = self.evaluation_history[-10:]  # Últimas 10 evaluaciones
        return sum(recent_evals) / len(recent_evals)

    def get_recent_reflections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtiene reflexiones recientes."""
        return self.reflections[-limit:] if self.reflections else []

    def get_top_knowledge_gaps(self, limit: int = 5) -> List[KnowledgeGap]:
        """Obtiene los gaps de conocimiento más importantes."""
        sorted_gaps = sorted(
            self.curiosity_engine.knowledge_gaps,
            key=lambda g: g.importance,
            reverse=True,
        )
        return sorted_gaps[:limit]

    # ==================== MÉTODOS PRIVADOS ====================

    def _identify_success_factors(self, experience: Dict[str, Any]) -> List[str]:
        """Identifica factores que contribuyeron al éxito."""
        factors = []

        # Factores contextuales
        context = experience.get("context", {})
        if context.get("resources") == "abundant":
            factors.append("abundant_resources")

        if context.get("complexity") == "low":
            factors.append("low_complexity")

        # Factores de acción
        if experience.get("action_quality") == "high":
            factors.append("well_executed_action")

        # Factores de preparación
        if experience.get("preparation") == "thorough":
            factors.append("thorough_preparation")

        return factors if factors else ["favorable_conditions"]

    def _identify_failure_factors(self, experience: Dict[str, Any]) -> List[str]:
        """Identifica factores que contribuyeron al fracaso."""
        factors = []

        # Factores contextuales
        context = experience.get("context", {})
        if context.get("resources") == "scarce":
            factors.append("insufficient_resources")

        if context.get("complexity") == "high":
            factors.append("excessive_complexity")

        # Factores de acción
        if experience.get("action_quality") == "low":
            factors.append("poor_execution")

        # Factores de conocimiento
        if "knowledge_gap" in experience:
            factors.append("insufficient_knowledge")

        return factors if factors else ["unfavorable_conditions"]

    def _measure_discrepancy(self, expected: Any, actual: Any) -> float:
        """Mide discrepancia entre expectativa y realidad."""
        # Simple: comparación de valores numéricos
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            max_val = max(abs(expected), abs(actual), 1.0)
            return abs(expected - actual) / max_val

        # Comparación de diccionarios
        if isinstance(expected, dict) and isinstance(actual, dict):
            differences = sum(
                1 for key in expected if expected.get(key) != actual.get(key)
            )
            return differences / max(len(expected), 1.0)

        # Comparación de strings
        if isinstance(expected, str) and isinstance(actual, str):
            return 0.0 if expected == actual else 1.0

        return 0.5  # Por defecto: moderada discrepancia

    def _identify_improvement_opportunities(
        self, experience: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identifica oportunidades de mejora basadas en experiencia."""
        opportunities = []

        # Si hubo dificultades, son oportunidades
        if "difficulties" in experience:
            for difficulty in experience["difficulties"]:
                opportunities.append(
                    {
                        "skill": f"handle_{difficulty}",
                        "impact": 0.7,
                        "evidence": f"Struggled with {difficulty}",
                    }
                )

        # Si hubo errores, son oportunidades
        if "errors" in experience:
            for error in experience["errors"]:
                opportunities.append(
                    {
                        "skill": f"avoid_{error.get('type', 'error')}",
                        "impact": 0.8,
                        "evidence": f"Made error: {error.get('description')}",
                    }
                )

        return opportunities
    
    def _suggest_better_strategy(self, current_strategy: str) -> str:
        """Sugiere una mejor estrategia cognitiva basada en performance."""
        # Evaluar performance de todas las estrategias conocidas
        strategies = ["balanced", "exploitative", "explorative", "conservative", "aggressive"]
        
        best_strategy = current_strategy
        best_avg = 0.0
        
        for strategy in strategies:
            if strategy in self.strategy_performance and len(self.strategy_performance[strategy]) >= 3:
                avg = sum(self.strategy_performance[strategy][-5:]) / min(5, len(self.strategy_performance[strategy]))
                if avg > best_avg:
                    best_avg = avg
                    best_strategy = strategy
        
        return best_strategy if best_strategy != current_strategy else "explorative"
    
    def _update_cognitive_load(self, control_actions: Dict[str, Any]) -> None:
        """Actualiza la carga cognitiva basada en acciones de control."""
        # Calcular carga basada en número de ajustes
        num_adjustments = len(control_actions)
        
        # Más ajustes = mayor carga
        load_increase = num_adjustments * 0.05
        
        # Decaimiento temporal de carga
        decay = 0.1
        
        self.state.cognitive_load = min(1.0, 
            (self.state.cognitive_load * (1 - decay)) + load_increase
        )
        
        # Si hay curiosidad activa, aumenta carga
        if "active_question" in control_actions:
            self.state.cognitive_load = min(1.0, self.state.cognitive_load + 0.1)
    
    def detect_cognitive_biases(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detecta sesgos cognitivos en decisiones recientes.
        
        Args:
            decisions: Lista de decisiones tomadas
            
        Returns:
            Lista de sesgos detectados con evidencia
        """
        biases: List[Dict[str, Any]] = []
        
        if len(decisions) < 3:
            return biases
        
        # Detectar sesgo de confirmación (buscar solo evidencia que confirma creencia)
        confirmation_count = sum(1 for d in decisions if d.get("seeks_confirmation", False))
        if confirmation_count / len(decisions) > 0.7:
            biases.append({
                "type": "confirmation_bias",
                "severity": confirmation_count / len(decisions),
                "description": "Tendency to seek only confirming evidence",
                "recommendation": "Actively seek disconfirming evidence"
            })
        
        # Detectar sesgo de recencia (dar más peso a experiencias recientes)
        recent_weight = sum(d.get("recency_weight", 0.5) for d in decisions[-3:]) / 3
        if recent_weight > 0.8:
            biases.append({
                "type": "recency_bias",
                "severity": recent_weight,
                "description": "Overweighting recent experiences",
                "recommendation": "Consider historical patterns equally"
            })
        
        # Detectar sesgo de disponibilidad (usar info fácilmente disponible)
        availability_count = sum(1 for d in decisions if d.get("used_cached_info", False))
        if availability_count / len(decisions) > 0.8:
            biases.append({
                "type": "availability_bias",
                "severity": availability_count / len(decisions),
                "description": "Relying too heavily on easily available information",
                "recommendation": "Seek diverse information sources"
            })
        
        # Actualizar score de sesgo
        if biases:
            self.state.bias_detection_score = sum(b["severity"] for b in biases) / len(biases)
            self.bias_patterns.extend(biases)
            self.bias_patterns = self.bias_patterns[-50:]  # Mantener últimos 50
        
        return biases
    
    def measure_cognitive_efficiency(self, task_performance: Dict[str, Any]) -> float:
        """
        Mide eficiencia cognitiva: output/input ratio.
        
        Args:
            task_performance: Métricas de rendimiento de tarea
            
        Returns:
            Score de eficiencia (0-1)
        """
        output_quality = task_performance.get("quality", 0.5)
        output_speed = task_performance.get("speed", 0.5)
        
        input_resources = task_performance.get("resources_used", 0.5)
        input_effort = task_performance.get("cognitive_effort", 0.5)
        
        # Eficiencia = (calidad + velocidad) / (recursos + esfuerzo)
        output = (output_quality + output_speed) / 2
        input_cost = (input_resources + input_effort) / 2
        
        efficiency = output / max(input_cost, 0.1)  # Evitar división por cero
        efficiency = min(1.0, efficiency)  # Cap a 1.0
        
        self.cognitive_efficiency_history.append(efficiency)
        self.cognitive_efficiency_history = self.cognitive_efficiency_history[-100:]
        
        return efficiency
    
    def get_meta_state(self) -> Dict[str, Any]:
        """Retorna estado completo de metacognición."""
        return {
            "state": {
                "wellbeing": self.state.wellbeing,
                "confidence": self.state.confidence,
                "learning_rate": self.state.learning_rate,
                "attention_focus": self.state.attention_focus,
                "cognitive_load": self.state.cognitive_load,
                "meta_awareness": self.state.meta_awareness,
                "bias_score": self.state.bias_detection_score,
                "strategy_effectiveness": self.state.strategy_effectiveness
            },
            "evaluation": {
                "recent_scores": self.evaluation_history[-10:] if self.evaluation_history else [],
                "avg_performance": sum(self.evaluation_history[-20:]) / len(self.evaluation_history[-20:]) if self.evaluation_history else 0.5
            },
            "curiosity": {
                "level": self.curiosity_engine.get_curiosity_level(),
                "knowledge_gaps": len(self.curiosity_engine.knowledge_gaps),
                "active_questions": len(self.curiosity_engine.active_questions),
                "most_curious": self.curiosity_engine.get_most_curious_about(5)
            },
            "reflections": {
                "total": len(self.reflections),
                "recent": self.reflections[-5:] if self.reflections else []
            },
            "biases": {
                "current_score": self.state.bias_detection_score,
                "recent_patterns": self.bias_patterns[-5:] if self.bias_patterns else []
            },
            "efficiency": {
                "current": self.cognitive_efficiency_history[-1] if self.cognitive_efficiency_history else 0.5,
                "avg_recent": sum(self.cognitive_efficiency_history[-10:]) / len(self.cognitive_efficiency_history[-10:]) if self.cognitive_efficiency_history else 0.5
            },
            "strategies": {
                strategy: sum(scores[-5:]) / len(scores[-5:]) if scores else 0.0
                for strategy, scores in self.strategy_performance.items()
            }
        }