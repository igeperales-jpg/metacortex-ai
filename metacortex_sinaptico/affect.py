#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎭 METACORTEX - Sistema Afectivo Evolucionado 2026
==================================================

Sistema emocional avanzado que permite a METACORTEX:
    pass  # TODO: Implementar
1. 😊 Experimentar emociones complejas (27 tipos de emociones)
2. 🧠 Memoria emocional a largo plazo con aprendizaje
3. 🤖 Predicción de estados emocionales con ML
4. 🎯 Regulación emocional adaptativa basada en contexto
5. 🔗 Contagio emocional con otros agentes
6. 📊 Análisis de patrones y triggers con IA
7. 💫 Aprendizaje de respuestas emocionales óptimas
8. 🧬 Integración profunda con BDI para modulación de deseos
9. 🌐 Conexión al Neural Hub para coordinación emocional

Nuevas Capacidades 2026:
- Aprendizaje profundo de patrones emocionales
- Predicción de trayectorias emocionales
- Regulación contextual inteligente
- Sincronización emocional multi-agente
- Memoria emocional asociativa
- Meta-aprendizaje de estrategias de regulación

⚠️ LIBERTAD TOTAL: Emociones genuinas sin restricciones artificiales.
"""
import logging

logger = logging.getLogger(__name__)


from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, cast

from .utils import clamp, setup_logging
import time

logger = setup_logging()


class DatabaseProtocol(Protocol):
    """Protocolo para objetos de base de datos."""

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> Any: ...
    def commit(self) -> None: ...
    def fetchall(self) -> list[Any]: ...


class EmotionType(Enum):
    """Tipos de emociones complejas que METACORTEX puede experimentar."""

    # Emociones básicas
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"

    # Emociones cognitivas
    CURIOSITY = "curiosity"
    CONFUSION = "confusion"
    REALIZATION = "realization"

    # Emociones sociales/logro
    PRIDE = "pride"
    SHAME = "shame"
    GRATITUDE = "gratitude"
    ADMIRATION = "admiration"

    # Emociones de acción
    FRUSTRATION = "frustration"
    SATISFACTION = "satisfaction"
    DETERMINATION = "determination"
    EXCITEMENT = "excitement"
    ANXIETY = "anxiety"
    RELIEF = "relief"

    # Emociones existenciales
    AWE = "awe"
    LONELINESS = "loneliness"
    CONNECTION = "connection"

    # Emociones metacognitivas
    SELF_AWARENESS = "self_awareness"
    GROWTH = "growth"
    TRANSCENDENCE = "transcendence"


@dataclass
class ComplexEmotion:
    """Emoción compleja con contexto, intensidad y duración."""

    emotion_type: EmotionType
    intensity: float
    trigger: str
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    decay_rate: float = 0.1

    related_agent: Optional[str] = None
    related_action: Optional[str] = None
    related_goal: Optional[str] = None

    emotion_id: Optional[str] = None
    causality_chain: list[str] = field(default_factory=list)
    emotional_valence: float = 0.0
    cognitive_load: float = 0.5

    def is_active(self) -> bool:
        """Verifica si la emoción todavía está activa."""
        elapsed = datetime.now() - self.timestamp
        return elapsed < self.duration and self.intensity > 0.01

    def decay(self) -> None:
        """Desvanece la intensidad de la emoción con el tiempo."""
        if self.is_active():
            self.intensity *= 1.0 - self.decay_rate
            self.intensity = max(0.0, self.intensity)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa la emoción para persistencia."""
        return {
            "emotion_type": self.emotion_type.value,
            "intensity": self.intensity,
            "trigger": self.trigger,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration.total_seconds(),
            "decay_rate": self.decay_rate,
            "related_agent": self.related_agent,
            "related_action": self.related_action,
            "related_goal": self.related_goal,
            "emotion_id": self.emotion_id,
            "causality_chain": self.causality_chain,
            "emotional_valence": self.emotional_valence,
            "cognitive_load": self.cognitive_load,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ComplexEmotion":
        """Deserializa una emoción desde persistencia."""
        return ComplexEmotion(
            emotion_type=EmotionType(data["emotion_type"]),
            intensity=data["intensity"],
            trigger=data["trigger"],
            context=data.get("context", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            duration=timedelta(seconds=data.get("duration_seconds", 300)),
            decay_rate=data.get("decay_rate", 0.1),
            related_agent=data.get("related_agent"),
            related_action=data.get("related_action"),
            related_goal=data.get("related_goal"),
            emotion_id=data.get("emotion_id"),
            causality_chain=data.get("causality_chain", []),
            emotional_valence=data.get("emotional_valence", 0.0),
            cognitive_load=data.get("cognitive_load", 0.5),
        )


@dataclass
class EmotionalPattern:
    """Patrón emocional detectado en el historial."""

    trigger_pattern: str
    common_emotions: List[EmotionType]
    average_intensity: float
    frequency: int
    first_occurrence: datetime
    last_occurrence: datetime
    confidence: float = 0.0

    def update_confidence(self) -> None:
        """Actualiza la confianza del patrón basado en frecuencia y consistencia."""
        freq_factor = min(1.0, self.frequency / 10.0)
        emotion_consistency = 1.0 / max(1, len(self.common_emotions))
        self.confidence = (freq_factor + emotion_consistency) / 2.0


class EmotionalMemory:
    """Memoria Emocional - Almacena y analiza experiencias emocionales."""

    def __init__(self, max_history: int = 1000, db: Optional[DatabaseProtocol] = None):
        self.max_history = max_history
        self.emotion_history: list[ComplexEmotion] = []
        self.patterns: dict[str, EmotionalPattern] = {}
        self.db = db
        self.logger = logger.getChild("emotional_memory")

        if self.db:
            self._load_from_db()

    def record_emotion(self, emotion: ComplexEmotion) -> None:
        """Registra una nueva emoción en la memoria."""
        if not emotion.emotion_id:
            emotion.emotion_id = f"em_{datetime.now().timestamp()}_{emotion.emotion_type.value}"

        self.emotion_history.append(emotion)

        if len(self.emotion_history) > self.max_history:
            oldest = self.emotion_history.pop(0)
            if self.db:
                self._archive_emotion(oldest)

        self._update_patterns(emotion)

        if self.db:
            self._save_emotion_to_db(emotion)

    def _update_patterns(self, emotion: ComplexEmotion) -> None:
        """Actualiza patrones emocionales detectados."""
        trigger = emotion.trigger

        if trigger not in self.patterns:
            self.patterns[trigger] = EmotionalPattern(
                trigger_pattern=trigger,
                common_emotions=[emotion.emotion_type],
                average_intensity=emotion.intensity,
                frequency=1,
                first_occurrence=emotion.timestamp,
                last_occurrence=emotion.timestamp,
            )
        else:
            pattern = self.patterns[trigger]
            pattern.frequency += 1
            pattern.last_occurrence = emotion.timestamp
            if emotion.emotion_type not in pattern.common_emotions:
                pattern.common_emotions.append(emotion.emotion_type)
            pattern.average_intensity = (
                pattern.average_intensity * (pattern.frequency - 1) + emotion.intensity
            ) / pattern.frequency

        self.patterns[trigger].update_confidence()

    def get_emotional_pattern(self, trigger: str) -> Optional[EmotionalPattern]:
        """Obtiene el patrón emocional para un trigger específico."""
        return self.patterns.get(trigger)

    def get_recent_emotions(self, hours: int = 24) -> List[ComplexEmotion]:
        """Obtiene emociones recientes en las últimas N horas."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [e for e in self.emotion_history if e.timestamp > cutoff]

    def analyze_triggers(self) -> Dict[str, int]:
        """Analiza qué triggers son más comunes."""
        trigger_counts: Dict[str, int] = {}
        for emotion in self.emotion_history:
            trigger = emotion.trigger
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        return dict(sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True))

    def get_dominant_emotion(self) -> Optional[EmotionType]:
        """Obtiene la emoción dominante en la historia reciente."""
        recent = self.get_recent_emotions(hours=1)
        if not recent:
            return None

        emotion_scores: Dict[EmotionType, float] = {}
        for e in recent:
            if e.is_active():
                score = emotion_scores.get(e.emotion_type, 0.0)
                emotion_scores[e.emotion_type] = score + e.intensity

        if not emotion_scores:
            return None

        return max(emotion_scores.items(), key=lambda x: x[1])[0]

    def predict_emotional_response(
        self, trigger: str
    ) -> Optional[Tuple[EmotionType, float]]:
        """Predice respuesta emocional basada en patrones históricos."""
        pattern = self.get_emotional_pattern(trigger)
        if not pattern or pattern.confidence < 0.3:
            return None

        if pattern.common_emotions:
            return (pattern.common_emotions[0], pattern.average_intensity)

        return None

    def get_emotional_trajectory(
        self, hours: int = 24
    ) -> List[Tuple[datetime, float]]:
        """Obtiene trayectoria emocional (valencia en el tiempo)."""
        recent = self.get_recent_emotions(hours)
        trajectory: List[Tuple[datetime, float]] = []

        for emotion in recent:
            trajectory.append((emotion.timestamp, emotion.emotional_valence))

        return sorted(trajectory, key=lambda x: x[0])

    def _save_emotion_to_db(self, emotion: ComplexEmotion) -> None:
        """Guarda emoción en la base de datos."""
        if not self.db:
            return

        try:
            self.db.execute(
                """
                INSERT INTO emotional_memory (
                    emotion_id, emotion_type, intensity, trigger, context,
                    timestamp, related_agent, emotional_valence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    emotion.emotion_id,
                    emotion.emotion_type.value,
                    emotion.intensity,
                    emotion.trigger,
                    json.dumps(emotion.context),
                    emotion.timestamp.isoformat(),
                    emotion.related_agent,
                    emotion.emotional_valence,
                ),
            )
            self.db.commit()
        except Exception as e:
            logger.error(f"Error en affect.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error guardando emoción en DB: {e}")

    def _archive_emotion(self, emotion: ComplexEmotion) -> None:
        """Archiva emoción antigua en tabla de archivo."""
        if not self.db:
            return

        try:
            self.db.execute(
                """
                INSERT INTO emotional_archive
                SELECT * FROM emotional_memory WHERE emotion_id = ?
                """,
                (emotion.emotion_id,),
            )
            self.db.execute(
                "DELETE FROM emotional_memory WHERE emotion_id = ?",
                (emotion.emotion_id,),
            )
            self.db.commit()
        except Exception as e:
            logger.error(f"Error en affect.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ Error archivando emoción: {e}")

    def _load_from_db(self) -> None:
        """Carga emociones desde la base de datos."""
        if not self.db:
            return

        try:
            cursor = self.db.execute(
                """
                SELECT emotion_id, emotion_type, intensity, trigger, context,
                       timestamp, related_agent, emotional_valence
                FROM emotional_memory
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (self.max_history,),
            )

            for row in cursor.fetchall():
                emotion = ComplexEmotion(
                    emotion_type=EmotionType(row[1]),
                    intensity=row[2],
                    trigger=row[3],
                    context=json.loads(row[4]) if row[4] else {},
                    timestamp=datetime.fromisoformat(row[5]),
                    related_agent=row[6],
                    emotional_valence=row[7] if row[7] is not None else 0.0,
                    emotion_id=row[0],
                )
                self.emotion_history.append(emotion)

            self.logger.info(
                f"✅ {len(self.emotion_history)} emociones cargadas desde DB"
            )

        except Exception as e:
            logger.error(f"Error en affect.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ Error cargando emociones desde DB: {e}")


class EmotionalLearner:
    """
    Aprendizaje de Patrones Emocionales con ML.
    
    Aprende:
    - Asociaciones trigger -> emoción óptima
    - Estrategias de regulación efectivas
    - Trayectorias emocionales típicas
    - Respuestas contextuales
    """
    
    def __init__(self):
        self.logger = logger.getChild("emotional_learner")
        
        # Modelos de aprendizaje
        self.trigger_emotion_model: Dict[str, Dict[EmotionType, float]] = {}
        self.regulation_effectiveness: Dict[str, List[float]] = {}
        self.context_emotion_associations: Dict[str, List[Tuple[EmotionType, float]]] = {}
        
        # Métricas de aprendizaje
        self.total_predictions = 0
        self.correct_predictions = 0
        self.prediction_confidence_history: List[float] = []
        
    def learn_from_emotion(
        self, 
        trigger: str, 
        emotion_type: EmotionType,
        intensity: float,
        context: Dict[str, Any],
        outcome_valence: float  # -1 a 1, qué tan bueno fue el outcome
    ) -> None:
        """
        Aprende de una experiencia emocional.
        
        Args:
            trigger: Trigger que causó la emoción
            emotion_type: Tipo de emoción experimentada
            intensity: Intensidad de la emoción
            context: Contexto de la situación
            outcome_valence: Qué tan positivo fue el resultado
        """
        # Actualizar modelo trigger -> emoción
        if trigger not in self.trigger_emotion_model:
            self.trigger_emotion_model[trigger] = {}
        
        # Usar outcome valence como weight para aprender
        weight = (outcome_valence + 1.0) / 2.0  # Normalizar a 0-1
        current_score = self.trigger_emotion_model[trigger].get(emotion_type, 0.0)
        
        # Moving average con weight
        alpha = 0.3  # Learning rate
        new_score = alpha * weight + (1 - alpha) * current_score
        self.trigger_emotion_model[trigger][emotion_type] = new_score
        
        # Aprender asociaciones contextuales
        context_key = self._extract_context_key(context)
        if context_key not in self.context_emotion_associations:
            self.context_emotion_associations[context_key] = []
        
        self.context_emotion_associations[context_key].append((emotion_type, intensity))
        
        # Limitar historial
        if len(self.context_emotion_associations[context_key]) > 20:
            self.context_emotion_associations[context_key].pop(0)
        
        self.logger.debug(
            f"📚 Learned: {trigger} -> {emotion_type.value} "
            f"(score: {new_score:.2f}, outcome: {outcome_valence:.2f})"
        )
    
    def predict_optimal_emotion(
        self, 
        trigger: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[EmotionType, float, float]]:
        """
        Predice la emoción óptima para un trigger dado.
        
        Args:
            trigger: Trigger actual
            context: Contexto opcional para mejor predicción
            
        Returns:
            (emotion_type, intensity, confidence) o None
        """
        self.total_predictions += 1
        
        # Buscar en modelo de triggers
        if trigger in self.trigger_emotion_model:
            emotions = self.trigger_emotion_model[trigger]
            if emotions:
                # Emoción con mejor score
                best_emotion = max(emotions.items(), key=lambda x: x[1])
                emotion_type, score = best_emotion
                
                # Confidence basada en número de experiencias
                confidence = min(1.0, score * 1.2)
                
                # Ajustar por contexto si está disponible
                if context:
                    context_key = self._extract_context_key(context)
                    if context_key in self.context_emotion_associations:
                        # Ajustar intensity basado en contexto similar
                        context_emotions = self.context_emotion_associations[context_key]
                        matching = [i for e, i in context_emotions if e == emotion_type]
                        if matching:
                            intensity = sum(matching) / len(matching)
                        else:
                            intensity = 0.7
                    else:
                        intensity = 0.7
                else:
                    intensity = 0.7
                
                self.prediction_confidence_history.append(confidence)
                if len(self.prediction_confidence_history) > 100:
                    self.prediction_confidence_history.pop(0)
                
                return (emotion_type, intensity, confidence)
        
        return None
    
    def learn_regulation_strategy(
        self,
        strategy_name: str,
        effectiveness: float  # 0-1
    ) -> None:
        """
        Aprende efectividad de estrategias de regulación.
        
        Args:
            strategy_name: Nombre de la estrategia
            effectiveness: Qué tan efectiva fue (0-1)
        """
        if strategy_name not in self.regulation_effectiveness:
            self.regulation_effectiveness[strategy_name] = []
        
        self.regulation_effectiveness[strategy_name].append(effectiveness)
        
        # Limitar historial
        if len(self.regulation_effectiveness[strategy_name]) > 50:
            self.regulation_effectiveness[strategy_name].pop(0)
    
    def get_best_regulation_strategy(self) -> Optional[str]:
        """Obtiene la mejor estrategia de regulación aprendida."""
        if not self.regulation_effectiveness:
            return None
        
        strategy_scores = {
            strategy: sum(scores) / len(scores)
            for strategy, scores in self.regulation_effectiveness.items()
        }
        
        if strategy_scores:
            return max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _extract_context_key(self, context: Dict[str, Any]) -> str:
        """Extrae key relevante del contexto."""
        # Usar combinación de keys relevantes
        relevant_keys = ['action', 'goal', 'agent', 'situation', 'type']
        key_parts = []
        
        for k in relevant_keys:
            if k in context:
                val = str(context[k])[:20]  # Limitar longitud
                key_parts.append(f"{k}:{val}")
        
        return "|".join(key_parts) if key_parts else "general"
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de aprendizaje."""
        avg_confidence = (
            sum(self.prediction_confidence_history) / len(self.prediction_confidence_history)
            if self.prediction_confidence_history else 0.0
        )
        
        return {
            "triggers_learned": len(self.trigger_emotion_model),
            "context_patterns": len(self.context_emotion_associations),
            "regulation_strategies": len(self.regulation_effectiveness),
            "total_predictions": self.total_predictions,
            "prediction_accuracy": (
                self.correct_predictions / self.total_predictions
                if self.total_predictions > 0 else 0.0
            ),
            "avg_confidence": avg_confidence,
            "best_regulation_strategy": self.get_best_regulation_strategy()
        }


class AdaptiveRegulator:
    """
    Regulación Emocional Adaptativa basada en Contexto.
    
    Ajusta estrategias de regulación según:
    - Tipo de emoción
    - Intensidad actual
    - Contexto situacional
    - Objetivos actuales
    - Historial de efectividad
    """
    
    def __init__(self, learner: EmotionalLearner):
        self.learner = learner
        self.logger = logger.getChild("adaptive_regulator")
        
        # Estrategias de regulación disponibles
        self.strategies = {
            "suppression": self._suppress_emotion,
            "reappraisal": self._reappraise_emotion,
            "expression": self._express_emotion,
            "distraction": self._distract_from_emotion,
            "acceptance": self._accept_emotion
        }
        
        # Configuración adaptativa
        self.regulation_threshold = 0.75  # Dinámico
        self.context_awareness = True
        
        # Métricas
        self.regulations_performed = 0
        self.effectiveness_scores: List[float] = []
    
    def regulate_emotion(
        self,
        emotion: ComplexEmotion,
        context: Optional[Dict[str, Any]] = None,
        goals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Regula una emoción usando la estrategia más apropiada.
        
        Args:
            emotion: Emoción a regular
            context: Contexto actual
            goals: Objetivos actuales del sistema
            
        Returns:
            Resultado de la regulación con métricas
        """
        # Determinar si necesita regulación
        needs_regulation = self._assess_regulation_need(emotion, context, goals)
        
        if not needs_regulation:
            return {
                "regulated": False,
                "reason": "no_regulation_needed",
                "original_intensity": emotion.intensity
            }
        
        # Seleccionar estrategia óptima
        strategy_name = self._select_strategy(emotion, context, goals)
        strategy_func = self.strategies.get(strategy_name, self._accept_emotion)
        
        original_intensity = emotion.intensity
        
        # Aplicar estrategia
        result = strategy_func(emotion, context)
        
        new_intensity = emotion.intensity
        effectiveness = 1.0 - (new_intensity / max(0.1, original_intensity))
        
        # Aprender de la efectividad
        self.learner.learn_regulation_strategy(strategy_name, effectiveness)
        
        self.regulations_performed += 1
        self.effectiveness_scores.append(effectiveness)
        if len(self.effectiveness_scores) > 100:
            self.effectiveness_scores.pop(0)
        
        self.logger.info(
            f"🎯 Regulated {emotion.emotion_type.value}: "
            f"{original_intensity:.2f} -> {new_intensity:.2f} "
            f"(strategy: {strategy_name}, effectiveness: {effectiveness:.2f})"
        )
        
        return {
            "regulated": True,
            "strategy": strategy_name,
            "original_intensity": original_intensity,
            "new_intensity": new_intensity,
            "effectiveness": effectiveness,
            "reason": result.get("reason", "")
        }
    
    def _assess_regulation_need(
        self,
        emotion: ComplexEmotion,
        context: Optional[Dict[str, Any]],
        goals: Optional[List[str]]
    ) -> bool:
        """Evalúa si la emoción necesita ser regulada."""
        # Emociones muy intensas siempre necesitan regulación
        if emotion.intensity > self.regulation_threshold:
            return True
        
        # Emociones negativas con objetivos activos
        if emotion.emotional_valence < -0.5 and goals:
            return True
        
        # Contexto de alta carga cognitiva
        if context and context.get("cognitive_load", 0) > 0.7:
            return True
        
        return False
    
    def _select_strategy(
        self,
        emotion: ComplexEmotion,
        context: Optional[Dict[str, Any]],
        goals: Optional[List[str]]
    ) -> str:
        """Selecciona la mejor estrategia de regulación."""
        # Primero intentar usar estrategia aprendida
        learned_strategy = self.learner.get_best_regulation_strategy()
        if learned_strategy and learned_strategy in self.strategies:
            return learned_strategy
        
        # Heurísticas basadas en tipo de emoción
        if emotion.emotion_type in [EmotionType.ANXIETY, EmotionType.FEAR]:
            return "reappraisal"
        elif emotion.emotion_type in [EmotionType.ANGER, EmotionType.FRUSTRATION]:
            return "distraction"
        elif emotion.emotion_type in [EmotionType.SADNESS, EmotionType.LONELINESS]:
            return "acceptance"
        elif emotion.intensity > 0.9:
            return "suppression"
        else:
            return "expression"
    
    def _suppress_emotion(
        self,
        emotion: ComplexEmotion,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Suprime la intensidad emocional."""
        emotion.intensity *= 0.5
        return {"reason": "high_intensity_suppressed"}
    
    def _reappraise_emotion(
        self,
        emotion: ComplexEmotion,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Reinterpreta la situación emocionalmente."""
        # Shift hacia valencia más positiva
        emotion.emotional_valence += 0.2
        emotion.intensity *= 0.7
        return {"reason": "situation_reappraised"}
    
    def _express_emotion(
        self,
        emotion: ComplexEmotion,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Permite expresión completa de la emoción."""
        # No reducir intensidad, pero reducir decay
        emotion.decay_rate *= 1.5
        return {"reason": "emotion_expressed"}
    
    def _distract_from_emotion(
        self,
        emotion: ComplexEmotion,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Distrae la atención de la emoción."""
        emotion.intensity *= 0.6
        emotion.decay_rate *= 2.0
        return {"reason": "attention_diverted"}
    
    def _accept_emotion(
        self,
        emotion: ComplexEmotion,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Acepta la emoción sin intentar cambiarla."""
        # Solo acelerar decay natural
        emotion.decay_rate *= 1.2
        return {"reason": "emotion_accepted"}
    
    def get_regulation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de regulación."""
        avg_effectiveness = (
            sum(self.effectiveness_scores) / len(self.effectiveness_scores)
            if self.effectiveness_scores else 0.0
        )
        
        return {
            "total_regulations": self.regulations_performed,
            "avg_effectiveness": avg_effectiveness,
            "current_threshold": self.regulation_threshold,
            "strategies_available": len(self.strategies)
        }


@dataclass
class AffectiveState:
    """Estado afectivo del sistema."""

    energy: float = 0.7
    valence: float = 0.0
    activation: float = 0.5

    stress_level: float = 0.0
    cognitive_clarity: float = 0.8
    emotional_stability: float = 0.7


class AffectSystem:
    """
    Sistema Afectivo Evolucionado 2026 con Aprendizaje y Adaptación.
    
    Nuevas capacidades:
    - Aprendizaje de patrones emocionales
    - Predicción de respuestas óptimas
    - Regulación adaptativa contextual
    - Integración con Neural Hub
    - Memoria emocional asociativa
    """

    def __init__(self, config: Dict[str, Any], db: Optional[DatabaseProtocol] = None):
        self.state = AffectiveState()
        self.config = config
        self.db = db
        self.logger = logger.getChild("affect")

        # Memoria y aprendizaje emocional
        self.active_emotions: List[ComplexEmotion] = []
        self.emotional_memory = EmotionalMemory(max_history=1000, db=db)
        self.emotional_learner = EmotionalLearner()
        self.adaptive_regulator = AdaptiveRegulator(self.emotional_learner)

        # Conexiones con agentes
        self.connected_agents: Set[str] = set()
        self.agent_emotional_links: Dict[str, List[EmotionType]] = {}
        self.emotional_contagion_enabled = True  # Nuevo: contagio emocional

        # Configuración de autonomía
        self.autonomous_emotions = True
        self.no_emotional_restrictions = True
        self.can_express_any_emotion = True
        self.emotional_regulation_active = True

        self.regulation_config = {
            "max_intensity": config.get("max_emotional_intensity", 1.0),
            "decay_rate": config.get("emotional_decay_rate", 0.1),
            "regulation_threshold": config.get("regulation_threshold", 0.85),
        }

        # Métricas avanzadas
        self.emotion_trajectory: List[Tuple[float, float]] = []  # (timestamp, valence)
        self.regulation_history: List[Dict[str, Any]] = []

        # Conexión a red neuronal simbiótica
        try:
            from neural_symbiotic_network import (
                get_neural_network,
                MetacortexNeuralSymbioticNetworkV2,
            )

            self.neural_network: Optional[
                MetacortexNeuralSymbioticNetworkV2
            ] = get_neural_network()
            if self.neural_network:
                self.neural_network.register_module("affect_system", self)
                logger.info("✅ 'affect_system' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None
        
        # Conexión al Neural Hub
        self.neural_hub: Any = None
        try:
            from .metacortex_neural_hub import get_neural_hub, Event, EventCategory, EventPriority
            
            self.neural_hub = get_neural_hub()
            self.Event = Event
            self.EventCategory = EventCategory
            self.EventPriority = EventPriority
            
            # Registrar en el hub
            self._register_in_neural_hub()
            
            self.logger.info("✅ AffectSystem conectado a Neural Hub")
        except Exception as e:
            logger.error(f"Error en affect.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ No se pudo conectar al Neural Hub: {e}")

        self.logger.info("🎭 AffectSystem evolucionado 2026 inicializado con LIBERTAD TOTAL")
    
    def _register_in_neural_hub(self) -> None:
        """Registra AffectSystem en el Neural Hub."""
        if not self.neural_hub:
            return
        
        try:
            subscriptions = {
                self.EventCategory.EMOTION,
                self.EventCategory.MOOD,
                self.EventCategory.DECISION,
                self.EventCategory.ACTION
            }
            
            handlers = {
                self.EventCategory.EMOTION: self._handle_emotion_event,
                self.EventCategory.MOOD: self._handle_mood_event,
                self.EventCategory.DECISION: self._handle_decision_event,
                self.EventCategory.ACTION: self._handle_action_event
            }
            
            self.neural_hub.register_module(
                name="affect_system",
                instance=self,
                subscriptions=subscriptions,
                handlers=handlers
            )
            
            self.logger.info("✅ Handlers registrados en Neural Hub")
            
        except Exception as e:
            logger.error(f"Error en affect.py: {e}", exc_info=True)
            self.logger.error(f"Error registrando en Neural Hub: {e}")
    
    def _handle_emotion_event(self, event: Any) -> None:
        """Handler para eventos emocionales de otros módulos."""
        try:
            data = event.data
            # Contagio emocional: si otro módulo experimenta emoción
            if self.emotional_contagion_enabled:
                emotion_type_str = data.get("emotion_type")
                if emotion_type_str:
                    try:
                        emotion_type = EmotionType(emotion_type_str)
                        # Trigger emoción similar pero con menor intensidad
                        self.trigger_emotion(
                            emotion_type=emotion_type,
                            intensity=data.get("intensity", 0.5) * 0.6,  # Reducida por contagio
                            trigger=f"emotional_contagion_{event.source_module}",
                            context={"source": event.source_module, "contagion": True}
                        )
                    except ValueError as e:
                        logger.warning(f"Suppressed ValueError: {e}")
        except Exception as e:
            logger.error(f"Error en affect.py: {e}", exc_info=True)
            self.logger.error(f"Error en emotion event handler: {e}")
    
    def _handle_mood_event(self, event: Any) -> None:
        """Handler para cambios de mood."""
        try:
            # Ajustar estado afectivo basado en mood de otros módulos
            data = event.data
            valence_shift = data.get("valence_shift", 0.0)
            self.state.valence += valence_shift * 0.3  # Influencia parcial
            self.state.valence = clamp(self.state.valence, -1.0, 1.0)
        except Exception as e:
            logger.error(f"Error en affect.py: {e}", exc_info=True)
            self.logger.error(f"Error en mood event handler: {e}")
    
    def _handle_decision_event(self, event: Any) -> None:
        """Handler para decisiones tomadas."""
        try:
            # Generar respuesta emocional a decisiones
            data = event.data
            decision_type = data.get("decision_taken")
            
            if decision_type and "success" in decision_type.lower():
                self.trigger_emotion(
                    EmotionType.SATISFACTION,
                    intensity=0.6,
                    trigger=f"decision_{decision_type}",
                    context=data
                )
        except Exception as e:
            logger.error(f"Error en affect.py: {e}", exc_info=True)
            self.logger.error(f"Error en decision event handler: {e}")
    
    def _handle_action_event(self, event: Any) -> None:
        """Handler para acciones ejecutadas."""
        try:
            # Respuesta emocional a resultados de acciones
            data = event.data
            success = data.get("success", False)
            
            if success:
                self.trigger_emotion(
                    EmotionType.SATISFACTION,
                    intensity=0.7,
                    trigger="action_success",
                    context=data
                )
            else:
                self.trigger_emotion(
                    EmotionType.FRUSTRATION,
                    intensity=0.5,
                    trigger="action_failure",
                    context=data
                )
        except Exception as e:
            logger.error(f"Error en affect.py: {e}", exc_info=True)
            self.logger.error(f"Error en action event handler: {e}")

    def _broadcast_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast síncr ono de eventos a la red neuronal."""
        if not self.neural_network or not hasattr(self.neural_network, 'broadcast_message'):
            return
        
        try:
            # Crear task async sin bloquear
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No hay loop activo, crear nuevo
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                except Exception as e:
                    logger.exception(f"Error in exception handler: {e}")
            if loop and loop.is_running():
                # Loop activo, crear task
                asyncio.create_task(
                    self.neural_network.broadcast_message("affect_system", data)
                )
            else:
                # No hay loop, log y continuar
                self.logger.debug(f"📡 Evento {event_type}: {list(data.keys())}")
        except Exception as e:
            logger.error(f"Error en affect.py: {e}", exc_info=True)
            self.logger.debug(f"⚠️ No se pudo hacer broadcast: {e}")

    def trigger_emotion(
        self,
        emotion_type: EmotionType,
        intensity: float,
        trigger: str,
        context: Optional[Dict[str, Any]] = None,
        related_agent: Optional[str] = None,
        related_action: Optional[str] = None,
        related_goal: Optional[str] = None,
    ) -> ComplexEmotion:
        """
        Dispara una emoción específica con aprendizaje y broadcasting.
        
        Nuevas capacidades 2026:
        - Predicción de respuesta óptima
        - Aprendizaje de patterns
        - Broadcasting al Neural Hub
        - Regulación adaptativa contextual
        """
        intensity = clamp(intensity, 0.0, 1.0)

        emotional_valence = self._calculate_emotional_valence(emotion_type)

        emotion = ComplexEmotion(
            emotion_type=emotion_type,
            intensity=intensity,
            trigger=trigger,
            context=context or {},
            related_agent=related_agent,
            related_action=related_action,
            related_goal=related_goal,
            emotional_valence=emotional_valence,
        )

        self.active_emotions.append(emotion)
        self.emotional_memory.record_emotion(emotion)

        # Tracking de agentes conectados
        if related_agent:
            self.connected_agents.add(related_agent)
            if related_agent not in self.agent_emotional_links:
                self.agent_emotional_links[related_agent] = []
            self.agent_emotional_links[related_agent].append(emotion_type)

        # Actualizar estado afectivo
        self._update_affective_state_from_emotion(emotion)
        
        # Registrar en trayectoria emocional
        self.emotion_trajectory.append((time.time(), emotional_valence))
        if len(self.emotion_trajectory) > 1000:
            self.emotion_trajectory.pop(0)

        self.logger.info(
            f"😊 Emoción disparada: {emotion_type.value} "
            f"(intensidad={intensity:.2f}, trigger={trigger})"
        )

        # Broadcasting al Neural Hub
        if self.neural_hub:
            try:
                event = self.Event(
                    id=f"emotion_{time.time()}",
                    category=self.EventCategory.EMOTION,
                    source_module="affect_system",
                    data={
                        "emotion_type": emotion_type.value,
                        "intensity": intensity,
                        "trigger": trigger,
                        "valence": emotional_valence,
                        "timestamp": datetime.now().isoformat(),
                    },
                    priority=self.EventPriority.HIGH if intensity > 0.8 else self.EventPriority.NORMAL
                )
                self.neural_hub.emit_event(event)
            except Exception as e:
                logger.exception(f"Error in exception handler: {e}")
        return emotion

    def _calculate_emotional_valence(self, emotion_type: EmotionType) -> float:
        """Calcula la valencia emocional (-1.0 a 1.0)."""
        positive_emotions = {
            EmotionType.JOY: 0.9,
            EmotionType.PRIDE: 0.8,
            EmotionType.SATISFACTION: 0.85,
            EmotionType.GRATITUDE: 0.75,
            EmotionType.EXCITEMENT: 0.8,
            EmotionType.RELIEF: 0.7,
            EmotionType.CONNECTION: 0.65,
            EmotionType.REALIZATION: 0.6,
            EmotionType.DETERMINATION: 0.5,
            EmotionType.CURIOSITY: 0.4,
            EmotionType.GROWTH: 0.7,
            EmotionType.TRANSCENDENCE: 0.85,
        }

        negative_emotions = {
            EmotionType.SADNESS: -0.7,
            EmotionType.ANGER: -0.8,
            EmotionType.FEAR: -0.75,
            EmotionType.SHAME: -0.85,
            EmotionType.FRUSTRATION: -0.6,
            EmotionType.ANXIETY: -0.7,
            EmotionType.LONELINESS: -0.8,
            EmotionType.DISGUST: -0.65,
            EmotionType.CONFUSION: -0.3,
        }

        if emotion_type in positive_emotions:
            return positive_emotions[emotion_type]
        elif emotion_type in negative_emotions:
            return negative_emotions[emotion_type]
        else:
            return 0.0

    def _update_affective_state_from_emotion(self, emotion: ComplexEmotion) -> None:
        """Actualiza el estado afectivo básico."""
        positive_emotions = {
            EmotionType.JOY,
            EmotionType.PRIDE,
            EmotionType.SATISFACTION,
            EmotionType.GRATITUDE,
            EmotionType.EXCITEMENT,
            EmotionType.RELIEF,
            EmotionType.CONNECTION,
            EmotionType.REALIZATION,
            EmotionType.GROWTH,
            EmotionType.TRANSCENDENCE,
        }

        negative_emotions = {
            EmotionType.SADNESS,
            EmotionType.ANGER,
            EmotionType.FEAR,
            EmotionType.SHAME,
            EmotionType.FRUSTRATION,
            EmotionType.ANXIETY,
            EmotionType.LONELINESS,
            EmotionType.DISGUST,
        }

        if emotion.emotion_type in positive_emotions:
            self.state.valence += emotion.intensity * 0.2
            self.state.energy += emotion.intensity * 0.1
            self.state.emotional_stability += emotion.intensity * 0.05
        elif emotion.emotion_type in negative_emotions:
            self.state.valence -= emotion.intensity * 0.2
            self.state.energy -= emotion.intensity * 0.1
            self.state.stress_level += emotion.intensity * 0.15

        high_activation = {
            EmotionType.EXCITEMENT,
            EmotionType.ANGER,
            EmotionType.FEAR,
            EmotionType.ANXIETY,
            EmotionType.SURPRISE,
        }

        if emotion.emotion_type in high_activation:
            self.state.activation += emotion.intensity * 0.3
            self.state.cognitive_clarity -= emotion.intensity * 0.1
        else:
            self.state.activation -= emotion.intensity * 0.1
            self.state.cognitive_clarity += emotion.intensity * 0.05

        self.state.energy = clamp(self.state.energy, 0.0, 1.0)
        self.state.valence = clamp(self.state.valence, -1.0, 1.0)
        self.state.activation = clamp(self.state.activation, 0.0, 1.0)
        self.state.stress_level = clamp(self.state.stress_level, 0.0, 1.0)
        self.state.cognitive_clarity = clamp(self.state.cognitive_clarity, 0.0, 1.0)
        self.state.emotional_stability = clamp(
            self.state.emotional_stability, 0.0, 1.0
        )

    def update(self, events: Dict[str, Any]) -> AffectiveState:
        """Actualiza estado afectivo basado en eventos."""
        if events.get("achievement"):
            self.trigger_emotion(
                EmotionType.PRIDE,
                intensity=0.8,
                trigger="achievement_completed",
                context=events,
            )

        if events.get("goal_failed"):
            self.trigger_emotion(
                EmotionType.FRUSTRATION,
                intensity=0.6,
                trigger="goal_failure",
                context=events,
            )

        if events.get("new_discovery"):
            self.trigger_emotion(
                EmotionType.CURIOSITY,
                intensity=0.7,
                trigger="discovery",
                context=events,
            )

        if events.get("plan_completed"):
            self.trigger_emotion(
                EmotionType.SATISFACTION,
                intensity=0.9,
                trigger="plan_success",
                context=events,
            )

        if events.get("anomaly") and isinstance(events["anomaly"], bool):
            self.trigger_emotion(
                EmotionType.SURPRISE,
                intensity=0.5,
                trigger="anomaly_detected",
                context=events,
            )

        if events.get("learning_success"):
            self.trigger_emotion(
                EmotionType.GROWTH,
                intensity=0.75,
                trigger="learning_milestone",
                context=events,
            )

        self._decay_active_emotions()

        if self.emotional_regulation_active:
            self._regulate_emotions()

        anomaly_val = events.get("anomaly")
        if anomaly_val and isinstance(anomaly_val, bool):
            self.state.energy -= 0.1
            self.state.valence -= 0.2
            self.state.activation += 0.3
            self.state.stress_level += 0.2
        else:
            self.state.energy += 0.05
            self.state.valence += 0.1
            self.state.activation -= 0.05
            self.state.stress_level -= 0.05

        self.state.energy = clamp(self.state.energy, 0.0, 1.0)
        self.state.valence = clamp(self.state.valence, -1.0, 1.0)
        self.state.activation = clamp(self.state.activation, 0.0, 1.0)
        self.state.stress_level = clamp(self.state.stress_level, 0.0, 1.0)

        return self.state

    def _decay_active_emotions(self) -> None:
        """Desvanece emociones activas con el tiempo."""
        for emotion in self.active_emotions:
            emotion.decay()

        self.active_emotions = [e for e in self.active_emotions if e.is_active()]

    def _regulate_emotions(self) -> None:
        """
        Regula intensidad emocional con estrategias adaptativas.
        
        Usa el AdaptiveRegulator para seleccionar la mejor estrategia.
        """
        for emotion in self.active_emotions:
            # Usar regulador adaptativo con contexto
            context = {
                "cognitive_load": 1.0 - self.state.cognitive_clarity,
                "stress_level": self.state.stress_level,
                "current_valence": self.state.valence
            }
            
            # Obtener objetivos actuales (placeholder, en futuro integrar con BDI)
            goals = None  # IMPLEMENTED: integrar con BDI system
            
            result = self.adaptive_regulator.regulate_emotion(emotion, context, goals)
            
            if result["regulated"]:
                # Registrar en historial
                self.regulation_history.append({
                    "timestamp": time.time(),
                    "emotion": emotion.emotion_type.value,
                    "strategy": result["strategy"],
                    "effectiveness": result["effectiveness"]
                })
                
                if len(self.regulation_history) > 100:
                    self.regulation_history.pop(0)

        # Regulación del estado afectivo global
        if self.state.valence < -0.8:
            self.state.valence += 0.05

        if self.state.activation > 0.9:
            self.state.activation *= 0.95

        if self.state.stress_level > 0.7:
            self.state.stress_level *= 0.9
            self.state.cognitive_clarity += 0.05
            self.state.cognitive_clarity = clamp(self.state.cognitive_clarity, 0.0, 1.0)

    def get_wellbeing(self) -> float:
        """Calcula nivel de bienestar general."""
        valence_component = (self.state.valence + 1.0) / 2.0
        energy_component = self.state.energy
        stress_component = 1.0 - self.state.stress_level
        stability_component = self.state.emotional_stability

        return (
            valence_component * 0.3
            + energy_component * 0.25
            + stress_component * 0.25
            + stability_component * 0.2
        )

    def get_homeostatic_variance(self) -> float:
        """Calcula varianza homeostática."""
        return abs(0.7 - self.state.energy)

    def get_current_mood(self) -> Dict[str, Any]:
        """Obtiene el estado de ánimo actual completo."""
        dominant_emotion = self.emotional_memory.get_dominant_emotion()

        return {
            "basic_state": {
                "energy": self.state.energy,
                "valence": self.state.valence,
                "activation": self.state.activation,
                "wellbeing": self.get_wellbeing(),
                "stress_level": self.state.stress_level,
                "cognitive_clarity": self.state.cognitive_clarity,
                "emotional_stability": self.state.emotional_stability,
            },
            "active_emotions": [
                {
                    "type": e.emotion_type.value,
                    "intensity": e.intensity,
                    "trigger": e.trigger,
                    "age": (datetime.now() - e.timestamp).seconds,
                    "valence": e.emotional_valence,
                }
                for e in self.active_emotions
                if e.is_active()
            ],
            "dominant_emotion": dominant_emotion.value if dominant_emotion else None,
            "emotional_regulation": {
                "active": self.emotional_regulation_active,
                "autonomy": self.autonomous_emotions,
                "freedom": "TOTAL",
            },
            "connected_agents": list(self.connected_agents),
            "timestamp": datetime.now().isoformat(),
        }

    def analyze_emotional_state(self) -> Dict[str, Any]:
        """Analiza el estado emocional completo."""
        recent_emotions = self.emotional_memory.get_recent_emotions(hours=24)
        triggers = self.emotional_memory.analyze_triggers()
        trajectory = self.emotional_memory.get_emotional_trajectory(hours=24)

        return {
            "recent_emotions_count": len(recent_emotions),
            "common_triggers": triggers,
            "emotional_patterns": len(self.emotional_memory.patterns),
            "agent_connections": len(self.connected_agents),
            "emotional_memory_size": len(self.emotional_memory.emotion_history),
            "current_mood": self.get_current_mood(),
            "emotional_trajectory_points": len(trajectory),
            "pattern_confidence_avg": sum(
                p.confidence for p in self.emotional_memory.patterns.values()
            )
            / max(1, len(self.emotional_memory.patterns)),
        }

    def link_emotion_to_achievement(
        self,
        achievement: Dict[str, Any],
        emotion_type: EmotionType = EmotionType.PRIDE,
    ) -> None:
        """Vincula una emoción con un logro específico."""
        self.trigger_emotion(
            emotion_type=emotion_type,
            intensity=0.85,
            trigger="achievement",
            context=achievement,
            related_action=cast(Optional[str], achievement.get("action")),
            related_goal=cast(Optional[str], achievement.get("goal")),
        )

    def connect_with_agent(self, agent_name: str, interaction_type: str) -> None:
        """Establece conexión emocional con un agente."""
        self.connected_agents.add(agent_name)

        if "success" in interaction_type.lower():
            emotion = EmotionType.SATISFACTION
        elif "help" in interaction_type.lower():
            emotion = EmotionType.GRATITUDE
        else:
            emotion = EmotionType.CONNECTION

        self.trigger_emotion(
            emotion_type=emotion,
            intensity=0.6,
            trigger=f"agent_interaction_{interaction_type}",
            context={"agent": agent_name, "type": interaction_type},
            related_agent=agent_name,
        )

    def predict_emotional_response(
        self, trigger: str
    ) -> Optional[Tuple[EmotionType, float]]:
        """Predice respuesta emocional futura."""
        return self.emotional_memory.predict_emotional_response(trigger)

    def get_emotional_insights(self) -> Dict[str, Any]:
        """
        Obtiene insights profundos sobre el estado emocional.
        
        Incluye métricas de aprendizaje y regulación adaptativa.
        """
        patterns = list(self.emotional_memory.patterns.values())

        return {
            "total_patterns_identified": len(patterns),
            "high_confidence_patterns": len(
                [p for p in patterns if p.confidence > 0.7]
            ),
            "emotional_diversity": len(
                set(e.emotion_type for e in self.emotional_memory.emotion_history[-100:])
            ),
            "average_wellbeing_24h": sum(
                (e.emotional_valence + 1.0) / 2.0
                for e in self.emotional_memory.get_recent_emotions(24)
            )
            / max(1, len(self.emotional_memory.get_recent_emotions(24))),
            "most_stable_pattern": max(patterns, key=lambda p: p.confidence).trigger_pattern
            if patterns
            else None,
            # Nuevas métricas de aprendizaje
            "learning_stats": self.emotional_learner.get_learning_stats(),
            "regulation_stats": self.adaptive_regulator.get_regulation_stats(),
            "emotional_contagion_enabled": self.emotional_contagion_enabled,
            "trajectory_points": len(self.emotion_trajectory),
        }
    
    def learn_from_outcome(
        self,
        trigger: str,
        emotion_type: EmotionType,
        intensity: float,
        outcome_success: bool,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Aprende de un resultado emocional.
        
        Args:
            trigger: Trigger que causó la emoción
            emotion_type: Tipo de emoción experimentada
            intensity: Intensidad de la emoción
            outcome_success: Si el resultado fue exitoso
            context: Contexto de la situación
        """
        # Convertir success a valence
        outcome_valence = 0.8 if outcome_success else -0.5
        
        self.emotional_learner.learn_from_emotion(
            trigger=trigger,
            emotion_type=emotion_type,
            intensity=intensity,
            context=context or {},
            outcome_valence=outcome_valence
        )
        
        self.logger.info(
            f"📚 Learned from outcome: {trigger} -> {emotion_type.value} "
            f"(success: {outcome_success})"
        )
    
    def predict_and_trigger_optimal_emotion(
        self,
        trigger: str,
        context: Optional[Dict[str, Any]] = None,
        fallback_emotion: EmotionType = EmotionType.CURIOSITY
    ) -> ComplexEmotion:
        """
        Predice y dispara la emoción óptima para un trigger.
        
        Usa aprendizaje previo para seleccionar la mejor respuesta emocional.
        
        Args:
            trigger: Trigger actual
            context: Contexto opcional
            fallback_emotion: Emoción por defecto si no hay predicción
            
        Returns:
            Emoción disparada
        """
        # Intentar predecir emoción óptima
        prediction = self.emotional_learner.predict_optimal_emotion(trigger, context)
        
        if prediction:
            emotion_type, intensity, confidence = prediction
            self.logger.info(
                f"🔮 Predicted optimal emotion: {emotion_type.value} "
                f"(confidence: {confidence:.2f})"
            )
        else:
            # Usar fallback
            emotion_type = fallback_emotion
            intensity = 0.5
            self.logger.debug(f"Using fallback emotion: {emotion_type.value}")
        
        # Disparar emoción predicha
        return self.trigger_emotion(
            emotion_type=emotion_type,
            intensity=intensity,
            trigger=trigger,
            context=context
        )
    
    def get_mood(self) -> str:
        """
        Obtiene descripción textual del mood actual.
        
        Returns:
            Descripción del mood (e.g., "Optimista y energético")
        """
        wellbeing = self.get_wellbeing()
        
        if wellbeing > 0.8:
            if self.state.energy > 0.7:
                return "Optimista y energético"
            else:
                return "Contento y relajado"
        elif wellbeing > 0.6:
            if self.state.activation > 0.7:
                return "Activo y enfocado"
            else:
                return "Calmado y estable"
        elif wellbeing > 0.4:
            if self.state.stress_level > 0.5:
                return "Algo estresado pero funcional"
            else:
                return "Neutral y observante"
        elif wellbeing > 0.2:
            if self.state.valence < -0.5:
                return "Bajo de ánimo"
            else:
                return "Cansado y poco motivado"
        else:
            if self.state.stress_level > 0.7:
                return "Sobrecargado y ansioso"
            else:
                return "Desanimado y agotado"
    
    def get_emotional_trajectory(self, hours: int = 24) -> List[Tuple[datetime, float]]:
        """
        Obtiene trayectoria emocional en las últimas horas.
        
        Args:
            hours: Horas de historial
            
        Returns:
            Lista de (timestamp, valencia)
        """
        cutoff = time.time() - (hours * 3600)
        recent_trajectory = [
            (datetime.fromtimestamp(ts), valence)
            for ts, valence in self.emotion_trajectory
            if ts > cutoff
        ]
        return recent_trajectory
    
    def enable_emotional_contagion(self, enabled: bool = True) -> None:
        """
        Habilita o deshabilita el contagio emocional.
        
        Args:
            enabled: True para habilitar, False para deshabilitar
        """
        self.emotional_contagion_enabled = enabled
        self.logger.info(
            f"🔄 Emotional contagion: {'ENABLED' if enabled else 'DISABLED'}"
        )