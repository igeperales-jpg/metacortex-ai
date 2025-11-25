#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol
from dataclasses import dataclass, field

from .emotional_models import Emotion, EmotionalTrait
from .metacortex_neural_hub import MetacortexNeuralHub

# Configuración del logger
logger = logging.getLogger(__name__)


def clip(value: float, min_val: float, max_val: float) -> float:
    """Una implementación simple de clip sin numpy."""
    return max(min_val, min(value, max_val))


class DatabaseProtocol(Protocol):
    """Protocolo para un conector de base de datos genérico."""
    def save(self, data: Any) -> None:
        ...
    def load(self) -> Any:
        ...


@dataclass
class EmotionalState:
    """Representa el estado emocional actual del agente."""
    emotion: str = "neutral"
    intensity: float = 0.5
    valence: float = 0.0
    arousal: float = 0.5
    activation: float = 0.5  # Añadido para compatibilidad
    mood: str = "calm"
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def energy(self) -> float:
        return self.intensity

    def update_emotion(self, new_emotion: Emotion, intensity: float):
        """Actualiza la emoción y su intensidad."""
        self.emotion = new_emotion.value
        self.intensity = clip(intensity, 0.0, 1.0)
        self.last_update = datetime.now()

    def dampen(self, amount: float):
        """Reduce la intensidad de la emoción actual."""
        self.intensity = clip(self.intensity - amount, 0.0, 1.0)

    def get_state(self) -> Dict[str, Any]:
        """Devuelve el estado emocional actual."""
        return {
            "emotion": self.emotion,
            "intensity": self.intensity,
            "last_update": self.last_update.isoformat(),
        }

    def __repr__(self) -> str:
        return f"EmotionalState(emotion={self.emotion}, intensity={self.intensity:.2f})"


class Personality:
    """
    Define la personalidad del agente usando el modelo de los Cinco Grandes (OCEAN).
    """

    def __init__(self, traits: Optional[Dict[EmotionalTrait, float]] = None):
        """
        Inicializa la personalidad.
        """
        if traits:
            self.traits: Dict[EmotionalTrait, float] = {trait: clip(value, 0.0, 1.0) for trait, value in traits.items()}
        else:
            self.traits: Dict[EmotionalTrait, float] = self._generate_random_traits()

        self._validate_traits()

    def _validate_traits(self):
        """Valida que todos los rasgos necesarios estén presentes."""
        for trait in EmotionalTrait:
            if trait not in self.traits:
                raise ValueError(f"Missing required personality trait: {trait.value}")

    def _generate_random_traits(self) -> Dict[EmotionalTrait, float]:
        """Genera un conjunto de rasgos de personalidad aleatorios."""
        return {trait: random.uniform(0.0, 1.0) for trait in EmotionalTrait}

    def get_trait(self, trait: EmotionalTrait) -> float:
        """
        Obtiene el valor de un rasgo de personalidad específico.
        """
        return self.traits.get(trait, 0.5)

    def get_all_traits(self) -> Dict[str, float]:
        """Devuelve todos los rasgos de personalidad."""
        return {trait.value: value for trait, value in self.traits.items()}

    def __repr__(self) -> str:
        traits_str = ", ".join(f"{trait.value}={value:.2f}" for trait, value in self.traits.items())
        return f"Personality({traits_str})"


class EmotionalMemory:
    """
    Almacena y recupera eventos emocionales significativos.
    """
    def __init__(self, max_history: int = 1000, db: Optional[DatabaseProtocol] = None):
        self.events: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.db = db
        if self.db:
            self.events = self.db.load() or []

    def add_event(self, event: Dict[str, Any]):
        """
        Añade un nuevo evento emocional a la memoria.
        """
        if not all(k in event for k in ["emotion", "intensity", "timestamp"]):
            logger.warning("Evento emocional descartado por falta de claves requeridas.")
            return

        try:
            # Asegurarse de que la emoción es un miembro del Enum si es un string
            if isinstance(event['emotion'], str):
                event['emotion'] = Emotion(event['emotion'])
        except ValueError:
            logger.warning(f"Evento emocional descartado por emoción desconocida: {event.get('emotion')}")
            return

        self.events.append(event)
        if len(self.events) > self.max_history:
            self.events.pop(0)
        if self.db:
            self.db.save(self.events)

    def get_recent_events(self, period: timedelta) -> List[Dict[str, Any]]:
        """Obtiene eventos dentro de un período de tiempo reciente."""
        now = datetime.now()
        return [event for event in self.events if now - event.get("timestamp", now) <= period]

    def get_events_by_emotion(self, emotion: Emotion) -> List[Dict[str, Any]]:
        """Filtra eventos por una emoción específica."""
        return [event for event in self.events if event.get("emotion") == emotion]

    def get_emotional_summary(self, period: timedelta) -> Dict[str, float]:
        """
        Calcula la prevalencia de cada emoción en un período de tiempo.
        """
        recent_events = self.get_recent_events(period)
        if not recent_events:
            return {}

        summary: Dict[Emotion, float] = {emotion: 0.0 for emotion in Emotion}
        total_intensity = 0.0

        for event in recent_events:
            emotion = event.get("emotion")
            intensity = event.get("intensity", 0.0)
            if isinstance(emotion, Emotion):
                summary[emotion] += intensity
                total_intensity += intensity

        if total_intensity == 0:
            return {emotion.value: 0.0 for emotion in Emotion}

        return {emotion.value: (intensity / total_intensity) for emotion, intensity in summary.items()}


class EmotionalLearner:
    """
    Aprende y adapta las respuestas emocionales del agente.
    """
    def __init__(self, personality: Personality, neural_hub: MetacortexNeuralHub):
        self.personality = personality
        self.neural_hub = neural_hub
        self.emotional_patterns: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    async def learn_from_feedback(self, feedback: Dict[str, Any]):
        """
        Aprende de la retroalimentación para ajustar las respuestas emocionales.
        """
        self.logger.info(f"Aprendizaje emocional a partir de feedback: {feedback}")
        await self.neural_hub.publish("emotional_learning", {"updated_patterns": self.emotional_patterns})

    def get_learned_patterns(self) -> Dict[str, Any]:
        return self.emotional_patterns


class AdaptiveRegulator:
    """
    Regula las emociones para mantener la homeostasis y la estabilidad cognitiva.
    """
    def __init__(self, emotional_state: EmotionalState, personality: Personality, neural_hub: MetacortexNeuralHub, emotional_learner: EmotionalLearner):
        self.emotional_state = emotional_state
        self.personality = personality
        self.neural_hub = neural_hub
        self.emotional_learner = emotional_learner
        self.logger = logging.getLogger(__name__)

    async def regulate(self):
        """
        Ejecuta el ciclo de regulación emocional.
        """
        neuroticism = self.personality.get_trait(EmotionalTrait.NEUROTICISM)
        if self.emotional_state.intensity > (0.8 - neuroticism * 0.2):
            self.logger.info(f"Regulando emoción intensa: {self.emotional_state.current_emotion.value}")
            self.emotional_state.dampen(0.1 + neuroticism * 0.1)
            await self.neural_hub.publish("emotion_regulation", {"action": "dampen", "new_intensity": self.emotional_state.intensity})

    def get_homeostatic_variance(self) -> float:
        """
        Calcula una métrica de cuán lejos está el sistema de su equilibrio emocional.
        """
        base_variance = (self.emotional_state.intensity - 0.5) ** 2
        neuroticism = self.personality.get_trait(EmotionalTrait.NEUROTICISM)
        variance = base_variance * (1 + neuroticism)
        return variance


class AffectSystem:
    """
    Sistema central que gestiona el estado afectivo y la personalidad del agente.
    """
    def __init__(self, neural_hub: MetacortexNeuralHub, personality_traits: Optional[Dict[EmotionalTrait, float]] = None):
        self.logger = logging.getLogger(__name__)
        self.neural_hub = neural_hub
        self.personality = Personality(traits=personality_traits)
        self.emotional_state = EmotionalState()
        self.emotional_memory = EmotionalMemory()
        self.emotional_learner = EmotionalLearner(self.personality, self.neural_hub)
        self.regulator = AdaptiveRegulator(self.emotional_state, self.personality, self.neural_hub, self.emotional_learner)
        self.active_emotions: List[Emotion] = []
        self.emotional_patterns: Dict[str, Any] = {}

        self._setup_event_listeners()

    @property
    def state(self) -> EmotionalState:
        """Propiedad para compatibilidad, devuelve el estado emocional."""
        return self.emotional_state

    def _setup_event_listeners(self):
        self.neural_hub.subscribe("perception_event", self.handle_perception_event)
        self.neural_hub.subscribe("cognitive_feedback", self.handle_feedback_event)

    async def update(self, events: List[Dict[str, Any]]):
        """
        Procesa una lista de eventos afectivos.
        """
        for event in events:
            # Este es un manejador simplificado. Se puede expandir.
            await self.handle_perception_event(event)

    async def handle_perception_event(self, event_data: Dict[str, Any]):
        """
        Procesa un evento de percepción y genera una respuesta emocional.
        """
        valence = event_data.get("valence", 0.0)
        arousal = event_data.get("arousal", 0.0)

        if valence > 0.5 and arousal > 0.5:
            emotion = Emotion.JOY
        elif valence < -0.5 and arousal > 0.5:
            emotion = Emotion.ANGER
        elif valence < -0.5 and arousal < 0.5:
            emotion = Emotion.SADNESS
        else:
            emotion = Emotion.NEUTRAL
        
        intensity = (abs(valence) + arousal) / 2
        self.emotional_state.update_emotion(emotion, intensity)

        event = {
            "emotion": emotion,
            "intensity": intensity,
            "timestamp": datetime.now(),
            "source": event_data.get("source"),
        }
        self.emotional_memory.add_event(event)
        await self.neural_hub.publish("emotional_response", self.emotional_state.get_state())

    async def handle_feedback_event(self, feedback: Dict[str, Any]):
        """
        Maneja el feedback cognitivo para el aprendizaje emocional.
        """
        await self.emotional_learner.learn_from_feedback(feedback)

    async def run_affective_cycle(self):
        """
        Ejecuta un ciclo completo del sistema afectivo.
        """
        await self.regulator.regulate()
        self.logger.debug(f"Ciclo afectivo completado. Estado actual: {self.emotional_state}")

    def get_current_emotional_state(self) -> Dict[str, Any]:
        """
        Devuelve el estado emocional actual del agente.
        """
        return self.emotional_state.get_state()

    def get_personality_traits(self) -> Dict[str, float]:
        """
        Devuelve los rasgos de personalidad del agente.
        """
        return self.personality.get_all_traits()

    def get_emotional_insights(self, period: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """
        Genera un análisis profundo del estado emocional reciente.
        """
        summary = self.emotional_memory.get_emotional_summary(period)
        variance = self.regulator.get_homeostatic_variance()
        
        insights = {
            "summary": summary,
            "homeostatic_variance": variance,
            "learned_patterns": self.emotional_learner.get_learned_patterns()
        }
        
        self.logger.info(f"Generando insights emocionales: {insights}")
        return insights

    def get_wellbeing(self) -> float:
        """
        Calcula un score de bienestar basado en el estado emocional actual.
        
        Returns:
            float: Score de bienestar entre 0.0 (muy mal) y 1.0 (muy bien)
        """
        state = self.emotional_state.get_state()
        
        # Calcular bienestar basado en valencia y arousal
        valence = state.get("valence", 0.0)
        arousal = state.get("arousal", 0.5)
        
        # Bienestar alto = valencia positiva + arousal moderado
        # Bienestar bajo = valencia negativa
        wellbeing = (valence + 1.0) / 2.0  # Normalizar de [-1, 1] a [0, 1]
        
        # Penalizar arousal muy alto o muy bajo
        arousal_penalty = abs(arousal - 0.5) * 0.2
        wellbeing = max(0.0, min(1.0, wellbeing - arousal_penalty))
        
        return wellbeing

# Ejemplo de uso (si se ejecuta el archivo directamente)
async def main():
    hub = MetacortexNeuralHub()
    hub.start()
    affect_system = AffectSystem(hub)

    perception_event = {"valence": 0.7, "arousal": 0.8, "source": "user_praise"}
    await affect_system.handle_perception_event(perception_event)

    print(f"Estado emocional current: {affect_system.get_current_emotional_state()}")

    await affect_system.run_affective_cycle()

    print(f"Insights emocionales: {affect_system.get_emotional_insights()}")

    feedback = {"reward": 0.9, "context": "user_praise_response"}
    await affect_system.handle_feedback_event(feedback)

    hub.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())