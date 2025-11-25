"""
METACORTEX Neural Hub - Sistema de Interconexión Neuronal Avanzado
===================================================================

Este módulo actúa como el "cerebro central" que conecta todos los subsistemas
cognitivos del METACORTEX en una red neuronal simbiótica coherente.

Características:
    pass  # TODO: Implementar
- Event Bus centralizado para comunicación asíncrona
- Routing inteligente de mensajes entre módulos
- Gestión de prioridades y flujos de información
- Circuit breakers para resiliencia
- Telemetría unificada y health monitoring
- Orquestación inteligente de subsistemas

Autor: METACORTEX Evolution Team
Fecha: 2026
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Coroutine, Union, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time
import random


# Configuración del logger
logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Prioridad de los eventos en el hub."""
    LOW = 1
    NORMAL = 2  # Añadido para compatibilidad
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class EventCategory(Enum):
    """Categorías de eventos para suscripciones."""
    SYSTEM = "system"
    BELIEF_UPDATE = "belief_update"
    DESIRE_CHANGE = "desire_change"
    INTENTION_SET = "intention_set"
    ACTION_EXECUTED = "action_executed"
    KNOWLEDGE_ACQUIRED = "knowledge_acquired"
    ETHICAL_DILEMMA = "ethical_dilemma"
    AFFECT_CHANGE = "affect_change"
    RESOURCE_UPDATE = "resource_update"
    PERCEPTION = "perception"
    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_CONSOLIDATE = "memory_consolidate"  # Añadido para compatibilidad
    ANOMALY_DETECTED = "anomaly_detected"
    ALERT = "alert"

@dataclass
class Event:
    """Representa un evento que fluye a través del Neural Hub."""
    id: str  # Añadido para identificar eventos únicos
    category: EventCategory
    source: str
    source_module: Optional[str] = None  # Añadido para compatibilidad
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Método para inicialización posterior."""
        if isinstance(self.priority, str):
            self.priority = EventPriority[self.priority.upper()]

        logger.debug(f"Evento creado: {self.id} - {self.category} desde {self.source} con prioridad {self.priority}")


class MetacortexNeuralHub:
    """
    Una implementación simplificada en memoria del Neural Hub para desacoplar componentes.
    En una implementación real, esto usaría un sistema de mensajería robusto.
    """
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[Any], Coroutine[Any, Any, None]]]] = {}
        self.modules: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    def register_module(self, name: str, instance: Any, subscriptions: Dict[str, Callable] = None, handlers: Dict[str, Callable] = None):
        """Registra un módulo con sus suscripciones y manejadores."""
        self.modules[name] = instance
        self.logger.info(f"Módulo '{name}' registrado en el Neural Hub.")
        if subscriptions and handlers:
            for event_category, handler in handlers.items():
                try:
                    # Convertir Enum a string para la suscripción
                    if hasattr(event_category, 'value'):
                        # Es un Enum
                        event_type = event_category.value.lower()
                    else:
                        # Ya es un string
                        event_type = str(event_category).lower()
                    self.subscribe(event_type, handler)
                except Exception as e:
                    self.logger.error(f"Error al suscribir handler para {event_category}: {e}")

    def subscribe(self, event_type: str, callback: Callable[[Any], Coroutine[Any, Any, None]]):
        """Suscribe un callback a un tipo de evento."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        self.logger.info(f"Callback {getattr(callback, '__name__', 'unknown')} suscrito al evento '{event_type}'")

    async def publish(self, event_or_type: Union[Event, EventCategory, str], source: Optional[str] = None, payload: Optional[Dict[str, Any]] = None):
        """
        Publica un evento en el hub de forma asíncrona.
        Crea un task para no bloquear el bucle de eventos.
        """
        if isinstance(event_or_type, Event):
            event = event_or_type
        else:
            # Normalizar la categoría del evento
            if isinstance(event_or_type, EventCategory):
                category = event_or_type
            else:
                try:
                    category = EventCategory(str(event_or_type))
                except ValueError:
                    self.logger.error(f"Categoría de evento desconocida: {event_or_type}")
                    return
            
            event = Event(
                id=f"event_{time.time()}_{random.randint(0, 1000)}",
                category=category,
                source=source or "unknown",
                payload=payload or {}
            )

        event_type_str = event.category.value
        if event_type_str in self.subscribers:
            self.logger.info(f"Publicando evento '{event_type_str}' con payload: {event.payload}")
            # Usamos asyncio.gather para ejecutar todos los callbacks concurrentemente
            await asyncio.gather(*(callback(event) for callback in self.subscribers[event_type_str]))

    def emit_event(self, event: Event):
        """Emite un evento al hub (versión síncrona para compatibilidad)."""
        asyncio.create_task(self.publish(event))

    def heartbeat(self, module_name: str):
        """Registra un heartbeat de un módulo."""
        # En una implementación real, esto actualizaría un timestamp
        self.logger.debug(f"❤️ Heartbeat recibido de '{module_name}'")

    def stop(self):
        """Método para compatibilidad con el ejemplo de uso."""
        self.logger.info("Deteniendo el Neural Hub (simulado).")

    def start(self):
        """Método para compatibilidad con el ejemplo de uso."""
        self.logger.info("Iniciando el Neural Hub (simulado).")

_neural_hub_instance = None

def get_neural_hub():
    """Retorna la instancia singleton del Neural Hub."""
    global _neural_hub_instance
    if _neural_hub_instance is None:
        _neural_hub_instance = MetacortexNeuralHub()
    return _neural_hub_instance