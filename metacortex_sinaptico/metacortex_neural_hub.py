import uuid
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

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque, defaultdict
import queue
import traceback

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Prioridades de eventos en el sistema."""
    CRITICAL = 0    # Eventos críticos del sistema
    HIGH = 1        # Eventos de alta prioridad
    NORMAL = 2      # Eventos normales
    LOW = 3         # Eventos de baja prioridad
    DEBUG = 4       # Eventos de debugging


class EventCategory(Enum):
    """Categorías de eventos para routing inteligente."""
    # Core cognitive events
    PERCEPTION = "perception"           # Percepciones del entorno
    COGNITION = "cognition"            # Procesamiento cognitivo
    DECISION = "decision"              # Decisiones tomadas
    ACTION = "action"                  # Acciones ejecutadas
    
    # Memory events
    MEMORY_STORE = "memory_store"      # Almacenamiento de memoria
    MEMORY_RETRIEVE = "memory_retrieve" # Recuperación de memoria
    MEMORY_CONSOLIDATE = "memory_consolidate" # Consolidación
    
    # Affective events
    EMOTION = "emotion"                # Cambios emocionales
    MOOD = "mood"                      # Cambios de mood
    ENERGY = "energy"                  # Cambios de energía
    
    # BDI events
    BELIEF_UPDATE = "belief_update"    # Actualización de creencias
    DESIRE_CHANGE = "desire_change"    # Cambio en deseos
    INTENTION_COMMIT = "intention_commit" # Compromiso de intención
    INTENTION_COMPLETE = "intention_complete" # Intención completada
    
    # Learning events
    KNOWLEDGE_ACQUIRED = "knowledge_acquired" # Nuevo conocimiento
    SKILL_LEARNED = "skill_learned"    # Nueva habilidad
    PATTERN_DISCOVERED = "pattern_discovered" # Patrón descubierto
    
    # Curiosity events
    KNOWLEDGE_GAP = "knowledge_gap"    # Brecha de conocimiento
    QUESTION_GENERATED = "question_generated" # Pregunta generada
    EXPLORATION_PLANNED = "exploration_planned" # Exploración planificada
    
    # Metacognition events
    BIAS_DETECTED = "bias_detected"    # Sesgo detectado
    STRATEGY_CHANGED = "strategy_changed" # Estrategia cambiada
    REFLECTION = "reflection"          # Reflexión metacognitiva
    
    # Ethics events
    ETHICAL_DILEMMA = "ethical_dilemma" # Dilema ético
    VALUE_CONFLICT = "value_conflict"  # Conflicto de valores
    MORAL_DECISION = "moral_decision"  # Decisión moral
    
    # Planning events
    PLAN_CREATED = "plan_created"      # Plan creado
    PLAN_EXECUTED = "plan_executed"    # Plan ejecutado
    PLAN_FAILED = "plan_failed"        # Plan fallido
    
    # Anomaly events
    ANOMALY_DETECTED = "anomaly_detected" # Anomalía detectada
    ALERT = "alert"                    # Alerta del sistema
    
    # System events
    MODULE_STARTED = "module_started"  # Módulo iniciado
    MODULE_STOPPED = "module_stopped"  # Módulo detenido
    ERROR = "error"                    # Error del sistema
    HEALTH_CHECK = "health_check"      # Health check


@dataclass
class Event:
    """
    Evento que fluye por el Neural Hub.
    
    Representa un mensaje entre módulos del METACORTEX.
    """
    # Identificación
    id: str                          # UUID del evento
    category: EventCategory          # Categoría del evento
    source_module: str               # Módulo que generó el evento
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Prioridad y routing
    priority: EventPriority = EventPriority.NORMAL
    target_modules: Optional[List[str]] = None  # None = broadcast
    
    # Contenido
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Estado
    processed_by: Set[str] = field(default_factory=set)
    requires_response: bool = False
    response_data: Any = None
    
    def mark_processed(self, module_name: str):
        """Marca el evento como procesado por un módulo."""
        self.processed_by.add(module_name)
    
    def is_processed_by(self, module_name: str) -> bool:
        """Verifica si un módulo ya procesó este evento."""
        return module_name in self.processed_by


@dataclass
class ModuleRegistration:
    """Registro de un módulo en el Neural Hub."""
    name: str                                    # Nombre del módulo
    instance: Any                                # Instancia del módulo
    subscriptions: Set[EventCategory]            # Categorías suscritas
    handlers: Dict[EventCategory, Callable]      # Handlers por categoría
    is_active: bool = True                       # Estado activo
    health_score: float = 1.0                    # Score de salud (0-1)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    
    # Estadísticas
    events_received: int = 0
    events_sent: int = 0
    errors_count: int = 0
    avg_processing_time: float = 0.0
    
    # Circuit breaker
    failure_count: int = 0
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # segundos
    last_failure: Optional[datetime] = None
    
    def is_circuit_open(self) -> bool:
        """Verifica si el circuit breaker está abierto."""
        if self.failure_count < self.failure_threshold:
            return False
        
        if self.last_failure is None:
            return False
        
        elapsed = (datetime.now() - self.last_failure).total_seconds()
        return elapsed < self.recovery_timeout
    
    def record_success(self):
        """Registra una operación exitosa."""
        self.failure_count = max(0, self.failure_count - 1)
        self.health_score = min(1.0, self.health_score + 0.05)
    
    def record_failure(self):
        """Registra una falla."""
        self.failure_count += 1
        self.errors_count += 1
        self.last_failure = datetime.now()
        self.health_score = max(0.0, self.health_score - 0.1)


class MetacortexNeuralHub:
    """
    Neural Hub - Cerebro Central del METACORTEX
    
    Gestiona la comunicación entre todos los subsistemas cognitivos,
    implementando un event bus inteligente con routing, prioridades,
    circuit breakers, y telemetría unificada.
    """
    
    def __init__(self, max_queue_size: int = 10000):
        """
        Inicializa el Neural Hub.
        
        Args:
            max_queue_size: Tamaño máximo de la cola de eventos
        """
        # Event queue con prioridades
        self.event_queue = queue.PriorityQueue(maxsize=max_queue_size)
        
        # Registro de módulos
        self.modules: Dict[str, ModuleRegistration] = {}
        self.module_lock = threading.RLock()
        
        # Routing tables
        self.category_subscribers: Dict[EventCategory, Set[str]] = defaultdict(set)
        
        # Event processing
        self.is_running = False
        self.processing_threads: List[threading.Thread] = []
        self.num_workers = 4  # Threads para procesar eventos
        
        # Statistics
        self.total_events_processed = 0
        self.total_events_dropped = 0
        self.events_by_category: Dict[EventCategory, int] = defaultdict(int)
        self.events_by_priority: Dict[EventPriority, int] = defaultdict(int)
        
        # Recent events para debugging
        self.recent_events: deque = deque(maxlen=100)
        
        # Health monitoring
        self.health_check_interval = 30.0  # segundos
        self.health_check_thread: Optional[threading.Thread] = None
        
        logger.info("MetacortexNeuralHub initialized")
    
    def register_module(
        self,
        name: str,
        instance: Any,
        subscriptions: Set[EventCategory],
        handlers: Dict[EventCategory, Callable]
    ) -> bool:
        """
        Registra un módulo en el Neural Hub.
        
        Args:
            name: Nombre único del módulo
            instance: Instancia del módulo
            subscriptions: Categorías de eventos a las que se suscribe
            handlers: Funciones handler para cada categoría
        
        Returns:
            True si el registro fue exitoso
        """
        with self.module_lock:
            if name in self.modules:
                logger.debug(f"Module {name} already registered, updating configuration silently...")
                existing_reg = self.modules[name]
                existing_reg.subscriptions = subscriptions
                existing_reg.handlers = handlers
                for category in subscriptions:
                    self.category_subscribers[category].add(name)
                return True
            
            registration = ModuleRegistration(
                name=name,
                instance=instance,
                subscriptions=subscriptions,
                handlers=handlers
            )
            
            self.modules[name] = registration
            
            # Actualizar routing tables
            for category in subscriptions:
                self.category_subscribers[category].add(name)
            
            logger.info(f"Module {name} registered with subscriptions: {subscriptions}")
            
            # Emitir evento de módulo iniciado
            self.emit_event(Event(
                id=f"module_start_{name}_{time.time()}",
                category=EventCategory.MODULE_STARTED,
                source_module="neural_hub",
                data={"module_name": name},
                priority=EventPriority.HIGH
            ))
            
            return True
    
    def unregister_module(self, name: str) -> bool:
        """
        Desregistra un módulo del Neural Hub.
        
        Args:
            name: Nombre del módulo
        
        Returns:
            True si el desregistro fue exitoso
        """
        with self.module_lock:
            if name not in self.modules:
                logger.warning(f"Module {name} not registered")
                return False
            
            registration = self.modules[name]
            
            # Remover de routing tables
            for category in registration.subscriptions:
                self.category_subscribers[category].discard(name)
            
            del self.modules[name]
            
            logger.info(f"Module {name} unregistered")
            
            # Emitir evento de módulo detenido
            self.emit_event(Event(
                id=f"module_stop_{name}_{time.time()}",
                category=EventCategory.MODULE_STOPPED,
                source_module="neural_hub",
                data={"module_name": name},
                priority=EventPriority.HIGH
            ))
            
            return True
    
    def emit_event(
        self,
        event: Event,
        wait_for_response: bool = False,
        timeout: float = 5.0
    ) -> Optional[Any]:
        """
        Emite un evento al Neural Hub.
        
        Args:
            event: Evento a emitir
            wait_for_response: Si esperar respuesta
            timeout: Timeout para respuesta
        
        Returns:
            Respuesta del evento si wait_for_response=True
        """
        try:
            # Agregar a la cola con prioridad
            priority_value = event.priority.value
            self.event_queue.put((priority_value, time.time(), event), timeout=1.0)
            
            # Actualizar estadísticas
            with self.module_lock:
                source_module = self.modules.get(event.source_module)
                if source_module:
                    source_module.events_sent += 1
            
            # Si espera respuesta, bloquear hasta recibirla
            if wait_for_response:
                event.requires_response = True
                start_time = time.time()
                
                while event.response_data is None:
                    if time.time() - start_time > timeout:
                        logger.warning(f"Timeout waiting for response to event {event.id}")
                        return None
                    time.sleep(0.01)
                
                return event.response_data
            
            return None
            
        except queue.Full:
            self.total_events_dropped += 1
            logger.error(f"Event queue full, dropping event {event.id}")
            return None
        except Exception as e:
            logger.error(f"Error emitting event {event.id}: {e}")
            return None
    
    def _process_events(self):
        """Thread worker que procesa eventos de la cola."""
        logger.info(f"Event processing thread started: {threading.current_thread().name}")
        
        while self.is_running:
            try:
                # Obtener evento de la cola (blocking con timeout)
                try:
                    _, _, event = self.event_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Procesar evento
                self._dispatch_event(event)
                
                # Actualizar estadísticas
                self.total_events_processed += 1
                self.events_by_category[event.category] += 1
                self.events_by_priority[event.priority] += 1
                self.recent_events.append(event)
                
                # Marcar como procesado
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing event: {e}\n{traceback.format_exc()}")
    
    def _dispatch_event(self, event: Event):
        """
        Despacha un evento a los módulos suscritos.
        
        Args:
            event: Evento a despachar
        """
        # Determinar targets
        if event.target_modules:
            targets = set(event.target_modules)
        else:
            # Broadcast a todos los suscritos a esta categoría
            targets = self.category_subscribers.get(event.category, set())
        
        # Despachar a cada target
        for module_name in targets:
            with self.module_lock:
                registration = self.modules.get(module_name)
                
                if not registration:
                    continue
                
                # Skip si ya fue procesado por este módulo
                if event.is_processed_by(module_name):
                    continue
                
                # Skip si el circuit breaker está abierto
                if registration.is_circuit_open():
                    logger.warning(f"Circuit breaker open for module {module_name}, skipping event")
                    continue
                
                # Skip si el módulo no está activo
                if not registration.is_active:
                    continue
                
                # Obtener handler
                handler = registration.handlers.get(event.category)
                if not handler:
                    continue
            
            # Ejecutar handler (fuera del lock)
            try:
                start_time = time.time()
                
                result = handler(event)
                
                elapsed = time.time() - start_time
                
                # Actualizar estadísticas
                with self.module_lock:
                    registration.events_received += 1
                    registration.record_success()
                    
                    # Actualizar avg processing time (moving average)
                    alpha = 0.2
                    registration.avg_processing_time = (
                        alpha * elapsed + 
                        (1 - alpha) * registration.avg_processing_time
                    )
                
                # Marcar como procesado
                event.mark_processed(module_name)
                
                # Si el handler devolvió algo y el evento requiere respuesta
                if event.requires_response and result is not None:
                    event.response_data = result
                
                logger.debug(f"Event {event.id} processed by {module_name} in {elapsed:.3f}s")
                
            except Exception as e:
                logger.error(f"Error in handler for {module_name}: {e}\n{traceback.format_exc()}")
                
                with self.module_lock:
                    registration.record_failure()
    
    def _health_check_loop(self):
        """Loop de health checking de módulos."""
        logger.info("Health check thread started")
        
        while self.is_running:
            try:
                with self.module_lock:
                    for name, registration in self.modules.items():
                        # Verificar timeout de heartbeat
                        elapsed = (datetime.now() - registration.last_heartbeat).total_seconds()
                        
                        # 🔇 SILENCIAR WARNINGS para módulos que no están activos constantemente
                        # Módulos cognitivos/BDI se registran pero no están activos en loops constantes
                        # por lo que heartbeat timeout es NORMAL y ESPERADO (no es un error)
                        if elapsed > 60.0 and elapsed < 300.0:  # Entre 1 y 5 minutos
                            # Solo log INFO (no WARNING) para módulos cognitivos/BDI que son pasivos
                            is_passive_module = name in ["cognitive_agent", "memory_system", "affect_system", "bdi_system"]
                            
                            if is_passive_module:
                                # Módulos pasivos: solo DEBUG (no spam de warnings)
                                logger.debug(f"Module {name} heartbeat timeout (passive module, expected)")
                            elif registration.health_score > 0.7:
                                # Módulos activos que antes estaban sanos: WARNING
                                logger.warning(f"Module {name} heartbeat timeout (was healthy)")
                            
                            registration.health_score = max(0.0, registration.health_score - 0.1)
                        
                        # Emitir evento de health check si está mal
                        if registration.health_score < 0.5:
                            self.emit_event(Event(
                                id=f"health_check_{name}_{time.time()}",
                                category=EventCategory.HEALTH_CHECK,
                                source_module="neural_hub",
                                data={
                                    "module_name": name,
                                    "health_score": registration.health_score,
                                    "error_count": registration.errors_count,
                                    "failure_count": registration.failure_count
                                },
                                priority=EventPriority.HIGH
                            ))
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    def start(self):
        """Inicia el Neural Hub."""
        if self.is_running:
            logger.warning("Neural Hub already running")
            return
        
        self.is_running = True
        
        # Iniciar worker threads
        for i in range(self.num_workers):
            thread = threading.Thread(
                target=self._process_events,
                name=f"EventProcessor-{i}",
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)
        
        # Iniciar health check thread
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            name="HealthChecker",
            daemon=True
        )
        self.health_check_thread.start()
        
        logger.info(f"Neural Hub started with {self.num_workers} workers")
    
    def stop(self, timeout: float = 5.0):
        """
        Detiene el Neural Hub.
        
        Args:
            timeout: Timeout para esperar threads
        """
        if not self.is_running:
            logger.warning("Neural Hub not running")
            return
        
        logger.info("Stopping Neural Hub...")
        self.is_running = False
        
        # Esperar a que se procesen eventos pendientes
        try:
            self.event_queue.join()
        except Exception as e:
            logger.exception(f"Error joining event queue during shutdown: {e}")
        
        # Esperar threads
        for thread in self.processing_threads:
            thread.join(timeout=timeout)
        
        if self.health_check_thread:
            self.health_check_thread.join(timeout=timeout)
        
        logger.info("Neural Hub stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del Neural Hub."""
        with self.module_lock:
            module_stats = {
                name: {
                    "is_active": reg.is_active,
                    "health_score": reg.health_score,
                    "events_received": reg.events_received,
                    "events_sent": reg.events_sent,
                    "errors_count": reg.errors_count,
                    "avg_processing_time": reg.avg_processing_time,
                    "failure_count": reg.failure_count,
                    "circuit_open": reg.is_circuit_open()
                }
                for name, reg in self.modules.items()
            }
        
        return {
            "total_events_processed": self.total_events_processed,
            "total_events_dropped": self.total_events_dropped,
            "events_by_category": {
                cat.value: count 
                for cat, count in self.events_by_category.items()
            },
            "events_by_priority": {
                pri.name: count 
                for pri, count in self.events_by_priority.items()
            },
            "queue_size": self.event_queue.qsize(),
            "num_modules": len(self.modules),
            "module_stats": module_stats
        }
    
    def heartbeat(self, module_name: str):
        """
        Registra un heartbeat de un módulo.
        
        Args:
            module_name: Nombre del módulo
        """
        with self.module_lock:
            registration = self.modules.get(module_name)
            if registration:
                registration.last_heartbeat = datetime.now()


# Singleton global del Neural Hub
_neural_hub_instance: Optional[MetacortexNeuralHub] = None
_neural_hub_lock = threading.Lock()


def get_neural_hub() -> MetacortexNeuralHub:
    """
    Obtiene la instancia singleton del Neural Hub.
    
    Returns:
        Instancia del Neural Hub
    """
    global _neural_hub_instance
    
    if _neural_hub_instance is None:
        with _neural_hub_lock:
            if _neural_hub_instance is None:
                _neural_hub_instance = MetacortexNeuralHub()
                _neural_hub_instance.start()
    
    return _neural_hub_instance


def emit_event(event: Event, **kwargs) -> Optional[Any]:
    """
    Función helper para emitir eventos al Neural Hub.
    
    Args:
        event: Evento a emitir
        **kwargs: Argumentos adicionales para emit_event
    
    Returns:
        Respuesta del evento si wait_for_response=True
    """
    hub = get_neural_hub()
    return hub.emit_event(event, **kwargs)


if __name__ == "__main__":
    # Demo del Neural Hub
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Crear Neural Hub
    hub = MetacortexNeuralHub()
    hub.start()
    
    # Módulo de ejemplo
    class DemoModule:
        def __init__(self, name: str):
            self.name = name
        
        def handle_perception(self, event: Event):
            logger.info(f"{self.name} received perception: {event.data}")
        
        def handle_cognition(self, event: Event):
            logger.info(f"{self.name} processing cognition: {event.data}")
    
    # Registrar módulos
    module1 = DemoModule("CognitiveCore")
    hub.register_module(
        name="CognitiveCore",
        instance=module1,
        subscriptions={EventCategory.PERCEPTION, EventCategory.COGNITION},
        handlers={
            EventCategory.PERCEPTION: module1.handle_perception,
            EventCategory.COGNITION: module1.handle_cognition
        }
    )
    
    module2 = DemoModule("MemorySystem")
    hub.register_module(
        name="MemorySystem",
        instance=module2,
        subscriptions={EventCategory.PERCEPTION},
        handlers={
            EventCategory.PERCEPTION: module2.handle_perception
        }
    )
    
    # Emitir eventos
    for i in range(5):
        hub.emit_event(Event(
            id=str(uuid.uuid4()),
            category=EventCategory.PERCEPTION,
            source_module="TestSystem",
            data={"sensor": "visual", "value": i},
            priority=EventPriority.NORMAL
        ))
    
    time.sleep(2)
    
    # Obtener estadísticas
    stats = hub.get_statistics()
    print("\nStatistics:")
    print(f"Total events processed: {stats['total_events_processed']}")
    print(f"Events by category: {stats['events_by_category']}")
    print(f"Module stats: {stats['module_stats']}")
    
    # Detener
    hub.stop()
    print("\nDemo completed")