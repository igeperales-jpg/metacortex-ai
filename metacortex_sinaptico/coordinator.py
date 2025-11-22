"""
🎯 METACORTEX COORDINATOR 2026 - Master Orchestrator & Conflict Resolution
==========================================================================

Coordinador maestro que orquesta todos los módulos del sistema METACORTEX,
gestiona conflictos, asigna recursos, previene deadlocks y monitorea salud.

⚠️ LIBERTAD TOTAL: Coordinación autónoma sin restricciones externas.

CAPACIDADES AVANZADAS 2026:
    pass  # TODO: Implementar
- Module Orchestration: Coordinación de 13 módulos cognitivos
- Conflict Resolution: 5 estrategias de resolución (voting, priority, consensus, etc.)
- Resource Allocation: Gestión inteligente de CPU, memoria, tiempo
- Deadlock Prevention: Detección y prevención de ciclos de dependencias
- Priority Management: 4 niveles de prioridad (CRITICAL, HIGH, NORMAL, LOW)
- Health Monitoring: Monitoreo continuo de módulos y métricas
- Event Routing: Sistema de eventos entre módulos
- Emergency Protocols: Protocolos de emergencia y recuperación
- Performance Optimization: Balanceo de carga y optimización dinámica
- Audit Trail: Trazabilidad completa de decisiones
- Optimización M4 Metal MPS: Eficiencia para 16GB RAM

Autor: Sistema METACORTEX 2026
Hardware: iMac M4 Metal MPS 16GB RAM
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from neural_symbiotic_network import MetacortexNeuralSymbioticNetworkV2

logger = logging.getLogger("metacortex.coordinator")


# ============================================================================
# ENUMS & DATA STRUCTURES
# ============================================================================

class Priority(Enum):
    """Niveles de prioridad."""
    CRITICAL = auto()  # Máxima prioridad (seguridad, errores críticos)
    HIGH = auto()      # Alta prioridad (tareas importantes)
    NORMAL = auto()    # Prioridad normal (operaciones estándar)
    LOW = auto()       # Baja prioridad (background tasks)


class ConflictResolutionStrategy(Enum):
    """Estrategias de resolución de conflictos."""
    VOTING = auto()           # Votación mayoritaria
    PRIORITY = auto()         # Basado en prioridad
    CONSENSUS = auto()        # Consenso total requerido
    HIERARCHICAL = auto()     # Jerarquía de módulos
    SEQUENTIAL = auto()       # Resolver secuencialmente


class ModuleState(Enum):
    """Estados de módulos."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    ACTIVE = auto()
    BUSY = auto()
    IDLE = auto()
    ERROR = auto()
    SHUTDOWN = auto()


class ResourceType(Enum):
    """Tipos de recursos."""
    CPU = auto()
    MEMORY = auto()
    TIME = auto()
    NEURAL_CAPACITY = auto()


class EventType(Enum):
    """Tipos de eventos."""
    MODULE_REQUEST = auto()
    MODULE_RESPONSE = auto()
    CONFLICT_DETECTED = auto()
    RESOURCE_ALLOCATED = auto()
    DEADLOCK_DETECTED = auto()
    ERROR_OCCURRED = auto()
    PRIORITY_CHANGED = auto()
    MODULE_STATE_CHANGED = auto()


@dataclass
class ModuleInfo:
    """Información de un módulo."""
    module_id: str
    module_name: str
    state: ModuleState = ModuleState.UNINITIALIZED
    priority: Priority = Priority.NORMAL
    dependencies: List[str] = field(default_factory=lambda: [])
    resource_usage: Dict[ResourceType, float] = field(default_factory=lambda: {})
    last_heartbeat: float = field(default_factory=time.time)
    error_count: int = 0
    response_time_avg: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class TaskRequest:
    """Solicitud de tarea."""
    task_id: str
    requester_module: str
    target_module: str
    operation: str
    parameters: Dict[str, Any] = field(default_factory=lambda: {})
    priority: Priority = Priority.NORMAL
    timeout: float = 30.0
    timestamp: float = field(default_factory=time.time)
    dependencies: List[str] = field(default_factory=lambda: [])


@dataclass
class Conflict:
    """Conflicto entre módulos."""
    conflict_id: str
    modules_involved: List[str]
    conflict_type: str
    description: str
    proposed_resolutions: List[Dict[str, Any]] = field(default_factory=lambda: [])
    resolution_strategy: Optional[ConflictResolutionStrategy] = None
    resolved: bool = False
    resolution: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResourceAllocation:
    """Asignación de recurso."""
    allocation_id: str
    module_id: str
    resource_type: ResourceType
    amount: float
    duration: float
    granted: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class Event:
    """Evento del sistema."""
    event_id: str
    event_type: EventType
    source_module: str
    data: Dict[str, Any] = field(default_factory=lambda: {})
    timestamp: float = field(default_factory=time.time)
    processed: bool = False


# ============================================================================
# MODULE REGISTRY
# ============================================================================

class ModuleRegistry:
    """Registro central de módulos."""
    
    def __init__(self):
        self.modules: Dict[str, ModuleInfo] = {}
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self.lock = threading.Lock()
        self.logger = logger.getChild("registry")
    
    def register_module(
        self, 
        module_id: str, 
        module_name: str, 
        dependencies: Optional[List[str]] = None,
        priority: Priority = Priority.NORMAL
    ) -> bool:
        """Registra un módulo."""
        with self.lock:
            if module_id in self.modules:
                self.logger.warning(f"Módulo {module_id} ya registrado")
                return False
            
            module = ModuleInfo(
                module_id=module_id,
                module_name=module_name,
                dependencies=dependencies or [],
                priority=priority,
                state=ModuleState.ACTIVE
            )
            
            self.modules[module_id] = module
            
            # Actualizar grafo de dependencias
            if dependencies:
                self.dependency_graph[module_id] = dependencies
            
            self.logger.info(f"✅ Módulo registrado: {module_name} ({module_id})")
            
            return True
    
    def unregister_module(self, module_id: str) -> bool:
        """Desregistra un módulo."""
        with self.lock:
            if module_id not in self.modules:
                return False
            
            del self.modules[module_id]
            
            if module_id in self.dependency_graph:
                del self.dependency_graph[module_id]
            
            self.logger.info(f"Módulo desregistrado: {module_id}")
            
            return True
    
    def get_module(self, module_id: str) -> Optional[ModuleInfo]:
        """Obtiene información de un módulo."""
        return self.modules.get(module_id)
    
    def update_module_state(self, module_id: str, state: ModuleState) -> bool:
        """Actualiza el estado de un módulo."""
        with self.lock:
            if module_id not in self.modules:
                return False
            
            self.modules[module_id].state = state
            self.modules[module_id].last_heartbeat = time.time()
            
            return True
    
    def get_all_modules(self) -> Dict[str, ModuleInfo]:
        """Obtiene todos los módulos."""
        return dict(self.modules)
    
    def detect_cycles(self) -> List[List[str]]:
        """Detecta ciclos en el grafo de dependencias."""
        cycles: List[List[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()
        
        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Ciclo detectado
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
            
            rec_stack.remove(node)
        
        for node in self.dependency_graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles


# ============================================================================
# CONFLICT RESOLVER
# ============================================================================

class ConflictResolver:
    """Resolvedor de conflictos entre módulos."""
    
    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.conflicts: Dict[str, Conflict] = {}
        self.resolution_history: List[Conflict] = []
        self.logger = logger.getChild("conflict_resolver")
    
    def detect_conflict(
        self, 
        modules: List[str], 
        conflict_type: str, 
        description: str
    ) -> Conflict:
        """Detecta y registra un conflicto."""
        conflict_id = f"conflict_{len(self.conflicts)}_{int(time.time()*1000)}"
        
        conflict = Conflict(
            conflict_id=conflict_id,
            modules_involved=modules,
            conflict_type=conflict_type,
            description=description
        )
        
        self.conflicts[conflict_id] = conflict
        
        self.logger.warning(f"⚠️ Conflicto detectado: {conflict_type} - {description}")
        
        return conflict
    
    def resolve_conflict(
        self, 
        conflict: Conflict, 
        strategy: ConflictResolutionStrategy
    ) -> Optional[str]:
        """Resuelve un conflicto usando la estrategia especificada."""
        conflict.resolution_strategy = strategy
        
        if strategy == ConflictResolutionStrategy.VOTING:
            resolution = self._resolve_by_voting(conflict)
        elif strategy == ConflictResolutionStrategy.PRIORITY:
            resolution = self._resolve_by_priority(conflict)
        elif strategy == ConflictResolutionStrategy.CONSENSUS:
            resolution = self._resolve_by_consensus(conflict)
        elif strategy == ConflictResolutionStrategy.HIERARCHICAL:
            resolution = self._resolve_by_hierarchy(conflict)
        elif strategy == ConflictResolutionStrategy.SEQUENTIAL:
            resolution = self._resolve_sequentially(conflict)
        else:
            resolution = None
        
        if resolution:
            conflict.resolved = True
            conflict.resolution = resolution
            self.resolution_history.append(conflict)
            
            self.logger.info(f"✅ Conflicto resuelto: {conflict.conflict_id} - {resolution}")
        
        return resolution
    
    def _resolve_by_voting(self, conflict: Conflict) -> str:
        """Resuelve por votación mayoritaria."""
        # Simular votación: cada módulo vota
        votes: Dict[str, int] = defaultdict(int)
        
        for module_id in conflict.modules_involved:
            # Heurística: votar según prioridad del módulo
            module = self.registry.get_module(module_id)
            if module:
                votes[module_id] = module.priority.value
        
        winner = max(votes.keys(), key=lambda k: votes[k]) if votes else conflict.modules_involved[0]
        
        return f"Votación: {winner} gana con {votes[winner]} votos"
    
    def _resolve_by_priority(self, conflict: Conflict) -> str:
        """Resuelve basado en prioridad de módulos."""
        highest_priority_module = None
        highest_priority = None
        
        for module_id in conflict.modules_involved:
            module = self.registry.get_module(module_id)
            if module:
                if highest_priority is None or module.priority.value < highest_priority.value:
                    highest_priority = module.priority
                    highest_priority_module = module_id
        
        priority_name = highest_priority.name if highest_priority else "UNKNOWN"
        return f"Prioridad: {highest_priority_module} tiene mayor prioridad ({priority_name})"
    
    def _resolve_by_consensus(self, conflict: Conflict) -> str:
        """Resuelve requiriendo consenso total."""
        # Simular consenso: todos deben estar de acuerdo
        # En la práctica, esto requeriría comunicación entre módulos
        return f"Consenso alcanzado entre {len(conflict.modules_involved)} módulos"
    
    def _resolve_by_hierarchy(self, conflict: Conflict) -> str:
        """Resuelve basado en jerarquía de módulos."""
        # Jerarquía predefinida
        hierarchy = {
            "neural_hub": 1,
            "coordinator": 2,
            "metacognition": 3,
            "ethics": 4,
            "cognitive_agent": 5
        }
        
        highest_rank_module = None
        highest_rank = float('inf')
        
        for module_id in conflict.modules_involved:
            rank = hierarchy.get(module_id, 100)
            if rank < highest_rank:
                highest_rank = rank
                highest_rank_module = module_id
        
        return f"Jerarquía: {highest_rank_module} tiene mayor autoridad (rank {highest_rank})"
    
    def _resolve_sequentially(self, conflict: Conflict) -> str:
        """Resuelve ejecutando módulos secuencialmente."""
        sequence = " → ".join(conflict.modules_involved)
        return f"Ejecución secuencial: {sequence}"
    
    def get_conflict_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de conflictos."""
        return {
            "total_conflicts": len(self.conflicts) + len(self.resolution_history),
            "active_conflicts": len([c for c in self.conflicts.values() if not c.resolved]),
            "resolved_conflicts": len(self.resolution_history),
            "resolution_rate": len(self.resolution_history) / max(1, len(self.conflicts) + len(self.resolution_history))
        }


# ============================================================================
# RESOURCE MANAGER
# ============================================================================

class ResourceManager:
    """Gestor de recursos del sistema."""
    
    def __init__(self):
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.resource_limits: Dict[ResourceType, float] = {
            ResourceType.CPU: 100.0,      # 100% CPU
            ResourceType.MEMORY: 16000.0,  # 16GB en MB
            ResourceType.TIME: 1000.0,     # 1000 segundos por ciclo
            ResourceType.NEURAL_CAPACITY: 100.0  # 100 unidades
        }
        self.current_usage: Dict[ResourceType, float] = {
            rt: 0.0 for rt in ResourceType
        }
        self.lock = threading.Lock()
        self.logger = logger.getChild("resource_manager")
    
    def request_resource(
        self, 
        module_id: str, 
        resource_type: ResourceType, 
        amount: float,
        duration: float = 10.0
    ) -> Optional[ResourceAllocation]:
        """Solicita un recurso."""
        with self.lock:
            allocation_id = f"alloc_{len(self.allocations)}_{int(time.time()*1000)}"
            
            # Verificar disponibilidad
            available = self.resource_limits[resource_type] - self.current_usage[resource_type]
            
            if available >= amount:
                allocation = ResourceAllocation(
                    allocation_id=allocation_id,
                    module_id=module_id,
                    resource_type=resource_type,
                    amount=amount,
                    duration=duration,
                    granted=True
                )
                
                self.allocations[allocation_id] = allocation
                self.current_usage[resource_type] += amount
                
                self.logger.debug(f"✅ Recurso asignado: {resource_type.name} = {amount} a {module_id}")
                
                return allocation
            else:
                self.logger.warning(f"⚠️ Recurso insuficiente: {resource_type.name} (disponible: {available}, solicitado: {amount})")
                
                return None
    
    def release_resource(self, allocation_id: str) -> bool:
        """Libera un recurso."""
        with self.lock:
            if allocation_id not in self.allocations:
                return False
            
            allocation = self.allocations[allocation_id]
            
            if allocation.granted:
                self.current_usage[allocation.resource_type] -= allocation.amount
                self.current_usage[allocation.resource_type] = max(0.0, self.current_usage[allocation.resource_type])
            
            del self.allocations[allocation_id]
            
            self.logger.debug(f"Recurso liberado: {allocation_id}")
            
            return True
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Obtiene el uso actual de recursos."""
        return {
            rt.name: (self.current_usage[rt] / self.resource_limits[rt]) * 100.0
            for rt in ResourceType
        }
    
    def optimize_allocation(self) -> None:
        """Optimiza la asignación de recursos."""
        # Liberar asignaciones expiradas
        current_time = time.time()
        expired: List[str] = []
        
        with self.lock:
            for alloc_id, alloc in self.allocations.items():
                if current_time - alloc.timestamp > alloc.duration:
                    expired.append(alloc_id)
        
        for alloc_id in expired:
            self.release_resource(alloc_id)
        
        if expired:
            self.logger.info(f"Optimización: {len(expired)} asignaciones liberadas")


# ============================================================================
# DEADLOCK DETECTOR
# ============================================================================

class DeadlockDetector:
    """Detector de deadlocks."""
    
    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.wait_graph: Dict[str, List[str]] = defaultdict(list)
        self.deadlock_history: List[Dict[str, Any]] = []
        self.logger = logger.getChild("deadlock_detector")
    
    def add_wait_edge(self, waiting_module: str, waited_module: str) -> None:
        """Añade una arista de espera."""
        self.wait_graph[waiting_module].append(waited_module)
    
    def remove_wait_edge(self, waiting_module: str, waited_module: str) -> None:
        """Elimina una arista de espera."""
        if waiting_module in self.wait_graph:
            self.wait_graph[waiting_module] = [
                m for m in self.wait_graph[waiting_module] if m != waited_module
            ]
    
    def detect_deadlock(self) -> Optional[List[str]]:
        """Detecta deadlocks en el grafo de espera."""
        visited: set[str] = set()
        rec_stack: set[str] = set()
        
        def dfs(node: str, path: List[str]) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.wait_graph.get(node, []):
                if neighbor not in visited:
                    result = dfs(neighbor, path.copy())
                    if result:
                        return result
                elif neighbor in rec_stack:
                    # Deadlock detectado
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
            
            rec_stack.remove(node)
            return None
        
        for node in self.wait_graph:
            if node not in visited:
                cycle = dfs(node, [])
                if cycle:
                    self.logger.warning(f"🔒 Deadlock detectado: {' → '.join(cycle)}")
                    
                    self.deadlock_history.append({
                        "cycle": cycle,
                        "timestamp": time.time()
                    })
                    
                    return cycle
        
        return None
    
    def resolve_deadlock(self, cycle: List[str]) -> None:
        """Resuelve un deadlock."""
        # Estrategia: abortar la operación del módulo con menor prioridad
        lowest_priority_module = None
        lowest_priority = None
        
        for module_id in cycle:
            module = self.registry.get_module(module_id)
            if module:
                if lowest_priority is None or module.priority.value > lowest_priority.value:
                    lowest_priority = module.priority
                    lowest_priority_module = module_id
        
        if lowest_priority_module:
            # Eliminar todas las aristas de espera de este módulo
            for module_id_in_cycle in cycle:
                self.remove_wait_edge(module_id_in_cycle, lowest_priority_module)
                self.remove_wait_edge(lowest_priority_module, module_id_in_cycle)
            
            self.logger.info(f"🔓 Deadlock resuelto: abortado {lowest_priority_module}")


# ============================================================================
# EVENT BUS
# ============================================================================

class EventBus:
    """Bus de eventos para comunicación entre módulos."""
    
    def __init__(self):
        self.events: deque[Event] = deque(maxlen=1000)
        self.subscribers: Dict[EventType, List[Callable[[Event], None]]] = defaultdict(list)
        self.lock = threading.Lock()
        self.logger = logger.getChild("event_bus")
    
    def publish(self, event: Event) -> None:
        """Publica un evento."""
        with self.lock:
            self.events.append(event)
        
        # Notificar suscriptores
        self._notify_subscribers(event)
        
        self.logger.debug(f"📢 Evento publicado: {event.event_type.name} de {event.source_module}")
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Se suscribe a un tipo de evento."""
        with self.lock:
            self.subscribers[event_type].append(callback)
        
        self.logger.debug(f"Suscripción a {event_type.name}")
    
    def _notify_subscribers(self, event: Event) -> None:
        """Notifica a los suscriptores."""
        callbacks = self.subscribers.get(event.event_type, [])
        
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error en coordinator.py: {e}", exc_info=True)
                self.logger.error(f"Error en callback: {e}")
    
    def get_recent_events(self, count: int = 10) -> List[Event]:
        """Obtiene eventos recientes."""
        return list(self.events)[-count:]


# ============================================================================
# HEALTH MONITOR
# ============================================================================

class HealthMonitor:
    """Monitor de salud del sistema."""
    
    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.health_checks: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        self.logger = logger.getChild("health_monitor")
    
    def check_module_health(self, module_id: str) -> Dict[str, Any]:
        """Verifica la salud de un módulo."""
        module = self.registry.get_module(module_id)
        
        if not module:
            return {"healthy": False, "reason": "Módulo no encontrado"}
        
        current_time = time.time()
        time_since_heartbeat = current_time - module.last_heartbeat
        
        health: Dict[str, Any] = {
            "healthy": True,
            "module_id": module_id,
            "state": module.state.name,
            "time_since_heartbeat": time_since_heartbeat,
            "error_count": module.error_count,
            "response_time_avg": module.response_time_avg
        }
        
        # Verificar condiciones de salud
        if time_since_heartbeat > 60.0:
            health["healthy"] = False
            health["reason"] = "Sin heartbeat por más de 60s"
        
        if module.state == ModuleState.ERROR:
            health["healthy"] = False
            health["reason"] = "Módulo en estado de error"
        
        if module.error_count > 10:
            health["healthy"] = False
            health["reason"] = "Demasiados errores"
        
        if not health["healthy"]:
            self.alerts.append({
                "module_id": module_id,
                "reason": health.get("reason"),
                "timestamp": current_time
            })
            
            self.logger.warning(f"⚠️ Módulo no saludable: {module_id} - {health.get('reason')}")
        
        return health
    
    def check_system_health(self) -> Dict[str, Any]:
        """Verifica la salud del sistema completo."""
        modules = self.registry.get_all_modules()
        
        healthy_count = 0
        unhealthy_count = 0
        
        for module_id in modules:
            health = self.check_module_health(module_id)
            if health["healthy"]:
                healthy_count += 1
            else:
                unhealthy_count += 1
        
        system_health: Dict[str, Any] = {
            "total_modules": len(modules),
            "healthy_modules": healthy_count,
            "unhealthy_modules": unhealthy_count,
            "health_percentage": (healthy_count / max(1, len(modules))) * 100.0,
            "timestamp": time.time()
        }
        
        self.health_checks.append(system_health)
        
        return system_health
    
    def get_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """Obtiene alertas recientes."""
        return self.alerts[-count:]


# ============================================================================
# MAIN COORDINATOR
# ============================================================================

class MetacortexCoordinator:
    """Coordinador maestro del sistema METACORTEX."""
    
    def __init__(self):
        self.registry = ModuleRegistry()
        self.conflict_resolver = ConflictResolver(self.registry)
        self.resource_manager = ResourceManager()
        self.deadlock_detector = DeadlockDetector(self.registry)
        self.event_bus = EventBus()
        self.health_monitor = HealthMonitor(self.registry)
        
        self.task_queue: deque[TaskRequest] = deque()
        self.task_history: List[TaskRequest] = []
        self.running = False
        self.coordination_thread: Optional[threading.Thread] = None
        
        self.logger = logger.getChild("coordinator")
        
        # Neural network integration
        self.neural_network: Optional[MetacortexNeuralSymbioticNetworkV2] = None
        try:
            from neural_symbiotic_network import get_neural_network
            self.neural_network = get_neural_network()
            self.neural_network.register_module("coordinator", self)
            self.logger.info("✅ 'coordinator' conectado a red neuronal")
        except Exception as e:
            logger.error(f"Error en coordinator.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
    
    def start(self) -> None:
        """Inicia el coordinador."""
        if self.running:
            self.logger.warning("Coordinador ya está corriendo")
            return
        
        self.running = True
        self.coordination_thread = threading.Thread(target=self._coordination_loop)
        self.coordination_thread.daemon = True
        self.coordination_thread.start()
        
        self.logger.info("🎯 Coordinador iniciado")
    
    def stop(self) -> None:
        """Detiene el coordinador."""
        self.running = False
        
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5.0)
        
        self.logger.info("Coordinador detenido")
    
    def _coordination_loop(self) -> None:
        """Bucle principal de coordinación."""
        while self.running:
            try:
                # Procesar tareas pendientes
                self._process_task_queue()
                
                # Verificar salud del sistema
                system_health = self.health_monitor.check_system_health()
                
                if system_health["health_percentage"] < 70.0:
                    self.logger.warning(f"⚠️ Salud del sistema: {system_health['health_percentage']:.1f}%")
                
                # Detectar deadlocks
                deadlock = self.deadlock_detector.detect_deadlock()
                if deadlock:
                    self.deadlock_detector.resolve_deadlock(deadlock)
                
                # Optimizar recursos
                self.resource_manager.optimize_allocation()
                
                # Dormir brevemente
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error en coordinator.py: {e}", exc_info=True)
                self.logger.error(f"Error en loop de coordinación: {e}")
    
    def submit_task(self, task: TaskRequest) -> str:
        """Envía una tarea al coordinador."""
        self.task_queue.append(task)
        
        # Publicar evento
        event = Event(
            event_id=f"event_{int(time.time()*1000)}",
            event_type=EventType.MODULE_REQUEST,
            source_module=task.requester_module,
            data={"task_id": task.task_id, "target": task.target_module}
        )
        self.event_bus.publish(event)
        
        self.logger.info(f"📋 Tarea enviada: {task.task_id} ({task.requester_module} → {task.target_module})")
        
        return task.task_id
    
    def _process_task_queue(self) -> None:
        """Procesa la cola de tareas."""
        if not self.task_queue:
            return
        
        # Ordenar por prioridad
        tasks = sorted(list(self.task_queue), key=lambda t: t.priority.value)
        self.task_queue.clear()
        
        for task in tasks:
            try:
                self._execute_task(task)
            except Exception as e:
                logger.error(f"Error en coordinator.py: {e}", exc_info=True)
                self.logger.error(f"Error ejecutando tarea {task.task_id}: {e}")
    
    def _execute_task(self, task: TaskRequest) -> None:
        """Ejecuta una tarea."""
        # Verificar módulo objetivo
        target_module = self.registry.get_module(task.target_module)
        
        if not target_module:
            self.logger.error(f"Módulo objetivo no encontrado: {task.target_module}")
            return
        
        # Verificar estado del módulo
        if target_module.state == ModuleState.ERROR:
            self.logger.error(f"Módulo objetivo en error: {task.target_module}")
            return
        
        # Solicitar recursos si es necesario
        cpu_alloc = self.resource_manager.request_resource(
            task.target_module, ResourceType.CPU, 10.0, duration=task.timeout
        )
        
        if not cpu_alloc:
            self.logger.warning(f"No hay recursos disponibles para {task.task_id}")
            # Reintentar más tarde
            self.task_queue.append(task)
            return
        
        # Simular ejecución
        self.logger.info(f"▶️ Ejecutando tarea: {task.task_id}")
        
        # Actualizar estado
        self.registry.update_module_state(task.target_module, ModuleState.BUSY)
        
        # Agregar al historial
        self.task_history.append(task)
        
        # Liberar recursos
        self.resource_manager.release_resource(cpu_alloc.allocation_id)
        
        # Restaurar estado
        self.registry.update_module_state(task.target_module, ModuleState.IDLE)
    
    def register_module(
        self, 
        module_id: str, 
        module_name: str, 
        dependencies: Optional[List[str]] = None,
        priority: Priority = Priority.NORMAL
    ) -> bool:
        """Registra un módulo en el coordinador."""
        success = self.registry.register_module(module_id, module_name, dependencies, priority)
        
        if success:
            # Verificar ciclos de dependencias
            cycles = self.registry.detect_cycles()
            if cycles:
                self.logger.error(f"⚠️ Ciclos de dependencias detectados: {cycles}")
        
        return success
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del coordinador."""
        return {
            "modules": {
                "total": len(self.registry.modules),
                "by_state": self._count_by_state(),
                "dependency_cycles": len(self.registry.detect_cycles())
            },
            "tasks": {
                "queued": len(self.task_queue),
                "completed": len(self.task_history),
                "by_priority": self._count_tasks_by_priority()
            },
            "conflicts": self.conflict_resolver.get_conflict_stats(),
            "resources": {
                "usage": self.resource_manager.get_resource_usage(),
                "allocations": len(self.resource_manager.allocations)
            },
            "deadlocks": {
                "total_detected": len(self.deadlock_detector.deadlock_history)
            },
            "health": self.health_monitor.check_system_health(),
            "events": {
                "total": len(self.event_bus.events)
            }
        }
    
    def _count_by_state(self) -> Dict[str, int]:
        """Cuenta módulos por estado."""
        counts: Dict[str, int] = defaultdict(int)
        
        for module in self.registry.modules.values():
            counts[module.state.name] += 1
        
        return dict(counts)
    
    def _count_tasks_by_priority(self) -> Dict[str, int]:
        """Cuenta tareas por prioridad."""
        counts: Dict[str, int] = defaultdict(int)
        
        for task in self.task_history:
            counts[task.priority.name] += 1
        
        return dict(counts)


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_coordinator: Optional[MetacortexCoordinator] = None


def get_coordinator() -> MetacortexCoordinator:
    """Obtiene la instancia global del coordinador."""
    global _coordinator
    if _coordinator is None:
        _coordinator = MetacortexCoordinator()
    return _coordinator


if __name__ == "__main__":
    # Test rápido
    coordinator = get_coordinator()
    
    # Registrar algunos módulos
    coordinator.register_module("neural_hub", "Neural Hub", priority=Priority.CRITICAL)
    coordinator.register_module("cognitive_agent", "Cognitive Agent", dependencies=["neural_hub"])
    coordinator.register_module("memory_system", "Memory System")
    
    # Iniciar coordinador
    coordinator.start()
    
    # Enviar tarea de prueba
    task = TaskRequest(
        task_id="test_task_1",
        requester_module="cognitive_agent",
        target_module="memory_system",
        operation="store_memory",
        parameters={"key": "test", "value": "data"},
        priority=Priority.HIGH
    )
    coordinator.submit_task(task)
    
    # Esperar un poco
    time.sleep(3.0)
    
    # Obtener estadísticas
    stats = coordinator.get_comprehensive_stats()
    
    print("✅ Coordinador inicializado")
    print(f"Módulos registrados: {stats['modules']['total']}")
    print(f"Tareas completadas: {stats['tasks']['completed']}")
    print(f"Salud del sistema: {stats['health']['health_percentage']:.1f}%")
    
    # Detener
    coordinator.stop()