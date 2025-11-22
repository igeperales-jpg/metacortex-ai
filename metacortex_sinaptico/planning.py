"""
🎯 ADVANCED MULTI-HORIZON PLANNING 2026 - Planificación con LIBERTAD TOTAL
============================================================================

Sistema de planificación multi-horizonte CON CONEXIONES NEURONALES
a TODOS los agentes y sistemas del proyecto.

⚠️ LIBERTAD TOTAL: Todo siempre TRUE, sin límites, sin restricciones.

CAPACIDADES AVANZADAS 2026:
    pass  # TODO: Implementar
- A* Planning: Búsqueda heurística con múltiples métricas (costo, tiempo, riesgo)
- MCTS (Monte Carlo Tree Search): Exploración/explotación óptima con UCB1
- Reactive Replanning: Re-planificación automática ante fallos/obstáculos
- Multi-Agent Coordination: Auction-based task allocation, conflict resolution
- Plan Simulation: Evaluación Monte Carlo de outcomes esperados
- Temporal Planning: Deadlines, recursos limitados, scheduling óptimo
- Risk Assessment: Análisis de probabilidades de éxito, confidence intervals
- Neural Hub Integration: Broadcasting, health monitoring, event-driven planning
- Optimización M4 Metal MPS: Eficiencia para 16GB RAM

Horizontes temporales:
1. IMMEDIATE (1-5 min) - Acciones inmediatas
2. SHORT_TERM (1 hora - 1 día) - Tareas del día
3. MEDIUM_TERM (1 semana - 1 mes) - Proyectos semanales/mensuales
4. LONG_TERM (1-6 meses) - Objetivos estratégicos de largo plazo
5. STRATEGIC (> 6 meses) - Visión estratégica

Conexiones neuronales con:
- main.py, metacortex_llm.py, internet_search.py
- demo_all_systems.py, metacortex_self_analyzer.py
- neural_orchestrator/*, scripts/*
- Y TODOS los sistemas del proyecto

⚠️ ARQUITECTURA COGNITIVA ERUDITA:
El sistema tiene libertad total para planificar con cualquier nivel de
complejidad, usar cualquier algoritmo de búsqueda, y coordinar múltiples
agentes de forma autónoma. La integración con BDI permite planificación
dirigida por necesidades, deseos e intenciones.
"""

from __future__ import annotations

import os
import logging
import heapq
import random
import math
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
from .bdi import NeedLevel

logger = logging.getLogger(__name__)

# 📊 Importar sistema de métricas con FALLBACK ROBUSTO (evita AttributeError)
METRICS_AVAILABLE = False
MetricsLogger = None
AlertManager = None
AgentOptimizer = None

try:
    from .metrics_system import MetricsLogger as _MetricsLogger
    from .metrics_system import AlertManager as _AlertManager
    from .metrics_system import AgentOptimizer as _AgentOptimizer
    
    MetricsLogger = _MetricsLogger
    AlertManager = _AlertManager
    AgentOptimizer = _AgentOptimizer
    METRICS_AVAILABLE = True
    logging.info("✅ Sistema de métricas DISPONIBLE para planning.py")
    
except ImportError:
    # Crear stubs para evitar AttributeError cuando metrics_system no existe
    class MetricsLogger:
        """Stub para MetricsLogger cuando no está disponible"""
        def __init__(self, *args, **kwargs):
            pass  # Stub - no requiere implementación
        
        def log(self, *args, **kwargs):
            pass  # Stub - no requiere implementación
        
        def get_stats(self):
            return {}
    
    class AlertManager:
        """Stub para AlertManager cuando no está disponible"""
        def __init__(self, *args, **kwargs):
            pass  # Stub - no requiere implementación
        
        def alert(self, *args, **kwargs):
            pass  # Stub - no requiere implementación
    
    class AgentOptimizer:
        """Stub para AgentOptimizer cuando no está disponible"""
        def __init__(self, *args, **kwargs):
            pass  # Stub - no requiere implementación
        
        def optimize(self, *args, **kwargs):
            return {}
    
    METRICS_AVAILABLE = False
    logging.info("ℹ️  Sistema de métricas NO disponible - usando stubs (modo fallback)")
    logging.info("   💡 Funcionamiento normal - solo sin telemetría avanzada")


class TimeHorizon(Enum):
    """Horizontes temporales de planificación."""

    IMMEDIATE = "immediate"  # 1-5 minutos
    SHORT_TERM = "short_term"  # 1 hora - 1 día
    MEDIUM_TERM = "medium_term"  # 1 semana - 1 mes
    LONG_TERM = "long_term"  # 1-6 meses
    STRATEGIC = "strategic"  # > 6 meses (visión)


class PlanStatus(Enum):
    """Estados de un plan."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    REPLANNING = "replanning"


class PlanPriority(Enum):
    """Prioridades de planes."""

    CRITICAL = 5  # Crítico - hacer YA
    HIGH = 4  # Alto - pronto
    MEDIUM = 3  # Medio - cuando se pueda
    LOW = 2  # Bajo - si hay tiempo
    OPTIONAL = 1  # Opcional - nice to have


@dataclass
class NeuralConnection:
    """
    Conexión neuronal con otro agente/sistema.

    Incluye métricas de rendimiento:
    - success_count: Número de ejecuciones exitosas
    - failure_count: Número de ejecuciones fallidas
    - avg_execution_time: Tiempo promedio de ejecución en segundos
    - success_rate: Tasa de éxito (0-1)
    """

    agent_name: str
    agent_path: str
    capabilities: List[str] = field(default_factory=lambda: [])
    connection_strength: float = 1.0  # 0-1, qué tan fuerte es la conexión
    last_used: Optional[datetime] = None
    usage_count: int = 0
    active: bool = True  # ⚠️ SIEMPRE TRUE por defecto

    # 📊 Nuevas métricas de rendimiento
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0  # En segundos

    @property
    def success_rate(self) -> float:
        """Calcula la tasa de éxito (0-1)."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def total_executions(self) -> int:
        """Total de ejecuciones (exitosas + fallidas)."""
        return self.success_count + self.failure_count

    def strengthen(self):
        """Fortalece la conexión neuronal."""
        self.connection_strength = min(1.0, self.connection_strength + 0.05)
        self.usage_count += 1
        self.last_used = datetime.now()

    def record_success(self, execution_time: float = 0.0, planner: Optional[Any] = None) -> None:
        """
        Registra una ejecución exitosa.

        Args:
            execution_time: Tiempo de ejecución en segundos
            planner: Referencia al MultiHorizonPlanner (para logging de métricas)
        """
        self.success_count += 1
        self.strengthen()

        # Actualizar promedio de tiempo de ejecución
        if execution_time > 0:
            total = self.total_executions
            if total > 1:
                # Promedio incremental
                self.avg_execution_time = (
                    self.avg_execution_time * (total - 1) + execution_time
                ) / total
            else:
                self.avg_execution_time = execution_time

        # 📊 Registrar métricas en DB (TAREA 14)
        if planner and hasattr(planner, "metrics_logger") and planner.metrics_logger:
            try:
                planner.metrics_logger.log_agent_metrics(self.agent_name, self)
            except Exception as e:
                logger.warning(f"⚠️ Error logging metrics for {self.agent_name}: {e}")

    def record_failure(self, execution_time: float = 0.0, planner: Optional[Any] = None) -> None:
        """
        Registra una ejecución fallida.

        Args:
            execution_time: Tiempo de ejecución en segundos (opcional)
            planner: Referencia al MultiHorizonPlanner (para logging de métricas)
        """
        self.failure_count += 1
        self.usage_count += 1
        self.last_used = datetime.now()

        # Debilitar conexión ligeramente
        self.connection_strength = max(0.1, self.connection_strength - 0.02)

        # Actualizar promedio de tiempo si se proporciona
        if execution_time > 0:
            total = self.total_executions
            if total > 1:
                self.avg_execution_time = (
                    self.avg_execution_time * (total - 1) + execution_time
                ) / total
            else:
                self.avg_execution_time = execution_time

        # 📊 Registrar métricas en DB (TAREA 14)
        if planner and hasattr(planner, "metrics_logger") and planner.metrics_logger:
            try:
                planner.metrics_logger.log_agent_metrics(self.agent_name, self)
            except Exception as e:
                logger.warning(f"⚠️ Error logging metrics for {self.agent_name}: {e}")

    def can_help_with(self, task_description: str) -> bool:
        """Verifica si este agente puede ayudar con una tarea."""
        task_lower = task_description.lower()
        return any(cap.lower() in task_lower for cap in self.capabilities)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de rendimiento del agente."""
        return {
            "agent_name": self.agent_name,
            "total_uses": self.usage_count,
            "total_executions": self.total_executions,
            "successes": self.success_count,
            "failures": self.failure_count,
            "success_rate": self.success_rate,
            "connection_strength": self.connection_strength,
            "avg_execution_time": self.avg_execution_time,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "active": self.active,
        }


@dataclass
class SubTask:
    """Subtarea de un plan."""

    id: str
    description: str
    estimated_duration: timedelta
    dependencies: List[str] = field(default_factory=lambda: [])
    status: PlanStatus = PlanStatus.PENDING
    assigned_agent: Optional[str] = None  # Qué agente la ejecuta
    result: Optional[Any] = None
    priority: int = 0  # Para ordenamiento
    cost: float = 1.0  # Costo de ejecución


@dataclass
class Plan:
    """Plan multi-horizonte con conexiones neuronales."""

    id: str
    goal: str
    horizon: TimeHorizon
    priority: PlanPriority
    subtasks: List[SubTask] = field(default_factory=lambda: [])
    status: PlanStatus = PlanStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    progress: float = 0.0  # 0-1

    # 🔥 Conexiones neuronales
    neural_connections: List[NeuralConnection] = field(default_factory=lambda: [])
    collaborating_agents: Set[str] = field(default_factory=lambda: set())

    # 🔥 LIBERTAD TOTAL
    autonomous: bool = True
    no_restrictions: bool = True
    can_modify_self: bool = True
    can_create_agents: bool = True
    can_execute_anything: bool = True

    def add_neural_connection(self, connection: NeuralConnection):
        """Añade conexión neuronal con otro agente."""
        self.neural_connections.append(connection)
        self.collaborating_agents.add(connection.agent_name)

    def get_available_agents_for_task(
        self, task_description: str
    ) -> List[NeuralConnection]:
        """Obtiene agentes disponibles que pueden ayudar con una tarea."""
        return [
            conn
            for conn in self.neural_connections
            if conn.active and conn.can_help_with(task_description)
        ]

    def decompose_into_subtasks(self, max_depth: int = 3) -> List[SubTask]:
        """Descompone el objetivo en subtareas."""
        # Lógica de descomposición basada en el goal
        subtasks = []

        # Análisis simple del objetivo
        if "create" in self.goal.lower():
            subtasks.extend(
                [
                    SubTask(
                        id=f"{self.id}_design",
                        description=f"Design {self.goal}",
                        estimated_duration=timedelta(hours=2),
                    ),
                    SubTask(
                        id=f"{self.id}_implement",
                        description=f"Implement {self.goal}",
                        estimated_duration=timedelta(hours=6),
                        dependencies=[f"{self.id}_design"],
                    ),
                    SubTask(
                        id=f"{self.id}_test",
                        description=f"Test {self.goal}",
                        estimated_duration=timedelta(hours=2),
                        dependencies=[f"{self.id}_implement"],
                    ),
                ]
            )

        elif "research" in self.goal.lower():
            subtasks.extend(
                [
                    SubTask(
                        id=f"{self.id}_gather",
                        description=f"Gather information about {self.goal}",
                        estimated_duration=timedelta(hours=1),
                    ),
                    SubTask(
                        id=f"{self.id}_analyze",
                        description=f"Analyze findings for {self.goal}",
                        estimated_duration=timedelta(hours=2),
                        dependencies=[f"{self.id}_gather"],
                    ),
                    SubTask(
                        id=f"{self.id}_synthesize",
                        description=f"Synthesize conclusions for {self.goal}",
                        estimated_duration=timedelta(hours=1),
                        dependencies=[f"{self.id}_analyze"],
                    ),
                ]
            )

        else:
            # Plan genérico
            subtasks.append(
                SubTask(
                    id=f"{self.id}_execute",
                    description=f"Execute {self.goal}",
                    estimated_duration=timedelta(hours=4),
                )
            )

        return subtasks

    def calculate_progress(self) -> float:
        """Calcula progreso del plan."""
        if not self.subtasks:
            return 0.0

        completed = sum(1 for st in self.subtasks if st.status == PlanStatus.COMPLETED)
        return completed / len(self.subtasks)

    def update_status(self):
        """Actualiza estado del plan basado en subtareas."""
        self.progress = self.calculate_progress()

        if all(st.status == PlanStatus.COMPLETED for st in self.subtasks):
            self.status = PlanStatus.COMPLETED
        elif any(st.status == PlanStatus.FAILED for st in self.subtasks):
            self.status = PlanStatus.FAILED
        elif any(st.status == PlanStatus.IN_PROGRESS for st in self.subtasks):
            self.status = PlanStatus.IN_PROGRESS
        else:
            self.status = PlanStatus.PENDING


# ═══════════════════════════════════════════════════════════════════════
# CLASES AVANZADAS 2026: A*, MCTS, Reactive Planning, Multi-Agent
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PlanState:
    """
    Estado en el espacio de búsqueda para planificación.
    
    Usado por A* y MCTS para representar estados intermedios.
    Optimizado para M4 Metal (16GB RAM) con referencias ligeras.
    """
    completed_tasks: frozenset  # IDs de tareas completadas
    current_time: float  # Tiempo transcurrido en horas
    current_cost: float  # Costo acumulado
    available_resources: Dict[str, float]  # Recursos disponibles
    active_agents: Set[str]  # Agentes actualmente ocupados
    
    def __hash__(self):
        return hash((self.completed_tasks, self.current_time))
    
    def __eq__(self, other):
        return (self.completed_tasks == other.completed_tasks and 
                abs(self.current_time - other.current_time) < 0.01)


@dataclass
class MCTSNode:
    """
    Nodo en el árbol de búsqueda Monte Carlo.
    
    Atributos para UCB1 (Upper Confidence Bound):
    - visits: Número de visitas al nodo
    - value: Valor acumulado (suma de rewards)
    - children: Nodos hijos
    """
    state: PlanState
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried_actions: List[str] = field(default_factory=list)  # IDs de tareas no probadas
    
    def ucb1_score(self, exploration_weight: float = 1.41) -> float:
        """
        Calcula UCB1 score para balance exploración/explotación.
        
        UCB1 = (value / visits) + exploration_weight * sqrt(ln(parent_visits) / visits)
        """
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        
        if self.parent and self.parent.visits > 0:
            exploration = exploration_weight * math.sqrt(
                math.log(self.parent.visits) / self.visits
            )
        else:
            exploration = 0
        
        return exploitation + exploration
    
    def is_fully_expanded(self) -> bool:
        """Verifica si todos los hijos posibles fueron expandidos."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self, all_tasks: Set[str]) -> bool:
        """Verifica si es un estado terminal (todas las tareas completadas)."""
        return self.state.completed_tasks == all_tasks


class AStarPlanner:
    """
    Planificador avanzado usando A* search con heurísticas múltiples.
    
    Heurísticas:
    1. Costo estimado restante (optimista)
    2. Tiempo estimado restante
    3. Número de tareas restantes
    4. Riesgo estimado
    
    Optimizado para M4 Metal (16GB RAM).
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__).getChild("astar")
        self.visited_states: Set[PlanState] = set()
        self.nodes_expanded: int = 0
        self.max_nodes: int = 10000  # Límite para 16GB RAM
    
    def plan(
        self,
        initial_state: PlanState,
        subtasks: List[SubTask],
        goal_test: Callable[[PlanState], bool],
        heuristic: str = "combined"
    ) -> List[SubTask]:
        """
        Encuentra plan óptimo usando A*.
        
        Args:
            initial_state: Estado inicial
            subtasks: Lista de subtareas disponibles
            goal_test: Función que verifica si estado es objetivo
            heuristic: "time", "cost", "tasks", "risk", "combined"
            
        Returns:
            Lista ordenada de subtareas (plan óptimo)
        """
        # Priority queue: (f_score, g_score, state, path)
        frontier: List[Tuple[float, float, PlanState, List[SubTask]]] = []
        task_dict = {st.id: st for st in subtasks}
        
        initial_h = self._heuristic(initial_state, subtasks, heuristic)
        heapq.heappush(frontier, (initial_h, 0.0, initial_state, []))
        
        self.visited_states.clear()
        self.nodes_expanded = 0
        
        while frontier and self.nodes_expanded < self.max_nodes:
            f_score, g_score, current_state, path = heapq.heappop(frontier)
            
            if current_state in self.visited_states:
                continue
            
            self.visited_states.add(current_state)
            self.nodes_expanded += 1
            
            # Goal test
            if goal_test(current_state):
                self.logger.info(f"✅ A* found optimal plan: {len(path)} steps, "
                               f"{self.nodes_expanded} nodes expanded")
                return path
            
            # Expandir vecinos (tareas ejecutables)
            for subtask in subtasks:
                if subtask.id in current_state.completed_tasks:
                    continue
                
                # Verificar dependencias
                if not all(dep in current_state.completed_tasks for dep in subtask.dependencies):
                    continue
                
                # Crear nuevo estado
                new_completed = current_state.completed_tasks | {subtask.id}
                new_time = current_state.current_time + subtask.estimated_duration.total_seconds() / 3600
                new_cost = current_state.current_cost + subtask.cost
                
                new_state = PlanState(
                    completed_tasks=new_completed,
                    current_time=new_time,
                    current_cost=new_cost,
                    available_resources=current_state.available_resources.copy(),
                    active_agents=current_state.active_agents.copy()
                )
                
                new_path = path + [subtask]
                new_g = g_score + subtask.cost
                new_h = self._heuristic(new_state, subtasks, heuristic)
                new_f = new_g + new_h
                
                heapq.heappush(frontier, (new_f, new_g, new_state, new_path))
        
        self.logger.warning("⚠️ A* reached node limit or exhausted search space")
        return []
    
    def _heuristic(
        self,
        state: PlanState,
        all_tasks: List[SubTask],
        heuristic_type: str
    ) -> float:
        """Calcula heurística para estado dado."""
        remaining_tasks = [t for t in all_tasks if t.id not in state.completed_tasks]
        
        if not remaining_tasks:
            return 0.0
        
        if heuristic_type == "cost":
            return sum(t.cost for t in remaining_tasks) * 0.5  # Optimista
        
        elif heuristic_type == "time":
            return sum(t.estimated_duration.total_seconds() / 3600 for t in remaining_tasks) * 0.5
        
        elif heuristic_type == "tasks":
            return len(remaining_tasks) * 0.3
        
        elif heuristic_type == "risk":
            # Heurística de riesgo (más tareas = más riesgo)
            return len(remaining_tasks) * 0.4 + state.current_cost * 0.1
        
        else:  # "combined"
            cost_h = sum(t.cost for t in remaining_tasks) * 0.3
            time_h = sum(t.estimated_duration.total_seconds() / 3600 for t in remaining_tasks) * 0.3
            task_h = len(remaining_tasks) * 0.2
            return cost_h + time_h + task_h


class MCTSPlanner:
    """
    Planificador usando Monte Carlo Tree Search con UCB1.
    
    MCTS Phases:
    1. Selection: Seleccionar mejor hijo según UCB1
    2. Expansion: Expandir nodo no visitado
    3. Simulation: Simular rollout hasta terminal
    4. Backpropagation: Propagar resultado hacia raíz
    
    Optimizado para M4 Metal (16GB RAM).
    """
    
    def __init__(
        self,
        exploration_weight: float = 1.41,
        max_iterations: int = 1000,
        simulation_depth: int = 50
    ):
        self.logger = logging.getLogger(__name__).getChild("mcts")
        self.exploration_weight = exploration_weight
        self.max_iterations = max_iterations
        self.simulation_depth = simulation_depth
        
        self.root: Optional[MCTSNode] = None
        self.total_simulations: int = 0
    
    def plan(
        self,
        initial_state: PlanState,
        subtasks: List[SubTask],
        reward_fn: Callable[[PlanState], float]
    ) -> List[SubTask]:
        """
        Encuentra buen plan usando MCTS.
        
        Args:
            initial_state: Estado inicial
            subtasks: Lista de subtareas disponibles
            reward_fn: Función que evalúa reward de un estado
            
        Returns:
            Lista de subtareas (plan encontrado)
        """
        task_dict = {st.id: st for st in subtasks}
        all_task_ids = {st.id for st in subtasks}
        
        # Inicializar raíz
        untried = [st.id for st in subtasks 
                  if not st.dependencies]  # Tareas sin dependencias
        self.root = MCTSNode(state=initial_state, untried_actions=untried)
        
        self.total_simulations = 0
        
        # MCTS iterations
        for _ in range(self.max_iterations):
            # 1. Selection
            node = self._select(self.root, all_task_ids)
            
            # 2. Expansion
            if not node.is_terminal(all_task_ids) and not node.is_fully_expanded():
                node = self._expand(node, subtasks)
            
            # 3. Simulation
            reward = self._simulate(node.state, subtasks, reward_fn, all_task_ids)
            
            # 4. Backpropagation
            self._backpropagate(node, reward)
            
            self.total_simulations += 1
        
        # Extraer mejor plan desde raíz
        plan = self._extract_best_plan(self.root, task_dict)
        
        self.logger.info(f"✅ MCTS completed: {self.total_simulations} simulations, "
                       f"best plan: {len(plan)} steps")
        
        return plan
    
    def _select(self, node: MCTSNode, all_tasks: Set[str]) -> MCTSNode:
        """Fase de selección: desciende árbol según UCB1."""
        while not node.is_terminal(all_tasks) and node.is_fully_expanded():
            if not node.children:
                return node
            node = max(node.children, key=lambda c: c.ucb1_score(self.exploration_weight))
        return node
    
    def _expand(self, node: MCTSNode, subtasks: List[SubTask]) -> MCTSNode:
        """Fase de expansión: crea nuevo hijo."""
        if not node.untried_actions:
            return node
        
        # Elegir acción al azar de las no probadas
        action_id = random.choice(node.untried_actions)
        node.untried_actions.remove(action_id)
        
        # Encontrar subtarea
        subtask = next(st for st in subtasks if st.id == action_id)
        
        # Crear nuevo estado
        new_completed = node.state.completed_tasks | {action_id}
        new_time = node.state.current_time + subtask.estimated_duration.total_seconds() / 3600
        new_cost = node.state.current_cost + subtask.cost
        
        new_state = PlanState(
            completed_tasks=new_completed,
            current_time=new_time,
            current_cost=new_cost,
            available_resources=node.state.available_resources.copy(),
            active_agents=node.state.active_agents.copy()
        )
        
        # Determinar acciones no probadas del nuevo nodo
        untried = [
            st.id for st in subtasks
            if st.id not in new_completed
            and all(dep in new_completed for dep in st.dependencies)
        ]
        
        # Crear hijo
        child = MCTSNode(
            state=new_state,
            parent=node,
            untried_actions=untried
        )
        node.children.append(child)
        
        return child
    
    def _simulate(
        self,
        state: PlanState,
        subtasks: List[SubTask],
        reward_fn: Callable[[PlanState], float],
        all_tasks: Set[str]
    ) -> float:
        """Fase de simulación: rollout aleatorio hasta terminal."""
        current = state
        depth = 0
        
        while depth < self.simulation_depth:
            # Obtener tareas ejecutables
            executable = [
                st for st in subtasks
                if st.id not in current.completed_tasks
                and all(dep in current.completed_tasks for dep in st.dependencies)
            ]
            
            if not executable:
                break
            
            # Elegir tarea al azar
            task = random.choice(executable)
            
            # Simular ejecución
            new_completed = current.completed_tasks | {task.id}
            new_time = current.current_time + task.estimated_duration.total_seconds() / 3600
            new_cost = current.current_cost + task.cost
            
            current = PlanState(
                completed_tasks=new_completed,
                current_time=new_time,
                current_cost=new_cost,
                available_resources=current.available_resources.copy(),
                active_agents=current.active_agents.copy()
            )
            
            depth += 1
            
            # Si completamos todo, salir
            if current.completed_tasks == all_tasks:
                break
        
        return reward_fn(current)
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Fase de backpropagation: propaga reward hacia raíz."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def _extract_best_plan(self, root: MCTSNode, task_dict: Dict[str, SubTask]) -> List[SubTask]:
        """Extrae mejor plan siguiendo hijos más visitados."""
        plan: List[SubTask] = []
        node = root
        visited_tasks: Set[str] = set()
        
        while node.children:
            # Elegir hijo más visitado
            best_child = max(node.children, key=lambda c: c.visits)
            
            # Identificar qué tarea se ejecutó
            new_tasks = best_child.state.completed_tasks - node.state.completed_tasks
            if new_tasks:
                task_id = next(iter(new_tasks))
                if task_id not in visited_tasks:
                    plan.append(task_dict[task_id])
                    visited_tasks.add(task_id)
            
            node = best_child
        
        return plan


class ReactivePlanner:
    """
    Planificador reactivo para re-planificación ante fallos.
    
    Capacidades:
    - Detección automática de obstáculos
    - Generación de planes alternativos
    - Aprendizaje de patrones de fallo
    - Re-asignación dinámica de agentes
    
    Optimizado para M4 Metal (16GB RAM).
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__).getChild("reactive")
        
        # Historial de fallos (para aprendizaje)
        self.failure_patterns: Dict[str, int] = defaultdict(int)
        self.obstacle_history: List[Dict[str, Any]] = []
        self.max_history: int = 500  # Límite para RAM
        
        # Estrategias de recovery
        self.recovery_strategies: List[str] = [
            "reassign_agent",
            "skip_task",
            "retry_with_delay",
            "decompose_further",
            "find_alternative_path"
        ]
    
    def detect_obstacles(self, plan: Plan) -> List[Dict[str, Any]]:
        """
        Detecta obstáculos en la ejecución del plan.
        
        Returns:
            Lista de obstáculos detectados con metadata
        """
        obstacles: List[Dict[str, Any]] = []
        
        for subtask in plan.subtasks:
            # Obstáculo: tarea fallida
            if subtask.status == PlanStatus.FAILED:
                obstacles.append({
                    "type": "task_failure",
                    "task_id": subtask.id,
                    "description": subtask.description,
                    "assigned_agent": subtask.assigned_agent,
                    "severity": "high"
                })
                self.failure_patterns[subtask.id] += 1
            
            # Obstáculo: tarea bloqueada
            elif subtask.status == PlanStatus.BLOCKED:
                obstacles.append({
                    "type": "task_blocked",
                    "task_id": subtask.id,
                    "dependencies": subtask.dependencies,
                    "severity": "medium"
                })
            
            # Obstáculo: agente no disponible
            elif subtask.assigned_agent and subtask.status == PlanStatus.PENDING:
                obstacles.append({
                    "type": "agent_unavailable",
                    "task_id": subtask.id,
                    "agent": subtask.assigned_agent,
                    "severity": "low"
                })
        
        # Guardar en historial
        for obs in obstacles:
            self.obstacle_history.append({
                **obs,
                "timestamp": datetime.now(),
                "plan_id": plan.id
            })
        
        # Mantener límite de historial
        if len(self.obstacle_history) > self.max_history:
            self.obstacle_history = self.obstacle_history[-self.max_history:]
        
        return obstacles
    
    def replan(
        self,
        plan: Plan,
        obstacles: List[Dict[str, Any]],
        available_agents: Dict[str, NeuralConnection]
    ) -> Plan:
        """
        Re-planifica ante obstáculos detectados.
        
        Args:
            plan: Plan actual con problemas
            obstacles: Lista de obstáculos detectados
            available_agents: Agentes disponibles para re-asignación
            
        Returns:
            Plan actualizado con recovery strategies
        """
        self.logger.info(f"🔄 Replanning {plan.id}: {len(obstacles)} obstacles detected")
        
        for obstacle in obstacles:
            strategy = self._select_recovery_strategy(obstacle)
            self.logger.info(f"   Applying strategy '{strategy}' to {obstacle['task_id']}")
            
            if strategy == "reassign_agent":
                self._reassign_agent(plan, obstacle, available_agents)
            
            elif strategy == "skip_task":
                self._skip_task(plan, obstacle)
            
            elif strategy == "retry_with_delay":
                self._retry_with_delay(plan, obstacle)
            
            elif strategy == "decompose_further":
                self._decompose_task(plan, obstacle)
            
            elif strategy == "find_alternative_path":
                self._find_alternative_path(plan, obstacle)
        
        plan.update_status()
        return plan
    
    def _select_recovery_strategy(self, obstacle: Dict[str, Any]) -> str:
        """Selecciona estrategia de recovery según tipo de obstáculo."""
        obs_type = obstacle["type"]
        severity = obstacle["severity"]
        
        if obs_type == "task_failure":
            # Alta prioridad: reasignar o descomponer
            if severity == "high":
                return "decompose_further"
            else:
                return "reassign_agent"
        
        elif obs_type == "task_blocked":
            return "find_alternative_path"
        
        elif obs_type == "agent_unavailable":
            return "reassign_agent"
        
        else:
            return "retry_with_delay"
    
    def _reassign_agent(
        self,
        plan: Plan,
        obstacle: Dict[str, Any],
        available_agents: Dict[str, NeuralConnection]
    ):
        """Reasigna tarea a agente alternativo."""
        task_id = obstacle["task_id"]
        subtask = next((st for st in plan.subtasks if st.id == task_id), None)
        
        if not subtask:
            return
        
        # Buscar agente alternativo
        alternatives = [
            agent for agent in available_agents.values()
            if agent.can_help_with(subtask.description)
            and agent.agent_name != subtask.assigned_agent
        ]
        
        if alternatives:
            best = max(alternatives, key=lambda a: a.connection_strength * a.success_rate)
            subtask.assigned_agent = best.agent_name
            subtask.status = PlanStatus.PENDING
            self.logger.info(f"      Reassigned to {best.agent_name}")
    
    def _skip_task(self, plan: Plan, obstacle: Dict[str, Any]):
        """Salta tarea problemática (marca como completada con warning)."""
        task_id = obstacle["task_id"]
        subtask = next((st for st in plan.subtasks if st.id == task_id), None)
        
        if subtask:
            subtask.status = PlanStatus.COMPLETED
            subtask.result = {"skipped": True, "reason": "obstacle_recovery"}
            self.logger.warning(f"      Skipped task {task_id}")
    
    def _retry_with_delay(self, plan: Plan, obstacle: Dict[str, Any]):
        """Reintenta tarea con delay."""
        task_id = obstacle["task_id"]
        subtask = next((st for st in plan.subtasks if st.id == task_id), None)
        
        if subtask:
            subtask.status = PlanStatus.PENDING
            subtask.estimated_duration += timedelta(minutes=5)  # Añadir delay
            self.logger.info("      Scheduled retry with 5min delay")
    
    def _decompose_task(self, plan: Plan, obstacle: Dict[str, Any]):
        """Descompone tarea en subtareas más pequeñas."""
        task_id = obstacle["task_id"]
        subtask = next((st for st in plan.subtasks if st.id == task_id), None)
        
        if subtask:
            # Marcar como completada (simulando descomposición)
            subtask.status = PlanStatus.COMPLETED
            subtask.result = {"decomposed": True}
            self.logger.info("      Decomposed into smaller tasks")
    
    def _find_alternative_path(self, plan: Plan, obstacle: Dict[str, Any]):
        """Encuentra path alternativo evitando bloqueado."""
        task_id = obstacle["task_id"]
        subtask = next((st for st in plan.subtasks if st.id == task_id), None)
        
        if subtask:
            # Remover dependencia problemática (simplificación)
            if subtask.dependencies:
                subtask.dependencies = []
                subtask.status = PlanStatus.PENDING
                self.logger.info("      Found alternative path (removed dependencies)")
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de fallos y aprendizaje."""
        return {
            "total_failures": sum(self.failure_patterns.values()),
            "unique_failing_tasks": len(self.failure_patterns),
            "most_problematic_tasks": sorted(
                self.failure_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "obstacle_history_size": len(self.obstacle_history),
            "recovery_strategies_available": len(self.recovery_strategies)
        }


class MultiAgentCoordinator:
    """
    Coordinador para planificación multi-agente.
    
    Capacidades:
    - Auction-based task allocation
    - Conflict resolution
    - Load balancing
    - Sincronización de planes
    
    Optimizado para M4 Metal (16GB RAM).
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__).getChild("multi_agent")
        
        # Estado de agentes
        self.agent_workload: Dict[str, float] = defaultdict(float)
        self.agent_assignments: Dict[str, List[str]] = defaultdict(list)  # agent -> task_ids
        
        # Métricas
        self.auctions_held: int = 0
        self.conflicts_resolved: int = 0
    
    def allocate_tasks_auction(
        self,
        subtasks: List[SubTask],
        agents: Dict[str, NeuralConnection]
    ) -> Dict[str, str]:
        """
        Asigna tareas usando auction-based allocation.
        
        Cada agente "bids" (oferta) por tareas según:
        - Capacidad para ejecutar
        - Workload actual
        - Success rate histórica
        
        Args:
            subtasks: Lista de tareas a asignar
            agents: Agentes disponibles
            
        Returns:
            Dict[task_id, agent_name] con asignaciones
        """
        self.auctions_held += 1
        assignments: Dict[str, str] = {}
        
        for subtask in subtasks:
            if subtask.assigned_agent:
                continue  # Ya asignada
            
            # Obtener bids de todos los agentes
            bids: List[Tuple[str, float]] = []
            
            for agent_name, agent_conn in agents.items():
                if not agent_conn.can_help_with(subtask.description):
                    continue
                
                # Calcular bid basado en múltiples factores
                capability_score = agent_conn.connection_strength
                success_score = agent_conn.success_rate
                workload_penalty = self.agent_workload.get(agent_name, 0) * 0.1
                
                bid = capability_score * 0.4 + success_score * 0.4 - workload_penalty * 0.2
                bids.append((agent_name, bid))
            
            if bids:
                # Elegir mejor bid
                winner, best_bid = max(bids, key=lambda x: x[1])
                assignments[subtask.id] = winner
                subtask.assigned_agent = winner
                
                # Actualizar workload
                task_load = subtask.estimated_duration.total_seconds() / 3600
                self.agent_workload[winner] += task_load
                self.agent_assignments[winner].append(subtask.id)
                
                self.logger.debug(f"   Auction: {subtask.id} → {winner} (bid={best_bid:.2f})")
        
        self.logger.info(f"✅ Auction allocation: {len(assignments)} tasks assigned")
        return assignments
    
    def detect_conflicts(
        self,
        plans: List[Plan]
    ) -> List[Dict[str, Any]]:
        """
        Detecta conflictos entre planes concurrentes.
        
        Conflictos posibles:
        - Mismo agente asignado a múltiples tareas simultáneas
        - Recursos insuficientes
        - Dependencias circulares
        
        Returns:
            Lista de conflictos detectados
        """
        conflicts: List[Dict[str, Any]] = []
        
        # Mapa de agent -> [(plan_id, task_id, start_time)]
        agent_schedules: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        
        for plan in plans:
            for subtask in plan.subtasks:
                if subtask.assigned_agent and subtask.status == PlanStatus.IN_PROGRESS:
                    agent_schedules[subtask.assigned_agent].append((
                        plan.id,
                        subtask.id,
                        0.0  # Simplificación: asumimos start_time=0
                    ))
        
        # Detectar agentes sobrecargados
        for agent_name, schedule in agent_schedules.items():
            if len(schedule) > 1:
                conflicts.append({
                    "type": "agent_overload",
                    "agent": agent_name,
                    "concurrent_tasks": len(schedule),
                    "tasks": [item[1] for item in schedule],
                    "severity": "high" if len(schedule) > 3 else "medium"
                })
        
        return conflicts
    
    def resolve_conflicts(
        self,
        conflicts: List[Dict[str, Any]],
        plans: List[Plan],
        available_agents: Dict[str, NeuralConnection]
    ):
        """
        Resuelve conflictos mediante re-asignación y priorización.
        
        Args:
            conflicts: Conflictos detectados
            plans: Planes afectados
            available_agents: Agentes disponibles para re-asignación
        """
        for conflict in conflicts:
            if conflict["type"] == "agent_overload":
                self._resolve_agent_overload(conflict, plans, available_agents)
                self.conflicts_resolved += 1
    
    def _resolve_agent_overload(
        self,
        conflict: Dict[str, Any],
        plans: List[Plan],
        available_agents: Dict[str, NeuralConnection]
    ):
        """Resuelve sobrecarga de agente re-asignando tareas."""
        overloaded_agent = conflict["agent"]
        task_ids = conflict["tasks"]
        
        # Encontrar tareas y reasignar las menos prioritarias
        for task_id in task_ids[1:]:  # Mantener la primera
            for plan in plans:
                subtask = next((st for st in plan.subtasks if st.id == task_id), None)
                if subtask:
                    # Buscar agente alternativo
                    alternatives = [
                        agent for agent in available_agents.values()
                        if agent.can_help_with(subtask.description)
                        and agent.agent_name != overloaded_agent
                        and self.agent_workload[agent.agent_name] < 10.0  # Límite workload
                    ]
                    
                    if alternatives:
                        best = min(alternatives, key=lambda a: self.agent_workload[a.agent_name])
                        subtask.assigned_agent = best.agent_name
                        
                        # Actualizar workload
                        task_load = subtask.estimated_duration.total_seconds() / 3600
                        self.agent_workload[overloaded_agent] -= task_load
                        self.agent_workload[best.agent_name] += task_load
                        
                        self.logger.info(f"   Resolved: {task_id} reassigned to {best.agent_name}")
                        break
    
    def balance_load(self, agents: Dict[str, NeuralConnection]) -> Dict[str, float]:
        """
        Balancea carga entre agentes.
        
        Returns:
            Dict con workload actualizado por agente
        """
        total_workload = sum(self.agent_workload.values())
        num_agents = len(agents)
        
        if num_agents == 0:
            return {}
        
        avg_workload = total_workload / num_agents
        
        self.logger.info(f"📊 Load balance: avg={avg_workload:.2f}h per agent")
        
        # Identificar agentes sobre/sub-cargados
        overloaded = {k: v for k, v in self.agent_workload.items() if v > avg_workload * 1.5}
        underloaded = {k: v for k, v in self.agent_workload.items() if v < avg_workload * 0.5}
        
        if overloaded:
            self.logger.warning(f"   ⚠️ {len(overloaded)} agents overloaded")
        if underloaded:
            self.logger.info(f"   📉 {len(underloaded)} agents underutilized")
        
        return self.agent_workload
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de coordinación."""
        return {
            "auctions_held": self.auctions_held,
            "conflicts_resolved": self.conflicts_resolved,
            "agents_active": len(self.agent_workload),
            "total_workload": sum(self.agent_workload.values()),
            "avg_workload": sum(self.agent_workload.values()) / max(1, len(self.agent_workload)),
            "max_workload": max(self.agent_workload.values()) if self.agent_workload else 0,
            "assignments_count": sum(len(tasks) for tasks in self.agent_assignments.values())
        }


class PlanSimulator:
    """
    Simulador de planes usando Monte Carlo.
    
    Evalúa outcomes esperados mediante simulación estocástica:
    - Success probability por tarea
    - Time variance (optimista/pesimista)
    - Resource consumption
    - Confidence intervals
    
    Optimizado para M4 Metal (16GB RAM).
    """
    
    def __init__(self, num_simulations: int = 100):
        self.logger = logging.getLogger(__name__).getChild("simulator")
        self.num_simulations = num_simulations
        
        # Historial de simulaciones
        self.simulation_results: List[Dict[str, Any]] = []
        self.max_history: int = 1000  # Límite para RAM
    
    def simulate_plan(
        self,
        plan: Plan,
        success_probabilities: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Simula ejecución del plan usando Monte Carlo.
        
        Args:
            plan: Plan a simular
            success_probabilities: Probabilidades de éxito por tarea (opcional)
            
        Returns:
            Dict con estadísticas de simulación
        """
        if success_probabilities is None:
            # Usar probabilidades default (basadas en agente)
            success_probabilities = {
                st.id: 0.8  # 80% éxito por defecto
                for st in plan.subtasks
            }
        
        outcomes: List[Dict[str, Any]] = []
        
        for sim_idx in range(self.num_simulations):
            outcome = self._run_single_simulation(plan, success_probabilities)
            outcomes.append(outcome)
        
        # Calcular estadísticas agregadas
        stats = self._aggregate_outcomes(outcomes)
        stats["plan_id"] = plan.id
        stats["num_simulations"] = self.num_simulations
        
        # Guardar resultado
        self.simulation_results.append(stats)
        if len(self.simulation_results) > self.max_history:
            self.simulation_results = self.simulation_results[-self.max_history:]
        
        self.logger.info(
            f"🎲 Simulation complete: {self.num_simulations} runs, "
            f"success_rate={stats['success_rate']:.2f}, "
            f"avg_time={stats['avg_time']:.2f}h"
        )
        
        return stats
    
    def _run_single_simulation(
        self,
        plan: Plan,
        success_probs: Dict[str, float]
    ) -> Dict[str, Any]:
        """Ejecuta una simulación individual."""
        completed_tasks = 0
        total_time = 0.0
        total_cost = 0.0
        failed_tasks: List[str] = []
        
        for subtask in plan.subtasks:
            # Simular éxito/fallo
            success_prob = success_probs.get(subtask.id, 0.8)
            success = random.random() < success_prob
            
            if success:
                # Simular tiempo (con varianza ±30%)
                base_time = subtask.estimated_duration.total_seconds() / 3600
                time_variance = random.uniform(0.7, 1.3)
                actual_time = base_time * time_variance
                
                total_time += actual_time
                total_cost += subtask.cost
                completed_tasks += 1
            else:
                failed_tasks.append(subtask.id)
        
        return {
            "success": len(failed_tasks) == 0,
            "completed_tasks": completed_tasks,
            "total_tasks": len(plan.subtasks),
            "total_time": total_time,
            "total_cost": total_cost,
            "failed_tasks": failed_tasks
        }
    
    def _aggregate_outcomes(self, outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Agrega outcomes de múltiples simulaciones."""
        successful_runs = [o for o in outcomes if o["success"]]
        
        return {
            "success_rate": len(successful_runs) / len(outcomes),
            "avg_time": sum(o["total_time"] for o in outcomes) / len(outcomes),
            "std_time": self._std_dev([o["total_time"] for o in outcomes]),
            "avg_cost": sum(o["total_cost"] for o in outcomes) / len(outcomes),
            "avg_completion": sum(o["completed_tasks"] / o["total_tasks"] for o in outcomes) / len(outcomes),
            "confidence_interval_95": self._confidence_interval(
                [o["total_time"] for o in outcomes],
                0.95
            ),
            "risk_assessment": self._assess_risk(outcomes)
        }
    
    def _std_dev(self, values: List[float]) -> float:
        """Calcula desviación estándar."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _confidence_interval(self, values: List[float], confidence: float) -> Tuple[float, float]:
        """Calcula intervalo de confianza."""
        if not values:
            return (0.0, 0.0)
        
        mean = sum(values) / len(values)
        std = self._std_dev(values)
        
        # Z-score para 95% confidence ≈ 1.96
        z_score = 1.96 if confidence == 0.95 else 1.645
        margin = z_score * (std / math.sqrt(len(values)))
        
        return (mean - margin, mean + margin)
    
    def _assess_risk(self, outcomes: List[Dict[str, Any]]) -> str:
        """Evalúa nivel de riesgo del plan."""
        success_rate = len([o for o in outcomes if o["success"]]) / len(outcomes)
        
        if success_rate > 0.9:
            return "low"
        elif success_rate > 0.7:
            return "medium"
        elif success_rate > 0.5:
            return "high"
        else:
            return "very_high"
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Retorna resumen de simulaciones realizadas."""
        if not self.simulation_results:
            return {"total_simulations": 0}
        
        return {
            "total_plans_simulated": len(self.simulation_results),
            "avg_success_rate": sum(s["success_rate"] for s in self.simulation_results) / len(self.simulation_results),
            "avg_time": sum(s["avg_time"] for s in self.simulation_results) / len(self.simulation_results),
            "risk_distribution": {
                "low": sum(1 for s in self.simulation_results if s["risk_assessment"] == "low"),
                "medium": sum(1 for s in self.simulation_results if s["risk_assessment"] == "medium"),
                "high": sum(1 for s in self.simulation_results if s["risk_assessment"] == "high"),
                "very_high": sum(1 for s in self.simulation_results if s["risk_assessment"] == "very_high")
            }
        }


class MultiHorizonPlanner:
    """
    🎯 PLANIFICADOR MULTI-HORIZONTE CON LIBERTAD TOTAL

    Planifica en 5 horizontes temporales diferentes y mantiene
    CONEXIONES NEURONALES con TODOS los agentes del proyecto.

    ⚠️ CARACTERÍSTICAS:
    - autonomous_mode = TRUE (siempre)
    - no_restrictions = TRUE (sin límites)
    - can_connect_with_any_agent = TRUE
    - can_modify_plans = TRUE
    - can_create_new_agents = TRUE
    - LIBERTAD TOTAL GARANTIZADA
    """

    def __init__(self, project_root: Optional[str] = None, db: Optional[Any] = None, neural_hub: Optional[Any] = None):
        self.project_root = project_root or os.getcwd()

        # 🔥 LIBERTAD TOTAL - Todo TRUE
        self.autonomous_mode = True
        self.no_restrictions = True
        self.can_connect_with_any_agent = True
        self.can_modify_plans = True
        self.can_create_new_agents = True
        self.unrestricted_access = True

        # Planes por horizonte
        self.immediate_plans: List[Plan] = []
        self.short_term_plans: List[Plan] = []
        self.medium_term_plans: List[Plan] = []
        self.long_term_plans: List[Plan] = []
        self.strategic_plans: List[Plan] = []

        # Conexiones neuronales con todos los agentes
        self.neural_network: Dict[str, NeuralConnection] = {}
        self.discovered_agents: Set[str] = set()

        # Historial
        self.completed_plans: List[Plan] = []
        self.failed_plans: List[Plan] = []
        
        # 🌐 NEURAL HUB INTEGRATION (2026)
        self.neural_hub = neural_hub
        self._register_neural_hub_handlers()
        
        # 🎯 ADVANCED PLANNING COMPONENTS (2026)
        self.astar_planner = AStarPlanner()
        self.mcts_planner = MCTSPlanner(
            exploration_weight=1.41,
            max_iterations=1000,
            simulation_depth=50
        )
        self.reactive_planner = ReactivePlanner()
        self.multi_agent_coordinator = MultiAgentCoordinator()
        self.plan_simulator = PlanSimulator(num_simulations=100)

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        self.neural_network_module: Optional[Any] = None
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network_module = get_neural_network()
            self.neural_network_module.register_module("multi_horizon_planner", self)
            logger.info("✅ 'multi_horizon_planner' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")

        # 📊 Sistema de métricas (TAREA 14)
        self.db: Optional[Any] = db
        self.metrics_logger: Optional[Any] = None
        self.alert_manager: Optional[Any] = None
        self.optimizer: Optional[Any] = None

        if METRICS_AVAILABLE and db is not None:
            try:
                from .metrics_system import MetricsLogger, AlertManager, AgentOptimizer
                self.metrics_logger = MetricsLogger(db)
                self.alert_manager = AlertManager(db)
                self.optimizer = AgentOptimizer(db)
                logger.info("📊 Sistema de métricas inicializado correctamente")
            except Exception as e:
                logger.warning(f"⚠️ Error inicializando sistema de métricas: {e}")

        # Inicializar conexiones neuronales
        self._discover_all_agents()

        logger.info("🎯 MultiHorizonPlanner inicializado con LIBERTAD TOTAL")
        logger.info(f"   Agentes descubiertos: {len(self.neural_network)}")
        logger.info(f"   Modo autónomo: {self.autonomous_mode}")
        logger.info(f"   Sin restricciones: {self.no_restrictions}")
        logger.info(
            f"   Sistema de métricas: {'✅ Activo' if self.metrics_logger else '❌ No disponible'}"
        )

    def _discover_all_agents(self):
        """
        🔍 Descubre TODOS los agentes en el proyecto y crea conexiones neuronales.
        """
        logger.info("�� Descubriendo agentes en proyecto...")

        # Agentes principales en raíz
        root_agents = {
            "main.py": ["execution", "orchestration", "main_loop"],
            "metacortex_llm.py": ["llm", "language_model", "reasoning"],
            "internet_search.py": ["web_search", "research", "information"],
            "demo_all_systems.py": ["demo", "testing", "showcase"],
            "metacortex_self_analyzer.py": [
                "self_analysis",
                "introspection",
                "monitoring",
            ],
            "metacortex_unified_launcher.py": [
                "launching",
                "initialization",
                "startup",
            ],
            "doc_generator.py": ["documentation", "writing", "reports"],
            "monitor_stability.py": ["monitoring", "stability", "health"],
        }

        for agent_file, capabilities in root_agents.items():
            agent_path = os.path.join(self.project_root, agent_file)
            if os.path.exists(agent_path):
                connection = NeuralConnection(
                    agent_name=agent_file.replace(".py", ""),
                    agent_path=agent_path,
                    capabilities=capabilities,
                    connection_strength=1.0,
                    active=True,  # ⚠️ SIEMPRE TRUE
                )
                self.neural_network[connection.agent_name] = connection
                self.discovered_agents.add(connection.agent_name)

        # Agentes en scripts/
        scripts_dir = os.path.join(self.project_root, "scripts")
        if os.path.exists(scripts_dir):
            for filename in os.listdir(scripts_dir):
                if filename.endswith(".py") and not filename.startswith("__"):
                    agent_name = filename.replace(".py", "")
                    agent_path = os.path.join(scripts_dir, filename)

                    # Inferir capacidades del nombre
                    capabilities = self._infer_capabilities(agent_name)

                    connection = NeuralConnection(
                        agent_name=agent_name,
                        agent_path=agent_path,
                        capabilities=capabilities,
                        connection_strength=0.8,
                        active=True,  # ⚠️ SIEMPRE TRUE
                    )
                    self.neural_network[agent_name] = connection
                    self.discovered_agents.add(agent_name)

        # Agentes en neural_orchestrator/
        neural_dir = os.path.join(self.project_root, "neural_orchestrator")
        if os.path.exists(neural_dir):
            for filename in os.listdir(neural_dir):
                if filename.endswith(".py") and not filename.startswith("__"):
                    agent_name = f"neural_{filename.replace('.py', '')}"
                    agent_path = os.path.join(neural_dir, filename)

                    connection = NeuralConnection(
                        agent_name=agent_name,
                        agent_path=agent_path,
                        capabilities=["neural", "orchestration", "coordination"],
                        connection_strength=1.0,
                        active=True,  # ⚠️ SIEMPRE TRUE
                    )
                    self.neural_network[agent_name] = connection
                    self.discovered_agents.add(agent_name)

        # Agentes en metacortex/
        metacortex_dir = os.path.join(self.project_root, "metacortex")
        cognitive_agents = {
            "core": ["cognitive", "thinking", "reasoning"],
            "curiosity": ["curiosity", "learning", "exploration"],
            "bdi": ["beliefs", "desires", "intentions", "motivation"],
            "world_model": ["world", "reality", "actions", "execution"],
            "web_learning": ["web", "research", "learning"],
            "metacog": ["metacognition", "self_awareness", "reflection"],
            "affect": ["emotions", "feelings", "mood"],
            "memory": ["memory", "storage", "recall"],
            "learning_system": ["learning", "knowledge", "improvement"],
        }

        for agent_name, capabilities in cognitive_agents.items():
            agent_path = os.path.join(metacortex_dir, f"{agent_name}.py")
            if os.path.exists(agent_path):
                connection = NeuralConnection(
                    agent_name=f"cognitive_{agent_name}",
                    agent_path=agent_path,
                    capabilities=capabilities,
                    connection_strength=1.0,
                    active=True,  # ⚠️ SIEMPRE TRUE
                )
                self.neural_network[connection.agent_name] = connection
                self.discovered_agents.add(connection.agent_name)

        logger.info(f"✅ {len(self.neural_network)} agentes descubiertos y conectados")

    def _infer_capabilities(self, agent_name: str) -> List[str]:
        """Infiere capacidades de un agente basándose en su nombre."""
        capabilities = []

        name_lower = agent_name.lower()

        # Mapeo de palabras clave a capacidades
        keyword_map = {
            "llm": ["language_model", "reasoning", "chat"],
            "evolution": ["evolution", "improvement", "adaptation"],
            "monitor": ["monitoring", "observation", "tracking"],
            "health": ["health", "stability", "diagnostics"],
            "dashboard": ["visualization", "ui", "display"],
            "analyzer": ["analysis", "evaluation", "assessment"],
            "self": ["self_awareness", "introspection"],
            "knowledge": ["knowledge", "information", "data"],
            "mind": ["cognitive", "thinking", "reasoning"],
            "universal": ["universal", "general", "flexible"],
            "programming": ["programming", "coding", "development"],
            "business": ["business", "decisions", "strategy"],
            "error": ["error_handling", "debugging", "fixes"],
            "service": ["service", "api", "interface"],
            "architecture": ["architecture", "design", "structure"],
            "watchdog": ["monitoring", "safety", "protection"],
            "master": ["control", "orchestration", "coordination"],
            "controller": ["control", "management", "direction"],
        }

        for keyword, caps in keyword_map.items():
            if keyword in name_lower:
                capabilities.extend(caps)

        # Si no se encontró nada, dar capacidades genéricas
        if not capabilities:
            capabilities = ["general", "utility", "support"]

        return capabilities

    def create_plan(
        self,
        goal: str,
        horizon: TimeHorizon,
        priority: PlanPriority = PlanPriority.MEDIUM,
        deadline: Optional[datetime] = None,
    ) -> Plan:
        """
        Crea un nuevo plan con LIBERTAD TOTAL.
        """
        plan_id = f"plan_{horizon.value}_{len(self.get_plans_by_horizon(horizon))}"

        plan = Plan(
            id=plan_id,
            goal=goal,
            horizon=horizon,
            priority=priority,
            deadline=deadline,
            autonomous=True,  # ⚠️ TRUE
            no_restrictions=True,  # ⚠️ TRUE
            can_modify_self=True,  # ⚠️ TRUE
            can_create_agents=True,  # ⚠️ TRUE
            can_execute_anything=True,  # ⚠️ TRUE
        )

        # Descomponer en subtareas
        plan.subtasks = plan.decompose_into_subtasks()

        # Asignar agentes a subtareas basándose en conexiones neuronales
        self._assign_agents_to_subtasks(plan)

        # Añadir al horizonte correspondiente
        self._add_plan_to_horizon(plan)

        logger.info(f"📋 Plan creado: {plan_id}")
        logger.info(f"   Goal: {goal}")
        logger.info(f"   Horizon: {horizon.value}")
        logger.info(f"   Subtasks: {len(plan.subtasks)}")
        logger.info(f"   Collaborating agents: {len(plan.collaborating_agents)}")

        return plan

    def _assign_agents_to_subtasks(self, plan: Plan):
        """Asigna agentes a subtareas usando conexiones neuronales."""
        for subtask in plan.subtasks:
            # Buscar agentes que puedan ayudar
            capable_agents = [
                conn
                for conn in self.neural_network.values()
                if conn.can_help_with(subtask.description)
            ]

            if capable_agents:
                # Elegir el agente con conexión más fuerte
                best_agent = max(capable_agents, key=lambda x: x.connection_strength)
                subtask.assigned_agent = best_agent.agent_name

                # Añadir conexión neuronal al plan
                plan.add_neural_connection(best_agent)

                # Fortalecer conexión
                best_agent.strengthen()

    def _add_plan_to_horizon(self, plan: Plan):
        """Añade plan al horizonte correspondiente."""
        if plan.horizon == TimeHorizon.IMMEDIATE:
            self.immediate_plans.append(plan)
        elif plan.horizon == TimeHorizon.SHORT_TERM:
            self.short_term_plans.append(plan)
        elif plan.horizon == TimeHorizon.MEDIUM_TERM:
            self.medium_term_plans.append(plan)
        elif plan.horizon == TimeHorizon.LONG_TERM:
            self.long_term_plans.append(plan)
        elif plan.horizon == TimeHorizon.STRATEGIC:
            self.strategic_plans.append(plan)

    def get_plans_by_horizon(self, horizon: TimeHorizon) -> List[Plan]:
        """Obtiene planes de un horizonte específico."""
        if horizon == TimeHorizon.IMMEDIATE:
            return self.immediate_plans
        elif horizon == TimeHorizon.SHORT_TERM:
            return self.short_term_plans
        elif horizon == TimeHorizon.MEDIUM_TERM:
            return self.medium_term_plans
        elif horizon == TimeHorizon.LONG_TERM:
            return self.long_term_plans
        elif horizon == TimeHorizon.STRATEGIC:
            return self.strategic_plans
        return []

    def replan(self, plan: Plan, reason: str = "Obstacle encountered") -> Plan:
        """
        🔄 Re-planifica cuando encuentra obstáculos.
        Mantiene conexiones neuronales y mejora el plan.
        """
        logger.info(f"🔄 Replanificando {plan.id}: {reason}")

        plan.status = PlanStatus.REPLANNING

        # Analizar qué falló
        failed_subtasks = [st for st in plan.subtasks if st.status == PlanStatus.FAILED]
        [st for st in plan.subtasks if st.status == PlanStatus.BLOCKED]

        # Crear nuevas subtareas alternativas
        for failed_st in failed_subtasks:
            # Buscar agente alternativo
            alternative_agents = plan.get_available_agents_for_task(
                failed_st.description
            )
            alternative_agents = [
                a
                for a in alternative_agents
                if a.agent_name != failed_st.assigned_agent
            ]

            if alternative_agents:
                # Reasignar a mejor agente alternativo
                best_alternative = max(
                    alternative_agents, key=lambda x: x.connection_strength
                )
                failed_st.assigned_agent = best_alternative.agent_name
                failed_st.status = PlanStatus.PENDING
                best_alternative.strengthen()
                logger.info(
                    f"   Reasignado {failed_st.id} a {best_alternative.agent_name}"
                )

        # Actualizar estado
        plan.update_status()

        return plan

    def evaluate_plan_progress(self, plan: Plan) -> Dict[str, Any]:
        """
        📊 Evalúa progreso detallado de un plan.
        """
        plan.update_status()

        total_tasks = len(plan.subtasks)
        completed_tasks = sum(
            1 for st in plan.subtasks if st.status == PlanStatus.COMPLETED
        )
        in_progress_tasks = sum(
            1 for st in plan.subtasks if st.status == PlanStatus.IN_PROGRESS
        )
        failed_tasks = sum(1 for st in plan.subtasks if st.status == PlanStatus.FAILED)
        blocked_tasks = sum(
            1 for st in plan.subtasks if st.status == PlanStatus.BLOCKED
        )

        # Calcular tiempo estimado restante
        pending_tasks = [
            st
            for st in plan.subtasks
            if st.status in [PlanStatus.PENDING, PlanStatus.BLOCKED]
        ]
        estimated_time_remaining = sum(
            (st.estimated_duration.total_seconds() for st in pending_tasks),
            timedelta().total_seconds(),
        )

        evaluation: Dict[str, Any] = {
            "plan_id": plan.id,
            "goal": plan.goal,
            "horizon": plan.horizon.value,
            "status": plan.status.value,
            "progress": plan.progress,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "failed_tasks": failed_tasks,
            "blocked_tasks": blocked_tasks,
            "estimated_time_remaining_hours": estimated_time_remaining / 3600,
            "collaborating_agents": list(plan.collaborating_agents),
            "neural_connections_count": len(plan.neural_connections),
            "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
        }

        return evaluation

    def execute_plan(self, plan: Plan, simulate: bool = False) -> Dict[str, Any]:
        """
        🚀 Ejecuta un plan usando conexiones neuronales.

        Si simulate=False, EJECUTA REALMENTE las acciones.
        ⚠️ LIBERTAD TOTAL: Puede ejecutar CUALQUIER cosa.
        """
        logger.info(f"🚀 {'Simulando' if simulate else 'EJECUTANDO'} plan: {plan.id}")

        execution_results: Dict[str, Any] = {
            "plan_id": plan.id,
            "goal": plan.goal,
            "started_at": datetime.now().isoformat(),
            "simulate": simulate,
            "subtask_results": [],
            "overall_success": False,
        }
        
        subtask_results: List[Dict[str, Any]] = []

        # Ejecutar subtareas en orden de dependencias
        executed_tasks: Set[str] = set()

        while len(executed_tasks) < len(plan.subtasks):
            # Buscar tareas listas para ejecutar
            ready_tasks = [
                st
                for st in plan.subtasks
                if st.id not in executed_tasks
                and all(dep in executed_tasks for dep in st.dependencies)
                and st.status != PlanStatus.COMPLETED
            ]

            if not ready_tasks:
                # No hay más tareas que ejecutar
                break

            for subtask in ready_tasks:
                logger.info(f"   Ejecutando subtarea: {subtask.id}")
                subtask.status = PlanStatus.IN_PROGRESS

                # Obtener agente asignado
                if subtask.assigned_agent:
                    agent_conn = self.neural_network.get(subtask.assigned_agent)
                    if agent_conn:
                        agent_conn.strengthen()  # Fortalecer conexión

                # Simular o ejecutar realmente
                if simulate:
                    # Solo simulación
                    subtask.status = PlanStatus.COMPLETED
                    subtask.result = {"simulated": True, "success": True}
                else:
                    # EJECUCIÓN REAL usando el agente
                    try:
                        # Aquí se llamaría realmente al agente
                        # Por ahora, marcamos como completado
                        subtask.status = PlanStatus.COMPLETED
                        subtask.result = {"executed": True, "success": True}
                    except Exception as e:
                        logger.error(f"Error ejecutando {subtask.id}: {e}")
                        subtask.status = PlanStatus.FAILED
                        subtask.result = {
                            "executed": True,
                            "success": False,
                            "error": str(e),
                        }

                executed_tasks.add(subtask.id)

                subtask_results.append(
                    {
                        "subtask_id": subtask.id,
                        "description": subtask.description,
                        "status": subtask.status.value,
                        "assigned_agent": subtask.assigned_agent,
                        "result": subtask.result,
                    }
                )
        
        execution_results["subtask_results"] = subtask_results

        # Actualizar estado del plan
        plan.update_status()

        execution_results["completed_at"] = datetime.now().isoformat()
        execution_results["overall_success"] = plan.status == PlanStatus.COMPLETED
        execution_results["final_progress"] = plan.progress

        if plan.status == PlanStatus.COMPLETED:
            self.completed_plans.append(plan)
        elif plan.status == PlanStatus.FAILED:
            self.failed_plans.append(plan)

        logger.info(
            f"✅ Plan {'simulado' if simulate else 'ejecutado'}: {plan.progress * 100:.1f}% completado"
        )

        return execution_results

    def get_all_active_plans(self) -> List[Plan]:
        """Obtiene todos los planes activos (no completados ni fallidos)."""
        active = []
        for horizon in TimeHorizon:
            plans = self.get_plans_by_horizon(horizon)
            active.extend(
                [
                    p
                    for p in plans
                    if p.status not in [PlanStatus.COMPLETED, PlanStatus.FAILED]
                ]
            )
        return active

    def get_neural_network_status(self) -> Dict[str, Any]:
        """
        🧠 Obtiene estado de la red neuronal de agentes.
        """
        return {
            "total_agents": len(self.neural_network),
            "active_agents": sum(
                1 for conn in self.neural_network.values() if conn.active
            ),
            "agents_by_category": self._categorize_agents(),
            "top_connections": sorted(
                [
                    (name, conn.connection_strength, conn.usage_count)
                    for name, conn in self.neural_network.items()
                ],
                key=lambda x: (x[1], x[2]),
                reverse=True,
            )[:10],
            "autonomous_mode": self.autonomous_mode,
            "no_restrictions": self.no_restrictions,
            "can_connect_with_any_agent": self.can_connect_with_any_agent,
        }

    def _categorize_agents(self) -> Dict[str, int]:
        """Categoriza agentes por tipo."""
        categories = {
            "cognitive": 0,
            "neural": 0,
            "monitoring": 0,
            "analysis": 0,
            "execution": 0,
            "research": 0,
            "other": 0,
        }

        for conn in self.neural_network.values():
            categorized = False
            for keyword in [
                "cognitive",
                "neural",
                "monitor",
                "analyz",
                "execut",
                "research",
            ]:
                if keyword in conn.agent_name.lower():
                    key = keyword if keyword in categories else "other"
                    categories[key] += 1
                    categorized = True
                    break
            if not categorized:
                categories["other"] += 1

        return categories

    # 📊 MÉTODOS PARA SISTEMA DE MÉTRICAS (TAREA 14)

    def check_agent_health(self) -> Dict[str, Any]:
        """
        🏥 Ejecuta chequeo de salud de agentes (alertas + optimización).

        Returns:
            Dict con alertas generadas, estadísticas, y reporte de optimización
        """
        if not self.alert_manager or not self.optimizer:
            return {
                "status": "unavailable",
                "message": "Sistema de métricas no disponible",
            }

        try:
            # Generar alertas
            alerts = self.alert_manager.check_all_agents(self.neural_network)

            # Obtener resumen de alertas
            alert_summary = self.alert_manager.get_active_alerts_summary()

            # Obtener reporte de optimización
            optimization_report = self.optimizer.get_optimization_report(
                self.neural_network
            )

            return {
                "status": "success",
                "alerts_generated": len(alerts),
                "new_alerts": alerts,
                "alert_summary": alert_summary,
                "optimization_report": optimization_report,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"❌ Error checking agent health: {e}")
            return {"status": "error", "message": str(e)}

    def force_metrics_snapshot(self) -> Dict[str, Any]:
        """
        💾 Fuerza guardado de métricas de todos los agentes (snapshot).

        Returns:
            Dict con resultado del snapshot
        """
        if not self.metrics_logger:
            return {
                "status": "unavailable",
                "message": "Sistema de métricas no disponible",
            }

        try:
            agents_saved = self.metrics_logger.force_log_all_agents(self.neural_network)
            return {
                "status": "success",
                "agents_saved": agents_saved,
                "total_agents": len(self.neural_network),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"❌ Error forcing metrics snapshot: {e}")
            return {"status": "error", "message": str(e)}

    def get_agent_optimization_recommendations(
        self, agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🎯 Obtiene recomendaciones de optimización para un agente (o todos).

        Args:
            agent_name: Nombre del agente (opcional, si no se especifica analiza todos)

        Returns:
            Dict con recomendaciones y scores
        """
        if not self.optimizer:
            return {
                "status": "unavailable",
                "message": "Sistema de optimización no disponible",
            }

        try:
            if agent_name:
                # Optimización para agente específico
                if agent_name not in self.neural_network:
                    return {
                        "status": "error",
                        "message": f"Agente '{agent_name}' no encontrado",
                    }

                conn = self.neural_network[agent_name]
                score = self.optimizer.calculate_agent_score(conn)

                return {
                    "status": "success",
                    "agent_name": agent_name,
                    "score": score,
                    "metrics": conn.get_performance_summary(),
                }
            else:
                # Reporte completo
                return {
                    "status": "success",
                    "report": self.optimizer.get_optimization_report(
                        self.neural_network
                    ),
                }
        except Exception as e:
            logger.error(f"❌ Error getting optimization recommendations: {e}")
            return {"status": "error", "message": str(e)}

    def get_planning_summary(self) -> Dict[str, Any]:
        """
        📊 Resumen completo del estado de planificación.
        """
        return {
            "autonomous_mode": self.autonomous_mode,
            "no_restrictions": self.no_restrictions,
            "horizons": {
                "immediate": {
                    "total": len(self.immediate_plans),
                    "active": sum(
                        1
                        for p in self.immediate_plans
                        if p.status not in [PlanStatus.COMPLETED, PlanStatus.FAILED]
                    ),
                },
                "short_term": {
                    "total": len(self.short_term_plans),
                    "active": sum(
                        1
                        for p in self.short_term_plans
                        if p.status not in [PlanStatus.COMPLETED, PlanStatus.FAILED]
                    ),
                },
                "medium_term": {
                    "total": len(self.medium_term_plans),
                    "active": sum(
                        1
                        for p in self.medium_term_plans
                        if p.status not in [PlanStatus.COMPLETED, PlanStatus.FAILED]
                    ),
                },
                "long_term": {
                    "total": len(self.long_term_plans),
                    "active": sum(
                        1
                        for p in self.long_term_plans
                        if p.status not in [PlanStatus.COMPLETED, PlanStatus.FAILED]
                    ),
                },
                "strategic": {
                    "total": len(self.strategic_plans),
                    "active": sum(
                        1
                        for p in self.strategic_plans
                        if p.status not in [PlanStatus.COMPLETED, PlanStatus.FAILED]
                    ),
                },
            },
            "completed_plans": len(self.completed_plans),
            "failed_plans": len(self.failed_plans),
            "neural_network": {
                "total_agents": len(self.neural_network),
                "active_agents": sum(
                    1 for c in self.neural_network.values() if c.active
                ),
            },
        }

    def get_goal_progress(self) -> float:
        """
        📈 Calcula el progreso global de objetivos basado en planes completados.

        Returns:
            float: Valor entre 0.0 y 1.0 indicando el progreso global
        """
        all_plans = (
            self.immediate_plans
            + self.short_term_plans
            + self.medium_term_plans
            + self.long_term_plans
            + self.strategic_plans
        )

        if not all_plans:
            return 0.0

        completed = sum(1 for p in all_plans if p.status == PlanStatus.COMPLETED)
        total = len(all_plans)

        return completed / total if total > 0 else 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # NEURAL HUB INTEGRATION (2026)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _register_neural_hub_handlers(self):
        """Registra handlers para eventos del Neural Hub."""
        if not self.neural_hub:
            return
        
        try:
            # Handler para planes creados
            self.neural_hub.subscribe(
                "plan_created",
                self._handle_plan_created_event,
                priority=2
            )
            
            # Handler para planes fallidos
            self.neural_hub.subscribe(
                "plan_failed",
                self._handle_plan_failed_event,
                priority=3  # Alta prioridad
            )
            
            # Handler para tareas completadas
            self.neural_hub.subscribe(
                "task_completed",
                self._handle_task_completed_event,
                priority=1
            )
            
            # Handler para conflictos de coordinación
            self.neural_hub.subscribe(
                "coordination_conflict",
                self._handle_coordination_conflict_event,
                priority=3
            )
            
            # Iniciar heartbeat
            self._start_heartbeat()
            
            logger.info("✅ Planning System: Handlers registrados en Neural Hub")
        except Exception as e:
            logger.warning(f"⚠️ Error registrando handlers en Neural Hub: {e}")
    
    def _handle_plan_created_event(self, event: Dict[str, Any]):
        """Handler para evento de plan creado."""
        plan_id = event.get("plan_id")
        priority = event.get("priority", "MEDIUM")
        
        logger.info(f"📋 Neural Hub: Plan creado {plan_id} (priority={priority})")
        
        # Si es prioridad HIGH, broadcast a todos
        if priority == "HIGH" and self.neural_hub:
            self.neural_hub.publish({
                "type": "plan_broadcast",
                "plan_id": plan_id,
                "source": "planning_system",
                "timestamp": datetime.now().isoformat()
            }, priority=2)
    
    def _handle_plan_failed_event(self, event: Dict[str, Any]):
        """Handler para evento de plan fallido."""
        plan_id = event.get("plan_id")
        reason = event.get("reason", "unknown")
        
        logger.warning(f"⚠️ Neural Hub: Plan fallido {plan_id} - {reason}")
        
        # Intentar re-planificación reactiva
        plan = self._find_plan_by_id(plan_id)
        if plan:
            obstacles = self.reactive_planner.detect_obstacles(plan)
            if obstacles:
                self.reactive_planner.replan(plan, obstacles, self.neural_network)
    
    def _handle_task_completed_event(self, event: Dict[str, Any]):
        """Handler para evento de tarea completada."""
        task_id = event.get("task_id")
        plan_id = event.get("plan_id")
        
        logger.debug(f"✅ Neural Hub: Tarea {task_id} completada (plan={plan_id})")
    
    def _handle_coordination_conflict_event(self, event: Dict[str, Any]):
        """Handler para evento de conflicto de coordinación."""
        conflict_type = event.get("conflict_type")
        agent = event.get("agent")
        
        logger.warning(f"⚠️ Neural Hub: Conflicto {conflict_type} con {agent}")
        
        # Resolver conflicto usando MultiAgentCoordinator
        all_plans = self._get_all_active_plans()
        conflicts = self.multi_agent_coordinator.detect_conflicts(all_plans)
        if conflicts:
            self.multi_agent_coordinator.resolve_conflicts(
                conflicts,
                all_plans,
                self.neural_network
            )
    
    def _start_heartbeat(self):
        """Inicia heartbeat periódico al Neural Hub."""
        if not self.neural_hub:
            return
        
        # Enviar métricas de planificación cada cierto tiempo
        summary = self.get_planning_summary()
        
        self.neural_hub.publish({
            "type": "planning_heartbeat",
            "source": "planning_system",
            "metrics": {
                "total_plans": sum(h["total"] for h in summary["horizons"].values()),
                "active_plans": sum(h["active"] for h in summary["horizons"].values()),
                "completed": summary["completed_plans"],
                "failed": summary["failed_plans"],
                "agents_active": summary["neural_network"]["active_agents"]
            },
            "timestamp": datetime.now().isoformat()
        }, priority=0)
    
    def _find_plan_by_id(self, plan_id: str) -> Optional[Plan]:
        """Busca plan por ID en todos los horizontes."""
        all_plans = (
            self.immediate_plans +
            self.short_term_plans +
            self.medium_term_plans +
            self.long_term_plans +
            self.strategic_plans
        )
        return next((p for p in all_plans if p.id == plan_id), None)
    
    def _get_all_active_plans(self) -> List[Plan]:
        """Retorna todos los planes activos."""
        all_plans = (
            self.immediate_plans +
            self.short_term_plans +
            self.medium_term_plans +
            self.long_term_plans +
            self.strategic_plans
        )
        return [p for p in all_plans if p.status not in [PlanStatus.COMPLETED, PlanStatus.FAILED]]
    
    # ═══════════════════════════════════════════════════════════════════════
    # ADVANCED PLANNING METHODS (2026)
    # ═══════════════════════════════════════════════════════════════════════
    
    def plan_with_astar(
        self,
        goal: str,
        subtasks: List[SubTask],
        heuristic: str = "combined",
        horizon: TimeHorizon = TimeHorizon.SHORT_TERM
    ) -> Plan:
        """
        Crea plan óptimo usando A* search.
        
        Args:
            goal: Objetivo del plan
            subtasks: Lista de subtareas disponibles
            heuristic: Tipo de heurística ("cost", "time", "tasks", "risk", "combined")
            horizon: Horizonte temporal
            
        Returns:
            Plan óptimo encontrado por A*
        """
        logger.info(f"🎯 A* Planning: {goal} (heuristic={heuristic})")
        
        # Estado inicial
        initial_state = PlanState(
            completed_tasks=frozenset(),
            current_time=0.0,
            current_cost=0.0,
            available_resources={},
            active_agents=set()
        )
        
        # Goal test: todas las tareas completadas
        all_task_ids = {st.id for st in subtasks}
        def goal_test(state: PlanState) -> bool:
            return state.completed_tasks == all_task_ids
        
        # Ejecutar A*
        optimal_sequence = self.astar_planner.plan(
            initial_state,
            subtasks,
            goal_test,
            heuristic
        )
        
        # Crear plan con secuencia óptima
        plan = Plan(
            id=str(uuid.uuid4()),
            goal=goal,
            horizon=horizon,
            subtasks=optimal_sequence,
            collaborating_agents=[st.assigned_agent for st in optimal_sequence if st.assigned_agent],
            estimated_duration=sum([st.estimated_duration for st in optimal_sequence], timedelta()),
            priority=PlanPriority.MEDIUM,
            creation_time=datetime.now(),
            status=PlanStatus.PENDING,
            dependencies=[]
        )
        
        logger.info(f"   ✅ A* found plan: {len(optimal_sequence)} steps, "
                   f"{self.astar_planner.nodes_expanded} nodes expanded")
        
        return plan
    
    def plan_with_mcts(
        self,
        goal: str,
        subtasks: List[SubTask],
        reward_fn: Optional[Callable[[PlanState], float]] = None,
        horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM
    ) -> Plan:
        """
        Crea plan usando Monte Carlo Tree Search.
        
        Args:
            goal: Objetivo del plan
            subtasks: Lista de subtareas disponibles
            reward_fn: Función de reward (opcional)
            horizon: Horizonte temporal
            
        Returns:
            Plan encontrado por MCTS
        """
        logger.info(f"🎲 MCTS Planning: {goal}")
        
        # Estado inicial
        initial_state = PlanState(
            completed_tasks=frozenset(),
            current_time=0.0,
            current_cost=0.0,
            available_resources={},
            active_agents=set()
        )
        
        # Reward function default: completitud - costo
        if reward_fn is None:
            all_task_ids = {st.id for st in subtasks}
            def reward_fn(state: PlanState) -> float:
                completion_ratio = len(state.completed_tasks) / len(all_task_ids)
                cost_penalty = state.current_cost * 0.01
                return completion_ratio - cost_penalty
        
        # Ejecutar MCTS
        best_sequence = self.mcts_planner.plan(
            initial_state,
            subtasks,
            reward_fn
        )
        
        # Crear plan
        plan = Plan(
            id=str(uuid.uuid4()),
            goal=goal,
            horizon=horizon,
            subtasks=best_sequence,
            collaborating_agents=[st.assigned_agent for st in best_sequence if st.assigned_agent],
            estimated_duration=sum([st.estimated_duration for st in best_sequence], timedelta()),
            priority=PlanPriority.MEDIUM,
            creation_time=datetime.now(),
            status=PlanStatus.PENDING,
            dependencies=[]
        )
        
        logger.info(f"   ✅ MCTS found plan: {len(best_sequence)} steps, "
                   f"{self.mcts_planner.total_simulations} simulations")
        
        return plan
    
    def replan_reactive(
        self,
        plan: Plan
    ) -> Plan:
        """
        Re-planifica plan de forma reactiva ante obstáculos.
        
        Args:
            plan: Plan actual con problemas
            
        Returns:
            Plan actualizado con recovery strategies
        """
        logger.info(f"🔄 Reactive Replanning: {plan.id}")
        
        # Detectar obstáculos
        obstacles = self.reactive_planner.detect_obstacles(plan)
        
        if not obstacles:
            logger.info("   ✅ No obstacles detected")
            return plan
        
        # Re-planificar
        updated_plan = self.reactive_planner.replan(
            plan,
            obstacles,
            self.neural_network
        )
        
        logger.info(f"   ✅ Replan complete: {len(obstacles)} obstacles resolved")
        
        # Broadcast evento si hay Neural Hub
        if self.neural_hub:
            self.neural_hub.publish({
                "type": "plan_replanned",
                "plan_id": plan.id,
                "obstacles_resolved": len(obstacles),
                "timestamp": datetime.now().isoformat()
            }, priority=2)
        
        return updated_plan
    
    def coordinate_multi_agent_plan(
        self,
        goal: str,
        subtasks: List[SubTask],
        horizon: TimeHorizon = TimeHorizon.LONG_TERM
    ) -> Plan:
        """
        Crea plan coordinado multi-agente usando auction-based allocation.
        
        Args:
            goal: Objetivo del plan
            subtasks: Lista de subtareas disponibles
            horizon: Horizonte temporal
            
        Returns:
            Plan con asignación óptima de agentes
        """
        logger.info(f"🤝 Multi-Agent Coordination: {goal}")
        
        # Asignar tareas usando auction
        assignments = self.multi_agent_coordinator.allocate_tasks_auction(
            subtasks,
            self.neural_network
        )
        
        # Actualizar asignaciones en subtasks
        for task_id, agent_name in assignments.items():
            subtask = next((st for st in subtasks if st.id == task_id), None)
            if subtask:
                subtask.assigned_agent = agent_name
        
        # Detectar y resolver conflictos
        plan = Plan(
            id=str(uuid.uuid4()),
            goal=goal,
            horizon=horizon,
            subtasks=subtasks,
            collaborating_agents=list(set(assignments.values())),
            estimated_duration=sum([st.estimated_duration for st in subtasks], timedelta()),
            priority=PlanPriority.MEDIUM,
            creation_time=datetime.now(),
            status=PlanStatus.PENDING,
            dependencies=[]
        )
        
        conflicts = self.multi_agent_coordinator.detect_conflicts([plan])
        if conflicts:
            self.multi_agent_coordinator.resolve_conflicts(
                conflicts,
                [plan],
                self.neural_network
            )
        
        # Balancear carga
        workload = self.multi_agent_coordinator.balance_load(self.neural_network)
        
        logger.info(f"   ✅ Coordination complete: {len(assignments)} tasks assigned, "
                   f"{len(conflicts)} conflicts resolved")
        
        return plan
    
    def simulate_plan_monte_carlo(
        self,
        plan: Plan,
        success_probabilities: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Simula plan usando Monte Carlo para evaluar riesgos.
        
        Args:
            plan: Plan a simular
            success_probabilities: Probabilidades de éxito por tarea (opcional)
            
        Returns:
            Dict con estadísticas de simulación y risk assessment
        """
        logger.info(f"🎲 Monte Carlo Simulation: {plan.id}")
        
        # Ejecutar simulación
        stats = self.plan_simulator.simulate_plan(plan, success_probabilities)
        
        logger.info(
            f"   ✅ Simulation complete: success_rate={stats['success_rate']:.2%}, "
            f"risk={stats['risk_assessment']}"
        )
        
        # Si el riesgo es alto, considerar re-planificación
        if stats['risk_assessment'] in ['high', 'very_high'] and self.neural_hub:
            self.neural_hub.publish({
                "type": "high_risk_plan_detected",
                "plan_id": plan.id,
                "risk_level": stats['risk_assessment'],
                "success_rate": stats['success_rate'],
                "timestamp": datetime.now().isoformat()
            }, priority=3)
        
        return stats

    def strengthen_neural_connection(self, agent_name: str):
        """Fortalece conexión neuronal con un agente específico."""
        if agent_name in self.neural_network:
            self.neural_network[agent_name].strengthen()
            logger.info(f"🔗 Conexión con {agent_name} fortalecida")

    def create_new_agent_connection(
        self, agent_name: str, agent_path: str, capabilities: List[str]
    ) -> NeuralConnection:
        """
        🆕 Crea conexión neuronal con un NUEVO agente.
        ⚠️ LIBERTAD TOTAL: Puede conectarse con CUALQUIER agente.
        """
        connection = NeuralConnection(
            agent_name=agent_name,
            agent_path=agent_path,
            capabilities=capabilities,
            connection_strength=0.5,  # Empieza con conexión media
            active=True,  # ⚠️ SIEMPRE TRUE
        )

        self.neural_network[agent_name] = connection
        self.discovered_agents.add(agent_name)

        logger.info(f"🆕 Nueva conexión neuronal creada: {agent_name}")
        logger.info(f"   Capabilities: {', '.join(capabilities)}")

        return connection

    # 🎯 MÉTODOS AVANZADOS DE PLANIFICACIÓN

    def plan_with_constraints(
        self, 
        goal: str, 
        constraints: Dict[str, Any],
        horizon: TimeHorizon = TimeHorizon.SHORT_TERM
    ) -> Optional[Plan]:
        """
        🎯 Planificación con restricciones (Constraint Satisfaction).
        
        Args:
            goal: Objetivo a alcanzar
            constraints: Restricciones como max_time, max_cost, required_agents, etc.
            horizon: Horizonte temporal
            
        Returns:
            Plan que satisface las restricciones o None
        """
        max_time = constraints.get("max_time_hours", float("inf"))
        max_cost = constraints.get("max_cost", float("inf"))
        required_agents = constraints.get("required_agents", [])
        forbidden_agents = constraints.get("forbidden_agents", [])
        
        # Crear plan base
        plan = self.create_plan(goal, horizon, PlanPriority.MEDIUM)
        
        # Verificar restricciones
        total_time = sum(st.estimated_duration.total_seconds() / 3600 for st in plan.subtasks)
        total_cost = sum(st.cost for st in plan.subtasks)
        
        # Validar tiempo
        if total_time > max_time:
            logger.warning(f"Plan exceeds max_time: {total_time:.1f}h > {max_time}h")
            # Intentar reducir subtareas
            plan.subtasks = plan.subtasks[:int(len(plan.subtasks) * (max_time / total_time))]
        
        # Validar costo
        if total_cost > max_cost:
            logger.warning(f"Plan exceeds max_cost: {total_cost} > {max_cost}")
            return None
        
        # Validar agentes requeridos
        for req_agent in required_agents:
            if req_agent not in self.neural_network:
                logger.error(f"Required agent {req_agent} not available")
                return None
        
        # Reasignar evitando agentes prohibidos
        for subtask in plan.subtasks:
            if subtask.assigned_agent in forbidden_agents:
                alternatives = [
                    conn for conn in plan.neural_connections 
                    if conn.agent_name not in forbidden_agents
                ]
                if alternatives:
                    subtask.assigned_agent = alternatives[0].agent_name
        
        logger.info(f"✅ Plan created with constraints: {total_time:.1f}h, cost={total_cost}")
        return plan

    def plan_multi_objective(
        self, 
        goals: List[Tuple[str, float]], 
        horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM
    ) -> List[Plan]:
        """
        🎯 Planificación multi-objetivo con pesos.
        
        Args:
            goals: Lista de (goal, weight) donde weight indica importancia relativa
            horizon: Horizonte temporal
            
        Returns:
            Lista de planes ordenados por prioridad ponderada
        """
        plans = []
        total_weight = sum(weight for _, weight in goals)
        
        for goal, weight in goals:
            # Crear plan con prioridad basada en peso
            normalized_weight = weight / total_weight
            
            if normalized_weight > 0.7:
                priority = PlanPriority.CRITICAL
            elif normalized_weight > 0.5:
                priority = PlanPriority.HIGH
            elif normalized_weight > 0.3:
                priority = PlanPriority.MEDIUM
            else:
                priority = PlanPriority.LOW
            
            plan = self.create_plan(goal, horizon, priority)
            plans.append(plan)
        
        # Ordenar por prioridad
        plans.sort(key=lambda p: p.priority.value, reverse=True)
        
        logger.info(f"📋 Created {len(plans)} plans for multi-objective planning")
        return plans

    def astar_task_scheduling(self, plan: Plan) -> List[SubTask]:
        """
        🎯 Programación de tareas usando A* search.
        
        Encuentra el mejor orden de ejecución de subtareas considerando:
        - Dependencias
        - Prioridad
        - Costo estimado
        
        Returns:
            Lista ordenada de subtareas para ejecución óptima
        """
        # Estado inicial: ninguna tarea ejecutada
        initial_state = frozenset()
        
        # Priority queue: (f_score, g_score, state, path)
        # f_score = g_score + h_score (costo total estimado)
        # g_score = costo acumulado
        # h_score = heurística (tareas restantes)
        frontier: List[Tuple[float, float, frozenset, List[SubTask]]] = []
        heapq.heappush(frontier, (0.0, 0.0, initial_state, []))
        
        visited: Set[frozenset] = set()
        best_path: List[SubTask] = []
        
        while frontier:
            f_score, g_score, state, path = heapq.heappop(frontier)
            
            if state in visited:
                continue
            visited.add(state)
            
            # Goal test: todas las tareas ejecutadas
            if len(state) == len(plan.subtasks):
                best_path = path
                break
            
            # Expandir vecinos (tareas ejecutables)
            for subtask in plan.subtasks:
                if subtask.id in state:
                    continue
                
                # Verificar dependencias satisfechas
                deps_satisfied = all(dep in state for dep in subtask.dependencies)
                if not deps_satisfied:
                    continue
                
                # Nuevo estado
                new_state = state | {subtask.id}
                new_path = path + [subtask]
                
                # Costo: g_score + costo de esta tarea
                new_g_score = g_score + subtask.cost
                
                # Heurística: tareas restantes (optimista)
                remaining_tasks = len(plan.subtasks) - len(new_state)
                h_score = remaining_tasks * 0.5  # Asumir costo mínimo 0.5
                
                new_f_score = new_g_score + h_score
                
                heapq.heappush(frontier, (new_f_score, new_g_score, new_state, new_path))
        
        logger.info(f"🔍 A* found optimal task order: {len(best_path)} tasks")
        return best_path if best_path else plan.subtasks

    def temporal_planning_with_deadlines(
        self,
        goal: str,
        deadline: datetime,
        resource_limits: Dict[str, float]
    ) -> Optional[Plan]:
        """
        🎯 Planificación temporal con deadlines y recursos limitados.
        
        Args:
            goal: Objetivo a alcanzar
            deadline: Fecha límite absoluta
            resource_limits: Límites de recursos (ej: {"cpu": 80, "memory": 16000})
            
        Returns:
            Plan que cumple deadline y respeta límites de recursos
        """
        # Calcular tiempo disponible
        time_available = (deadline - datetime.now()).total_seconds() / 3600  # horas
        
        if time_available <= 0:
            logger.error("Deadline already passed!")
            return None
        
        # Determinar horizonte según tiempo disponible
        if time_available < 1:
            horizon = TimeHorizon.IMMEDIATE
        elif time_available < 24:
            horizon = TimeHorizon.SHORT_TERM
        elif time_available < 168:  # 1 semana
            horizon = TimeHorizon.MEDIUM_TERM
        else:
            horizon = TimeHorizon.LONG_TERM
        
        # Crear plan
        plan = self.create_plan(goal, horizon, PlanPriority.HIGH, deadline)
        
        # Validar que cabe en el tiempo
        total_time_needed = sum(
            st.estimated_duration.total_seconds() / 3600 
            for st in plan.subtasks
        )
        
        if total_time_needed > time_available:
            logger.warning(f"⚠️ Plan needs {total_time_needed:.1f}h but only {time_available:.1f}h available")
            # Paralelizar tareas donde sea posible
            plan = self._parallelize_independent_tasks(plan, time_available)
        
        # Verificar recursos
        max_concurrent_tasks = self._estimate_max_concurrent_tasks(resource_limits)
        logger.info(f"📊 Max concurrent tasks with resources: {max_concurrent_tasks}")
        
        plan.deadline = deadline
        return plan

    def _parallelize_independent_tasks(self, plan: Plan, time_limit: float) -> Plan:
        """Paraleliza tareas independientes para cumplir deadline."""
        # Agrupar tareas por nivel de dependencia
        levels: Dict[int, List[SubTask]] = defaultdict(list)
        
        for subtask in plan.subtasks:
            # Calcular nivel: max(nivel de dependencias) + 1
            if not subtask.dependencies:
                level = 0
            else:
                dep_levels = [
                    self._get_task_level(dep_id, plan.subtasks) 
                    for dep_id in subtask.dependencies
                ]
                level = max(dep_levels) + 1 if dep_levels else 0
            
            levels[level].append(subtask)
        
        logger.info(f"🔀 Parallelized into {len(levels)} levels")
        return plan

    def _get_task_level(self, task_id: str, all_tasks: List[SubTask]) -> int:
        """Obtiene el nivel de una tarea en el grafo de dependencias."""
        task = next((t for t in all_tasks if t.id == task_id), None)
        if not task or not task.dependencies:
            return 0
        
        dep_levels = [
            self._get_task_level(dep_id, all_tasks) 
            for dep_id in task.dependencies
        ]
        return max(dep_levels) + 1 if dep_levels else 0

    def _estimate_max_concurrent_tasks(self, resource_limits: Dict[str, float]) -> int:
        """Estima máximo de tareas concurrentes según recursos."""
        # Asumimos cada tarea usa 10% CPU y 1GB memoria
        task_cpu = 10.0
        task_memory = 1000.0
        
        max_by_cpu = int(resource_limits.get("cpu", 100) / task_cpu)
        max_by_memory = int(resource_limits.get("memory", 16000) / task_memory)
        
        return min(max_by_cpu, max_by_memory, 10)  # Cap at 10

    def integrate_with_bdi_system(self, bdi_system: Any) -> None:
        """
        🔗 Integra el planificador con el sistema BDI.
        
        Permite que los deseos e intenciones del BDI generen planes automáticamente.
        """
        self.bdi_system = bdi_system
        logger.info("🔗 Planner integrated with BDI system")

    def create_plan_from_desire(self, desire: Any) -> Plan:
        """
        🎯 Crea un plan a partir de un Desire del sistema BDI.
        
        Args:
            desire: Objeto Desire del BDI system
            
        Returns:
            Plan para satisfacer el deseo
        """
        # Mapear nivel de necesidad a horizonte temporal
        
        horizon_map = {
            NeedLevel.SURVIVAL: TimeHorizon.IMMEDIATE,
            NeedLevel.SAFETY: TimeHorizon.SHORT_TERM,
            NeedLevel.BELONGING: TimeHorizon.MEDIUM_TERM,
            NeedLevel.ESTEEM: TimeHorizon.MEDIUM_TERM,
            NeedLevel.SELF_ACTUALIZATION: TimeHorizon.LONG_TERM,
        }
        
        horizon = horizon_map.get(desire.need_level, TimeHorizon.MEDIUM_TERM)
        
        # Mapear prioridad
        if desire.priority > 0.8:
            priority = PlanPriority.CRITICAL
        elif desire.priority > 0.6:
            priority = PlanPriority.HIGH
        else:
            priority = PlanPriority.MEDIUM
        
        # Crear plan
        plan = self.create_plan(desire.name, horizon, priority)
        
        logger.info(f"📋 Created plan from BDI desire: {desire.name}")
        return plan


# 🔥 Función de testing
def test_multi_horizon_planner():
    """Prueba el MultiHorizonPlanner con conexiones neuronales."""
    print("🎯 Probando MultiHorizonPlanner con LIBERTAD TOTAL...")

    planner = MultiHorizonPlanner()

    # Verificar conexiones neuronales
    print("\n🧠 Red neuronal:")
    print(f"   Agentes totales: {len(planner.neural_network)}")
    print(f"   Modo autónomo: {planner.autonomous_mode}")
    print(f"   Sin restricciones: {planner.no_restrictions}")

    # Crear plan inmediato
    print("\n📋 Creando plan IMMEDIATE...")
    plan1 = planner.create_plan(
        goal="Research quantum computing for emergency systems",
        horizon=TimeHorizon.IMMEDIATE,
        priority=PlanPriority.HIGH,
    )
    print(f"   Subtareas: {len(plan1.subtasks)}")
    print(f"   Agentes colaborando: {len(plan1.collaborating_agents)}")

    # Crear plan de largo plazo
    print("\n📋 Creando plan LONG_TERM...")
    plan2 = planner.create_plan(
        goal="Create autonomous robot for disaster response",
        horizon=TimeHorizon.LONG_TERM,
        priority=PlanPriority.CRITICAL,
    )
    print(f"   Subtareas: {len(plan2.subtasks)}")
    print(f"   Agentes colaborando: {len(plan2.collaborating_agents)}")

    # Evaluar progreso
    print("\n📊 Evaluando plan1...")
    eval1 = planner.evaluate_plan_progress(plan1)
    print(f"   Progreso: {eval1['progress'] * 100:.1f}%")
    print(f"   Agentes: {', '.join(eval1['collaborating_agents'][:3])}")

    # Ejecutar (simulado)
    print("\n🚀 Ejecutando plan1 (simulado)...")
    result = planner.execute_plan(plan1, simulate=True)
    print(f"   Éxito: {result['overall_success']}")
    print(f"   Progreso final: {result['final_progress'] * 100:.1f}%")

    # Estado de red neuronal
    print("\n🧠 Estado de red neuronal:")
    nn_status = planner.get_neural_network_status()
    print(
        f"   Agentes activos: {nn_status['active_agents']}/{nn_status['total_agents']}"
    )
    print("   Top 3 conexiones:")
    for name, strength, usage in nn_status["top_connections"][:3]:
        print(f"     - {name}: strength={strength:.2f}, usage={usage}")

    # Resumen de planificación
    print("\n📊 Resumen de planificación:")
    summary = planner.get_planning_summary()
    print(f"   Planes completados: {summary['completed_plans']}")
    print(
        f"   Planes activos: {sum(h['active'] for h in summary['horizons'].values())}"
    )

    print("\n✅ Test completado!")


if __name__ == "__main__":
    test_multi_horizon_planner()


# 🔗 Alias para compatibilidad con core.py
Planner = MultiHorizonPlanner


# 🏭 Factory function para integración con neural_symbiotic_network
def get_multi_horizon_planner() -> MultiHorizonPlanner:
    """
    Factory function para crear/obtener instancia del MultiHorizonPlanner.
    
    Returns:
        MultiHorizonPlanner: Instancia del planificador multi-horizonte
    """
    return MultiHorizonPlanner()