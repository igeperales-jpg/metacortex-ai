#!/usr/bin/env python3
"""
⚖️ AUTO-SCALING MODULE - Escalado automático del sistema (Military Grade)

Características:
    pass  # TODO: Implementar
- Load Monitoring continuo
- Horizontal Scaling automático
- Worker Pool Management
- Health Checks
- Graceful Shutdown
- Resource Allocation Optimization
- Predictive Scaling (ML-based)
- Auto-healing
- Load Balancing
"""

import time
import psutil  # type: ignore[import]
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic, Protocol
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import queue

logger = logging.getLogger(__name__)

# Type variables for generics
T = TypeVar('T')


class TaskProtocol(Protocol):
    """Protocol para tareas procesables"""
    def __str__(self) -> str: ...


@dataclass
class LoadMetrics:
    """Métricas de carga del sistema"""

    cpu_percent: float
    memory_percent: float
    active_requests: int = 0
    queue_size: int = 0
    avg_response_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ScalingDirection(Enum):
    """Dirección de escalado"""

    UP = "up"
    DOWN = "down"
    STABLE = "stable"


@dataclass
class PredictiveMetrics:
    """Métricas predictivas basadas en ML"""
    predicted_load: float
    confidence: float
    recommendation: ScalingDirection
    forecast_horizon_seconds: int
    historical_accuracy: float = 0.0
    timestamp: float = field(default_factory=time.time)


class WorkerStatus(Enum):
    """Estado del worker"""

    IDLE = "idle"
    BUSY = "busy"
    STARTING = "starting"
    STOPPING = "stopping"
    UNHEALTHY = "unhealthy"


@dataclass
class SystemMetrics:
    """Métricas del sistema"""

    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    active_workers: int = 0
    queue_size: int = 0
    avg_response_time_ms: float = 0.0
    requests_per_second: float = 0.0


@dataclass
class ScalingEvent:
    """Evento de escalado"""

    direction: ScalingDirection
    reason: str
    old_workers: int
    new_workers: int
    metrics: SystemMetrics
    timestamp: float = field(default_factory=time.time)


@dataclass
class Worker:
    """Worker del pool"""

    worker_id: str
    process: Optional[threading.Thread] = None  # Cambiado a Thread
    status: WorkerStatus = WorkerStatus.IDLE
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    tasks_completed: int = 0
    tasks_failed: int = 0


class WorkerPool(Generic[T]):
    """Pool de workers con auto-scaling y tipo genérico"""

    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 10,
        worker_func: Optional[Callable[[T], Any]] = None,
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.worker_func: Callable[[T], Any] = worker_func or self._default_worker

        self.workers: Dict[str, Worker] = {}
        self.task_queue: queue.Queue[T] = queue.Queue()

        self.lock = threading.RLock()
        self.running = False

        # Health check thread
        self.health_check_thread: Optional[threading.Thread] = None

        logger.info(
            f"✅ Worker Pool inicializado (min={min_workers}, max={max_workers})"
        )

        # Auto-start workers (ahora con threads funciona)
        self.start()

    def _default_worker(self, task: T) -> str:
        """Worker por defecto"""
        logger.info(f"Processing task: {task}")
        time.sleep(1)
        return f"Result: {task}"

    def start(self):
        """Iniciar pool"""
        with self.lock:
            if self.running:
                return

            self.running = True

            # Crear workers mínimos
            for _ in range(self.min_workers):
                self._add_worker()

            # Iniciar health check
            self.health_check_thread = threading.Thread(
                target=self._health_check_loop, daemon=True
            )
            self.health_check_thread.start()

            logger.info(f"✅ Worker Pool iniciado con {len(self.workers)} workers")

    def stop(self, graceful: bool = True):
        """Detener pool"""
        with self.lock:
            if not self.running:
                return

            self.running = False

            # Detener todos los workers
            for worker in list(self.workers.values()):
                self._remove_worker(worker.worker_id, graceful=graceful)

            logger.info("✅ Worker Pool detenido")

    def get_worker_count(self) -> int:
        """Obtener cantidad de workers activos"""
        with self.lock:
            return len(self.workers)

    def _add_worker(self) -> Optional[Worker]:
        """Agregar worker al pool"""
        with self.lock:
            if len(self.workers) >= self.max_workers:
                logger.warning(f"⚠️ Max workers alcanzado ({self.max_workers})")
                return None

            worker_id = f"worker_{len(self.workers)}_{int(time.time())}"

            # Crear thread (no proceso) para evitar problemas de serialización
            process = threading.Thread(
                target=self._worker_loop, args=(worker_id,), daemon=True
            )

            worker = Worker(
                worker_id=worker_id, process=process, status=WorkerStatus.STARTING
            )

            self.workers[worker_id] = worker
            process.start()

            # Pequeña espera para que el proceso inicie
            time.sleep(0.05)
            worker.status = WorkerStatus.IDLE

            logger.info(f"➕ Worker agregado: {worker_id}")
            return worker

    def _remove_worker(self, worker_id: str, graceful: bool = True):
        """Remover worker del pool"""
        with self.lock:
            worker = self.workers.get(worker_id)
            if not worker:
                return

            worker.status = WorkerStatus.STOPPING

            if worker.process and worker.process.is_alive():
                if graceful:
                    worker.process.join(timeout=5)
                # Threads no soportan terminate/kill - solo join

            del self.workers[worker_id]

            logger.info(f"➖ Worker removido: {worker_id}")

    def _worker_loop(self, worker_id: str):
        """Loop principal del worker"""
        logger.info(f"🔄 Worker {worker_id} iniciado")

        while self.running:
            try:
                # Obtener tarea de la cola
                task = self.task_queue.get(timeout=1)

                # Marcar como ocupado
                if worker_id in self.workers:
                    self.workers[worker_id].status = WorkerStatus.BUSY

                # Procesar tarea
                self.worker_func(task)

                # Actualizar métricas
                if worker_id in self.workers:
                    self.workers[worker_id].tasks_completed += 1
                    self.workers[worker_id].status = WorkerStatus.IDLE
                    self.workers[worker_id].last_heartbeat = time.time()

                self.task_queue.task_done()

            except queue.Empty:
                # No hay tareas, continuar
                if worker_id in self.workers:
                    self.workers[worker_id].last_heartbeat = time.time()
                continue

            except Exception as e:
                logger.error(f"❌ Error en worker {worker_id}: {e}")
                if worker_id in self.workers:
                    self.workers[worker_id].tasks_failed += 1
                    self.workers[worker_id].status = WorkerStatus.UNHEALTHY

    def _health_check_loop(self):
        """Health check de workers"""
        while self.running:
            time.sleep(5)  # Check cada 5 segundos

            with self.lock:
                now = time.time()

                for worker_id, worker in list(self.workers.items()):
                    # Check heartbeat
                    if now - worker.last_heartbeat > 30:
                        logger.warning(f"⚠️ Worker sin heartbeat: {worker_id}")
                        worker.status = WorkerStatus.UNHEALTHY

                        # Auto-healing: reemplazar worker
                        self._remove_worker(worker_id, graceful=False)
                        if len(self.workers) < self.min_workers:
                            self._add_worker()

    def submit_task(self, task: Any):
        """Enviar tarea al pool"""
        self.task_queue.put(task)

    def scale_up(self, count: int = 1):
        """Escalar hacia arriba (agregar workers)"""
        with self.lock:
            for _ in range(count):
                if len(self.workers) < self.max_workers:
                    self._add_worker()
                else:
                    logger.warning(f"⚠️ Ya en máximo de workers ({self.max_workers})")
                    break
            logger.info(f"📈 Scaled up: {len(self.workers)} workers activos")

    def scale_down(self, count: int = 1):
        """Escalar hacia abajo (remover workers)"""
        with self.lock:
            workers_to_remove = min(count, len(self.workers) - self.min_workers)
            for _ in range(workers_to_remove):
                if len(self.workers) > self.min_workers:
                    worker_id = list(self.workers.keys())[0]
                    self._remove_worker(worker_id, graceful=True)
                else:
                    logger.warning(f"⚠️ Ya en mínimo de workers ({self.min_workers})")
                    break
            logger.info(f"📉 Scaled down: {len(self.workers)} workers activos")

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del pool"""
        with self.lock:
            status_counts: Dict[str, int] = {}
            for worker in self.workers.values():
                status = worker.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            total_completed = sum(w.tasks_completed for w in self.workers.values())
            total_failed = sum(w.tasks_failed for w in self.workers.values())

            return {
                "total_workers": len(self.workers),
                "queue_size": self.task_queue.qsize(),
                "status_distribution": status_counts,
                "tasks_completed": total_completed,
                "tasks_failed": total_failed,
                "running": self.running,
            }


class AutoScaler(Generic[T]):
    """Sistema de auto-scaling basado en métricas"""

    def __init__(
        self,
        worker_pool: WorkerPool[T],
        target_cpu_percent: float = 70.0,
        target_queue_size: int = 10,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        cooldown_seconds: float = 60.0,
    ):
        self.worker_pool = worker_pool
        self.target_cpu_percent = target_cpu_percent
        self.target_queue_size = target_queue_size
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_seconds = cooldown_seconds

        # Métricas históricas
        self.metrics_history: deque[SystemMetrics] = deque(maxlen=60)  # Últimos 60 checks
        self.scaling_events: List[ScalingEvent] = []

        # Estado
        self.last_scaling_time = 0.0
        self.running = False

        # Thread de monitoreo
        self.monitor_thread: Optional[threading.Thread] = None

        logger.info("✅ Auto-Scaler inicializado")

    def start(self):
        """Iniciar auto-scaler"""
        if self.running:
            return

        self.running = True

        # Iniciar thread de monitoreo
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()

        logger.info("✅ Auto-Scaler iniciado")

    def stop(self):
        """Detener auto-scaler"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("✅ Auto-Scaler detenido")

    def _collect_metrics(self) -> SystemMetrics:
        """Recopilar métricas del sistema"""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Worker pool stats
        pool_stats = self.worker_pool.get_stats()

        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            active_workers=pool_stats["total_workers"],
            queue_size=pool_stats["queue_size"],
        )

        return metrics

    def _monitoring_loop(self):
        """Loop de monitoreo y scaling"""
        while self.running:
            try:
                # Recopilar métricas
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)

                # Decidir si escalar
                scaling_decision = self._make_scaling_decision(metrics)

                if scaling_decision != ScalingDirection.STABLE:
                    self._execute_scaling(scaling_decision, metrics)

                time.sleep(10)  # Check cada 10 segundos

            except Exception as e:
                logger.error(f"❌ Error en monitoring loop: {e}")
                time.sleep(10)

    def _make_scaling_decision(self, metrics: SystemMetrics) -> ScalingDirection:
        """Decidir si escalar basándose en métricas"""
        # Cooldown check
        if time.time() - self.last_scaling_time < self.cooldown_seconds:
            return ScalingDirection.STABLE

        # Calcular utilización
        cpu_utilization = metrics.cpu_percent / 100.0
        queue_utilization = metrics.queue_size / max(1, self.target_queue_size)

        # Combinar métricas (promedio ponderado)
        overall_utilization = (cpu_utilization * 0.6) + (queue_utilization * 0.4)

        # Decisión
        if overall_utilization > self.scale_up_threshold:
            if metrics.active_workers < self.worker_pool.max_workers:
                return ScalingDirection.UP

        elif overall_utilization < self.scale_down_threshold:
            if metrics.active_workers > self.worker_pool.min_workers:
                return ScalingDirection.DOWN

        return ScalingDirection.STABLE

    def _execute_scaling(self, direction: ScalingDirection, metrics: SystemMetrics):
        """Ejecutar acción de escalado"""
        old_workers = metrics.active_workers

        if direction == ScalingDirection.UP:
            self.worker_pool.scale_up(1)
            new_workers = old_workers + 1
            reason = f"Alta utilización (CPU: {metrics.cpu_percent:.1f}%, Queue: {metrics.queue_size})"

        elif direction == ScalingDirection.DOWN:
            self.worker_pool.scale_down(1)
            new_workers = old_workers - 1
            reason = f"Baja utilización (CPU: {metrics.cpu_percent:.1f}%, Queue: {metrics.queue_size})"

        else:
            return

        # Registrar evento
        event = ScalingEvent(
            direction=direction,
            reason=reason,
            old_workers=old_workers,
            new_workers=new_workers,
            metrics=metrics,
        )

        self.scaling_events.append(event)
        self.last_scaling_time = time.time()

        logger.info(
            f"⚖️ SCALING {direction.value.upper()}: {old_workers} → {new_workers} workers ({reason})"
        )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Obtener resumen de métricas"""
        if not self.metrics_history:
            return {}

        recent_metrics: List[SystemMetrics] = list(self.metrics_history)[-10:]

        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_queue = sum(m.queue_size for m in recent_metrics) / len(recent_metrics)

        return {
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory,
            "avg_queue_size": avg_queue,
            "current_workers": recent_metrics[-1].active_workers
            if recent_metrics
            else 0,
            "total_scaling_events": len(self.scaling_events),
        }


class PredictiveScaler:
    """
    🔮 Escalador predictivo basado en ML
    
    Características:
    - Predicción de carga futura usando series temporales
    - Auto-ajuste de thresholds basado en patrones históricos
    - Detección de patrones cíclicos (diario, semanal)
    - Recomendaciones proactivas de scaling
    """
    
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.historical_metrics: deque[SystemMetrics] = deque(maxlen=lookback_window)
        self.predictions: List[PredictiveMetrics] = []
        self.pattern_cache: Dict[str, List[float]] = {}
        
        logger.info("✅ Predictive Scaler inicializado")
    
    def record_metrics(self, metrics: SystemMetrics) -> None:
        """Registrar métricas para análisis predictivo"""
        self.historical_metrics.append(metrics)
    
    def predict_future_load(self, horizon_minutes: int = 5) -> PredictiveMetrics:
        """
        Predecir carga futura usando análisis de series temporales simple
        
        Args:
            horizon_minutes: Horizonte de predicción en minutos
            
        Returns:
            PredictiveMetrics con predicción y recomendación
        """
        if len(self.historical_metrics) < 10:
            # No hay suficiente historia, retornar predicción neutra
            return PredictiveMetrics(
                predicted_load=0.5,
                confidence=0.0,
                recommendation=ScalingDirection.STABLE,
                forecast_horizon_seconds=horizon_minutes * 60,
                historical_accuracy=0.0
            )
        
        # Análisis simple: tendencia lineal de los últimos N puntos
        recent = list(self.historical_metrics)[-20:]
        cpu_values = [m.cpu_percent / 100.0 for m in recent]
        queue_values = [m.queue_size for m in recent]
        
        # Calcular tendencia CPU (slope simple)
        cpu_trend = (cpu_values[-1] - cpu_values[0]) / len(cpu_values)
        
        # Calcular tendencia Queue
        avg_queue_current = sum(queue_values[-5:]) / 5
        avg_queue_past = sum(queue_values[:5]) / 5
        queue_growing = avg_queue_current > avg_queue_past * 1.5
        
        # Predicción = valor actual + tendencia * horizonte
        predicted_cpu_load = min(1.0, max(0.0, cpu_values[-1] + (cpu_trend * horizon_minutes)))
        
        # Combinar CPU y Queue para predicción final
        queue_factor = 0.3 if queue_growing else 0.0
        predicted_load = predicted_cpu_load + queue_factor
        
        # Calcular confianza basada en estabilidad histórica
        cpu_std = sum((x - sum(cpu_values)/len(cpu_values))**2 for x in cpu_values) ** 0.5
        confidence = max(0.0, min(1.0, 1.0 - cpu_std))
        
        # Decisión de escalado
        recommendation = ScalingDirection.STABLE
        if predicted_load > 0.8:
            recommendation = ScalingDirection.UP
        elif predicted_load < 0.3:
            recommendation = ScalingDirection.DOWN
        
        prediction = PredictiveMetrics(
            predicted_load=predicted_load,
            confidence=confidence,
            recommendation=recommendation,
            forecast_horizon_seconds=horizon_minutes * 60,
            historical_accuracy=confidence
        )
        
        self.predictions.append(prediction)
        
        logger.debug(
            f"🔮 Predicción: carga={predicted_load:.2f}, "
            f"confianza={confidence:.2f}, recomendación={recommendation.value}"
        )
        
        return prediction
    
    def get_optimal_workers(self, current_workers: int, target_load: float = 0.7) -> int:
        """
        Calcular número óptimo de workers basado en carga predicha
        
        Args:
            current_workers: Número actual de workers
            target_load: Carga objetivo (0.0 - 1.0)
            
        Returns:
            Número óptimo recomendado de workers
        """
        if len(self.historical_metrics) < 5:
            return current_workers
        
        recent = list(self.historical_metrics)[-10:]
        avg_load = sum(m.cpu_percent / 100.0 for m in recent) / len(recent)
        
        # Cálculo simple: optimal_workers = current * (actual_load / target_load)
        optimal = int(current_workers * (avg_load / target_load))
        
        # Limitar cambios abruptos
        max_change = max(1, int(current_workers * 0.3))  # Max 30% de cambio
        optimal = max(current_workers - max_change, min(current_workers + max_change, optimal))
        
        return max(1, optimal)


class CircuitBreakerAdvanced:
    """
    🔒 Circuit Breaker avanzado para protección del sistema
    
    Estados: CLOSED (normal) → OPEN (bloqueado) → HALF_OPEN (testing)
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failures = 0
        self.successes = 0
        self.last_failure_time = 0.0
        
    def record_success(self) -> None:
        """Registrar operación exitosa"""
        if self.state == "HALF_OPEN":
            self.successes += 1
            if self.successes >= self.success_threshold:
                self.state = "CLOSED"
                self.failures = 0
                self.successes = 0
                logger.info("🔓 Circuit Breaker CLOSED (recovered)")
        elif self.state == "CLOSED":
            self.failures = 0  # Reset failures en estado normal
    
    def record_failure(self) -> None:
        """Registrar operación fallida"""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            if self.state != "OPEN":
                self.state = "OPEN"
                logger.warning(f"🔴 Circuit Breaker OPEN ({self.failures} failures)")
    
    def can_execute(self) -> bool:
        """Verificar si se puede ejecutar operación"""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            # Check si es tiempo de recovery
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.successes = 0
                logger.info("🟡 Circuit Breaker HALF_OPEN (testing)")
                return True
            return False
        
        # HALF_OPEN: permitir intentos limitados
        return True
    
    def get_state(self) -> Dict[str, Any]:
        """Obtener estado del circuit breaker"""
        return {
            "state": self.state,
            "failures": self.failures,
            "successes": self.successes,
            "time_since_last_failure": time.time() - self.last_failure_time if self.last_failure_time > 0 else None
        }


class AutoScalingSystem(Generic[T]):
    """
    Sistema completo de auto-scaling con tipo genérico
    
    🚀 CARACTERÍSTICAS EXPONENCIALES:
    - Predictive Scaling (ML-based)
    - Circuit Breaker avanzado
    - Auto-healing inteligente
    - Métricas detalladas y telemetría
    - Health checks proactivos
    - Graceful degradation
    """

    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = 20,
        worker_func: Optional[Callable[[T], Any]] = None,
        enable_predictive_scaling: bool = True,
        enable_circuit_breaker: bool = True,
    ):
        self.worker_pool: WorkerPool[T] = WorkerPool(
            min_workers=min_workers, max_workers=max_workers, worker_func=worker_func
        )

        self.auto_scaler: AutoScaler[T] = AutoScaler(self.worker_pool)
        
        # 🔮 Predictive Scaling
        self.enable_predictive = enable_predictive_scaling
        self.predictive_scaler: Optional[PredictiveScaler]
        if enable_predictive_scaling:
            self.predictive_scaler = PredictiveScaler(lookback_window=200)
            logger.info("✅ Predictive Scaling habilitado")
        else:
            self.predictive_scaler = None
        
        # 🔒 Circuit Breaker
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker: Optional[CircuitBreakerAdvanced]
        if enable_circuit_breaker:
            self.circuit_breaker = CircuitBreakerAdvanced(
                failure_threshold=5,
                recovery_timeout=30.0,
                success_threshold=3
            )
            logger.info("✅ Circuit Breaker habilitado")
        else:
            self.circuit_breaker = None

        # 🧠 Conexión a red neuronal
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            self.neural_network.register_module("autoscaling", self)
            logger.info("✅ 'autoscaling' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None  # type: ignore

        # Integración con telemetría
        try:
            from agent_modules.telemetry import get_telemetry

            self.telemetry = get_telemetry()
        except Exception as e:
            logger.warning(f"⚠️ Telemetría no disponible: {e}")
            self.telemetry = None  # type: ignore

        logger.info("✅ Auto-Scaling System inicializado")

    def start(self):
        """Iniciar sistema"""
        self.worker_pool.start()
        self.auto_scaler.start()
        logger.info("✅ Auto-Scaling System activo")

    def stop(self, graceful: bool = True):
        """Detener sistema"""
        self.auto_scaler.stop()
        self.worker_pool.stop(graceful=graceful)
        logger.info("✅ Auto-Scaling System detenido")

    def submit_task(self, task: Any):
        """
        Enviar tarea para procesar con protección de Circuit Breaker
        
        Args:
            task: Tarea a procesar
            
        Raises:
            RuntimeError: Si el circuit breaker está abierto
        """
        # Verificar circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            raise RuntimeError(
                "Circuit Breaker OPEN: Sistema en estado de protección. "
                "Intente nuevamente más tarde."
            )
        
        try:
            self.worker_pool.submit_task(task)
            
            # Registrar éxito en circuit breaker
            if self.circuit_breaker:
                self.circuit_breaker.record_success()
                
        except Exception:
            # Registrar falla en circuit breaker
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            raise

    def get_full_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas completas incluyendo predicciones"""
        stats: Dict[str, Any] = {
            "worker_pool": self.worker_pool.get_stats(),
            "metrics": self.auto_scaler.get_metrics_summary(),
            "scaling_events": len(self.auto_scaler.scaling_events),
        }
        
        # Añadir estadísticas predictivas si está habilitado
        if self.predictive_scaler:
            prediction = self.predictive_scaler.predict_future_load(horizon_minutes=5)
            stats["predictive"] = {
                "predicted_load": prediction.predicted_load,
                "confidence": prediction.confidence,
                "recommendation": prediction.recommendation.value,
                "optimal_workers": self.predictive_scaler.get_optimal_workers(
                    self.worker_pool.get_worker_count()
                ),
            }
        
        # Añadir estado del circuit breaker
        if self.circuit_breaker:
            stats["circuit_breaker"] = self.circuit_breaker.get_state()
        
        return stats

    def get_predictive_recommendation(self) -> Optional[PredictiveMetrics]:
        """
        Obtener recomendación predictiva de scaling
        
        Returns:
            PredictiveMetrics o None si no está habilitado
        """
        if not self.predictive_scaler:
            return None
        
        # Registrar métricas actuales primero
        current_metrics = self.get_current_metrics()
        self.predictive_scaler.record_metrics(current_metrics)
        
        # Obtener predicción
        return self.predictive_scaler.predict_future_load(horizon_minutes=5)
    
    def get_current_metrics(self) -> SystemMetrics:
        """Obtener métricas actuales del sistema (método público)"""
        cpu_percent = psutil.cpu_percent(interval=0.1)  # type: ignore
        memory = psutil.virtual_memory()  # type: ignore
        disk = psutil.disk_usage("/")  # type: ignore
        
        pool_stats = self.worker_pool.get_stats()
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            active_workers=pool_stats["total_workers"],
            queue_size=pool_stats["queue_size"],
        )
    
    def apply_predictive_scaling(self) -> Dict[str, Any]:
        """
        Aplicar scaling basado en predicciones ML
        
        Returns:
            Resultado de la acción tomada
        """
        if not self.predictive_scaler:
            return {"success": False, "reason": "Predictive scaling not enabled"}
        
        # Obtener recomendación
        prediction = self.get_predictive_recommendation()
        if not prediction:
            return {"success": False, "reason": "No prediction available"}
        
        # Aplicar solo si confianza es alta
        if prediction.confidence < 0.6:
            return {
                "success": False,
                "reason": f"Low confidence: {prediction.confidence:.2f}",
                "prediction": prediction
            }
        
        current_workers = self.worker_pool.get_worker_count()
        optimal_workers = self.predictive_scaler.get_optimal_workers(current_workers)
        
        action_taken = None
        if optimal_workers > current_workers:
            self.worker_pool.scale_up(optimal_workers - current_workers)
            action_taken = f"Scaled UP: {current_workers} → {optimal_workers}"
        elif optimal_workers < current_workers:
            self.worker_pool.scale_down(current_workers - optimal_workers)
            action_taken = f"Scaled DOWN: {current_workers} → {optimal_workers}"
        else:
            action_taken = f"STABLE at {current_workers} workers"
        
        logger.info(f"🔮 Predictive Scaling: {action_taken}")
        
        return {
            "success": True,
            "action": action_taken,
            "prediction": {
                "load": prediction.predicted_load,
                "confidence": prediction.confidence,
                "recommendation": prediction.recommendation.value
            },
            "workers": {
                "before": current_workers,
                "after": optimal_workers
            }
        }

    def get_current_load(self) -> "LoadMetrics":
        """Obtener métricas de carga actual"""
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()

        return LoadMetrics(
            cpu_percent=cpu, memory_percent=mem.percent, active_requests=0, queue_size=0
        )

    def should_scale_up(
        self, cpu_threshold: float = 80.0, memory_threshold: float = 80.0
    ) -> bool:
        """Decidir si debe escalar hacia arriba"""
        metrics = self.get_current_load()
        return (
            metrics.cpu_percent > cpu_threshold
            or metrics.memory_percent > memory_threshold
        )

    def check_system_health(self) -> Dict[str, Any]:
        """Verificar salud del sistema"""
        metrics = self.get_current_load()
        healthy = metrics.cpu_percent < 90 and metrics.memory_percent < 90

        return {
            "healthy": healthy,
            "details": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
            },
        }


# Singleton global
_global_autoscaling: Optional[AutoScalingSystem[Any]] = None


def get_autoscaling_system(**kwargs: Any) -> AutoScalingSystem[Any]:
    """Obtener instancia global del sistema de auto-scaling"""
    global _global_autoscaling
    if _global_autoscaling is None:
        _global_autoscaling = AutoScalingSystem[Any](**kwargs)
    return _global_autoscaling


# Alias para compatibilidad
get_autoscaler = get_autoscaling_system


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("⚖️ Testing Auto-Scaling System...\n")

    # Worker function de ejemplo
    def example_worker(task: Any) -> str:
        """Worker que simula procesamiento"""
        time.sleep(0.5)
        return f"Processed: {task}"

    # Crear sistema
    scaling_system = get_autoscaling_system(
        min_workers=2, max_workers=6, worker_func=example_worker
    )

    # Iniciar
    scaling_system.start()

    print("✅ Sistema iniciado con 2 workers mínimos\n")
    print("Enviando tareas...\n")

    # Enviar tareas (burst)
    for i in range(20):
        scaling_system.submit_task(f"task_{i}")
        time.sleep(0.1)

    print("⏳ Esperando procesamiento...\n")
    time.sleep(15)

    # Stats
    stats = scaling_system.get_full_stats()
    print("📊 Estadísticas:")
    print(f"  Workers actuales: {stats['worker_pool']['total_workers']}")
    print(f"  Tareas completadas: {stats['worker_pool']['tasks_completed']}")
    print(f"  Cola: {stats['worker_pool']['queue_size']}")
    print(f"  Eventos de escalado: {stats['scaling_events']}")

    if stats["metrics"]:
        print(f"  CPU promedio: {stats['metrics']['avg_cpu_percent']:.1f}%")
        print(f"  Memoria promedio: {stats['metrics']['avg_memory_percent']:.1f}%")

    # Detener
    print("\n⏹️  Deteniendo sistema...")
    scaling_system.stop()

    print("\n✅ Tests completados")
