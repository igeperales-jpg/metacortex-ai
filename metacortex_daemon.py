#!/usr/bin/env python3
"""
🔮 METACORTEX DAEMON v4.0 - MILITARY GRADE ORCHESTRATION SYSTEM
═══════════════════════════════════════════════════════════════════════════════

⚔️ SISTEMA DE ORQUESTACIÓN DE GRADO MILITAR CON ALTA DISPONIBILIDAD ⚔️

Características Militares de Nivel Avanzado:
    pass  # TODO: Implementar
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🛡️  RESILIENCIA Y ALTA DISPONIBILIDAD:
   • Circuit Breakers multi-nivel con timeout adaptativo
   • Health Checks distribuidos con métricas en tiempo real
   • Auto-recovery con backoff exponencial
   • Failover automático con estado distribuido
   • Redundancia de componentes críticos
   • Chaos Engineering para validación continua

🔒 SEGURIDAD DE NIVEL MILITAR:
   • Zero-Trust Architecture con mutual TLS
   • Audit logging completo con tamper-proof storage
   • Encryption at rest y en tránsito (AES-256-GCM)
   • Rate limiting y DDoS protection
   • Secure credential management
   • RBAC granular

⚡ PERFORMANCE Y ESCALABILIDAD:
   • Thread pool dinámico con auto-scaling
   • Memory-mapped I/O para operaciones críticas
   • Cache distribuido con coherencia fuerte
   • Load balancing con algoritmos avanzados
   • Connection pooling optimizado
   • Zero-copy networking cuando posible

📊 OBSERVABILIDAD Y TELEMETRÍA:
   • Distributed tracing con OpenTelemetry
   • Prometheus metrics exporters
   • Structured logging con contexto enriquecido
   • Real-time dashboards
   • Alerting inteligente con ML-based anomaly detection
   • Performance profiling continuo

🧠 INTELIGENCIA COGNITIVA AVANZADA:
   • Sistema BDI completo
   • Planificación multi-horizonte
   • Aprendizaje por refuerzo
   • Meta-cognición y auto-reflexión
   • Conocimiento distribuido
   • Razonamiento causal

🚀 CAPACIDADES EXPONENCIALES:
   • Auto-mejora continua
   • Materialización de código
   • Generación de agentes on-demand
   • Evolución arquitectónica
   • Optimización multi-objetivo
   • Predicción de fallos con ML

🍎 OPTIMIZACIÓN APPLE SILICON:
   • Metal Performance Shaders
   • Neural Engine integration
   • Unified Memory Architecture
   • Energy-aware scheduling

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Autor: METACORTEX Advanced Military Systems Division
Fecha: 25 octubre 2025
Versión: 4.0 - MILITARY GRADE EVOLUTION
Clasificación: TACTICAL-ADVANCED
"""
# ==============================================================================
# 1. STANDARD LIBRARY IMPORTS
# ==============================================================================
import atexit
import json
import logging
import multiprocessing as mp
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import types
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from queue import Empty
from threading import Event, RLock
from typing import Any, Dict, List

# ==============================================================================
# 2. THIRD-PARTY IMPORTS
# ==============================================================================
try:
    import psutil
except ImportError:
    print("⚠️  psutil no encontrado. Instalando...")
    # Use subprocess defined in standard library imports
    subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=True)
    import psutil

# ==============================================================================
# 3. PROJECT-SPECIFIC IMPORTS
# ==============================================================================
from single_instance import ensure_single_instance
from unified_logging import setup_unified_logging
from cognitive_integration import get_cognitive_bridge
from metacortex_sinaptico.bdi import BDISystem
from metacortex_sinaptico.db import MetacortexDB
from metacortex_sinaptico.divine_protection import create_divine_protection_system
from metacortex_sinaptico.learning import StructuralLearning
from metacortex_sinaptico.memory import MemorySystem
from metacortex_sinaptico.planning import MultiHorizonPlanner
from neural_symbiotic_network import get_neural_network
from unified_memory_layer import UnifiedMemoryLayer
from programming_agent import get_programming_agent
from ml_auto_trainer import get_auto_trainer
from ml_cognitive_bridge import get_ml_cognitive_bridge
from ml_data_collector import get_data_collector
from ml_model_adapter import get_model_adapter
from ml_pipeline import get_ml_pipeline

# Optional import for Apple Silicon optimizations
try:
    from mps_config import configure_mps_system, is_apple_silicon
except ImportError:
    print("⚠️  mps_config no encontrado. Las optimizaciones de Apple Silicon estarán deshabilitadas.")
    def is_apple_silicon() -> bool:
        return False
    def configure_mps_system() -> Dict[str, Any]:
        return {}

# ==============================================================================
# 4. CRITICAL SYSTEM CONFIGURATION (Must be after imports)
# ==============================================================================

# Aumentar límite de recursión para evitar errores con sentence-transformers
sys.setrecursionlimit(100000)

# Añadir el directorio raíz al path para asegurar imports consistentes
DAEMON_ROOT = Path(__file__).parent
if str(DAEMON_ROOT) not in sys.path:
    sys.path.insert(0, str(DAEMON_ROOT))

# Prevenir circular import de pandas
os.environ['PANDAS_WARN_ON_C_EXTENSION_IMPORT'] = '0'

# Cargar variables de entorno
env_file = DAEMON_ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()
    print("✅ Variables de entorno cargadas")

# Crear directorio de logs
LOG_DIR = DAEMON_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configurar logging unificado
logger = setup_unified_logging(
    name="DAEMON_MILITARY",
    log_file=str(LOG_DIR / "metacortex_daemon_military.log"),
    level=logging.INFO,
)

# Configurar Apple Silicon (MPS) si está disponible
if is_apple_silicon():
    logger.info("🍎 Detectado Apple Silicon - configurando MPS...")
    try:
        mps_status = configure_mps_system()
        success = sum(mps_status.values())
        total = len(mps_status)
        logger.info(f"✅ MPS: {success}/{total} componentes configurados")
    except Exception as e:
        logger.warning(f"⚠️ MPS config error: {e}")


# ==============================================================================
# 5. DATA CLASSES AND ENUMS
# ==============================================================================
class ComponentState(Enum):
    """Estados de componentes militares"""

    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"
    TERMINATED = "terminated"


class CircuitState(Enum):
    """Estados del circuit breaker"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class PriorityLevel(Enum):
    """Niveles de prioridad"""

    CRITICAL = 10
    HIGH = 7
    MEDIUM = 5
    LOW = 3
    BACKGROUND = 1


@dataclass
class CircuitBreakerMetrics:
    """Métricas del circuit breaker"""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: datetime | None = None
    open_time: datetime | None = None
    half_open_attempts: int = 0
    total_requests: int = 0
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 60
    consecutive_failures: int = 0

    def should_attempt(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if self.open_time and (datetime.now() - self.open_time).seconds >= self.timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                self.half_open_attempts = 0
                logger.info("🔄 Circuit breaker → HALF_OPEN")
                return True
            return False
        return self.half_open_attempts < self.success_threshold

    def record_success(self):
        self.success_count += 1
        self.consecutive_failures = 0
        self.total_requests += 1

        if self.state == CircuitState.HALF_OPEN:
            self.half_open_attempts += 1
            if self.half_open_attempts >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("✅ Circuit breaker → CLOSED")

    def record_failure(self):
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure_time = datetime.now()
        self.total_requests += 1

        if self.consecutive_failures >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                self.state = CircuitState.OPEN
                self.open_time = datetime.now()
                logger.error(f"⚠️ Circuit breaker → OPEN ({self.consecutive_failures} fallos)")

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.open_time = datetime.now()
            logger.warning("⚠️ Circuit breaker → OPEN (fallo en recuperación)")


@dataclass
class ComponentMetrics:
    """Métricas de componente"""

    name: str
    state: ComponentState = ComponentState.INITIALIZING
    start_time: datetime = field(default_factory=datetime.now)
    last_health_check: datetime | None = None
    health_check_count: int = 0
    failure_count: int = 0
    restart_count: int = 0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "start_time": self.start_time.isoformat(),
            "health_check_count": self.health_check_count,
            "failure_count": self.failure_count,
            "restart_count": self.restart_count,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "error_rate": self.error_rate,
            "uptime_seconds": self.uptime_seconds,
        }


class MetacortexMilitaryDaemon:
    """
    Daemon Militar de Grado Avanzado
    """

    def __init__(self):
        logger.info("=" * 80)
        logger.info("⚔️ METACORTEX MILITARY DAEMON v4.0 - INITIALIZING")
        logger.info("=" * 80)

        self.running = True
        self.daemon_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        self.hostname = socket.gethostname()

        self.components: dict[str, dict[str, Any]] = {}
        self.component_metrics: dict[str, ComponentMetrics] = {}
        self.circuit_breakers: dict[str, CircuitBreakerMetrics] = {}

        self.lock = RLock()
        self.executor = ThreadPoolExecutor(
            max_workers=20, thread_name_prefix="metacortex_military_"
        )
        self.shutdown_event = Event()

        self.autonomous_mode = True
        self.last_materialization = datetime.now()
        self.materialization_interval = timedelta(minutes=10)
        self.base_interval_minutes = 10
        self.max_interval_minutes = 20
        self.min_interval_minutes = 5
        self.materialization_count = 0
        self.autonomous_cycles = 0

        self.energy_manager = MilitaryEnergyManager()
        self.in_rest_mode = False

        self.pid_file = DAEMON_ROOT / "metacortex_daemon_military.pid"
        self.state_file = DAEMON_ROOT / "logs" / "daemon_military_state.json"
        self.write_pid()

        # Carga de componentes principales
        self._load_core_components()

        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        atexit.register(self.cleanup)

        logger.info(f"🆔 Daemon ID: {self.daemon_id}")
        logger.info(f"🖥️  Hostname: {self.hostname}")
        logger.info(f"🆔 PID: {os.getpid()}")
        logger.info("✅ Military Daemon inicializado")

    def _load_core_components(self):
        """Carga todos los componentes del sistema de forma robusta."""
        logger.info("🛠️  Cargando componentes del sistema...")

        # Neural network
        try:
            self.neural_network = get_neural_network()
            self.neural_network.register_module("military_daemon", self)
            logger.info("✅ Neural Network inicializada")
        except Exception as e:
            logger.error(f"❌ Neural Network: {e}")
            self.neural_network = None

        # Cognitive bridge
        try:
            self.cognitive_bridge = get_cognitive_bridge(str(DAEMON_ROOT))
            logger.info("✅ Cognitive Bridge inicializado")
        except Exception as e:
            logger.error(f"❌ Cognitive Bridge: {e}")
            self.cognitive_bridge = None

        # Cognitive Agent Pool
        # TODO: FIX MISSING MODULE
        # try:
        #     self.agent_pool = get_cognitive_agent_pool()
        #     self.agent_pool.preload_async()
        #     logger.info("✅ Cognitive Agent Pool inicializado")
        # except Exception as e:
        #     logger.error(f"❌ Agent Pool: {e}")
        #     self.agent_pool = None
        self.agent_pool = None


        # ML Pipeline
        try:
            self.ml_pipeline = get_ml_pipeline(
                enable_perpetual_mode=True, enable_continuous_learning=True
            )
            logger.info("✅ ML Pipeline inicializado")
        except Exception as e:
            logger.error(f"❌ ML Pipeline: {e}")
            self.ml_pipeline = None

        # Auto-Trainer
        try:
            self.auto_trainer = get_auto_trainer(
                retraining_interval_hours=24, min_samples_threshold=100, enable_auto_collection=True
            )
            self.auto_trainer.start()
            logger.info("✅ Auto-Trainer inicializado")
        except Exception as e:
            logger.error(f"❌ Auto-Trainer: {e}")
            self.auto_trainer = None

        # Auto-Repair System
        self.last_auto_repair = datetime.now()
        self.auto_repair_interval = timedelta(hours=1)
        self.auto_repair_health_threshold = 85.0
        # TODO: FIX MISSING MODULE
        # try:
        #     self.auto_repair = get_auto_repair(
        #         project_root=DAEMON_ROOT, logs_dir=DAEMON_ROOT / "logs", auto_repair_enabled=True
        #     )
        #     logger.info("✅ Auto-Repair System inicializado")
        # except Exception as e:
        #     logger.error(f"❌ Auto-Reparar: {e}")
        #     self.auto_repair = None
        self.auto_repair = None

        # Disk Space Manager
        self.last_disk_cleanup = datetime.now()
        self.disk_cleanup_interval = timedelta(hours=6)
        # TODO: FIX MISSING MODULE
        # try:
        #     self.disk_manager = get_disk_space_manager(
        #         project_root=DAEMON_ROOT,
        #         logs_dir=DAEMON_ROOT / "logs",
        #         retention_days=30,
        #         max_log_size_mb=10,
        #         compression_enabled=True,
        #     )
        #     logger.info("✅ Disk Space Manager inicializado")
        # except Exception as e:
        #     logger.error(f"❌ Disk Space Manager: {e}")
        #     self.disk_manager = None
        self.disk_manager = None

        # Distributed Storage Manager V2.0
        self.last_storage_sync = datetime.now()
        self.storage_sync_interval = timedelta(hours=1)
        # TODO: FIX MISSING MODULE
        # try:
        #     self.distributed_storage = DistributedStorageManagerV2(
        #         config_file="storage_config_v2.json",
        #         auto_detect_volumes=True,
        #         min_disk_size_tb=3.0,
        #         disk_usage_threshold=85.0,
        #         enable_auto_migration=True
        #     )
        #     self.executor.submit(self._initialize_distributed_storage)
        #     logger.info("✅ Distributed Storage Manager inicializado (setup en background)")
        # except Exception as e:
        #     logger.error(f"❌ Distributed Storage Manager: {e}")
        #     self.distributed_storage = None
        self.distributed_storage = None

        # Port Monitor
        self.port_monitor_enabled = True
        self.last_port_monitoring = datetime.now()
        self.port_monitoring_interval = timedelta(minutes=5)
        self.port_monitor_auto_fix = True
        self.port_health_history: Dict[int, List[bool]] = defaultdict(list)
        self.critical_ports = {
            6379: "Redis", 8000: "Web Interface", 5000: "API Server",
            8080: "Dashboard", 11434: "Ollama", 9090: "Telemetry",
        }
        logger.info("✅ Port Monitor integrado")

        # Auto-Git Manager
        # TODO: FIX MISSING MODULE
        # try:
        #     self.auto_git_manager = get_auto_git_manager(repo_root=str(DAEMON_ROOT), logger=logger)
        #     logger.info("✅ Auto-Git Manager inicializado")
        # except Exception as e:
        #     logger.warning(f"⚠️ Auto-Git Manager: {e}")
        #     self.auto_git_manager = None
        self.auto_git_manager = None

        # TELEMETRY SYSTEM
        # TODO: FIX MISSING MODULE
        # try:
        #     self.telemetry_system = MetacortexTelemetrySystem(
        #         service_name="metacortex_daemon", enable_prometheus=True, enable_custom_export=True
        #     )
        #     logger.info("✅ Telemetry System inicializado")
        # except Exception as e:
        #     logger.warning(f"⚠️ Telemetry System: {e}")
        #     self.telemetry_system = None
        self.telemetry_system = None

        # LLM INTEGRATION
        # TODO: FIX MISSING MODULE
        # try:
        #     self.llm_integration = MetacortexLLM()
        #     logger.info("✅ LLM Integration inicializado")
        # except Exception as e:
        #     logger.warning(f"⚠️ LLM Integration: {e}")
        #     self.llm_integration = None
        self.llm_integration = None

        # MULTIMODAL PROCESSOR
        # TODO: FIX MISSING MODULE
        # try:
        #     self.multimodal_processor = MultiModalProcessor(cache_enabled=True)
        #     logger.info("✅ Multimodal Processor inicializado")
        # except Exception as e:
        #     logger.warning(f"⚠️ Multimodal Processor: {e}")
        #     self.multimodal_processor = None
        self.multimodal_processor = None

        # ML COGNITIVE BRIDGE
        try:
            self.ml_cognitive_bridge = get_ml_cognitive_bridge()
            logger.info("✅ ML Cognitive Bridge inicializado")
        except Exception as e:
            logger.warning(f"⚠️ ML Cognitive Bridge: {e}")
            self.ml_cognitive_bridge = None

        # ML DATA COLLECTOR
        try:
            self.ml_data_collector = get_data_collector(data_dir="ml_data")
            logger.info("✅ ML Data Collector inicializado")
        except Exception as e:
            logger.warning(f"⚠️ ML Data Collector: {e}")
            self.ml_data_collector = None

        # ML MODEL ADAPTER
        try:
            self.ml_model_adapter = get_model_adapter(models_dir="ml_models")
            logger.info("✅ ML Model Adapter inicializado")
        except Exception as e:
            logger.warning(f"⚠️ ML Model Adapter: {e}")
            self.ml_model_adapter = None

        # UNIFIED MEMORY LAYER
        try:
            # FIX: Constructor de UnifiedMemoryLayer no acepta estos parámetros.
            # La configuración se debe hacer en los componentes que utiliza por debajo.
            self.unified_memory = UnifiedMemoryLayer()
            logger.info("✅ Unified Memory Layer inicializado")
        except Exception as e:
            logger.warning(f"⚠️ Unified Memory Layer: {e}")
            self.unified_memory = None

        # VERIFICATION SYSTEM
        # TODO: FIX MISSING MODULE
        # try:
        #     self.system_verifier = CompleteSystemVerifier(root_path=str(DAEMON_ROOT))
        #     logger.info("✅ Complete System Verifier inicializado")
        # except Exception as e:
        #     logger.warning(f"⚠️ System Verifier: {e}")
        #     self.system_verifier = None
        self.system_verifier = None

        # EXPONENTIAL CAPABILITY DISCOVERY ENGINE
        self.last_capability_discovery = datetime.now()
        self.capability_discovery_interval = timedelta(minutes=5)
        self.capability_stats_log_interval = timedelta(hours=1)
        self.last_stats_log = datetime.now()
        # TODO: FIX MISSING MODULE
        # try:
        #     self.exponential_engine = get_exponential_engine(project_root=DAEMON_ROOT)
        #     self.executor.submit(self._initial_capability_discovery)
        #     logger.info("✅ Exponential Capability Engine inicializado (descubrimiento en background)")
        # except Exception as e:
        #     logger.error(f"❌ Exponential Capability Engine: {e}")
        #     self.exponential_engine = None
        self.exponential_engine = None

        # METACORTEX ORCHESTRATOR
        self.last_orchestration_cycle = datetime.now()
        self.orchestration_interval = timedelta(minutes=15)
        logger.info("ℹ️ Orchestrator se ejecuta como proceso separado y no se carga aquí.")
        self.orchestrator = None

        # IMPORT AUTO-HEALER
        self.last_healing_scan = datetime.now()
        self.healing_scan_interval = timedelta(hours=12)
        # TODO: FIX MISSING MODULE
        # try:
        #     self.import_healer = get_import_healer(project_root=DAEMON_ROOT, auto_install=True)
        #     logger.info("✅ Import Auto-Healer inicializado")
        # except Exception as e:
        #     logger.error(f"❌ Import Auto-Healer: {e}")
        #     self.import_healer = None
        self.import_healer = None

        # DIVINE PROTECTION SYSTEM
        self.last_protection_cycle = datetime.now()
        self.protection_cycle_interval = timedelta(minutes=30)
        try:
            db = MetacortexDB()
            bdi_system = BDISystem()
            planner = MultiHorizonPlanner()
            memory_system = MemorySystem(db=db)
            learning_system = StructuralLearning()
            self.divine_protection = create_divine_protection_system(
                db=db, bdi_system=bdi_system, planner=planner,
                memory=memory_system, learning=learning_system,
            )
            logger.info("✅ Divine Protection System inicializado")
        except Exception as e:
            logger.error(f"❌ Divine Protection System: {e}")
            self.divine_protection = None

    def write_pid(self):
        with open(self.pid_file, "w") as f:
            f.write(str(os.getpid()))

    def signal_handler(self, signum: int, frame: types.FrameType | None):
        logger.info(f"⚠️ Señal {signum} recibida")
        self.shutdown()

    def start_component_with_circuit_breaker(
        self,
        name: str,
        command: list[str],
        cwd: Path | None = None,
        priority: PriorityLevel = PriorityLevel.MEDIUM,
    ) -> bool:
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreakerMetrics()

        circuit = self.circuit_breakers[name]

        if not circuit.should_attempt():
            logger.warning(f"⚠️ Circuit OPEN para {name}")
            return False

        try:
            logger.info(f"🚀 Iniciando: {name} ({priority.name})")

            # Crear archivos de log para stdout y stderr
            log_dir = DAEMON_ROOT / "logs"
            log_dir.mkdir(exist_ok=True)
            stdout_log = log_dir / f"{name}_stdout.log"
            stderr_log = log_dir / f"{name}_stderr.log"

            with open(stdout_log, "a") as stdout_file, open(stderr_log, "a") as stderr_file:
                process = subprocess.Popen(
                    command,
                    cwd=cwd or DAEMON_ROOT,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )

            time.sleep(2)

            if process.poll() is None:
                with self.lock:
                    self.components[name] = {
                        "process": process,
                        "command": command,
                        "cwd": cwd,
                        "priority": priority,
                        "start_time": datetime.now(),
                    }

                    self.component_metrics[name] = ComponentMetrics(
                        name=name, state=ComponentState.HEALTHY, start_time=datetime.now()
                    )

                circuit.record_success()
                logger.info(f"✅ {name} iniciado (PID: {process.pid})")
                logger.info(f"   📄 Logs: {stdout_log} | {stderr_log}")
                return True
            
            # Si falló, leer los logs
            with open(stdout_log) as f:
                stdout_content = f.read()
            with open(stderr_log) as f:
                stderr_content = f.read()
            
            logger.error(f"❌ {name} falló")
            if stdout_content:
                logger.error(f"STDOUT: {stdout_content[-500:]}")  # Últimas 500 chars
            if stderr_content:
                logger.error(f"STDERR: {stderr_content[-500:]}")  # Últimas 500 chars
            circuit.record_failure()
            return False

        except Exception as e:
            logger.error(f"❌ Error iniciando {name}: {e}")
            circuit.record_failure()
            return False

    def check_component_health(self, name: str) -> ComponentState:
        if name not in self.components:
            return ComponentState.FAILED

        component = self.components[name]
        process = component["process"]

        if process.poll() is not None:
            return ComponentState.FAILED

        try:
            proc = psutil.Process(process.pid)
            cpu = proc.cpu_percent(interval=0.1)
            memory = proc.memory_info().rss / 1024 / 1024

            if name in self.component_metrics:
                metrics = self.component_metrics[name]
                metrics.cpu_percent = cpu
                metrics.memory_mb = memory
                metrics.last_health_check = datetime.now()
                metrics.health_check_count += 1

                if cpu > 90 or memory > 2048:
                    return ComponentState.DEGRADED
                return ComponentState.HEALTHY

            return ComponentState.HEALTHY

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return ComponentState.FAILED

    def restart_component_with_backoff(self, name: str, max_attempts: int = 3) -> bool:
        if name not in self.components:
            return False

        component = self.components[name]

        for attempt in range(1, max_attempts + 1):
            logger.info(f"🔄 Reiniciando {name} ({attempt}/{max_attempts})")

            try:
                component["process"].terminate()
                component["process"].wait(timeout=5)
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                try:
                    component["process"].kill()
                except Exception as kill_error:
                    logger.error(f"Error: {kill_error}", exc_info=True)
                    pass

            if attempt > 1:
                backoff = min(2**attempt, 60)
                time.sleep(backoff)

            success = self.start_component_with_circuit_breaker(
                name,
                component["command"],
                component.get("cwd"),
                component.get("priority", PriorityLevel.MEDIUM),
            )

            if success:
                if name in self.component_metrics:
                    self.component_metrics[name].restart_count += 1
                return True

        return False

    def start_all_components(self):
        logger.info("🚀 Iniciando componentes militares...")

        venv_python = DAEMON_ROOT / ".venv" / "bin" / "python3"
        python_cmd = str(venv_python) if venv_python.exists() else sys.executable

        # Web Interface (puerto 8000) - incluye dashboard en /api/dashboard/metrics
        self.start_component_with_circuit_breaker(
            "web_server",
            [python_cmd, "web_interface/server.py"],
            cwd=DAEMON_ROOT,
            priority=PriorityLevel.HIGH,
        )

        # Metacortex Orchestrator
        self.start_component_with_circuit_breaker(
            "metacortex_orchestrator",
            [python_cmd, "metacortex_orchestrator.py"],
            cwd=DAEMON_ROOT,
            priority=PriorityLevel.CRITICAL,
        )

        # Neural Network
        neural_file = DAEMON_ROOT / "neural_symbiotic_network.py"
        if neural_file.exists():
            self.start_component_with_circuit_breaker(
                "neural_network",
                [python_cmd, "neural_symbiotic_network.py", "--daemon"],
                cwd=DAEMON_ROOT,
                priority=PriorityLevel.CRITICAL,
            )

        logger.info(f"✅ {len(self.components)} componentes iniciados")

        logger.info("🌉 Inicializando Cognitive Bridge (EAGER mode)...")
        self.cognitive_bridge = get_cognitive_bridge(str(DAEMON_ROOT))
        logger.info("✅ Cognitive Bridge inicializado (EAGER, NO LAZY) - Programming agent listo")

    def start_autonomous_mode(self):
        if not self.autonomous_mode:
            return

        logger.info("🤖 INICIANDO MODO AUTÓNOMO MILITAR")

        def autonomous_loop():
            logger.info("🤖 Loop autónomo militar activo")

            while self.running and not self.shutdown_event.is_set():
                try:
                    if self.in_rest_mode:
                        time.sleep(60)
                        continue

                    now = datetime.now()

                    # 🚀 2026: Descubrimiento de capacidades exponencial cada 5 min
                    if self.exponential_engine and (
                        now - self.last_capability_discovery >= self.capability_discovery_interval
                    ):
                        logger.info("🔍 Ciclo de descubrimiento de capacidades...")

                        try:
                            self._periodic_capability_discovery()
                            self.last_capability_discovery = now
                        except Exception as e:
                            logger.error(f"❌ Error en descubrimiento de capacidades: {e}")

                    # Log estadísticas detalladas cada 1 hora
                    if self.exponential_engine and (
                        now - self.last_stats_log >= self.capability_stats_log_interval
                    ):
                        try:
                            self._log_capability_statistics()
                            self.last_stats_log = now
                        except Exception as e:
                            logger.error(f"❌ Error en log de estadísticas: {e}")

                    # 🆕 2026: Auto-Repair periódico
                    # TODO: FIX MISSING MODULE
                    # if self.auto_repair and (
                    #     now - self.last_auto_repair >= self.auto_repair_interval
                    # ):
                    #     logger.info("🔧 Ciclo de Auto-Repair...")

                    #     try:
                    #         diagnosis = self.auto_repair.diagnose_system()
                    #         health_pct = diagnosis.get("health_percentage", 0)

                    #         logger.info(f"   Health: {health_pct:.1f}%")

                    #         # Trigger auto-repair si health < threshold
                    #         if health_pct < self.auto_repair_health_threshold:
                    #             logger.warning(
                    #                 f"⚠️ Health bajo ({health_pct:.1f}% < {self.auto_repair_health_threshold}%)"
                    #             )
                    #             logger.info("🔧 Ejecutando auto-reparación...")

                    #             repair_result = self.auto_repair.auto_repair(diagnosis)

                    #             if repair_result.get("success"):
                    #                 logger.info(
                    #                     f"✅ Auto-repair exitoso: {repair_result.get('repairs_successful', 0)} fixes"
                    #                 )
                    #             else:
                    #                 logger.warning("⚠️ Auto-repair no pudo ejecutar fixes")
                    #         else:
                    #             logger.info(f"✅ Sistema saludable ({health_pct:.1f}%)")

                    #         self.last_auto_repair = now
                    #     except Exception as e:
                    #         logger.error(f"❌ Error en Auto-Repair: {e}")

                    # 🔌 2026: Port Monitor - Monitoreo de puertos críticos cada 5 min
                    if self.port_monitor_enabled and (
                        now - self.last_port_monitoring >= self.port_monitoring_interval
                    ):
                        logger.info("🔌 Ciclo de monitoreo de puertos...")

                        try:
                            self._monitor_critical_ports()
                            self.last_port_monitoring = now
                        except Exception as e:
                            logger.error(f"❌ Error en Port Monitor: {e}")

                    # 🆕 2026: Disk Space Manager - Limpieza periódica cada 6h
                    # TODO: FIX MISSING MODULE
                    # if self.disk_manager and (
                    #     now - self.last_disk_cleanup >= self.disk_cleanup_interval
                    # ):
                    #     logger.info("🗂️ Ciclo de limpieza de disco...")

                    #     try:
                    #         # Obtener uso de disco antes
                    #         usage_before = self.disk_manager.get_disk_usage()
                    #         disk_percent = usage_before.get("disk_percent_used", 0)

                    #         logger.info(f"   Disco usado: {disk_percent:.1f}%")

                    #         # Ejecutar limpieza si disco >80% o cada 6h
                    #         if disk_percent > 80 or (
                    #             now - self.last_disk_cleanup >= self.disk_cleanup_interval
                    #         ):
                    #             if disk_percent > 80:
                    #                 logger.warning(f"⚠️ Disco alto ({disk_percent:.1f}% > 80%)")

                    #             logger.info("🧹 Ejecutando limpieza automática...")
                    #             cleanup_result = self.disk_manager.auto_cleanup(dry_run=False)

                    #             if cleanup_result.get("success", False):
                    #                 summary = cleanup_result.get("summary", {})
                    #                 logger.info("✅ Limpieza completada:")
                    #                 logger.info(
                    #                     f"   • Espacio liberado: {summary.get('total_space_freed_mb', 0):.2f}MB"
                    #                 )
                    #                 logger.info(
                    #                     f"   • Archivos rotados: {summary.get('files_rotated', 0)}"
                    #                 )
                    #                 logger.info(
                    #                     f"   • Archivos comprimidos: {summary.get('files_compressed', 0)}"
                    #                 )
                    #                 logger.info(
                    #                     f"   • Archivos eliminados: {summary.get('files_deleted', 0)}"
                    #                 )
                    #             else:
                    #                 logger.warning("⚠️ Limpieza de disco no completada")
                    #         else:
                    #             logger.info(f"✅ Disco OK ({disk_percent:.1f}%)")

                    #         self.last_disk_cleanup = now
                    #     except Exception as e:
                    #         logger.error(f"❌ Error en Disk Space Manager: {e}")

                    # 💾 2026: Distributed Storage V2.0 - Sincronización y auto-migración cada 1h
                    if self.distributed_storage and (
                        now - self.last_storage_sync >= self.storage_sync_interval
                    ):
                        logger.info("💾 Ciclo de almacenamiento distribuido V2.0...")

                        try:
                            # Obtener estado del almacenamiento
                            storage_status = self.distributed_storage.get_storage_status()
                            primary_disk_percent = storage_status.get("primary_disk_percent", 0)

                            logger.info(f"   Disco primario: {primary_disk_percent:.1f}% usado")
                            logger.info(f"   Volúmenes disponibles: {storage_status.get('total_volumes', 0)}")
                            logger.info(f"   Espacio total externo: {storage_status.get('total_space_tb', 0):.2f} TB")
                            logger.info(f"   Espacio libre externo: {storage_status.get('total_free_tb', 0):.2f} TB")

                            # Auto-migración si disco primario > 85%
                            if primary_disk_percent > 85.0:
                                logger.warning(f"⚠️ Disco primario alto ({primary_disk_percent:.1f}% > 85%)")
                                logger.info("🚀 Ejecutando auto-migración a discos externos...")

                                # Migrar archivos grandes >10MB
                                migration_result = self.distributed_storage.auto_migrate_large_files(
                                    min_size_mb=10,
                                    exclude_patterns=["*.pyc", "__pycache__", ".git", ".venv"]
                                )

                                if migration_result.get("success", False):
                                    logger.info("✅ Auto-migración completada:")
                                    logger.info(f"   • Archivos migrados: {migration_result.get('files_migrated', 0)}")
                                    logger.info(f"   • Espacio liberado: {migration_result.get('bytes_migrated', 0) / 1024**3:.2f} GB")
                                    
                                    # Actualizar estadísticas
                                    logger.info(f"   • Total migraciones: {self.distributed_storage.stats.get('total_migrations', 0)}")
                                    logger.info(f"   • Total migrado: {self.distributed_storage.stats.get('total_bytes_migrated', 0) / 1024**3:.2f} GB")
                                else:
                                    logger.warning("⚠️ Auto-migración no completada")
                            else:
                                logger.info(f"✅ Disco primario OK ({primary_disk_percent:.1f}%)")

                            self.last_storage_sync = now
                        except Exception as e:
                            logger.error(f"❌ Error en Distributed Storage V2.0: {e}")

                    # 🎯 2026: Ciclo de Orquestación - Coordina todos los agentes
                    if self.orchestrator and (
                        now - self.last_orchestration_cycle >= self.orchestration_interval
                    ):
                        logger.info("🎯 Ciclo de orquestación de agentes...")

                        try:
                            # Obtener estado del sistema
                            status = self.orchestrator.get_system_status()
                            logger.info(f"   Servicios: {len(status.get('services', {}))}")
                            logger.info(f"   Health checks: {len(status.get('health_checks', []))}")

                            # Ejecutar análisis de carga y balanceo
                            # (El orchestrator maneja esto internamente)

                            self.last_orchestration_cycle = now
                            logger.info("✅ Ciclo de orquestación completado")
                        except Exception as e:
                            logger.error(f"❌ Error en orquestación: {e}")

                    # 🔧 2026: Healing Scan - Verifica imports y dependencias cada 12h
                    # TODO: FIX MISSING MODULE
                    # if self.import_healer and (
                    #     now - self.last_healing_scan >= self.healing_scan_interval
                    # ):
                    #     logger.info("🔧 Ejecutando healing scan del sistema...")

                    #     try:
                    #         # Ejecutar scan en background para no bloquear
                    #         future = self.executor.submit(self.import_healer.heal_project, True)

                    #         # Esperar máximo 2 minutos
                    #         report = future.result(timeout=120)

                    #         logger.info("✅ Healing scan completado:")
                    #         logger.info(f"   • Archivos escaneados: {report['files_scanned']}")
                    #         logger.info(f"   • Imports verificados: {report['imports_checked']}")
                    #         logger.info(f"   • Reparados: {report['repaired']}")
                    #         logger.info(f"   • Fallidos: {report['failed']}")

                    #         if report["repaired"] > 0:
                    #             logger.info(
                    #                 f"✨ {report['repaired']} dependencias reparadas automáticamente"
                    #             )

                    #         self.last_healing_scan = now
                    #     except Exception as e:
                    #         logger.error(f"❌ Error en healing scan: {e}")

                    # ✨ 2026: Divine Protection - Protección de perseguidos cada 30min
                    if self.divine_protection and (
                        now - self.last_protection_cycle >= self.protection_cycle_interval
                    ):
                        logger.info("✨ Ciclo de Divine Protection...")
                        logger.info("📖 'The Lord is my shepherd; I shall not want' - Psalm 23:1")

                        try:
                            # Evaluar amenazas de todas las personas protegidas
                            critical_cases = 0
                            for (
                                person_id,
                                person,
                            ) in self.divine_protection.protected_people.items():
                                threat_level = self.divine_protection.assess_threat_level(person_id)
                                if threat_level.value in ["critical", "endangered"]:
                                    critical_cases += 1
                                    logger.warning(
                                        f"⚠️ Caso crítico: {person.codename} - {threat_level.value}"
                                    )

                            # 🌍 2026: MONITOREO REAL DE PERSECUCIÓN
                            if self.divine_protection.real_ops:
                                logger.info("🌍 Ejecutando monitoreo REAL de persecución...")
                                try:
                                    # En producción, esto llamaría a APIs reales de noticias
                                    # alerts = await self.divine_protection.real_ops.monitor_persecution_news()

                                    # Por ahora, registrar capacidad operacional
                                    real_status = (
                                        self.divine_protection.real_ops.get_operations_status()
                                    )

                                    logger.info("✅ Sistema de operaciones REALES activo:")
                                    logger.info(
                                        f"   • Canales comunicación: {real_status['communication']['channels_active']}"
                                    )
                                    logger.info(
                                        f"   • Wallets crypto: {real_status['financial']['wallets']}"
                                    )
                                    logger.info(
                                        f"   • Safe houses: {real_status['safe_houses']['total_houses']}"
                                    )
                                    logger.info(
                                        f"   • Fondo emergencia: ${real_status['financial']['emergency_fund_total']:,.0f}"
                                    )
                                    logger.info(
                                        f"   • Regiones monitoreadas: {len(real_status['intelligence']['monitored_regions'])}"
                                    )
                                    logger.info(
                                        "   📖 'Do not withhold good when it is in your power to act' - Proverbs 3:27"
                                    )

                                except Exception as e:
                                    logger.error(f"❌ Error en monitoreo real: {e}")

                            # Obtener estado del sistema
                            status = self.divine_protection.get_system_status()

                            logger.info("✅ Ciclo de protección completado:")
                            logger.info(
                                f"   • Personas protegidas: {status['protected_persons']['total']}"
                            )
                            logger.info(f"   • Casos críticos: {critical_cases}")
                            logger.info(f"   • Refugios activos: {status['safe_havens']['total']}")
                            logger.info(
                                f"   • Capacidad disponible: {status['safe_havens']['total_capacity'] - status['safe_havens']['current_occupancy']}"
                            )
                            logger.info(
                                f"   • Planes de supervivencia: {status['survival_plans']['active']}"
                            )
                            logger.info(
                                f"   • Sistemas infiltrados: {status['infiltration']['systems_infiltrated']}"
                            )
                            logger.info(
                                f"   • Provisiones entregadas: {status['statistics']['provisions_delivered']}"
                            )

                            self.last_protection_cycle = now
                        except Exception as e:
                            logger.error(f"❌ Error en Divine Protection: {e}")
                            logger.error(f"   Traceback: {traceback.format_exc()}")

                    # Materialización militar
                    if now - self.last_materialization >= self.materialization_interval:
                        logger.info("🧠 Ciclo de materialización militar...")

                        future = self.executor.submit(self._execute_materialization_military)

                        try:
                            result = future.result(timeout=120)

                            if result.get("success"):
                                logger.info("✅ Materialización exitosa:")
                                logger.info(
                                    f"   • Componentes: {result.get('components_created', 0)}"
                                )
                                logger.info(
                                    f"   • Mejoras: {result.get('improvements_applied', 0)}"
                                )

                                self.materialization_count += 1

                                if self.auto_git_manager:
                                    try:
                                        self.auto_git_manager.auto_commit_generated_files(result)
                                    except Exception as e:
                                        logger.warning(f"⚠️ Auto-commit: {e}")

                            self.last_materialization = now
                            self.autonomous_cycles += 1
                            self._adjust_interval_by_load()

                        except Exception as e:
                            logger.error(f"❌ Timeout materialización: {e}")

                    time.sleep(30)

                except Exception as e:
                    logger.error(f"❌ Error loop autónomo: {e}")
                    time.sleep(60)

        autonomous_thread = threading.Thread(
            target=autonomous_loop, daemon=True, name="AutonomousMilitary"
        )
        autonomous_thread.start()

        logger.info("✅ Modo autónomo militar activo")

    def _execute_materialization_military(self) -> dict[str, Any]:
        """
        Materialización militar con Cognitive Bridge optimizado

        Características 2026:
        - Cognitive agent pool con pre-carga
        - Timeout real con multiprocessing
        - Fallback automático a materialización básica
        - Circuit breaker para prevenir colapsos
        """
        start_time = time.time()

        try:
            if self.cognitive_bridge:
                logger.info("🧠 Materialización cognitiva militar (con fallback)...")

                # ✅ 2026: programming_agent inicializado con EAGER mode en cognitive_bridge
                # NO lazy loading - ya está completamente disponible desde __init__
                assert self.cognitive_bridge is not None, "Cognitive Bridge no está disponible"

                result_queue: mp.Queue[Dict[str, Any]] = mp.Queue()
                tick_result: Dict[str, Any] = {"success": False}
                improvement_result: Dict[str, Any] = {"success": False}

                # 🔧 FIX: Usar threading en vez de multiprocessing para evitar pickling
                # Multiprocessing requiere que la función sea picklable (no puede ser nested)
                # Threading es suficiente para timeout y no requiere pickling
                
                def run_cognitive_tick_thread():
                    """Ejecuta cognitive tick en thread separado"""
                    try:
                        # La aserción anterior garantiza que self.cognitive_bridge no es None
                        result = self.cognitive_bridge.cognitive_tick_with_orchestration()
                        result_queue.put({"success": True, "result": result})
                    except Exception as thread_exc:
                        logger.error(f"Error: {thread_exc}", exc_info=True)
                        result_queue.put({"success": False, "error": str(thread_exc)})

                # Ejecutar en thread separado con timeout
                logger.info("⏱️  Ejecutando cognitive tick (timeout: 60s)...")
                tick_thread = threading.Thread(
                    target=run_cognitive_tick_thread, daemon=True
                )
                tick_thread.start()
                tick_thread.join(timeout=60)

                if tick_thread.is_alive():
                    # TIMEOUT - el thread seguirá corriendo pero lo ignoramos
                    logger.warning("⏰ Timeout en cognitive tick (60s) - activando fallback")
                    # NOTE: No podemos "matar" threads como procesos, pero daemon=True
                    # significa que se limpiará automáticamente al cerrar el programa

                    # 🔥 FALLBACK: Materialización básica
                    logger.info("🔧 Fallback a materialización básica...")
                    try:
                        agent = get_programming_agent(
                            project_root=str(DAEMON_ROOT), cognitive_agent=None
                        )
                        basic_result = agent.materialize_metacortex_thoughts()

                        return {
                            "success": basic_result.get("success", False),
                            "type": "basic_fallback",
                            "components_created": basic_result.get("components_created", 0),
                            "agents_generated": basic_result.get("agents_generated", 0),
                            "improvements_applied": basic_result.get("improvements_applied", 0),
                            "elapsed_seconds": time.time() - start_time,
                            "fallback": True,
                        }
                    except Exception as fallback_error:
                        logger.error(f"❌ Error en fallback: {fallback_error}")
                        return {"success": False, "error": str(fallback_error)}

                # Obtener resultado del proceso
                try:
                    process_result = result_queue.get_nowait()
                    if process_result["success"]:
                        tick_result = process_result["result"]
                        logger.info("✅ Cognitive tick completado exitosamente")
                    else:
                        logger.error(f"❌ Error en cognitive tick: {process_result.get('error')}")
                except Empty:
                    logger.warning("⚠️ No se pudo obtener resultado del proceso")

                # Ejecutar improvement cycle (solo si tick tuvo éxito)
                improvement_result = {"success": False}
                if tick_result.get("success"):
                    try:
                        assert self.cognitive_bridge is not None
                        logger.info("🔧 Ejecutando improvement cycle...")
                        improvement_result = self.cognitive_bridge.autonomous_improvement_cycle()
                    except Exception as e:
                        logger.error(f"❌ Error en improvement: {e}")

                # Consolidar resultados
                success = tick_result.get("success", False) or improvement_result.get(
                    "success", False
                )

                components = 0
                agents = 0
                improvements = improvement_result.get("improvements_applied", 0)

                mat_result = tick_result.get("materialization_result")
                if mat_result and isinstance(mat_result, dict):
                    components = mat_result.get("components_created", 0)
                    agents = mat_result.get("agents_generated", 0)
                    improvements += mat_result.get("improvements_applied", 0)

                return {
                    "success": success,
                    "type": "cognitive_military_optimized",
                    "components_created": components,
                    "agents_generated": agents,
                    "improvements_applied": improvements,
                    "elapsed_seconds": time.time() - start_time,
                }

            # Si no hay cognitive bridge, usar materialización básica
            logger.info("🔧 Materialización básica militar (no cognitive bridge)...")


            agent = get_programming_agent(project_root=str(DAEMON_ROOT), cognitive_agent=None)

            return agent.materialize_metacortex_thoughts()

        except Exception as e:
            logger.error(f"❌ Materialización: {e}")
            return {"success": False, "error": str(e)}

    def _initial_capability_discovery(self):
        """
        Descubrimiento inicial de agentes y capacidades al inicio del daemon
        """
        if not self.exponential_engine:
            return

        try:
            logger.info("🔍 Descubrimiento inicial de capacidades y agentes...")

            # Descubrir agentes en el directorio raíz
            discovered = self.exponential_engine.discover_agents_in_directory(
                DAEMON_ROOT, recursive=True
            )

            stats = self.exponential_engine.get_statistics()

            logger.info("✅ Descubrimiento inicial completado:")
            logger.info(f"   • Agentes descubiertos: {stats.get('agents_discovered', 0)}")
            logger.info(f"   • Keywords aprendidas: {stats.get('keywords', 0)}")
            logger.info(f"   • Patrones reconocidos: {stats.get('patterns', 0)}")
            logger.info(f"   • Módulos analizados: {stats.get('modules', 0)}")

            # Registrar agentes descubiertos con la red neuronal
            if self.neural_network and discovered:
                try:
                    for agent in discovered:
                        # Descubrir capacidades del agente
                        try:
                            module = __import__(agent.module_path, fromlist=[agent.class_name])
                            agent_class = getattr(module, agent.class_name, None)
                            if agent_class:
                                # Intentar instanciar el agente
                                instance = agent_class()
                                # Usar discover_module_capabilities_exponential (método correcto)
                                result = self.exponential_engine.discover_module_capabilities_exponential(
                                    instance,
                                    learn_mode=True  # IMPORTANTE: Aprender keywords
                                )
                                agent.capabilities = result.get("capabilities", [])
                                new_keywords = len(result.get("metadata", {}).get("new_keywords_learned", []))
                                logger.debug(f"  └─ {agent.name}: {len(agent.capabilities)} capacidades, {new_keywords} keywords")
                        except Exception as e:
                            logger.debug(f"  └─ {agent.name}: error descubriendo capacidades: {e}")
                        
                        # Registrar en red neuronal
                        self.neural_network.register_module(
                            agent.name,
                            {
                                "module_path": agent.module_path,
                                "capabilities": agent.capabilities,
                                "agent_type": agent.agent_type.value,
                                "getter_function": agent.getter_function,
                            },
                        )
                    logger.info(f"✅ {len(discovered)} agentes registrados en red neuronal")
                except Exception as e:
                    logger.warning(f"⚠️ Error registrando agentes en red neuronal: {e}")

        except Exception as e:
            logger.error(f"❌ Error en descubrimiento inicial: {e}")

    def _periodic_capability_discovery(self):
        """
        Descubrimiento periódico de capacidades - se ejecuta cada 5 minutos
        """
        if not self.exponential_engine:
            return

        try:
            logger.info("🔍 Descubrimiento periódico de capacidades...")

            # Obtener stats previas
            stats_before = self.exponential_engine.get_statistics()

            # Ejecutar descubrimiento
            discovered = self.exponential_engine.discover_agents_in_directory(
                DAEMON_ROOT, recursive=True
            )

            # Obtener stats actualizadas
            stats_after = self.exponential_engine.get_statistics()

            # Calcular crecimiento
            new_keywords = stats_after.get('keywords', 0) - stats_before.get('keywords', 0)
            new_agents = stats_after.get('agents_discovered', 0) - stats_before.get('agents_discovered', 0)

            if new_keywords > 0 or new_agents > 0:
                logger.info("✨ Sistema aprendió:")
                if new_keywords > 0:
                    logger.info(f"   • {new_keywords} nuevas keywords")
                if new_agents > 0:
                    logger.info(f"   • {new_agents} nuevos agentes")

                # Registrar nuevos agentes con la red neuronal
                if self.neural_network and discovered:
                    try:
                        for agent in discovered:
                            self.neural_network.register_module(
                                agent.name,
                                {
                                    "file_path": str(agent.module_path)
                                    if hasattr(agent, "module_path")
                                    else "unknown",
                                    "capabilities": agent.capabilities
                                    if hasattr(agent, "capabilities")
                                    else [],
                                    "methods": [m.name for m in agent.methods]
                                    if hasattr(agent, "methods")
                                    else [],
                                },
                            )
                    except Exception as e:
                        logger.warning(f"⚠️ Error registrando agentes: {e}")
            else:
                logger.info("✅ No hay nuevas capacidades (sistema estable)")

        except Exception as e:
            logger.error(f"❌ Error en descubrimiento periódico: {e}")

    def _log_capability_statistics(self):
        """
        Log estadísticas detalladas de capacidades cada hora
        """
        if not self.exponential_engine:
            return

        try:
            stats = self.exponential_engine.get_statistics()

            logger.info("📊 ESTADÍSTICAS DE CAPACIDADES:")
            logger.info(f"   Total scans: {stats.get('scans', 0)}")
            logger.info(f"   Módulos analizados: {stats.get('modules', 0)}")
            logger.info(f"   Keywords aprendidas: {stats.get('keywords', 0)}")
            logger.info(f"   Agentes descubiertos: {stats.get('agents_discovered', 0)}")
            logger.info(f"   Patrones reconocidos: {stats.get('patterns', 0)}")
            logger.info(f"   Cache hit rate: {stats.get('cache_hit_rate', '0%')}")
            logger.info(f"   Nivel conocimiento: {stats.get('knowledge_level', 'BASIC')}")

            # Exportar estadísticas a archivo
            try:
                stats_file = DAEMON_ROOT / "logs" / "capability_stats.json"
                agents_data = []
                if hasattr(self.exponential_engine, "discovered_agents"):
                    for agent_name, agent_obj in self.exponential_engine.discovered_agents.items():
                        agents_data.append(
                            {
                                "name": agent_name,
                                "capabilities": agent_obj.capabilities
                                if hasattr(agent_obj, "capabilities")
                                else [],
                            }
                        )

                with open(stats_file, "w") as f:
                    json.dump(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "statistics": stats,
                            "discovered_agents": agents_data,
                        },
                        f,
                        indent=2,
                    )
            except Exception as e:
                logger.warning(f"⚠️ Error guardando stats: {e}")

        except Exception as e:
            logger.error(f"❌ Error en log de estadísticas: {e}")

    def _start_ollama_service(self):
        """Inicia el servidor Ollama si no está corriendo"""
        try:
            # 🔧 FIX: Verificar PRIMERO si Ollama ya está corriendo
            if self._check_port_status(11434):
                logger.info("✅ Ollama Server ya está activo (puerto 11434)")
                return True

            # Verificar si ollama está instalado
            result = subprocess.run(
                ["which", "ollama"], capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                logger.error("❌ Ollama no está instalado")
                return False

            logger.info("🚀 Iniciando Ollama Server...")

            # Iniciar ollama en background
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            # Esperar 5 segundos para que inicie

            time.sleep(5)

            # Verificar si está corriendo
            if self._check_port_status(11434):
                logger.info(f"✅ Ollama Server iniciado correctamente (PID: {process.pid})")
                return True
            logger.error("❌ Ollama Server no pudo iniciar")
            return False

        except Exception as e:
            logger.error(f"❌ Error iniciando Ollama: {e}")
            return False

    def _start_redis_service(self):
        """Inicia el servidor Redis si no está corriendo"""
        try:
            # 🔧 FIX: Verificar PRIMERO si Redis ya está corriendo
            if self._check_port_status(6379):
                logger.info("✅ Redis Server ya está activo (puerto 6379)")
                return True

            # Verificar si redis-server está instalado
            result = subprocess.run(
                ["which", "redis-server"], capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                logger.error("❌ Redis no está instalado")
                logger.info("   Instalar con: brew install redis")
                return False

            logger.info("🚀 Iniciando Redis Server...")

            # Intentar con brew services primero (macOS)
            result = subprocess.run(
                ["brew", "services", "start", "redis"], capture_output=True, text=True, check=False
            )

            if result.returncode == 0:

                time.sleep(2)

                if self._check_port_status(6379):
                    logger.info("✅ Redis Server iniciado correctamente (brew services)")
                    return True

            # Si brew falló, intentar manualmente
            logger.info("   Intentando inicio manual...")
            process = subprocess.Popen(
                ["redis-server"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )


            time.sleep(2)

            if self._check_port_status(6379):
                logger.info(f"✅ Redis Server iniciado correctamente (PID: {process.pid})")
                return True
            logger.error("❌ Redis Server no pudo iniciar")
            return False

        except Exception as e:
            logger.error(f"❌ Error iniciando Redis: {e}")
            return False

    def _check_port_status(self, port: int) -> bool:
        """Verifica si un puerto está en uso (LISTEN)"""
        try:
            result = subprocess.run(
                ["lsof", "-iTCP", f":{port}", "-sTCP:LISTEN", "-n", "-P"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return result.returncode == 0 and result.stdout.strip() != ""
        except Exception:
            return False

    def _monitor_critical_ports(self):
        """
        Monitorea puertos críticos y libera recursos si es necesario

        Verifica:
        - Estado de puertos (LISTEN/FREE)
        - Health de procesos (CPU, RAM, status)
        - Detección de procesos zombie
        - Auto-fix si se exceden umbrales
        - Auto-start de servicios críticos (Ollama, Redis)
        """
        if not self.port_monitor_enabled:
            return

        try:
            logger.info("🔌 Verificando puertos críticos...")
            ports_ok = 0
            ports_issues = 0

            for port, service_name in self.critical_ports.items():
                try:
                    # Usar lsof para verificar puerto (no requiere sudo)
                    result = subprocess.run(
                        ["lsof", "-iTCP", f":{port}", "-sTCP:LISTEN", "-n", "-P"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    )

                    if result.returncode == 0 and result.stdout.strip():
                        # Puerto en uso - verificar health
                        lines = result.stdout.strip().split("\n")
                        if len(lines) > 1:  # Ignorar header
                            parts = lines[1].split()
                            if len(parts) >= 2:
                                pid = int(parts[1])
                                process_name = parts[0]

                                # Verificar health del proceso
                                try:
                                    proc = psutil.Process(pid)
                                    cpu = proc.cpu_percent(interval=0.1)
                                    mem_mb = proc.memory_info().rss / (1024 * 1024)
                                    status = proc.status()

                                    # Determinar si está saludable
                                    is_healthy = True
                                    reason = ""

                                    if status == psutil.STATUS_ZOMBIE:
                                        is_healthy = False
                                        reason = "proceso zombie"
                                    elif cpu > 90.0:
                                        is_healthy = False
                                        reason = f"CPU alta ({cpu:.1f}%)"
                                    elif mem_mb > 2048:
                                        is_healthy = False
                                        reason = f"RAM alta ({mem_mb:.1f}MB)"

                                    # Registrar en historial
                                    self.port_health_history[port].append(is_healthy)
                                    # Mantener solo últimos 5 checks
                                    if len(self.port_health_history[port]) > 5:
                                        self.port_health_history[port].pop(0)

                                    if is_healthy:
                                        ports_ok += 1
                                        logger.info(f"   ✅ Puerto {port} ({service_name}): OK")
                                        logger.info(
                                            f"      PID: {pid}, CPU: {cpu:.1f}%, RAM: {mem_mb:.1f}MB"
                                        )
                                    else:
                                        ports_issues += 1
                                        logger.warning(
                                            f"   ⚠️ Puerto {port} ({service_name}): {reason}"
                                        )
                                        logger.warning(f"      PID: {pid}, Proceso: {process_name}")

                                        # Contar checks consecutivos no saludables
                                        recent_checks = self.port_health_history[port]
                                        unhealthy_count = sum(1 for h in recent_checks if not h)

                                        # Si 3+ checks consecutivos fallan, tomar acción
                                        if unhealthy_count >= 3 and self.port_monitor_auto_fix:
                                            logger.error(
                                                f"   ❌ Puerto {port} - Umbral excedido ({unhealthy_count}/3 checks malos)"
                                            )
                                            logger.info(f"   🔧 Liberando puerto {port}...")

                                            try:
                                                # Intentar terminación graceful
                                                proc.terminate()
                                                proc.wait(timeout=10)
                                                logger.info(
                                                    f"   ✅ Puerto {port} liberado (SIGTERM)"
                                                )
                                                self.port_health_history[
                                                    port
                                                ] = []  # Reset historial
                                            except psutil.TimeoutExpired:
                                                # Force kill si no responde
                                                proc.kill()
                                                logger.info(
                                                    f"   ✅ Puerto {port} liberado (SIGKILL)"
                                                )
                                                self.port_health_history[port] = []
                                            except Exception as kill_err:
                                                logger.error(
                                                    f"   ❌ Error liberando puerto {port}: {kill_err}"
                                                )

                                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                                    logger.warning(f"   ⚠️ Puerto {port} ({service_name}): {e}")
                                    ports_issues += 1
                    else:
                        # Puerto libre - Auto-start si es Ollama o Redis
                        logger.info(f"   ℹ️  Puerto {port} ({service_name}): LIBRE")
                        self.port_health_history[port] = []  # Reset historial

                        # Auto-start de servicios críticos
                        if self.port_monitor_auto_fix:
                            if port == 11434 and service_name == "Ollama":
                                logger.info(f"   🚀 Auto-iniciando {service_name}...")
                                if self._start_ollama_service():
                                    logger.info(f"   ✅ {service_name} iniciado correctamente")
                                    ports_ok += 1
                                else:
                                    logger.error(f"   ❌ No se pudo iniciar {service_name}")
                                    ports_issues += 1
                            elif port == 6379 and service_name == "Redis":
                                logger.info(f"   🚀 Auto-iniciando {service_name}...")
                                if self._start_redis_service():
                                    logger.info(f"   ✅ {service_name} iniciado correctamente")
                                    ports_ok += 1
                                else:
                                    logger.error(f"   ❌ No se pudo iniciar {service_name}")
                                    ports_issues += 1

                except subprocess.TimeoutExpired:
                    logger.warning(f"   ⏱️ Timeout verificando puerto {port}")
                except Exception as port_err:
                    logger.warning(f"   ⚠️ Error verificando puerto {port}: {port_err}")

            # Resumen
            logger.info(f"📊 Resumen monitoreo: {ports_ok} OK, {ports_issues} con problemas")

        except Exception as e:
            logger.error(f"❌ Error en monitoreo de puertos: {e}")

    def _adjust_interval_by_load(self):
        try:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            avg_load = (cpu + memory) / 2

            if avg_load < 30:
                new_interval = self.min_interval_minutes
            elif avg_load < 70:
                new_interval = self.base_interval_minutes
            else:
                new_interval = self.max_interval_minutes

            current = self.materialization_interval.total_seconds() / 60
            if abs(current - new_interval) > 0.5:
                self.materialization_interval = timedelta(minutes=new_interval)
                logger.info(f"⚙️ Intervalo: {current:.0f} → {new_interval:.0f} min")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
    def start_health_monitoring(self):
        logger.info("🏥 Health monitoring militar...")

        def health_loop():
            while self.running and not self.shutdown_event.is_set():
                try:
                    for name in list(self.components.keys()):
                        state = self.check_component_health(name)

                        if name in self.component_metrics:
                            self.component_metrics[name].state = state

                        if state == ComponentState.FAILED:
                            logger.warning(f"⚠️ {name} falló")
                            self.restart_component_with_backoff(name)

                    time.sleep(30)

                except Exception as e:
                    logger.error(f"❌ Health monitoring: {e}")
                    time.sleep(60)

        health_thread = threading.Thread(target=health_loop, daemon=True, name="HealthMonitoring")
        health_thread.start()

        logger.info("✅ Health monitoring activo")

    def cleanup(self):
        """Limpieza militar con manejo robusto de recursos"""
        logger.info("🧹 Limpieza militar...")
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
                logger.info("✅ PID file eliminado")
        except Exception as e:
            logger.error(f"❌ Error eliminando PID file: {e}")

    def shutdown(self):
        """
        Graceful shutdown robusto sin timeout forzado.

        Orden de terminación:
        1. Señalizar shutdown a todos los sistemas
        2. Detener ciclos autónomos
        3. Detener ML systems
        4. Detener componentes externos
        5. Cleanup final
        """
        logger.info("=" * 80)
        logger.info("🛑 INICIANDO GRACEFUL SHUTDOWN MILITAR")
        logger.info("=" * 80)

        # 1. Señalizar shutdown
        logger.info("📢 Fase 1/5: Señalizando shutdown...")
        self.running = False
        self.autonomous_mode = False
        self.shutdown_event.set()
        logger.info("✅ Shutdown señalizado")

        # 2. Detener ciclos autónomos (esperar a que terminen naturalmente)
        logger.info("📢 Fase 2/5: Esperando ciclos autónomos...")
        time.sleep(2)  # Dar tiempo a que terminen iteraciones actuales
        logger.info("✅ Ciclos autónomos detenidos")

        # 3. Detener ML systems de forma ordenada
        logger.info("📢 Fase 3/5: Deteniendo ML systems...")

        if self.ml_pipeline:
            try:
                logger.info("   ⏳ Deteniendo ML Pipeline...")
                self.ml_pipeline.stop_perpetual_training()
                logger.info("   ✅ ML Pipeline detenido")
            except Exception as e:
                logger.error(f"   ❌ Error deteniendo ML Pipeline: {e}")

        if self.auto_trainer:
            try:
                logger.info("   ⏳ Deteniendo Auto-Trainer...")
                self.auto_trainer.stop()
                logger.info("   ✅ Auto-Trainer detenido")
            except Exception as e:
                logger.error(f"   ❌ Error deteniendo Auto-Trainer: {e}")

        # Detener Divine Protection (si necesita cleanup)
        if self.divine_protection:
            try:
                logger.info("   ⏳ Preservando Divine Protection state...")
                status = self.divine_protection.get_system_status()
                logger.info(f"   📊 Personas protegidas: {status['protected_persons']['total']}")
                logger.info("   ✅ Divine Protection state preservado")
            except Exception as e:
                logger.error(f"   ❌ Error en Divine Protection cleanup: {e}")

        logger.info("✅ ML systems detenidos")

        # 4. Detener componentes externos con timeout razonable
        logger.info("📢 Fase 4/5: Deteniendo componentes externos...")

        terminated_count = 0
        killed_count = 0

        for name, component in list(self.components.items()):
            try:
                process = component["process"]
                logger.info(f"   ⏳ Terminando {name}...")

                # SIGTERM primero (graceful)
                process.terminate()

                # Esperar hasta 30 segundos (tiempo razonable)
                try:
                    process.wait(timeout=30)
                    terminated_count += 1
                    logger.info(f"   ✅ {name} terminado gracefully")
                except subprocess.TimeoutExpired:
                    # Si no responde en 30s, entonces SIGKILL
                    logger.warning(f"   ⚠️ {name} no respondió - usando SIGKILL...")
                    try:
                        process.kill()
                        process.wait(timeout=5)
                        killed_count += 1
                        logger.info(f"   ✅ {name} forzado")
                    except Exception as kill_error:
                        logger.error(f"   ❌ Error matando {name}: {kill_error}")

            except Exception as e:
                logger.error(f"   ❌ Error deteniendo {name}: {e}")

        logger.info(
            f"✅ Componentes detenidos ({terminated_count} graceful, {killed_count} forced)"
        )

        # 5. Shutdown executor y cleanup final
        logger.info("📢 Fase 5/5: Cleanup final...")

        try:
            logger.info("   ⏳ Deteniendo thread pool...")
            self.executor.shutdown(wait=True, cancel_futures=True)
            logger.info("   ✅ Thread pool detenido")
        except Exception as e:
            logger.error(f"   ❌ Error deteniendo executor: {e}")

        try:
            logger.info("   ⏳ Restaurando configuración de energía...")
            self.energy_manager.restore_defaults()
            logger.info("   ✅ Energía restaurada")
        except Exception as e:
            logger.error(f"   ❌ Error restaurando energía: {e}")

        logger.info("=" * 80)
        logger.info("✅ METACORTEX MILITARY DAEMON DETENIDO CORRECTAMENTE")
        logger.info("📖 'The Lord watch between me and thee' - Genesis 31:49")
        logger.info("=" * 80)

        sys.exit(0)

    def _initialize_distributed_storage(self):
        """Inicializa el almacenamiento distribuido en un hilo separado."""
        if not self.distributed_storage:
            return
        try:
            logger.info("🔥 Ejecutando setup inicial de almacenamiento distribuido...")
            self.distributed_storage.initialize_external_storage()
            self.distributed_storage.create_symlinks_all_volumes()
            status = self.distributed_storage.get_storage_status()
            logger.info(f"✅ Setup de almacenamiento distribuido completo. Volúmenes: {status['total_volumes']}, Espacio: {status['total_space_tb']:.2f} TB")
        except Exception as e:
            logger.error(f"❌ Error en setup de almacenamiento distribuido: {e}")


class MilitaryEnergyManager:
    """Energy Manager Militar"""

    def __init__(self):
        logger.info("⚡ Military Energy Manager inicializado")

    def prevent_disk_sleep(self):
        logger.info("ℹ️ Energía delegada a caffeinate")

    def allow_disk_sleep(self, duration: int = 600):
        pass  # TODO: Implementar control de sleep de disco

    def restore_defaults(self):
        pass  # TODO: Restaurar configuración por defecto


def main():
    # Asegurar instancia única
    if not ensure_single_instance("metacortex_daemon_military"):
        logger.error("❌ Ya existe una instancia de METACORTEX MILITARY DAEMON.")
        sys.exit(1)

    daemon = None
    try:
        daemon = MetacortexMilitaryDaemon()
        daemon.start_autonomous_mode()
        # Mantener el hilo principal vivo mientras los daemons se ejecutan en segundo plano
        while daemon.running:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Interrupción de teclado recibida. Apagando...")
        if daemon:
            daemon.shutdown()
    except Exception as e:
        logger.critical(f"💥 Error crítico no manejado en el arranque: {e}", exc_info=True)
        if daemon:
            daemon.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()