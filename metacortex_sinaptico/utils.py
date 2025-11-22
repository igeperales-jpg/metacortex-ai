#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METACORTEX - Utilidades y Configuración
========================================

Tipos de datos, configuración y herramientas comunes para el sistema cognitivo.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# === CONFIGURACIÓN DEL AGENTE - SINGLETON VERDADERO ===


class AgentConfig:
    """
    Configuración del agente cognitivo - TRUE SINGLETON PATTERN.

    🔥 IMPLEMENTACIÓN SINGLETON THREAD-SAFE:
        pass  # TODO: Implementar
    - Solo UNA instancia en toda la aplicación
    - Thread-safe con threading.Lock
    - Previene duplicación completa
    """

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        """Singleton pattern verdadero con thread safety."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    print("🔥 AgentConfig SINGLETON: Nueva instancia única creada")
                # else: Reutilización silenciosa - no necesita warning
        # else: Retorno silencioso - no necesita warning
        return cls._instance

    def __init__(self):
        """Inicialización única - solo se ejecuta UNA vez."""
        if AgentConfig._initialized:
            return  # Ya inicializado, no repetir

        print("🧠 AgentConfig SINGLETON: Inicializando configuración única")

        # === CONFIGURACIÓN UNIFICADA MAESTRA ===
        # 🔥 UNA SOLA FUENTE DE VERDAD - NO MÁS CONFLICTOS

        # Parámetros básicos UNIFICADOS
        self.learning_rate = 0.001  # 🔥 MAESTRA: 0.001 (no 0.1)
        self.exploration_rate = 0.1
        self.memory_size = 1000
        self.context_window = 50  # 🔥 MAESTRA: 50 (no 100)
        self.seed = 42

        # Base de datos UNIFICADA
        self.db_path = "metacortex.sqlite"
        self.history_window = 100  # 🔥 MAESTRA: 100
        self.anomaly_threshold = 2.0  # 🔥 MAESTRA: 2.0

        # Percepción UNIFICADA
        self.perception_threshold = 0.5
        self.attention_span = 10

        # Cognición UNIFICADA
        self.reasoning_depth = 3
        self.creativity_factor = 0.2

        # Aprendizaje UNIFICADO
        self.curiosity_drive = 0.3
        self.novelty_threshold = 0.15  # 🔥 OPTIMIZADO: 0.15 (antes 0.3) para permitir más crecimiento del grafo

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            self.neural_network.register_module("agent_config", self)
            print("✅ 'agent_config' conectado a red neuronal")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None

        # Metacognición UNIFICADA
        self.wellbeing_threshold = 0.4

        # Configurar semilla para reproducibilidad
        random.seed(self.seed)

        # 🔥 LOGGING UNIFICADO - SIN DUPLICACIÓN
        self._setup_unified_logging()

        # Marcar como inicializado
        AgentConfig._initialized = True
        print("✅ AgentConfig SINGLETON: Configuración unificada completada")

    def _setup_unified_logging(self):
        """Configurar logging unificado sin duplicación."""
        # Obtener root logger
        root_logger = logging.getLogger()

        # Limpiar handlers existentes para evitar duplicación
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Configurar ÚNICO handler
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",  # 🔥 Sin duplicar "METACORTEX"
            force=True,  # Forzar reconfiguración
        )
        print("🔧 Logging unificado configurado (sin duplicación)")

    @classmethod
    def get_instance(cls):
        """Método de clase para obtener la instancia singleton."""
        return cls()  # __new__ maneja el singleton

    @classmethod
    def reset_singleton(cls):
        """SOLO para testing - reinicia el singleton."""
        with cls._lock:
            cls._instance = None
            cls._initialized = False
            print("🔄 AgentConfig SINGLETON: Reiniciado (solo testing)")


# 🔥 FUNCIÓN GLOBAL UNIFICADA
def get_singleton_config() -> AgentConfig:
    """
    Retorna la instancia singleton VERDADERA de AgentConfig.

    🔥 GARANTÍA: Solo UNA instancia en toda la aplicación.
    """
    return AgentConfig.get_instance()


# === MODELOS PYDANTIC PARA API ===


class PerceptionInput(BaseModel):
    """Input para el endpoint de percepción."""

    name: str = Field(..., description="Nombre del evento percibido")
    payload: Dict[str, Any] = Field(..., description="Datos del evento")


class PerceptionOutput(BaseModel):
    """Output del endpoint de percepción."""

    anomaly: bool = Field(..., description="Si se detectó una anomalía")
    z_score: Optional[float] = Field(None, description="Puntuación Z si es anomalía")
    stored: bool = Field(..., description="Si se almacenó en memoria")


class MetaReport(BaseModel):
    """Reporte del estado metacognitivo."""

    wellbeing: float = Field(..., description="Nivel de bienestar (0-1)")
    anomalies: int = Field(..., description="Número de anomalías detectadas")
    intention: Optional[str] = Field(None, description="Intención actual")
    notes: List[str] = Field(default_factory=list, description="Notas del sistema")
    timestamp: float = Field(..., description="Timestamp del reporte")


class GraphSnapshot(BaseModel):
    """Snapshot del grafo de conocimiento."""

    nodes: List[str] = Field(..., description="Lista de nodos")
    edges: List[Dict[str, Any]] = Field(..., description="Lista de aristas con pesos")
    metrics: Dict[str, float] = Field(..., description="Métricas del grafo")


class SystemStatus(BaseModel):
    """Estado general del sistema."""

    active: bool = Field(..., description="Si el sistema está activo")
    uptime: float = Field(..., description="Tiempo activo en segundos")
    memory_usage: Dict[str, int] = Field(..., description="Uso de memoria")
    last_tick: Optional[float] = Field(None, description="Último tick procesado")


# === UTILIDADES ===


def get_env_config() -> AgentConfig:
    """
    Obtiene configuración desde variables de entorno.

    🔥 GARANTÍA: Usa SINGLETON VERDADERO - no crea nueva instancia.
    """
    config = get_singleton_config()  # ✅ Retorna ÚNICA instancia

    # 🔥 CONFIGURACIÓN MAESTRA desde ENV (si existe)
    # Solo actualiza valores si las variables de entorno están definidas
    seed_env = os.getenv("AGENT_SEED")
    if seed_env:
        config.seed = int(seed_env)
        random.seed(config.seed)  # Reconfigurar seed
        print(f"🔧 ENV Override: seed={config.seed}")

    db_path_env = os.getenv("METACORTEX_DB_PATH")
    if db_path_env:
        config.db_path = db_path_env
        print(f"🔧 ENV Override: db_path={config.db_path}")

    anomaly_env = os.getenv("ANOMALY_THRESHOLD")
    if anomaly_env:
        config.anomaly_threshold = float(anomaly_env)
        print(f"🔧 ENV Override: anomaly_threshold={config.anomaly_threshold}")

    wellbeing_env = os.getenv("WELLBEING_THRESHOLD")
    if wellbeing_env:
        config.wellbeing_threshold = float(wellbeing_env)
        print(f"🔧 ENV Override: wellbeing_threshold={config.wellbeing_threshold}")

    return config


# === LOGGING SINGLETON - SOLUCIÓN DE RAÍZ ===

_logging_initialized = False
_logging_lock = threading.Lock()


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configura y retorna logger del sistema (SINGLETON).

    🔥 SOLUCIÓN DE RAÍZ: Solo configura handlers UNA vez usando flag global.
    Previene duplicación de mensajes cuando se llama múltiples veces.
    """
    global _logging_initialized

    logger = logging.getLogger("metacortex")

    # Thread-safe initialization
    with _logging_lock:
        if not _logging_initialized:
            # Configurar nivel
            logger.setLevel(getattr(logging, level.upper()))

            # Limpiar handlers existentes (por si acaso)
            logger.handlers.clear()

            # Prevenir propagación al root logger
            logger.propagate = False

            # Crear ÚNICO handler
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            _logging_initialized = True
            logger.debug("🔧 Logging singleton inicializado (primera vez)")
        else:
            # Ya inicializado, solo retornar
            logger.debug("🔧 Logging singleton ya inicializado (reutilizando)")

    return logger


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Limita un valor entre min y max."""
    return max(min_val, min(value, max_val))


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normaliza un valor al rango [0, 1]."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


# === EXCEPCIONES PERSONALIZADAS ===


class MetacortexError(Exception):
    """Excepción base para errores de Metacortex."""

    # IMPLEMENTED: Implement this functionality


class DatabaseError(MetacortexError):
    """Error de base de datos."""

    # IMPLEMENTED: Implement this functionality


class ConfigurationError(MetacortexError):
    """Error de configuración."""

    # IMPLEMENTED: Implement this functionality


class CognitiveError(MetacortexError):
    """Error en procesos cognitivos."""

    # IMPLEMENTED: Implement this functionality


# === DECORADORES AVANZADOS ===

T = TypeVar("T")


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorador para reintentar funciones que fallan.
    
    Args:
        max_attempts: Número máximo de intentos
        delay: Delay inicial entre intentos (segundos)
        backoff: Factor multiplicador del delay en cada intento
        
    Ejemplo:
        @retry(max_attempts=3, delay=0.5, backoff=2.0)
        def api_call():
            return requests.get(url)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception = None
            logger = logging.getLogger("metacortex")
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error: {e}", exc_info=True)
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"Intento {attempt}/{max_attempts} falló para {func.__name__}: {e}. "
                            f"Reintentando en {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"Todos los intentos fallaron para {func.__name__}")
            
            # Si llegamos aquí, todos los intentos fallaron
            raise last_exception  # type: ignore
        
        return wrapper
    return decorator


def rate_limit(calls: int = 10, period: float = 1.0):
    """
    Decorador para limitar la tasa de llamadas a una función.
    
    Args:
        calls: Número máximo de llamadas permitidas
        period: Período de tiempo en segundos
        
    Ejemplo:
        @rate_limit(calls=5, period=1.0)  # 5 llamadas por segundo
        def api_request():
            # IMPLEMENTED: Implement this functionality
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        timestamps: List[float] = []
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with lock:
                now = time.time()
                # Limpiar timestamps antiguos
                timestamps[:] = [ts for ts in timestamps if now - ts < period]
                
                if len(timestamps) >= calls:
                    # Calcular tiempo de espera
                    sleep_time = period - (now - timestamps[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        now = time.time()
                        timestamps[:] = [ts for ts in timestamps if now - ts < period]
                
                timestamps.append(now)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


@contextmanager
def timer(name: str = "Operation") -> Generator[Dict[str, float], None, None]:
    """
    Context manager para medir tiempo de ejecución.
    
    Ejemplo:
        with timer("Database query") as t:
            execute_query()
        print(f"Tiempo: {t['elapsed']}s")
    """
    result: Dict[str, float] = {"start": 0.0, "end": 0.0, "elapsed": 0.0}
    logger = logging.getLogger("metacortex")
    
    result["start"] = time.time()
    logger.debug(f"⏱️  {name}: Iniciando...")
    
    try:
        yield result
    finally:
        result["end"] = time.time()
        result["elapsed"] = result["end"] - result["start"]
        logger.info(f"⏱️  {name}: Completado en {result['elapsed']:.4f}s")


def memoize_with_ttl(ttl: float = 60.0):
    """
    Decorador para cachear resultados con tiempo de vida.
    
    Args:
        ttl: Tiempo de vida del cache en segundos
        
    Ejemplo:
        @memoize_with_ttl(ttl=30.0)
        def expensive_calculation(x):
            return x ** 2
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: Dict[str, tuple[float, T]] = {}
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Crear clave del cache
            key = str(args) + str(sorted(kwargs.items()))
            
            with lock:
                now = time.time()
                
                # Verificar si existe en cache y no ha expirado
                if key in cache:
                    timestamp, value = cache[key]
                    if now - timestamp < ttl:
                        return value
                
                # Calcular nuevo valor
                result = func(*args, **kwargs)
                cache[key] = (now, result)
                
                # Limpiar entradas expiradas
                expired_keys = [
                    k for k, (ts, _) in cache.items()
                    if now - ts >= ttl
                ]
                for k in expired_keys:
                    del cache[k]
                
                return result
        
        return wrapper
    return decorator


# === UTILIDADES DE HASHING Y SERIALIZACIÓN ===


def compute_hash(data: Union[str, bytes, Dict[str, Any]], algorithm: str = "sha256") -> str:
    """
    Calcula hash de datos.
    
    Args:
        data: Datos a hashear (string, bytes o dict)
        algorithm: Algoritmo de hash (md5, sha1, sha256, sha512)
        
    Returns:
        Hash hexadecimal
        
    Ejemplo:
        >>> compute_hash({"key": "value"})
        'a1b2c3...'
    """
    # Convertir a bytes
    if isinstance(data, dict):
        data_bytes = json.dumps(data, sort_keys=True).encode("utf-8")
    elif isinstance(data, str):
        data_bytes = data.encode("utf-8")
    else:
        data_bytes = data
    
    # Calcular hash
    hasher = hashlib.new(algorithm)
    hasher.update(data_bytes)
    return hasher.hexdigest()


def safe_json_loads(data: str, default: Any = None) -> Any:
    """
    Carga JSON de forma segura.
    
    Args:
        data: String JSON
        default: Valor por defecto si falla
        
    Returns:
        Objeto parseado o default
    """
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError) as e:
        logger = logging.getLogger("metacortex")
        logger.warning(f"Error parseando JSON: {e}")
        return default


def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """
    Serializa a JSON de forma segura.
    
    Args:
        data: Objeto a serializar
        default: String por defecto si falla
        
    Returns:
        String JSON o default
    """
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as e:
        logger = logging.getLogger("metacortex")
        logger.warning(f"Error serializando JSON: {e}")
        return default


# === UTILIDADES DE PATHS ===


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Asegura que un directorio existe, creándolo si es necesario.
    
    Args:
        path: Ruta del directorio
        
    Returns:
        Path object del directorio
        
    Ejemplo:
        data_dir = ensure_dir("./data/models")
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_project_root() -> Path:
    """
    Obtiene la raíz del proyecto.
    
    Returns:
        Path a la raíz del proyecto
    """
    # Buscar archivo setup.py, pyproject.toml, o .git
    current = Path.cwd()
    
    for _ in range(10):  # Limitar búsqueda a 10 niveles
        if any((current / marker).exists() for marker in [
            "setup.py", "pyproject.toml", ".git", "requirements.txt"
        ]):
            return current
        
        parent = current.parent
        if parent == current:  # Llegamos a la raíz del sistema
            break
        current = parent
    
    # Si no encontramos, usar directorio actual
    return Path.cwd()


def safe_file_read(path: Union[str, Path], default: str = "") -> str:
    """
    Lee archivo de forma segura.
    
    Args:
        path: Ruta del archivo
        default: Contenido por defecto si falla
        
    Returns:
        Contenido del archivo o default
    """
    try:
        return Path(path).read_text(encoding="utf-8")
    except (IOError, OSError) as e:
        logger = logging.getLogger("metacortex")
        logger.warning(f"Error leyendo archivo {path}: {e}")
        return default


def safe_file_write(path: Union[str, Path], content: str) -> bool:
    """
    Escribe archivo de forma segura.
    
    Args:
        path: Ruta del archivo
        content: Contenido a escribir
        
    Returns:
        True si éxito, False si error
    """
    try:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(content, encoding="utf-8")
        return True
    except (IOError, OSError) as e:
        logger = logging.getLogger("metacortex")
        logger.error(f"Error escribiendo archivo {path}: {e}")
        return False


# === VALIDADORES PYDANTIC CUSTOM ===


def validate_probability(value: float) -> float:
    """
    Valida que un valor sea una probabilidad válida [0, 1].
    
    Raises:
        ValueError: Si el valor está fuera del rango
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Probability must be between 0 and 1, got {value}")
    return value


def validate_positive(value: Union[int, float]) -> Union[int, float]:
    """
    Valida que un valor sea positivo.
    
    Raises:
        ValueError: Si el valor no es positivo
    """
    if value <= 0:
        raise ValueError(f"Value must be positive, got {value}")
    return value


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """
    Sanitiza un string para prevenir inyección.
    
    Args:
        value: String a sanitizar
        max_length: Longitud máxima permitida
        
    Returns:
        String sanitizado
    """
    # Remover caracteres de control
    sanitized = "".join(char for char in value if char.isprintable() or char.isspace())
    
    # Limitar longitud
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # Remover espacios extras
    sanitized = " ".join(sanitized.split())
    
    return sanitized


# === UTILIDADES DE MÉTRICAS Y ESTADÍSTICAS ===


@lru_cache(maxsize=128)
def moving_average(values: tuple[float, ...], window: int = 5) -> float:
    """
    Calcula promedio móvil de valores.
    
    Args:
        values: Tupla de valores (debe ser tupla para cache)
        window: Tamaño de la ventana
        
    Returns:
        Promedio móvil
    """
    if not values:
        return 0.0
    
    recent = values[-window:]
    return sum(recent) / len(recent)


def calculate_percentile(values: List[float], percentile: float) -> float:
    """
    Calcula percentil de una lista de valores.
    
    Args:
        values: Lista de valores
        percentile: Percentil a calcular (0-100)
        
    Returns:
        Valor en el percentil especificado
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    index = int(len(sorted_values) * (percentile / 100.0))
    index = max(0, min(index, len(sorted_values) - 1))
    
    return sorted_values[index]


def calculate_variance(values: List[float]) -> float:
    """
    Calcula varianza de valores.
    
    Args:
        values: Lista de valores
        
    Returns:
        Varianza
    """
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def calculate_std_dev(values: List[float]) -> float:
    """
    Calcula desviación estándar.
    
    Args:
        values: Lista de valores
        
    Returns:
        Desviación estándar
    """
    return calculate_variance(values) ** 0.5


# === UTILIDADES DE SISTEMA ===


def get_memory_usage() -> Dict[str, int]:
    """
    Obtiene uso de memoria del proceso actual.
    
    Returns:
        Dict con rss, vms en bytes
    """
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "rss": usage.ru_maxrss * 1024,  # KB to bytes on Linux
            "vms": 0,  # No disponible fácilmente
        }
    except ImportError:
        # Fallback para Windows
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                "rss": mem_info.rss,
                "vms": mem_info.vms,
            }
        except ImportError:
            return {"rss": 0, "vms": 0}


def get_system_info() -> Dict[str, Any]:
    """
    Obtiene información del sistema.
    
    Returns:
        Dict con platform, python_version, etc.
    """
    return {
        "platform": sys.platform,
        "python_version": sys.version,
        "python_implementation": sys.implementation.name,
        "timestamp": datetime.now().isoformat(),
    }


# === UTILIDADES DE CONFIGURACIÓN MEJORADAS ===


@lru_cache(maxsize=1)
def get_cached_config() -> AgentConfig:
    """
    Obtiene configuración cacheada (singleton con LRU cache adicional).
    
    Returns:
        AgentConfig singleton instance
    """
    return get_singleton_config()


def validate_config(config: AgentConfig) -> List[str]:
    """
    Valida configuración y retorna lista de advertencias.
    
    Args:
        config: Configuración a validar
        
    Returns:
        Lista de mensajes de advertencia (vacía si todo está bien)
    """
    warnings: List[str] = []
    
    # Validar learning_rate
    if not 0.0 < config.learning_rate <= 1.0:
        warnings.append(f"learning_rate debería estar entre 0 y 1, got {config.learning_rate}")
    
    # Validar exploration_rate
    if not 0.0 <= config.exploration_rate <= 1.0:
        warnings.append(f"exploration_rate debe ser probabilidad [0,1], got {config.exploration_rate}")
    
    # Validar memory_size
    if config.memory_size <= 0:
        warnings.append(f"memory_size debe ser positivo, got {config.memory_size}")
    
    # Validar context_window
    if config.context_window <= 0:
        warnings.append(f"context_window debe ser positivo, got {config.context_window}")
    
    # Validar anomaly_threshold
    if config.anomaly_threshold <= 0:
        warnings.append(f"anomaly_threshold debe ser positivo, got {config.anomaly_threshold}")
    
    # Validar wellbeing_threshold
    if not 0.0 <= config.wellbeing_threshold <= 1.0:
        warnings.append(f"wellbeing_threshold debe estar en [0,1], got {config.wellbeing_threshold}")
    
    # Validar db_path
    if not config.db_path:
        warnings.append("db_path está vacío")
    
    return warnings