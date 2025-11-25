#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import time
import logging
import threading
from typing import Dict, Optional, Tuple, Any, Deque, Callable
from dataclasses import dataclass
from collections import deque
from enum import Enum
import traceback

# Solución para el problema de importación relativa cuando se carga dinámicamente
# Añadir el directorio padre ('agent_modules') a sys.path
sys.path.append(str(Path(__file__).parent))

from redis_connection_pool import get_redis_client

"""
Advanced Rate Limiting System v2.1
"""

# Configuración del logger
logger = logging.getLogger(__name__)

# Redis opcional
try:
    import redis
    from redis.exceptions import ConnectionError as RedisConnectionError
    redis_available = True
except ImportError:
    redis = None # type: ignore
    RedisConnectionError = Exception # type: ignore
    redis_available = False
    logger.warning("⚠️ Redis no disponible - usando rate limiting local")


class RateLimitAlgorithm(Enum):
    """Algoritmos de rate limiting"""

    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitConfig:
    """Configuración de rate limit"""

    max_requests: int  # Máximo de requests
    window_seconds: float  # Ventana de tiempo en segundos
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    burst_size: Optional[int] = None  # Para token bucket
    adaptive: bool = False  # Rate limiting adaptativo


@dataclass
class RateLimitStatus:
    """Estado actual del rate limit"""

    allowed: bool
    remaining: int
    reset_timestamp: float
    retry_after_seconds: Optional[float] = None

    def to_headers(self) -> Dict[str, str]:
        """Convertir a headers HTTP"""
        headers = {
            "X-RateLimit-Limit": str(self.remaining + (0 if self.allowed else 1)),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_timestamp)),
        }

        if self.retry_after_seconds:
            headers["Retry-After"] = str(int(self.retry_after_seconds))

        return headers


class TokenBucket:
    """Token Bucket Algorithm - Permite burst de tráfico"""

    def __init__(
        self, max_requests: int, window_seconds: float, burst_size: Optional[int] = None
    ):
        self.capacity = burst_size or max_requests
        self.tokens = self.capacity
        self.refill_rate = max_requests / window_seconds  # tokens/segundo
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def _refill(self):
        """Rellenar tokens basado en tiempo transcurrido"""
        now = time.time()
        elapsed = now - self.last_refill

        # Calcular tokens a agregar
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> Tuple[bool, int]:
        """
        Intentar consumir tokens
        Returns: (permitido, tokens_restantes)
        """
        with self.lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, int(self.tokens)
            else:
                return False, int(self.tokens)

    def get_status(self) -> Dict[str, Any]:
        """Obtener estado actual"""
        with self.lock:
            self._refill()
            return {
                "tokens": int(self.tokens),
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
            }


class LeakyBucket:
    """Leaky Bucket Algorithm - Tráfico constante"""

    def __init__(self, max_requests: int, window_seconds: float):
        self.capacity = max_requests
        self.leak_rate = max_requests / window_seconds  # requests/segundo
        self.queue_size = 0
        self.last_leak = time.time()
        self.lock = threading.Lock()

    def _leak(self):
        """Vaciar el bucket basado en tiempo transcurrido"""
        now = time.time()
        elapsed = now - self.last_leak

        # Calcular cuánto se vació
        leaked = elapsed * self.leak_rate
        self.queue_size = max(0, self.queue_size - leaked)
        self.last_leak = now

    def consume(self, requests: int = 1) -> Tuple[bool, int]:
        """
        Intentar agregar requests al bucket
        Returns: (permitido, espacio_restante)
        """
        with self.lock:
            self._leak()

            if self.queue_size + requests <= self.capacity:
                self.queue_size += requests
                return True, int(self.capacity - self.queue_size)
            else:
                return False, int(self.capacity - self.queue_size)


class SlidingWindowCounter:
    """Sliding Window Algorithm - Más preciso que fixed window"""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Deque[float] = deque()
        self.lock = threading.Lock()

    def _cleanup_old_requests(self):
        """Eliminar requests fuera de la ventana"""
        now = time.time()
        cutoff = now - self.window_seconds

        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

    def consume(self, requests: int = 1) -> Tuple[bool, int]:
        """
        Intentar agregar requests
        Returns: (permitido, requests_restantes)
        """
        with self.lock:
            self._cleanup_old_requests()

            current_count = len(self.requests)
            remaining = self.max_requests - current_count

            if current_count < self.max_requests:
                # Agregar timestamp(s)
                now = time.time()
                for _ in range(requests):
                    self.requests.append(now)
                return True, remaining - requests
            else:
                return False, 0

    def get_reset_timestamp(self) -> float:
        """Obtener timestamp cuando se resetea"""
        if not self.requests:
            return time.time()
        return self.requests[0] + self.window_seconds


class AdaptiveRateLimiter:
    """Rate Limiter que se adapta a la carga del sistema"""

    def __init__(self, base_config: RateLimitConfig):
        self.base_config = base_config
        self.current_multiplier = 1.0
        self.lock = threading.Lock()

        # Métricas del sistema
        self.system_load_history: Deque[Tuple[float, float]] = deque(maxlen=60)  # Últimos 60 segundos

        # Thresholds
        self.high_load_threshold = 0.8  # 80% de carga
        self.low_load_threshold = 0.3  # 30% de carga

    def update_system_load(self, load: float):
        """Actualizar carga del sistema (0.0 - 1.0)"""
        with self.lock:
            self.system_load_history.append((time.time(), load))
            self._adjust_rate_limit()

    def _adjust_rate_limit(self):
        """Ajustar rate limit basado en carga"""
        if not self.system_load_history:
            return

        # Calcular carga promedio reciente (últimos 10s)
        recent_cutoff = time.time() - 10
        recent_loads = [
            load for ts, load in self.system_load_history if ts > recent_cutoff
        ]

        if not recent_loads:
            return

        avg_load = sum(recent_loads) / len(recent_loads)

        # Ajustar multiplicador
        if avg_load > self.high_load_threshold:
            # Alta carga - reducir rate limit
            self.current_multiplier = max(0.3, self.current_multiplier - 0.1)
            logger.info(
                f"🔻 Rate limit reducido: {self.current_multiplier:.2f}x (carga: {avg_load:.2%})"
            )
        elif avg_load < self.low_load_threshold:
            # Baja carga - aumentar rate limit
            self.current_multiplier = min(1.5, self.current_multiplier + 0.05)
            logger.info(
                f"🔺 Rate limit aumentado: {self.current_multiplier:.2f}x (carga: {avg_load:.2%})"
            )

    def get_adjusted_config(self) -> RateLimitConfig:
        """Obtener configuración ajustada"""
        with self.lock:
            adjusted_max = int(self.base_config.max_requests * self.current_multiplier)

            return RateLimitConfig(
                max_requests=adjusted_max,
                window_seconds=self.base_config.window_seconds,
                algorithm=self.base_config.algorithm,
                burst_size=self.base_config.burst_size,
            )


class RateLimitSystem:
    """Sistema completo de rate limiting"""

    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: int = 6379,
        enable_adaptive: bool = True,
    ):
        # Limiters por identificador (user_id, ip, etc)
        self.limiters: Dict[str, Any] = {}
        self.configs: Dict[str, RateLimitConfig] = {}
        self.adaptive_limiters: Dict[str, AdaptiveRateLimiter] = {}
        self.telemetry: Optional[Any] = None # Tipo explícito para telemetría

        self.lock = threading.RLock()

        # Redis para rate limiting distribuido (usando CONNECTION POOL)
        self.redis_client: Optional[Any] = None
        if redis_host and redis_available and redis:
            try:
                # CRITICAL: Usar connection pool para prevenir leaks
                
                logger.info(
                    f"🔧 Conectando Redis rate limiting: {redis_host}:{redis_port} (usando pool compartido)"
                )
                self.redis_client = get_redis_client(
                    host=redis_host,
                    port=redis_port,
                    db=0,
                    max_connections=50,
                    decode_responses=True,
                )
                ping_result = self.redis_client.ping()
                logger.info(
                    f"✅ Redis conectado para rate limiting: {redis_host}:{redis_port} (ping={ping_result}, pool compartido)"
                )
            except ImportError:
                # Fallback: conexión directa si pool no disponible
                logger.warning("⚠️ redis_connection_pool no disponible, usando conexión directa")
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True,
                    socket_connect_timeout=10,
                    socket_timeout=10,
                )
            except RedisConnectionError as e:
                logger.error(f"❌ Redis ConnectionError: {e}")

                logger.error(f"   Traceback: {traceback.format_exc()}")
                self.redis_client = None
            except Exception as e:
                logger.error(f"❌ Redis Exception: {type(e).__name__}: {e}")

                logger.error(f"   Traceback: {traceback.format_exc()}")
                self.redis_client = None

        self.enable_adaptive = enable_adaptive

        # 🧠 Conexión a red neuronal (FASE 1: Declaración)
        self.neural_network: Optional[Any] = None
        self._neural_ready = False
        logger.debug("🧠 Neural network: declarado, registro diferido")

    def connect_to_neural_network(self) -> bool:
        """
        🧠 FASE 2: Registro en red neuronal (llamar después de __init__)

        Sistema de 2 fases para evitar circular imports:
        - Fase 1 (__init__): Declaración de variables
        - Fase 2 (este método): Registro y conexión

        Returns:
            bool: True si conectó exitosamente
        """
        if self._neural_ready:
            logger.debug("✅ Rate limiting ya conectado a red neuronal")
            return True
        
        try:
            from neural_symbiotic_network import get_neural_network
            
            self.neural_network = get_neural_network()
            if self.neural_network:
                # Registro bidireccional
                self.neural_network.register_module("rate_limiting", self)
                self._neural_ready = True
                logger.info("✅ 'rate_limiting' conectado a red neuronal (bidireccional)")
                return True
            else:
                logger.warning("⚠️ get_neural_network() retornó None")
                return False
                
        except RecursionError as e:
            logger.error(f"🚨 RecursionError en neural connection: {e}")
            logger.error("   Sistema continuará sin conexión neuronal")
            return False
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None
            return False

        # Integración con telemetría
        try:
            from agent_modules.telemetry import get_telemetry

            self.telemetry = get_telemetry()
        except Exception as e:
            logger.warning(f"⚠️ Telemetría no disponible: {e}")
            self.telemetry = None

    def register_limit(self, identifier: str, config: RateLimitConfig):
        """Registrar configuración de rate limit"""
        with self.lock:
            self.configs[identifier] = config

            # Crear adaptive limiter si está habilitado
            if self.enable_adaptive and config.adaptive:
                self.adaptive_limiters[identifier] = AdaptiveRateLimiter(config)

            logger.info(
                f"✅ Rate limit registrado: {identifier} ({config.max_requests} req/{config.window_seconds}s)"
            )

    def _get_or_create_limiter(self, identifier: str, key: str) -> Any:
        """Obtener o crear limiter para un key específico"""
        limiter_key = f"{identifier}:{key}"

        if limiter_key in self.limiters:
            return self.limiters[limiter_key]

        # Obtener config (ajustada si es adaptativo)
        config = self.configs.get(identifier)
        if not config:
            raise ValueError(f"No config para: {identifier}")

        if self.enable_adaptive and identifier in self.adaptive_limiters:
            config = self.adaptive_limiters[identifier].get_adjusted_config()

        # Crear limiter según algoritmo
        limiter: Any # Usar Any para evitar errores de tipo
        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            limiter = TokenBucket(
                config.max_requests, config.window_seconds, config.burst_size
            )
        elif config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            limiter = LeakyBucket(config.max_requests, config.window_seconds)
        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            limiter = SlidingWindowCounter(config.max_requests, config.window_seconds)
        else:
            limiter = TokenBucket(config.max_requests, config.window_seconds)

        self.limiters[limiter_key] = limiter
        return limiter

    def check_rate_limit(self, user_id: str, config: RateLimitConfig) -> bool:
        """Verificar rate limit (signature compatible con tests)"""
        # Usar user_id como identifier único para aislamiento perfecto
        identifier = f"user_{user_id}"

        # Registrar config para este usuario si no existe
        if identifier not in self.configs:
            self.configs[identifier] = config

        # Usar el método interno con key única
        status = self._check_rate_limit_internal(identifier, user_id, 1)
        return status.allowed

    def _check_rate_limit_internal(
        self, identifier: str, key: str, tokens: int = 1
    ) -> RateLimitStatus:
        """
        Verificar rate limit

        Args:
            identifier: Tipo de límite (ej: "api", "ml_predictions")
            key: Key específica (ej: user_id, ip_address)
            tokens: Número de tokens a consumir

        Returns:
            RateLimitStatus con información del límite
        """
        with self.lock:
            limiter = self._get_or_create_limiter(identifier, key)
            allowed, remaining = limiter.consume(tokens)

            # Calcular reset timestamp
            if isinstance(limiter, SlidingWindowCounter):
                reset_ts = limiter.get_reset_timestamp()
            else:
                config = self.configs[identifier]
                reset_ts = time.time() + config.window_seconds

            # Calcular retry_after si fue rechazado
            retry_after = None
            if not allowed:
                retry_after = max(1, reset_ts - time.time())

            status = RateLimitStatus(
                allowed=allowed,
                remaining=max(0, remaining),
                reset_timestamp=reset_ts,
                retry_after_seconds=retry_after,
            )

            # Registrar en telemetría
            if self.telemetry:
                self.telemetry.record_metric(
                    "rate_limit_check",
                    1 if allowed else 0,
                    tags={"identifier": identifier, "allowed": str(allowed)},
                    metric_type="counter",
                )

            return status

    def update_system_load(self, identifier: str, load: float):
        """Actualizar carga del sistema para rate limiting adaptativo"""
        if identifier in self.adaptive_limiters:
            self.adaptive_limiters[identifier].update_system_load(load)

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de rate limiting"""
        with self.lock:
            return {
                "total_limiters": len(self.limiters),
                "total_configs": len(self.configs),
                "adaptive_limiters": len(self.adaptive_limiters),
                "redis_connected": self.redis_client is not None,
            }


# Singleton global
_global_rate_limiter: Optional[RateLimitSystem] = None


def get_rate_limiter(**kwargs: Any) -> RateLimitSystem:
    """Obtener instancia global del rate limiter

    Si se pasan kwargs y la instancia ya existe, se recrea con los nuevos parámetros.
    Esto permite configurar Redis después de la primera inicialización.
    """
    global _global_rate_limiter
    if _global_rate_limiter is None or kwargs:  # Recrear si hay kwargs
        _global_rate_limiter = RateLimitSystem(**kwargs)
    return _global_rate_limiter


# Decorator para rate limiting
def rate_limit(
    max_requests: int = 10, window_seconds: float = 60.0, identifier: str = "default"
) -> Callable[..., Any]:
    """
    Decorator para aplicar rate limiting a funciones

    Args:
        max_requests: Máximo de requests permitidos
        window_seconds: Ventana de tiempo en segundos
        identifier: Identificador único (opcional)
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generar key desde args (primer argumento si existe, sino 'default')
            key = str(args[0]) if args else "default"

            # Check rate limit con config inline
            limiter = get_rate_limiter()
            config = RateLimitConfig(
                max_requests=max_requests, window_seconds=window_seconds
            )
            # Corregido: check_rate_limit en lugar de check_rate_limiter
            allowed = limiter.check_rate_limit(key, config)

            if not allowed:
                raise Exception(f"Rate limit excedido para {key}")

            # Ejecutar función
            return func(*args, **kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("🚦 Testing Rate Limiting System...\n")

    limiter = get_rate_limiter(enable_adaptive=True)

    # Test 1: Token Bucket
    print("Test 1: Token Bucket (5 req/10s, burst=10)")
    config1 = RateLimitConfig(
        max_requests=5,
        window_seconds=10,
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        burst_size=10,
    )
    limiter.register_limit("test_api", config1)


    success_count = 0
    for i in range(12):
        allowed = limiter.check_rate_limit("user_123", config1)
        if allowed:
            success_count += 1
            print(f"  ✅ Request {i + 1}: Permitido")
        else:
            print(f"  ❌ Request {i + 1}: Rechazado")

    print(f"  Total permitidos: {success_count}/12\n")

    # Test 2: Sliding Window
    print("Test 2: Sliding Window (3 req/5s)")
    config2 = RateLimitConfig(
        max_requests=3,
        window_seconds=5,
        algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
    )
    limiter.register_limit("test_sliding", config2)


    for i in range(5):
        allowed = limiter.check_rate_limit("user_456", config2)
        print(f"  Request {i + 1}: {'✅ OK' if allowed else '❌ DENIED'}")
        time.sleep(1)

    print()

    # Test 3: Adaptive Rate Limiting
    print("Test 3: Adaptive Rate Limiting")
    config3 = RateLimitConfig(
        max_requests=10,
        window_seconds=10,
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        adaptive=True,
    )
    limiter.register_limit("adaptive_api", config3)


    # Simular alta carga
    print("  Simulando alta carga (90%)...")
    limiter.update_system_load("adaptive_api", 0.9)
    time.sleep(0.1)

    allowed = limiter.check_rate_limit("user_789", config3)
    print(f"  Con alta carga: {'✅ OK' if allowed else '❌ DENIED'}")

    # Simular baja carga
    print("  Simulando baja carga (20%)...")
    limiter.update_system_load("adaptive_api", 0.2)
    time.sleep(0.1)

    allowed = limiter.check_rate_limit("user_789", config3)
    print(f"  Con baja carga: {'✅ OK' if allowed else '❌ DENIED'}")

    # Stats
    print("\nTest 4: Estadísticas")
    stats = limiter.get_stats()
    print(f"  Total limiters: {stats['total_limiters']}")
    print(f"  Total configs: {stats['total_configs']}")
    print(f"  Adaptive limiters: {stats['adaptive_limiters']}")

    print("\n✅ Tests completados")
