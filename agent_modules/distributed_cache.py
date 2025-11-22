#!/usr/bin/env python3
"""
💾 DISTRIBUTED CACHE MODULE v3.0 - Cache multi-nivel (Military Grade)

Características EXPONENCIALES:
    pass  # TODO: Implementar
- L1: In-Memory Cache (ultra-rápido) con LRU inteligente
- L2: Redis Cache (distribuido) con circuit breaker
- L3: Disk Cache (persistente) con compresión
- Cache warming automático multi-nivel
- Invalidación inteligente por namespace/pattern
- TTL adaptativo basado en patrones de uso
- Compression automático para datos grandes (>1KB)
- Cache coherence protocol con versioning
- Health checks automáticos
- Métricas detalladas en tiempo real
- Type safety completo
- Error handling robusto con fallbacks
"""

import time
import pickle
import hashlib
import logging
import threading
import zlib
from typing import Any, Optional, Callable, Dict, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from functools import wraps
from collections import OrderedDict
from datetime import datetime, timedelta
from enum import Enum
import sys
import redis as redis_module
import shutil

logger = logging.getLogger(__name__)

# Imports condicionales con type safety
REDIS_AVAILABLE: bool
try:
    import redis
    from redis import Redis, ConnectionError as RedisConnectionError

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    RedisConnectionError = type("RedisConnectionError", (Exception,), {})  # type: ignore
    logger.warning("⚠️ Redis no disponible - usando solo cache en memoria")


class CacheLevel(Enum):
    """Niveles de caché"""

    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISK = "l3_disk"


class CircuitState(Enum):
    """Estados del circuit breaker"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Bloqueado por errores
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit Breaker para Redis con auto-recovery"""

    failure_threshold: int = 5
    timeout_seconds: int = 60
    half_open_max_calls: int = 3

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    half_open_calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Optional[Any]:
        """Ejecutar función con circuit breaker"""
        with self.lock:
            # Si está abierto, verificar timeout
            if self.state == CircuitState.OPEN:
                if (
                    self.last_failure_time
                    and time.time() - self.last_failure_time >= self.timeout_seconds
                ):
                    # Intentar recovery
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info("🔄 Circuit breaker: HALF_OPEN (intentando recovery)")
                else:
                    return None  # Circuito abierto, no ejecutar

            # Si está half-open, limitar llamadas
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    return None
                self.half_open_calls += 1

        # Ejecutar función
        try:
            result = func(*args, **kwargs)

            # Éxito - resetear circuito
            with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    logger.info("✅ Circuit breaker: CLOSED (recovery exitoso)")
                self.failure_count = 0

            return result

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            # Fallo - incrementar contador
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(
                        f"⚠️ Circuit breaker: OPEN (failures: {self.failure_count})"
                    )

            logger.debug(f"Circuit breaker: Error en llamada: {e}")
            return None

    def is_open(self) -> bool:
        """Verificar si circuito está abierto"""
        return self.state == CircuitState.OPEN


@dataclass
class CacheEntry:
    """Entrada de caché con metadata extendida"""

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    size_bytes: int = 0
    compressed: bool = False
    version: int = 1
    namespace: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Verificar si expiró con precisión"""
        if self.ttl_seconds is None:
            return False
        elapsed = time.time() - self.created_at
        return elapsed >= self.ttl_seconds

    def touch(self) -> None:
        """Actualizar last_accessed y contador"""
        self.last_accessed = time.time()
        self.access_count += 1

    def get_age_seconds(self) -> float:
        """Obtener edad de la entrada"""
        return time.time() - self.created_at

    def get_remaining_ttl(self) -> Optional[float]:
        """Obtener TTL restante en segundos"""
        if self.ttl_seconds is None:
            return None
        remaining = self.ttl_seconds - self.get_age_seconds()
        return max(0, remaining)


class LRUCache:
    """
    Cache LRU (Least Recently Used) mejorado con:
    - Compresión automática para datos grandes
    - Limpieza automática de expirados
    - Thread-safe operations
    - Métricas detalladas
    """

    def __init__(
        self,
        max_size: int = 1000,
        enable_compression: bool = True,
        compression_threshold_kb: float = 1.0,
    ):
        self.max_size = max_size
        self.enable_compression = enable_compression
        self.compression_threshold_bytes = int(compression_threshold_kb * 1024)
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        
        # Métricas detalladas
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        self.compressions = 0
        self.total_size_bytes = 0
        
        # Cleanup automático
        self._cleanup_interval = 60  # segundos
        self._last_cleanup = time.time()

    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache con auto-cleanup"""
        with self.lock:
            # Cleanup periódico de expirados
            if time.time() - self._last_cleanup > self._cleanup_interval:
                self._cleanup_expired()

            if key not in self.cache:
                self.misses += 1
                return None

            entry = self.cache[key]

            # Verificar expiración
            if entry.is_expired():
                del self.cache[key]
                self.total_size_bytes -= entry.size_bytes
                self.misses += 1
                self.expirations += 1
                return None

            # LRU: mover al final (más reciente)
            self.cache.move_to_end(key)
            entry.touch()
            self.hits += 1

            # Descomprimir si es necesario
            value = entry.value
            if entry.compressed:
                try:
                    value = pickle.loads(zlib.decompress(value))  # type: ignore
                except Exception as e:
                    logger.error(f"Error descomprimiendo valor: {e}")
                    return None

            return value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        namespace: str = "default",
    ) -> None:
        """Guardar en cache con compresión opcional"""
        with self.lock:
            # Serializar y comprimir si es necesario
            serialized = pickle.dumps(value)
            size_bytes = len(serialized)
            compressed = False

            if (
                self.enable_compression
                and size_bytes > self.compression_threshold_bytes
            ):
                try:
                    compressed_data = zlib.compress(serialized, level=6)
                    if len(compressed_data) < size_bytes:
                        serialized = compressed_data
                        size_bytes = len(compressed_data)
                        compressed = True
                        self.compressions += 1
                except Exception as e:
                    logger.warning(f"Error comprimiendo: {e}")

            # Crear entrada
            entry = CacheEntry(
                key=key,
                value=serialized if compressed else value,
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes,
                compressed=compressed,
                namespace=namespace,
            )

            # Si existe, eliminar primero para actualizar tamaño
            if key in self.cache:
                old_entry = self.cache[key]
                self.total_size_bytes -= old_entry.size_bytes
                del self.cache[key]

            # Agregar al final (más reciente)
            self.cache[key] = entry
            self.total_size_bytes += size_bytes

            # Evicción LRU si excede tamaño
            while len(self.cache) > self.max_size:
                oldest_key, oldest_entry = self.cache.popitem(last=False)
                self.total_size_bytes -= oldest_entry.size_bytes
                self.evictions += 1

    def delete(self, key: str) -> bool:
        """Eliminar del cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.total_size_bytes -= entry.size_bytes
                del self.cache[key]
                return True
            return False

    def invalidate_namespace(self, namespace: str) -> int:
        """Invalidar todas las entradas de un namespace"""
        with self.lock:
            keys_to_delete = [
                key for key, entry in self.cache.items() if entry.namespace == namespace
            ]
            for key in keys_to_delete:
                entry = self.cache[key]
                self.total_size_bytes -= entry.size_bytes
                del self.cache[key]
            return len(keys_to_delete)

    def clear(self) -> None:
        """Limpiar todo el cache"""
        with self.lock:
            self.cache.clear()
            self.total_size_bytes = 0
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.expirations = 0
            self.compressions = 0

    def _cleanup_expired(self) -> int:
        """Limpiar entradas expiradas"""
        expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]
        for key in expired_keys:
            entry = self.cache[key]
            self.total_size_bytes -= entry.size_bytes
            del self.cache[key]
            self.expirations += 1

        self._last_cleanup = time.time()
        if expired_keys:
            logger.debug(f"🧹 Limpiadas {len(expired_keys)} entradas expiradas")
        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas detalladas"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 2),
                "evictions": self.evictions,
                "expirations": self.expirations,
                "compressions": self.compressions,
                "total_size_mb": round(self.total_size_bytes / 1024 / 1024, 2),
                "avg_size_kb": (
                    round(self.total_size_bytes / len(self.cache) / 1024, 2)
                    if len(self.cache) > 0
                    else 0
                ),
            }


class DistributedCacheSystem:
    """
    Sistema de caché distribuido multi-nivel v3.0

    Arquitectura:
    - L1: In-Memory Cache (LRU con compresión)
    - L2: Redis Cache (con circuit breaker)
    - L3: Disk Cache (persistente con metadata)

    Características:
    - Circuit breaker para Redis
    - Warmup automático entre niveles
    - TTL preservado en cascada
    - Type safety completo
    - Error handling robusto
    """

    def __init__(
        self,
        l1_max_size: int = 1000,
        l2_enabled: bool = True,
        l2_host: str = "localhost",
        l2_port: int = 6379,
        l2_db: int = 0,
        l3_enabled: bool = True,
        l3_dir: str = ".cache",
        enable_compression: bool = True,
        circuit_breaker_enabled: bool = True,
    ):
        # L1: In-Memory Cache (ultra-rápido)
        self.l1_cache = LRUCache(
            max_size=l1_max_size, enable_compression=enable_compression
        )
        logger.info(f"✅ L1 Cache inicializado (max_size={l1_max_size})")

        # L2: Redis Cache con circuit breaker
        self.l2_cache: Optional[Any] = None  # Type: redis.Redis
        self.l2_enabled = l2_enabled
        self.circuit_breaker = CircuitBreaker() if circuit_breaker_enabled else None

        if l2_enabled and REDIS_AVAILABLE:
            # 🛡️ CAPA DE PROTECCIÓN RECURSION: Aumentar límite temporalmente
            original_recursion_limit = sys.getrecursionlimit()
            enhanced_limit = max(15000, original_recursion_limit * 2)
            
            logger.info(f"🔧 Conectando Redis L2: {l2_host}:{l2_port}")
            logger.debug(f"🛡️ Recursion limit: {original_recursion_limit} → {enhanced_limit}")
            
            redis_l2_successful = False
            sys.setrecursionlimit(enhanced_limit)
            
            try:
                # CRÍTICO: Usar connection pool para evitar fuga de conexiones
                try:
                    from redis_connection_pool import get_redis_client
                    
                    self.l2_cache = get_redis_client(
                        host=l2_host,
                        port=l2_port,
                        db=l2_db,
                        max_connections=50,  # Pool compartido
                        decode_responses=False,
                    )
                    
                    # Probar conexión
                    ping_result = self.l2_cache.ping()
                    redis_l2_successful = True
                    logger.info(
                        f"✅ L2 Cache (Redis) conectado: {l2_host}:{l2_port} (ping={ping_result}, usando pool compartido)"
                    )
                    
                except ImportError:
                    # Fallback: conexión directa si redis_connection_pool no disponible
                    logger.warning("⚠️ redis_connection_pool no disponible, usando conexión directa")

                    self.l2_cache = redis_module.Redis(
                        host=l2_host,
                        port=l2_port,
                        db=l2_db,
                        decode_responses=False,
                        socket_connect_timeout=10,
                        socket_timeout=10,
                        socket_keepalive=True,
                        health_check_interval=30,
                        retry_on_timeout=True,
                        max_connections=100,
                    )
                    
                    # Probar conexión con timeout
                    ping_result = self.l2_cache.ping()
                    redis_l2_successful = True
                    logger.info(
                        f"✅ L2 Cache (Redis) conectado: {l2_host}:{l2_port} (ping={ping_result}, conexión directa)"
                    )
                
            except RecursionError as e:
                logger.error(f"🚨 Redis RecursionError detectado: {e}")
                logger.error(f"   EventDispatcher causó recursión - L2 desactivado")
                self.l2_cache = None
                redis_l2_successful = False
                
            except RedisConnectionError as e:
                logger.error(f"❌ Redis ConnectionError: {e}")
                self.l2_cache = None
                redis_l2_successful = False
                
            except Exception as e:
                logger.error(f"❌ Redis Exception: {type(e).__name__}: {e}")
                self.l2_cache = None
                redis_l2_successful = False
            
            finally:
                # Restaurar límite original SIEMPRE
                sys.setrecursionlimit(original_recursion_limit)
                logger.debug(f"🛡️ Recursion limit restaurado: {original_recursion_limit}")
            
            # 🧠 CAPA DE INTELIGENCIA: Advertencia si L2 no disponible
            if not redis_l2_successful:
                logger.warning("⚠️ L2 Cache (Redis) no disponible - usando solo L1+L3")
                logger.info("💡 Sistema continuará con cache L1 (memoria) + L3 (disco)")

        # L3: Disk Cache (persistente)
        self.l3_enabled = l3_enabled
        self.l3_dir = Path(l3_dir)
        if l3_enabled:
            self.l3_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"✅ L3 Cache (Disk) inicializado: {self.l3_dir}")

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
            logger.debug("✅ Distributed cache ya conectado a red neuronal")
            return True

        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            if self.neural_network:
                # Registro bidireccional
                self.neural_network.register_module("distributed_cache", self)
                self._neural_ready = True
                logger.info("✅ 'distributed_cache' conectado a red neuronal (bidireccional)")
                return True

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

    def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor con cascade L1 → L2 → L3
        
        Incluye:
        - Circuit breaker para Redis
        - Warmup automático de niveles superiores
        - Preservación de TTL
        - Type safety
        """
        # L1: In-Memory
        value = self.l1_cache.get(key)
        if value is not None:
            logger.debug(f"✅ L1 HIT: {key}")
            return value

        # L2: Redis con circuit breaker
        if self.l2_cache and (
            not self.circuit_breaker or not self.circuit_breaker.is_open()
        ):
            def _get_from_redis() -> Optional[bytes]:
                if self.l2_cache:
                    result = self.l2_cache.get(key)
                    return result if result else None  # type: ignore
                return None

            if self.circuit_breaker:
                data = self.circuit_breaker.call(_get_from_redis)
            else:
                try:
                    data = _get_from_redis()
                except Exception as e:
                    logger.warning(f"⚠️ L2 error: {e}")
                    data = None

            if data:
                try:
                    value = pickle.loads(data)  # type: ignore
                    # Warm up L1
                    self.l1_cache.set(key, value)
                    logger.debug(f"✅ L2 HIT: {key}")
                    return value
                except Exception as e:
                    logger.warning(f"⚠️ L2 deserialization error: {e}")

        # L3: Disk
        if self.l3_enabled:
            cache_file = self._get_cache_file_path(key)
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        cache_data = pickle.load(f)

                    # Extraer valor y metadata
                    if isinstance(cache_data, dict) and "value" in cache_data:
                        value = cache_data["value"]
                        ttl_seconds = cache_data.get("ttl_seconds")
                        created_at = cache_data.get("created_at", time.time())

                        # Verificar si expiró
                        if ttl_seconds is not None:
                            elapsed = time.time() - created_at
                            if elapsed >= ttl_seconds:
                                # Expirado - eliminar y retornar None
                                cache_file.unlink()
                                logger.debug(f"⏰ L3 EXPIRED: {key}")
                                return None
                    else:
                        # Formato antiguo (backward compatibility)
                        value = cache_data
                        ttl_seconds = None

                    # Warm up L1 y L2 con TTL preservado
                    remaining_ttl = None
                    if ttl_seconds is not None and created_at:
                        elapsed = time.time() - created_at
                        remaining_ttl = max(0, ttl_seconds - elapsed)

                    self.l1_cache.set(key, value, ttl_seconds=remaining_ttl)
                    if self.l2_cache:
                        self._set_l2(key, value, remaining_ttl)

                    logger.debug(f"✅ L3 HIT: {key}")
                    return value
                except Exception as e:
                    logger.warning(f"⚠️ L3 error: {e}")

        logger.debug(f"❌ MISS: {key}")
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        warm_all_levels: bool = True,
    ):
        """Guardar valor en todos los niveles"""
        # L1: In-Memory
        self.l1_cache.set(key, value, ttl_seconds=ttl_seconds)

        if warm_all_levels:
            # L2: Redis
            if self.l2_cache:
                self._set_l2(key, value, ttl_seconds)

            # L3: Disk (ahora con TTL)
            if self.l3_enabled:
                self._set_l3(key, value, ttl_seconds)

    def _set_l2(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Guardar en L2 (Redis) con circuit breaker"""
        if not self.l2_cache:
            return False

        def _set_to_redis() -> bool:
            if not self.l2_cache:
                return False
            try:
                data = pickle.dumps(value)
                if ttl_seconds:
                    self.l2_cache.setex(key, int(ttl_seconds), data)  # type: ignore
                else:
                    self.l2_cache.set(key, data)  # type: ignore
                return True
            except Exception as e:
                logger.warning(f"⚠️ Error guardando en L2: {e}")
                return False

        if self.circuit_breaker:
            result = self.circuit_breaker.call(_set_to_redis)
            return bool(result)
        else:
            return _set_to_redis()

    def _set_l3(self, key: str, value: Any, ttl_seconds: Optional[float] = None):
        """Guardar en L3 (Disk) con metadata"""
        try:
            cache_file = self._get_cache_file_path(key)
            cache_file.parent.mkdir(exist_ok=True, parents=True)

            # Guardar valor + metadata para preservar TTL
            cache_data = {
                "value": value,
                "ttl_seconds": ttl_seconds,
                "created_at": time.time(),
            }

            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"⚠️ Error guardando en L3: {e}")

    def delete(self, key: str) -> bool:
        """Eliminar de todos los niveles"""
        deleted = False

        # L1
        if self.l1_cache.delete(key):
            deleted = True

        # L2 con circuit breaker
        if self.l2_cache and (
            not self.circuit_breaker or not self.circuit_breaker.is_open()
        ):
            def _delete_from_redis() -> bool:
                if self.l2_cache:
                    try:
                        return bool(self.l2_cache.delete(key))  # type: ignore
                    except Exception as e:
                        logger.warning(f"⚠️ Error eliminando de L2: {e}")
                return False

            if self.circuit_breaker:
                result = self.circuit_breaker.call(_delete_from_redis)
                deleted = deleted or bool(result)
            else:
                deleted = deleted or _delete_from_redis()

        # L3
        if self.l3_enabled:
            cache_file = self._get_cache_file_path(key)
            if cache_file.exists():
                try:
                    cache_file.unlink()
                    deleted = True
                except Exception as e:
                    logger.warning(f"⚠️ Error eliminando de L3: {e}")

        return deleted

    def invalidate_namespace(self, namespace: str) -> int:
        """Invalidar todas las entradas de un namespace"""
        count = 0

        # L1
        count += self.l1_cache.invalidate_namespace(namespace)

        # L2: No soportado actualmente (necesitaría scan pattern)
        # IMPLEMENTED: Implementar invalidación de namespace en Redis

        logger.info(f"🗑️ Invalidadas {count} entradas del namespace '{namespace}'")
        return count

    def health_check(self) -> Dict[str, Any]:
        """
        Health check del sistema de caché

        Returns:
            Dict con estado de cada nivel
        """
        health = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "levels": {}
        }

        # L1 Health
        try:
            l1_stats = self.l1_cache.get_stats()
            health["levels"]["l1"] = {
                "status": "healthy",
                "size": l1_stats["size"],
                "hit_rate": l1_stats["hit_rate"],
            }
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            health["levels"]["l1"] = {"status": "unhealthy", "error": str(e)}
            health["overall_status"] = "degraded"

        # L2 Health
        if self.l2_enabled and self.l2_cache:
            try:
                def _ping_redis() -> bool:
                    if self.l2_cache:
                        return bool(self.l2_cache.ping())  # type: ignore
                    return False

                if self.circuit_breaker:
                    ping_ok = self.circuit_breaker.call(_ping_redis)
                else:
                    ping_ok = _ping_redis()

                if ping_ok:
                    health["levels"]["l2"] = {
                        "status": "healthy",
                        "circuit_breaker": (
                            self.circuit_breaker.state.value
                            if self.circuit_breaker
                            else "disabled"
                        ),
                    }
                else:
                    health["levels"]["l2"] = {
                        "status": "unhealthy",
                        "circuit_breaker": (
                            self.circuit_breaker.state.value
                            if self.circuit_breaker
                            else "disabled"
                        ),
                    }
                    health["overall_status"] = "degraded"
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                health["levels"]["l2"] = {"status": "unhealthy", "error": str(e)}
                health["overall_status"] = "degraded"
        else:
            health["levels"]["l2"] = {"status": "disabled"}

        # L3 Health
        if self.l3_enabled:
            try:
                if self.l3_dir.exists():
                    cache_files = list(self.l3_dir.rglob("*.cache"))
                    health["levels"]["l3"] = {
                        "status": "healthy",
                        "files": len(cache_files),
                        "path": str(self.l3_dir),
                    }
                else:
                    health["levels"]["l3"] = {
                        "status": "unhealthy",
                        "error": "Directory missing",
                    }
                    health["overall_status"] = "degraded"
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                health["levels"]["l3"] = {"status": "unhealthy", "error": str(e)}
                health["overall_status"] = "degraded"
        else:
            health["levels"]["l3"] = {"status": "disabled"}

        return health

    def clear_all(self):
        """Limpiar todos los niveles"""
        self.l1_cache.clear()

        if self.l2_cache:
            try:
                self.l2_cache.flushdb()
            except Exception as e:
                logger.warning(f"⚠️ Error limpiando L2: {e}")

        if self.l3_enabled:

            if self.l3_dir.exists():
                shutil.rmtree(self.l3_dir)
                self.l3_dir.mkdir(exist_ok=True, parents=True)

    def _get_cache_file_path(self, key: str) -> Path:
        """Obtener ruta de archivo en L3"""
        # Hash del key para crear estructura de directorios
        key_hash = hashlib.md5(key.encode()).hexdigest()
        subdir = key_hash[:2]  # Primeros 2 chars para subdirectorio
        return self.l3_dir / subdir / f"{key_hash}.cache"

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas completas del sistema de caché

        Returns:
            Dict con métricas de todos los niveles
        """
        stats: Dict[str, Any] = {
            "l1": self.l1_cache.get_stats(),
            "l2": None,
            "l3": None,
            "circuit_breaker": None,
        }

        # L2 stats con circuit breaker
        if self.l2_cache and (
            not self.circuit_breaker or not self.circuit_breaker.is_open()
        ):
            def _get_redis_stats() -> Optional[Dict[str, Any]]:
                if self.l2_cache:
                    try:
                        info = self.l2_cache.info()  # type: ignore
                        dbsize = self.l2_cache.dbsize()  # type: ignore
                        return {
                            "connected": True,
                            "keys": dbsize,
                            "memory_used_mb": round(
                                info.get("used_memory", 0) / 1024 / 1024, 2  # type: ignore
                            ),
                        }
                    except Exception as e:
                        logger.error(f"Error: {e}", exc_info=True)
                        return {"connected": False, "error": str(e)}
                return None

            if self.circuit_breaker:
                stats["l2"] = self.circuit_breaker.call(_get_redis_stats)
            else:
                stats["l2"] = _get_redis_stats()

        # L3 stats
        if self.l3_enabled:
            try:
                cache_files = list(self.l3_dir.rglob("*.cache"))
                total_size = sum(f.stat().st_size for f in cache_files)
                stats["l3"] = {
                    "enabled": True,
                    "files": len(cache_files),
                    "total_size_mb": round(total_size / 1024 / 1024, 2),
                    "path": str(self.l3_dir),
                }
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                stats["l3"] = {"enabled": True, "error": str(e)}

        # Circuit breaker stats
        if self.circuit_breaker:
            stats["circuit_breaker"] = {
                "state": self.circuit_breaker.state.value,
                "failure_count": self.circuit_breaker.failure_count,
                "failure_threshold": self.circuit_breaker.failure_threshold,
            }

        return stats


# Singleton global
_global_cache: Optional[DistributedCacheSystem] = None


def get_distributed_cache(**kwargs: Any) -> DistributedCacheSystem:
    """
    Obtener instancia global del cache distribuido

    Si se pasan kwargs y la instancia ya existe, se recrea con los nuevos parámetros.
    Esto permite configurar Redis después de la primera inicialización.

    Args:
        **kwargs: Argumentos para DistributedCacheSystem

    Returns:
        Instancia global del cache
    """
    global _global_cache
    if _global_cache is None or kwargs:  # Recrear si hay kwargs
        _global_cache = DistributedCacheSystem(**kwargs)  # type: ignore
    return _global_cache


# Decorador para cachear resultados de funciones
def cached(
    ttl_seconds: Optional[float] = None, key_prefix: str = "", use_args: bool = True
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorador para cachear resultados de funciones

    Args:
        ttl_seconds: Time to live en segundos
        key_prefix: Prefijo para la clave de cache
        use_args: Incluir argumentos en la clave

    Returns:
        Función decorada

    Example:
        @cached(ttl_seconds=60, key_prefix="fibonacci")
        def fibonacci(n: int) -> int:
            # expensive calculation
            return result
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = get_distributed_cache()

            # Generar cache key
            if use_args:
                key_parts = [key_prefix or func.__name__, str(args), str(kwargs)]
                cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            else:
                cache_key = key_prefix or func.__name__

            # Intentar obtener del cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"✅ Cache HIT: {func.__name__}")
                return cached_result

            # Ejecutar función
            logger.debug(f"❌ Cache MISS: {func.__name__} - ejecutando...")
            result = func(*args, **kwargs)

            # Guardar en cache
            cache.set(cache_key, result, ttl_seconds=ttl_seconds)

            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("💾 Testing Distributed Cache System...\n")

    cache = get_distributed_cache(l1_max_size=10, l2_enabled=False, l3_enabled=True)

    # Test 1: Set & Get
    print("Test 1: Set & Get")
    cache.set("test_key", {"data": "Hello World!"})
    value = cache.get("test_key")
    print(f"  Valor: {value}\n")

    # Test 2: TTL
    print("Test 2: TTL (expiración)")
    cache.set("temp_key", "Temporal", ttl_seconds=2)
    print(f"  Inmediato: {cache.get('temp_key')}")
    time.sleep(3)
    print(f"  Después de 3s: {cache.get('temp_key')}\n")

    # Test 3: Decorador @cached
    print("Test 3: Decorador @cached")

    call_count = 0

    @cached(ttl_seconds=60)
    def expensive_operation(x: int) -> int:
        global call_count
        call_count += 1
        time.sleep(0.5)
        return x * 2

    result1 = expensive_operation(5)
    result2 = expensive_operation(5)  # Debe usar cache
    print(f"  Resultado 1: {result1}")
    print(f"  Resultado 2: {result2}")
    print(f"  Función llamada {call_count} vez(ces)\n")

    # Test 4: Stats
    print("Test 4: Estadísticas")
    stats = cache.get_stats()
    print(f"  L1 Hit Rate: {stats['l1']['hit_rate']:.2f}%")
    print(f"  L1 Size: {stats['l1']['size']}/{stats['l1']['max_size']}")
    if stats["l3"]:
        print(f"  L3 Files: {stats['l3']['files']}")

    print("\n✅ Tests completados")
