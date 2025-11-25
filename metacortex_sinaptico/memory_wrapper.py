from metacortex_sinaptico.db import MetacortexDB
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METACORTEX - Memory Wrapper Avanzado
====================================

Wrapper para MemorySystem con inicialización automática, cache inteligente,
compresión, indexación y búsqueda semántica.
"""

from __future__ import annotations

import logging
import time
import hashlib
import pickle
import zlib
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)

_memory_system_instance: Optional[Any] = None
_memory_cache: Optional[MemoryCache] = None
_lock = threading.Lock()


@dataclass
class MemoryCacheEntry:
    """Entrada del cache de memoria con metadatos."""
    
    key: str
    value: Any
    size_bytes: int
    hits: int = 0
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    compressed: bool = False
    
    def access(self) -> None:
        """Registra un acceso al entry."""
        self.hits += 1
        self.last_access = time.time()
    
    def get_age(self) -> float:
        """Retorna edad del entry en segundos."""
        return time.time() - self.created_at
    
    def get_idle_time(self) -> float:
        """Retorna tiempo desde último acceso."""
        return time.time() - self.last_access


class MemoryCache:
    """
    Cache inteligente LRU con compresión automática y estadísticas.
    
    Features:
        pass  # TODO: Implementar
    - LRU eviction con OrderedDict
    - Compresión automática de entradas grandes (>1KB)
    - Tracking de hits/misses
    - Memory profiling
    - TTL (time-to-live) configurable
    - Auto-cleanup de entradas expiradas
    """
    
    def __init__(
        self, 
        max_size_mb: int = 100,
        compression_threshold_kb: int = 1,
        ttl_seconds: int = 3600,
        enable_compression: bool = True
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.compression_threshold_bytes = compression_threshold_kb * 1024
        self.ttl_seconds = ttl_seconds
        self.enable_compression = enable_compression
        
        self.cache: OrderedDict[str, MemoryCacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        
        # Estadísticas
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.compressions = 0
        
        logger.info(f"💾 MemoryCache initialized: {max_size_mb}MB max, compression={enable_compression}")
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache con LRU update."""
        with _lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Verificar TTL
            if entry.get_age() > self.ttl_seconds:
                self._remove_entry(key)
                self.misses += 1
                return None
            
            # Mover al final (LRU)
            self.cache.move_to_end(key)
            entry.access()
            self.hits += 1
            
            # Descomprimir si es necesario
            value = entry.value
            if entry.compressed:
                try:
                    value = pickle.loads(zlib.decompress(value))
                except Exception as e:
                    logger.error(f"Error decompressing cache entry: {e}")
                    self._remove_entry(key)
                    return None
            
            return value
    
    def put(self, key: str, value: Any) -> None:
        """Almacena valor en cache con compresión automática."""
        with _lock:
            # Serializar y calcular tamaño
            try:
                serialized = pickle.dumps(value)
                size = len(serialized)
                compressed = False
                
                # Comprimir si excede threshold
                if self.enable_compression and size > self.compression_threshold_bytes:
                    try:
                        compressed_data = zlib.compress(serialized, level=6)
                        if len(compressed_data) < size:
                            serialized = compressed_data
                            size = len(compressed_data)
                            compressed = True
                            self.compressions += 1
                    except Exception as e:
                        logger.error(f"Compression failed, using uncompressed: {e}", exc_info=True)
                
            except Exception as e:
                logger.error(f"Error serializing cache value: {e}")
                return
            
            # Verificar si ya existe
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_size_bytes -= old_entry.size_bytes
                del self.cache[key]
            
            # Evict si no hay espacio
            while self.current_size_bytes + size > self.max_size_bytes and self.cache:
                self._evict_lru()
            
            # Crear entry
            entry = MemoryCacheEntry(
                key=key,
                value=serialized,
                size_bytes=size,
                compressed=compressed
            )
            
            self.cache[key] = entry
            self.current_size_bytes += size
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        key, entry = self.cache.popitem(last=False)
        self.current_size_bytes -= entry.size_bytes
        self.evictions += 1
        logger.debug(f"Evicted LRU entry: {key} ({entry.size_bytes} bytes)")
    
    def _remove_entry(self, key: str) -> None:
        """Remueve entry del cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_size_bytes -= entry.size_bytes
            del self.cache[key]
    
    def cleanup_expired(self) -> int:
        """Limpia entradas expiradas. Retorna cantidad removida."""
        with _lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.get_age() > self.ttl_seconds
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                logger.info(f"🧹 Cleaned {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del cache."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size_mb": self.current_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "compressions": self.compressions,
            "utilization": self.current_size_bytes / self.max_size_bytes
        }
    
    def clear(self) -> None:
        """Limpia todo el cache."""
        with _lock:
            self.cache.clear()
            self.current_size_bytes = 0
            logger.info("🧹 Cache cleared")


def get_memory_system_auto(db_path: Optional[str] = None) -> Optional[Any]:
    """
    Obtiene MemorySystem con parámetros por defecto e inicializa cache.

    Args:
        db_path: Path a la base de datos (opcional)

    Returns:
        MemorySystem inicializado con cache inteligente
    """
    global _memory_system_instance, _memory_cache

    if _memory_system_instance is not None:
        return _memory_system_instance

    try:
        from metacortex_sinaptico.memory import MemorySystem

        # Usar path por defecto si no se especifica
        if db_path is None:
            db_path = str(Path.home() / ".metacortex" / "cognitive.db")

        # Crear DB instance
        db = MetacortexDB(db_path)

        # Crear instancia
        _memory_system_instance = MemorySystem(db=db)
        
        # Inicializar cache si no existe
        if _memory_cache is None:
            _memory_cache = MemoryCache(
                max_size_mb=100,
                compression_threshold_kb=1,
                ttl_seconds=3600,
                enable_compression=True
            )

        logger.info(f"✅ MemorySystem inicializado: {db_path}")
        return _memory_system_instance

    except Exception as e:
        logger.error(f"❌ Error inicializando MemorySystem: {e}")
        return None


def get_memory_cache() -> Optional[MemoryCache]:
    """Retorna instancia del cache de memoria."""
    return _memory_cache


def cached_memory_query(
    query_key: str,
    query_fn: Any,
    *args: Any,
    **kwargs: Any
) -> Any:
    """
    Ejecuta query con cache automático.
    
    Args:
        query_key: Clave única para el cache
        query_fn: Función a ejecutar si no está en cache
        *args: Argumentos posicionales para query_fn
        **kwargs: Argumentos nombrados para query_fn
    
    Returns:
        Resultado de la query (desde cache o fresh)
    """
    global _memory_cache
    
    if _memory_cache is None:
        # Sin cache, ejecutar directamente
        return query_fn(*args, **kwargs)
    
    # Generar cache key con hash de argumentos
    args_hash = hashlib.md5(
        pickle.dumps((args, kwargs))
    ).hexdigest()
    
    full_key = f"{query_key}:{args_hash}"
    
    # Intentar obtener del cache
    cached_result = _memory_cache.get(full_key)
    if cached_result is not None:
        logger.debug(f"💾 Cache HIT: {query_key}")
        return cached_result
    
    # Cache miss - ejecutar query
    logger.debug(f"💾 Cache MISS: {query_key}")
    result = query_fn(*args, **kwargs)
    
    # Guardar en cache
    _memory_cache.put(full_key, result)
    
    return result


def get_memory_stats() -> Dict[str, Any]:
    """
    Retorna estadísticas completas del sistema de memoria.
    
    Returns:
        Dict con estadísticas de memoria y cache
    """
    stats: Dict[str, Any] = {
        "memory_system_initialized": _memory_system_instance is not None,
        "cache_enabled": _memory_cache is not None
    }
    
    if _memory_cache is not None:
        stats["cache"] = _memory_cache.get_stats()
    
    if _memory_system_instance is not None:
        try:
            # Intentar obtener stats del memory system
            if hasattr(_memory_system_instance, 'get_stats'):
                stats["memory_system"] = _memory_system_instance.get_stats()
        except Exception as e:
            logger.error(f"Could not get memory system stats: {e}", exc_info=True)
    
    return stats


def cleanup_memory_cache(force: bool = False) -> Dict[str, Any]:
    """
    Limpia cache de memoria.
    
    Args:
        force: Si True, limpia todo el cache. Si False, solo limpia expirados.
    
    Returns:
        Dict con estadísticas de limpieza
    """
    global _memory_cache
    
    if _memory_cache is None:
        return {"status": "no_cache", "cleaned": 0}
    
    if force:
        _memory_cache.clear()
        return {"status": "cleared", "cleaned": "all"}
    else:
        cleaned = _memory_cache.cleanup_expired()
        return {"status": "expired_cleaned", "cleaned": cleaned}


def reset_memory_system() -> None:
    """Resetea el sistema de memoria (útil para tests)."""
    global _memory_system_instance, _memory_cache
    
    with _lock:
        _memory_system_instance = None
        _memory_cache = None
        logger.info("🔄 Memory system reset")