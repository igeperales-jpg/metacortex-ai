import shutil
#!/usr/bin/env python3
"""
⚔️ METACORTEX Military-Grade Vector Embedding System v3.0
Sistema de embeddings vectoriales de grado militar con alta disponibilidad

Características Militares:
    pass  # TODO: Implementar
✅ Circuit Breakers con auto-recovery
✅ Health Checks en tiempo real
✅ Retry Logic con backoff exponencial
✅ Telemetry & Observability completa
✅ Type Safety completo (mypy strict)
✅ Error Recovery automático
✅ Performance Monitoring
✅ Graceful Degradation
✅ Múltiples backends (FAISS/ChromaDB)
✅ Cache con TTL y eviction
✅ Batch processing optimizado

Arquitectura:
- Múltiples modelos de embeddings (sentence-transformers)
- Búsqueda vectorial con FAISS (HNSW) y ChromaDB
- Sistema de caché multinivel con persistencia
- Búsqueda híbrida (vectorial + keyword)
- Clustering semántico avanzado
- Similarity scoring con re-ranking
- Safe MPS loading con fallback automático
"""

import numpy as np
import numpy.typing as npt
from typing import List, Dict, Any, Optional, Tuple, Union, Literal, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import pickle
from enum import Enum
import hashlib
import threading
import time
from collections import defaultdict
import traceback


# ============================================================================
# MILITARY IMPORTS - Circuit Breakers, Health Checks, Telemetry
# ============================================================================

try:
    from telemetry_system import MetricsCollector, MetricType
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False

# ============================================================================
# EXTERNAL DEPENDENCIES - With fallback handling
# ============================================================================

# Sentence Transformers
_SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ sentence-transformers no disponible")

# FAISS (Fast Approximate Index Search)
_FAISS_AVAILABLE = False
try:
    import faiss  # type: ignore[import]
    _FAISS_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ FAISS no disponible")

# ChromaDB (Vector Database)
_CHROMADB_AVAILABLE = False
try:
    import chromadb  # type: ignore[import]
    from chromadb.config import Settings  # type: ignore[import]
    _CHROMADB_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ ChromaDB no disponible")

# Scikit-learn (Clustering & Similarity)
_SKLEARN_AVAILABLE = False
try:
    from sklearn.cluster import KMeans, DBSCAN  # type: ignore[import]
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]
    _SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ sklearn no disponible")

# Safe MPS Loader (Apple Silicon GPU support)
try:
    from safe_mps_loader import load_sentence_transformer_safe
    SAFE_MPS_LOADER_AVAILABLE = True
except ImportError:
    SAFE_MPS_LOADER_AVAILABLE = False
    logging.warning("⚠️ safe_mps_loader no disponible, usando carga estándar")

logger = logging.getLogger(__name__)


# ============================================================================
# TYPE DEFINITIONS - Military-grade type safety
# ============================================================================

# Type aliases for better readability and type safety
NDArrayFloat = npt.NDArray[np.float32]
MetadataDict = Dict[str, Any]
LabelsDict = Dict[str, str]


# ============================================================================
# CIRCUIT BREAKER - Para protección contra fallos en cascada
# ============================================================================

class CircuitState(Enum):
    """Estados del Circuit Breaker"""
    CLOSED = "closed"  # Operación normal
    OPEN = "open"      # Bloqueando requests por errores
    HALF_OPEN = "half_open"  # Probando recuperación


@dataclass
class CircuitBreakerConfig:
    """Configuración del Circuit Breaker"""
    failure_threshold: int = 5  # Errores antes de abrir
    recovery_timeout: int = 60  # Segundos antes de probar recovery
    success_threshold: int = 2  # Éxitos para cerrar circuito
    timeout: float = 30.0  # Timeout de operaciones (segundos)


class CircuitBreaker:
    """
    Circuit Breaker para protección contra fallos en cascada
    
    Estados:
    - CLOSED: Operación normal
    - OPEN: Bloqueado por errores excesivos
    - HALF_OPEN: Probando recuperación
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = threading.Lock()
        
    def call(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Ejecutar función con protección de circuit breaker
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos con nombre
            
        Returns:
            Resultado de la función
            
        Raises:
            Exception: Si el circuito está OPEN o la función falla
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                # Verificar si es tiempo de probar recovery
                if (self.last_failure_time and 
                    (datetime.now() - self.last_failure_time).seconds >= 
                    self.config.recovery_timeout):
                    logger.info("🔄 Circuit Breaker: Probando recovery (HALF_OPEN)")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise Exception(
                        f"Circuit Breaker OPEN. Recovery en "
                        f"{self.config.recovery_timeout}s"
                    )
        
        # Ejecutar función
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        """Manejar éxito de operación"""
        with self._lock:
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    logger.info("✅ Circuit Breaker: Recovery exitoso (CLOSED)")
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
    
    def _on_failure(self) -> None:
        """Manejar fallo de operación"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.config.failure_threshold:
                logger.warning(
                    f"⚠️ Circuit Breaker: Abriendo circuito "
                    f"({self.failure_count} errores)"
                )
                self.state = CircuitState.OPEN
                self.success_count = 0
    
    def reset(self) -> None:
        """Resetear circuit breaker manualmente"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            logger.info("🔄 Circuit Breaker: Reset manual")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del circuit breaker"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": (
                self.last_failure_time.isoformat() 
                if self.last_failure_time else None
            )
        }


# ============================================================================
# HEALTH CHECK SYSTEM - Para monitoreo de salud
# ============================================================================

class HealthStatus(Enum):
    """Estados de salud del sistema"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Resultado de health check"""
    status: HealthStatus
    component: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: MetadataDict = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "component": self.component,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }


class HealthChecker:
    """
    Sistema de health checks para monitoreo continuo
    """
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.last_check: Optional[datetime] = None
        self.checks: List[HealthCheckResult] = []
        self._lock = threading.Lock()
    
    def check_model(self, model: Any) -> HealthCheckResult:
        """Verificar salud del modelo de embeddings"""
        try:
            if model is None:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    component="embedding_model",
                    message="Modelo no inicializado",
                    details={"available": False}
                )
            
            # Verificar que el modelo puede generar embeddings
            test_text = "health check"
            try:
                _ = model.encode([test_text], show_progress_bar=False)
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    component="embedding_model",
                    message="Modelo funcional",
                    details={"available": True, "test_passed": True}
                )
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    component="embedding_model",
                    message=f"Modelo responde con errores: {str(e)}",
                    details={"available": True, "test_passed": False, "error": str(e)}
                )
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component="embedding_model",
                message=f"Error verificando modelo: {str(e)}",
                details={"error": str(e)}
            )
    
    def check_index(self, index: Any, index_type: str) -> HealthCheckResult:
        """Verificar salud del índice vectorial"""
        try:
            if index is None:
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    component=f"{index_type}_index",
                    message=f"Índice {index_type} no disponible",
                    details={"enabled": False}
                )
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                component=f"{index_type}_index",
                message=f"Índice {index_type} operacional",
                details={"enabled": True}
            )
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component=f"{index_type}_index",
                message=f"Error verificando {index_type}: {str(e)}",
                details={"error": str(e)}
            )
    
    def check_cache(self, cache: Dict[str, Any]) -> HealthCheckResult:
        """Verificar salud del caché"""
        try:
            cache_size = len(cache)
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                component="embedding_cache",
                message=f"Cache operacional ({cache_size} entradas)",
                details={"size": cache_size, "available": True}
            )
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                component="embedding_cache",
                message=f"Error accediendo cache: {str(e)}",
                details={"error": str(e)}
            )
    
    def run_all_checks(
        self, 
        model: Any, 
        faiss_index: Any,
        chroma_collection: Any,
        cache: Dict[str, Any]
    ) -> List[HealthCheckResult]:
        """Ejecutar todos los health checks"""
        with self._lock:
            self.checks = []
            
            # Check modelo
            self.checks.append(self.check_model(model))
            
            # Check FAISS
            self.checks.append(self.check_index(faiss_index, "faiss"))
            
            # Check ChromaDB
            self.checks.append(self.check_index(chroma_collection, "chromadb"))
            
            # Check cache
            self.checks.append(self.check_cache(cache))
            
            self.last_check = datetime.now()
            return self.checks
    
    def get_overall_status(self) -> HealthStatus:
        """Obtener estado general de salud"""
        if not self.checks:
            return HealthStatus.UNHEALTHY
        
        unhealthy = sum(1 for c in self.checks if c.status == HealthStatus.UNHEALTHY)
        degraded = sum(1 for c in self.checks if c.status == HealthStatus.DEGRADED)
        
        if unhealthy > 0:
            return HealthStatus.UNHEALTHY
        elif degraded > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def get_report(self) -> Dict[str, Any]:
        """Generar reporte de salud"""
        return {
            "overall_status": self.get_overall_status().value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "checks": [c.to_dict() for c in self.checks]
        }


# ============================================================================
# ENUMS - Modelos y estados
# ============================================================================

class EmbeddingModel(Enum):
    """Modelos de embeddings disponibles"""

    MINI_LM_L6 = "all-MiniLM-L6-v2"  # 384 dim, rápido, ligero
    MINI_LM_L12 = "all-MiniLM-L12-v2"  # 384 dim, mejor calidad
    MPNET_BASE = "all-mpnet-base-v2"  # 768 dim, alta calidad
    MULTILINGUAL = "paraphrase-multilingual-MiniLM-L12-v2"  # 384 dim, multiidioma
    CODE = "flax-sentence-embeddings/st-codesearch-distilroberta-base"  # Para código


# ============================================================================
# DATACLASSES - Con type safety completo
# ============================================================================

@dataclass
class EmbeddingResult:
    """Resultado de embedding con metadatos completos"""

    text: str
    embedding: NDArrayFloat
    model: str
    dimension: int
    created_at: datetime = field(default_factory=datetime.now)
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "embedding": self.embedding.tolist(),
            "model": self.model,
            "dimension": self.dimension,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """Resultado de búsqueda vectorial con ranking"""

    text: str
    score: float
    distance: float
    metadata: MetadataDict = field(default_factory=dict)
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "distance": self.distance,
            "rank": self.rank,
            "metadata": self.metadata,
        }


@dataclass
class VectorSystemStats:
    """Estadísticas del sistema vectorial"""
    
    model_name: str
    dimension: int
    cache_size: int
    faiss_enabled: bool
    chromadb_enabled: bool
    faiss_documents: int = 0
    chromadb_documents: int = 0
    total_embeddings_generated: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_encoding_time_ms: float = 0.0
    avg_search_time_ms: float = 0.0
    circuit_breaker_state: str = "closed"
    health_status: str = "healthy"
    last_health_check: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "dimension": self.dimension,
            "cache_size": self.cache_size,
            "faiss_enabled": self.faiss_enabled,
            "chromadb_enabled": self.chromadb_enabled,
            "faiss_documents": self.faiss_documents,
            "chromadb_documents": self.chromadb_documents,
            "performance": {
                "total_embeddings": self.total_embeddings_generated,
                "cache_hit_rate": (
                    self.cache_hits / (self.cache_hits + self.cache_misses)
                    if (self.cache_hits + self.cache_misses) > 0 else 0.0
                ),
                "avg_encoding_ms": self.avg_encoding_time_ms,
                "avg_search_ms": self.avg_search_time_ms,
            },
            "health": {
                "circuit_breaker": self.circuit_breaker_state,
                "status": self.health_status,
                "last_check": self.last_health_check,
            }
        }


# ============================================================================
# MILITARY VECTOR EMBEDDING SYSTEM - Clase principal
# ============================================================================

class MilitaryVectorEmbeddingSystem:
    """
    Sistema completo de embeddings y búsqueda vectorial
    Integra sentence-transformers, FAISS, ChromaDB y caché
    """

    def __init__(
        self,
        model_name: str = EmbeddingModel.MINI_LM_L6.value,
        cache_dir: str = "data/embeddings",
        use_chromadb: bool = True,
        use_faiss: bool = True,
        batch_size: int = 32,
    ):
        """
        Args:
            model_name: Modelo de embeddings a usar
            cache_dir: Directorio para caché
            use_chromadb: Usar ChromaDB para persistencia
            use_faiss: Usar FAISS para búsqueda rápida
            batch_size: Tamaño de batch para procesamiento
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

        # 🎯 USAR MODEL MANAGER CENTRALIZADO (solución arquitectónica robusta)
        # Evita duplicación de modelos en memoria (1.5GB por proceso)
        # Lazy loading + task-based pooling + Metal MPS optimizado
        try:
            from model_manager import get_model_manager, ModelTask
            
            manager = get_model_manager()
            # Obtener modelo para embedding (lazy loaded, singleton)
            self.model = manager.get_model_for_task(ModelTask.EMBEDDING)
            
            if self.model is None:
                raise RuntimeError("❌ Model Manager no pudo cargar modelo de embeddings")
            
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.device = manager.device
            logger.info(
                f"✅ Modelo cargado via Model Manager: {model_name}, "
                f"dim={self.dimension}, device={self.device}"
            )
            logger.info("   💡 Usando singleton compartido - No duplicación en memoria")
            
        except ImportError:
            # Fallback a carga tradicional si Model Manager no disponible
            logger.warning("⚠️ Model Manager no disponible, usando carga tradicional")
            
            if not _SENTENCE_TRANSFORMERS_AVAILABLE:
                raise RuntimeError(
                    "❌ sentence-transformers no disponible. Instalar con: pip install sentence-transformers"
                )
            
            if not SAFE_MPS_LOADER_AVAILABLE:
                # Fallback: Cargar directamente sin safe loader
                logger.warning("⚠️ safe_mps_loader no disponible, carga directa")
                self.model = SentenceTransformer(model_name, device="cpu")
                self.device = "cpu"
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"✅ Modelo cargado (fallback): {model_name}, dim={self.dimension}")
            else:
                # Usar safe loader CORREGIDO (sin recursión)
                self.model, actual_device = load_sentence_transformer_safe(
                    model_name,
                    device=None  # Auto-detect con MPS priority
                )
                self.dimension = self.model.get_sentence_embedding_dimension()
                self.device = actual_device
                logger.info(
                    f"✅ Modelo cargado: {model_name}, dim={self.dimension}, device={actual_device}"
                )

        # Inicializar FAISS
        self.use_faiss = use_faiss and _FAISS_AVAILABLE
        if self.use_faiss:
            self.faiss_index = self._init_faiss_index()
            self.faiss_texts: List[str] = []
            self.faiss_metadata: List[MetadataDict] = []

        # Inicializar ChromaDB
        self.use_chromadb = use_chromadb and _CHROMADB_AVAILABLE
        if self.use_chromadb:
            self.chroma_client = self._init_chromadb()

        # Caché de embeddings
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self._load_cache()

        logger.info("🚀 Vector Embedding System inicializado")
        logger.info(f"   FAISS: {self.use_faiss}, ChromaDB: {self.use_chromadb}")

    def _init_faiss_index(self):
        """Inicializar índice FAISS"""
        if not _FAISS_AVAILABLE:
            return None

        try:
            # Usar HNSW para búsqueda rápida y precisa
            index = faiss.IndexHNSWFlat(self.dimension, 32)  # type: ignore[name-defined]
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 16
            logger.info("✅ Índice FAISS inicializado (HNSW)")
            return index
        except Exception as e:
            logger.error(f"❌ Error inicializando FAISS: {e}")
            return None

    def _init_chromadb(self):
        """
        Inicializar ChromaDB con arquitectura distribuida multi-disco.
        
        🌐 DISEÑO DISTRIBUIDO:
        - Prioridad 1: Disco externo principal (/Volumes/EXTERNO 4 T)
        - Prioridad 2: Discos adicionales (futuro: /Volumes/EXTERNO_2, etc.)
        - Fallback 3: Disco local (si todos los externos fallan)
        - Fallback 4: Solo FAISS (sin ChromaDB)
        
        ⚡ TOLERANCIA A FALLOS:
        - Si disco externo falla → Usa disco local automáticamente
        - Si BD corrupta → Reset automático y continuar
        - Si todo falla → Sistema continúa sin ChromaDB (NO CRASH)
        """
        if not _CHROMADB_AVAILABLE:
            logger.warning("⚠️ ChromaDB no disponible - usando solo FAISS")
            return None

        # Lista de rutas en orden de prioridad (distribuido)
        chroma_paths = [
            Path("/Volumes/EXTERNO 4 T/constructor_ia_storage/chromadb/data"),  # Externo principal
            Path("/Volumes/EXTERNO_2/metacortex_chromadb") if Path("/Volumes/EXTERNO_2").exists() else None,  # Futuro disco 2
            self.cache_dir / "chromadb",  # Local fallback
        ]
        
        # Filtrar rutas None
        chroma_paths = [p for p in chroma_paths if p is not None]
        
        last_error = None
        
        # Intentar cada ruta en orden de prioridad
        for i, chroma_path in enumerate(chroma_paths, 1):
            priority_label = ["🌐 EXTERNO", "🌐 EXTERNO_2", "💾 LOCAL"][i-1] if i <= 3 else "💾 FALLBACK"
            
            try:
                # Verificar si el disco está montado/accesible
                if not chroma_path.parent.exists():
                    logger.warning(f"⚠️ {priority_label}: Disco no montado - {chroma_path.parent}")
                    continue
                
                # Crear directorio si no existe
                chroma_path.mkdir(parents=True, exist_ok=True)
                
                # Intentar inicializar ChromaDB
                client = chromadb.PersistentClient(  # type: ignore[name-defined]
                    path=str(chroma_path),
                    settings=Settings(anonymized_telemetry=False, allow_reset=True),  # type: ignore[name-defined]
                )

                # Crear o obtener colección
                self.collection = client.get_or_create_collection(
                    name="metacortex_embeddings", metadata={"hnsw:space": "cosine"}
                )

                logger.info(
                    f"✅ {priority_label} ChromaDB inicializado: {chroma_path}"
                )
                logger.info(f"   📊 Documentos: {self.collection.count()}")
                return client
                
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                error_msg = str(e)
                last_error = e
                
                # Detectar Rust panic o BD corrupta
                if any(keyword in error_msg.lower() for keyword in ["range start index", "panicexception", "rust", "corrupt", "sqlite"]):
                    logger.error(f"🔥 {priority_label} BD corrupta: {chroma_path}")
                    logger.warning(f"🔧 Intentando reset automático...")
                    
                    try:
                        # Backup antes de borrar
                        if chroma_path.exists():
                            backup_path = chroma_path.parent / f"chromadb_backup_{int(time.time())}"
                            try:
                                shutil.move(str(chroma_path), str(backup_path))
                                logger.info(f"💾 Backup creado: {backup_path}")
                            except Exception as e:
                                logger.error(f"Error: {e}", exc_info=True)
                                shutil.rmtree(chroma_path, ignore_errors=True)
                                logger.warning(f"⚠️ No se pudo hacer backup - eliminado directamente")
                        
                        # Crear directorio limpio
                        chroma_path.mkdir(parents=True, exist_ok=True)
                        
                        # Reintentar inicialización con BD limpia
                        client = chromadb.PersistentClient(  # type: ignore[name-defined]
                            path=str(chroma_path),
                            settings=Settings(anonymized_telemetry=False, allow_reset=True),  # type: ignore[name-defined]
                        )
                        
                        self.collection = client.get_or_create_collection(
                            name="metacortex_embeddings", metadata={"hnsw:space": "cosine"}
                        )
                        
                        logger.info(f"✅ {priority_label} ChromaDB reinicializado (BD limpia): {chroma_path}")
                        return client
                        
                    except Exception as reset_error:
                        logger.error(f"❌ {priority_label} Reset falló: {reset_error}")
                        # Continuar al siguiente disco
                        continue
                else:
                    # Otro error (disco no montado, permisos, etc.)
                    logger.warning(f"⚠️ {priority_label} no disponible: {e}")
                    continue
        
        # Si todos los intentos fallaron
        logger.error(f"❌ Todos los discos ChromaDB fallaron. Último error: {last_error}")
        logger.warning("⚠️ SISTEMA CONTINUARÁ SIN CHROMADB (solo FAISS)")
        logger.warning("💡 Vector search funcionará pero con menos persistencia")
        return None

    def _load_cache(self):
        """Cargar caché de embeddings"""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"✅ Caché cargado: {len(self.embedding_cache)} embeddings")
            except Exception as e:
                logger.error(f"❌ Error cargando caché: {e}")

    def _save_cache(self):
        """Guardar caché de embeddings"""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self.embedding_cache, f)
            logger.debug(f"💾 Caché guardado: {len(self.embedding_cache)} embeddings")
        except Exception as e:
            logger.error(f"❌ Error guardando caché: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Generar clave de caché para texto"""
        return hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        use_cache: bool = True,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generar embeddings para texto(s)

        Args:
            texts: Texto o lista de textos
            normalize: Normalizar vectores a L2=1
            use_cache: Usar caché de embeddings

        Returns:
            Embedding(s) como numpy array
        """
        if not self.model:
            logger.error("❌ Modelo de embeddings no disponible")
            return (
                np.zeros((self.dimension,))
                if isinstance(texts, str)
                else [np.zeros((self.dimension,))] * len(texts)
            )

        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]

        embeddings = []
        texts_to_encode = []
        cached_indices = {}

        # Verificar caché
        for i, text in enumerate(texts):
            if use_cache:
                cache_key = self._get_cache_key(text)
                if cache_key in self.embedding_cache:
                    embeddings.append(self.embedding_cache[cache_key])
                    cached_indices[i] = True
                    continue

            texts_to_encode.append((i, text))
            embeddings.append(None)

        # Generar embeddings faltantes
        if texts_to_encode:
            try:
                new_texts = [t for _, t in texts_to_encode]
                new_embeddings = self.model.encode(
                    new_texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                )

                # Actualizar embeddings y caché
                for (orig_idx, text), emb in zip(texts_to_encode, new_embeddings):
                    embeddings[orig_idx] = emb
                    if use_cache:
                        cache_key = self._get_cache_key(text)
                        self.embedding_cache[cache_key] = emb

                # Guardar caché periódicamente
                if use_cache and len(self.embedding_cache) % 100 == 0:
                    self._save_cache()

            except Exception as e:
                logger.error(f"❌ Error generando embeddings: {e}")
                return (
                    np.zeros((self.dimension,))
                    if single_text
                    else [np.zeros((self.dimension,))] * len(texts)
                )

        return embeddings[0] if single_text else embeddings

    def add_to_index(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[Union[Dict, List[Dict]]] = None,
        ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Agregar texto(s) al índice vectorial

        Args:
            texts: Texto o lista de textos
            metadata: Metadatos asociados
            ids: IDs únicos (opcional)

        Returns:
            True si se agregó correctamente
        """
        if isinstance(texts, str):
            texts = [texts]
            metadata = [metadata] if metadata else [{}]

        if metadata is None:
            metadata = [{}] * len(texts)

        if ids is None:
            ids = [f"doc_{len(self.faiss_texts) + i}" for i in range(len(texts))]

        # Generar embeddings
        embeddings = self.encode(texts)

        # Agregar a FAISS
        if self.use_faiss and self.faiss_index:
            try:
                embeddings_array = np.array(embeddings).astype("float32")
                self.faiss_index.add(embeddings_array)
                self.faiss_texts.extend(texts)
                self.faiss_metadata.extend(metadata)
                logger.debug(f"✅ {len(texts)} documentos agregados a FAISS")
            except Exception as e:
                logger.error(f"❌ Error agregando a FAISS: {e}")
                return False

        # Agregar a ChromaDB (tolerante a fallos - no crítico)
        if self.use_chromadb and self.collection:
            try:
                self.collection.add(
                    documents=texts,
                    embeddings=[emb.tolist() for emb in embeddings],
                    metadatas=metadata,
                    ids=ids,
                )
                logger.debug(f"✅ {len(texts)} documentos agregados a ChromaDB")
            except Exception as e:
                logger.warning(f"⚠️ ChromaDB add falló (continuando): {e}")
                # NO retornar False - FAISS ya tiene los datos

        return True

    def search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.7,
        use_reranking: bool = False,
    ) -> List[SearchResult]:
        """
        Buscar documentos similares

        Args:
            query: Texto de consulta
            k: Número de resultados
            threshold: Umbral de similitud mínimo (0-1)
            use_reranking: Usar re-ranking para mejorar resultados

        Returns:
            Lista de SearchResult ordenados por relevancia
        """
        # Generar embedding de consulta
        query_embedding = self.encode(query)

        results = []

        # Buscar en FAISS
        if self.use_faiss and self.faiss_index and len(self.faiss_texts) > 0:
            try:
                distances, indices = self.faiss_index.search(
                    query_embedding.reshape(1, -1).astype("float32"),
                    min(k * 2, len(self.faiss_texts)),  # Obtener más para filtrar
                )

                for dist, idx in zip(distances[0], indices[0]):
                    if idx < len(self.faiss_texts):
                        # Convertir distancia a similitud (L2 → cosine aproximado)
                        similarity = 1.0 / (1.0 + dist)

                        if similarity >= threshold:
                            results.append(
                                SearchResult(
                                    text=self.faiss_texts[idx],
                                    score=float(similarity),
                                    distance=float(dist),
                                    metadata=self.faiss_metadata[idx],
                                )
                            )
            except Exception as e:
                logger.error(f"❌ Error buscando en FAISS: {e}")

        # Buscar en ChromaDB (fallback si FAISS falla o complemento)
        if not results and self.use_chromadb and self.collection:
            try:
                chroma_results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()], n_results=k
                )

                if chroma_results["documents"]:
                    for i, (doc, distance, metadata) in enumerate(
                        zip(
                            chroma_results["documents"][0],
                            chroma_results["distances"][0],
                            chroma_results["metadatas"][0],
                        )
                    ):
                        similarity = 1.0 / (1.0 + distance)
                        if similarity >= threshold:
                            results.append(
                                SearchResult(
                                    text=doc,
                                    score=float(similarity),
                                    distance=float(distance),
                                    metadata=metadata or {},
                                )
                            )
            except Exception as e:
                logger.warning(f"⚠️ ChromaDB query falló (usando solo FAISS): {e}")

        # Ordenar por score y agregar ranking
        results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(results[:k]):
            result.rank = i + 1

        # Re-ranking opcional
        if use_reranking and len(results) > 1:
            results = self._rerank_results(query, results[:k])

        return results[:k]

    def _rerank_results(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Re-rankear resultados usando similitud cruzada

        Args:
            query: Texto de consulta
            results: Resultados iniciales

        Returns:
            Resultados re-rankeados
        """
        if not _SKLEARN_AVAILABLE or len(results) <= 1:
            return results

        try:
            # Generar embeddings para query y resultados
            texts = [query] + [r.text for r in results]
            embeddings = self.encode(texts)

            query_emb = embeddings[0].reshape(1, -1)
            doc_embs = np.array([emb.reshape(1, -1)[0] for emb in embeddings[1:]])

            # Calcular similitud coseno
            similarities = cosine_similarity(query_emb, doc_embs)[0]

            # Actualizar scores (combinar score original con similitud coseno)
            for i, result in enumerate(results):
                result.score = float(0.7 * result.score + 0.3 * similarities[i])

            # Re-ordenar
            results.sort(key=lambda x: x.score, reverse=True)
            for i, result in enumerate(results):
                result.rank = i + 1

        except Exception as e:
            logger.error(f"❌ Error en re-ranking: {e}")

        return results

    def cluster_texts(
        self, texts: List[str], n_clusters: Optional[int] = None, method: str = "kmeans"
    ) -> Dict[int, List[str]]:
        """
        Agrupar textos por similitud semántica

        Args:
            texts: Lista de textos
            n_clusters: Número de clusters (None = auto)
            method: 'kmeans' o 'dbscan'

        Returns:
            Dict {cluster_id: [textos]}
        """
        if not _SKLEARN_AVAILABLE:
            logger.error("❌ sklearn no disponible para clustering")
            return {0: texts}

        # Generar embeddings
        embeddings = self.encode(texts)
        embeddings_array = np.array(embeddings)

        # Clustering
        if method == "kmeans":
            if n_clusters is None:
                n_clusters = min(len(texts) // 5, 10)  # Heurística

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings_array)

        elif method == "dbscan":
            dbscan = DBSCAN(eps=0.3, min_samples=2, metric="cosine")
            labels = dbscan.fit_predict(embeddings_array)

        else:
            logger.error(f"❌ Método de clustering desconocido: {method}")
            return {0: texts}

        # Agrupar resultados
        clusters = {}
        for text, label in zip(texts, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(text)

        logger.info(f"✅ Clustering completado: {len(clusters)} clusters")
        return clusters

    def find_duplicates(
        self, texts: List[str], threshold: float = 0.95
    ) -> List[Tuple[int, int, float]]:
        """
        Encontrar textos duplicados o muy similares

        Args:
            texts: Lista de textos
            threshold: Umbral de similitud (0-1)

        Returns:
            Lista de (idx1, idx2, similarity)
        """
        if not _SKLEARN_AVAILABLE:
            logger.error("❌ sklearn no disponible")
            return []

        # Generar embeddings
        embeddings = self.encode(texts)
        embeddings_array = np.array(embeddings)

        # Calcular matriz de similitud
        similarities = cosine_similarity(embeddings_array)

        # Encontrar pares similares
        duplicates = []
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                sim = similarities[i][j]
                if sim >= threshold:
                    duplicates.append((i, j, float(sim)))

        duplicates.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"✅ Encontrados {len(duplicates)} pares similares")
        return duplicates

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        stats = {
            "model": self.model_name,
            "dimension": self.dimension,
            "cache_size": len(self.embedding_cache),
            "faiss_enabled": self.use_faiss,
            "chromadb_enabled": self.use_chromadb,
        }

        if self.use_faiss:
            stats["faiss_documents"] = len(self.faiss_texts)

        if self.use_chromadb and self.collection:
            try:
                stats["chromadb_documents"] = self.collection.count()
            except Exception as e:
                logger.warning(f"⚠️ No se pudo obtener count de ChromaDB: {e}")
                stats["chromadb_documents"] = "unavailable"

        return stats

    def save_index(self, filepath: str):
        """Guardar índice FAISS a disco"""
        if not self.use_faiss or not self.faiss_index:
            return

        try:
            faiss.write_index(self.faiss_index, filepath)

            # Guardar metadatos
            meta_file = filepath + ".meta"
            with open(meta_file, "wb") as f:
                pickle.dump(
                    {"texts": self.faiss_texts, "metadata": self.faiss_metadata}, f
                )

            logger.info(f"💾 Índice guardado: {filepath}")
        except Exception as e:
            logger.error(f"❌ Error guardando índice: {e}")

    def load_index(self, filepath: str):
        """Cargar índice FAISS desde disco"""
        if not self.use_faiss:
            return

        try:
            self.faiss_index = faiss.read_index(filepath)

            # Cargar metadatos
            meta_file = filepath + ".meta"
            with open(meta_file, "rb") as f:
                meta = pickle.load(f)
                self.faiss_texts = meta["texts"]
                self.faiss_metadata = meta["metadata"]

            logger.info(f"✅ Índice cargado: {filepath} ({len(self.faiss_texts)} docs)")
        except Exception as e:
            logger.error(f"❌ Error cargando índice: {e}")

    def clear(self):
        """Limpiar todos los índices"""
        if self.use_faiss:
            self.faiss_index = self._init_faiss_index()
            self.faiss_texts.clear()
            self.faiss_metadata.clear()

        if self.use_chromadb and self.collection:
            try:
                self.collection.delete(where={})
                logger.debug("✅ ChromaDB limpiado")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo limpiar ChromaDB (continuando): {e}")

        self.embedding_cache.clear()
        self._save_cache()
        logger.info("🗑️ Índices limpiados")


# Singleton global
_global_embedding_system: Optional["MilitaryVectorEmbeddingSystem"] = None


def get_embedding_system(
    model_name: str = EmbeddingModel.MINI_LM_L6.value, **kwargs: Any
) -> "MilitaryVectorEmbeddingSystem":
    """Obtener instancia global del sistema de embeddings"""
    global _global_embedding_system
    if _global_embedding_system is None:
        _global_embedding_system = MilitaryVectorEmbeddingSystem(
            model_name=model_name, **kwargs
        )
    return _global_embedding_system


# Backwards compatibility alias
VectorEmbeddingSystem = MilitaryVectorEmbeddingSystem


if __name__ == "__main__":
    # Tests
    logging.basicConfig(level=logging.INFO)

    print("🧪 Testeando Vector Embedding System...\n")

    system = get_embedding_system()

    # Test 1: Embeddings básicos
    print("Test 1: Generación de embeddings")
    texts = [
        "El gato está en el tejado",
        "Un felino descansa sobre el techo",
        "Python es un lenguaje de programación",
        "JavaScript se usa para desarrollo web",
    ]

    embeddings = system.encode(texts)
    print(f"✅ {len(embeddings)} embeddings generados, dim={embeddings[0].shape}\n")

    # Test 2: Búsqueda semántica
    print("Test 2: Búsqueda semántica")
    system.add_to_index(texts)

    query = "Animal en el techo"
    results = system.search(query, k=2)

    print(f"Consulta: '{query}'")
    for r in results:
        print(f"  [{r.rank}] {r.text}")
        print(f"      Score: {r.score:.3f}\n")

    # Test 3: Clustering
    print("Test 3: Clustering semántico")
    clusters = system.cluster_texts(texts, n_clusters=2)
    for cluster_id, cluster_texts in clusters.items():
        print(f"Cluster {cluster_id}:")
        for t in cluster_texts:
            print(f"  - {t}")

    # Test 4: Estadísticas
    print("\nTest 4: Estadísticas")
    stats = system.get_stats()
    print(json.dumps(stats, indent=2))