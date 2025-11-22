# -*- coding: utf-8 -*-
"""
🌐 METACORTEX Internet Search v2.0
Compatible con v1.0 - Características v2.0 opcionales
=========================================================

Sistema avanzado de búsqueda integrado con la consciencia cognitiva de METACORTEX.

Características v2.0:
    pass  # TODO: Implementar
- Circuit breakers por fuente (Wikipedia, ArXiv, Google)
- Fetching asíncrono con aiohttp
- Rate limiting avanzado (requests/minuto por fuente)
- Caché de resultados con TTL y búsqueda semántica
- Métricas de rendimiento detalladas
- Fallback a modo síncrono v1.0

METACORTEX Project - 2025
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

try:
    import requests
except ImportError:
    requests = None  # type: ignore

# 🆕 v2.0: Imports condicionales
try:
    import aiohttp
    AIOHTTP_AVAILABLE: bool = True
except ImportError:
    aiohttp = None  # type: ignore
    AIOHTTP_AVAILABLE: bool = False

try:
    from advanced_cache_system import get_global_cache
    CACHE_V2_AVAILABLE: bool = True
except ImportError:
    get_global_cache = None  # type: ignore
    CACHE_V2_AVAILABLE: bool = False

try:
    from vector_embedding_system import get_embedding_system
    EMBEDDINGS_AVAILABLE: bool = True
except ImportError:
    get_embedding_system = None  # type: ignore
    EMBEDDINGS_AVAILABLE: bool = False

# === IMPORTACIONES METACORTEX ===
try:
    import wikipedia
    wikipedia.set_lang("es")  # type: ignore
    WIKIPEDIA_AVAILABLE: bool = True
except ImportError:
    wikipedia = None  # type: ignore
    WIKIPEDIA_AVAILABLE: bool = False
    print("⚠️ Wikipedia no disponible. Instala: pip install wikipedia")

try:
    import arxiv
    ARXIV_AVAILABLE: bool = True
except ImportError:
    arxiv = None  # type: ignore
    ARXIV_AVAILABLE: bool = False
    print("⚠️ ArXiv no disponible. Instala: pip install arxiv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/metacortex_search.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# === CONFIGURACIÓN GLOBAL ===
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
]


class SecureConnectionManager:
    """
    Gestor de conexiones web seguras con rotación de User-Agents y control de velocidad
    """

    def __init__(self):
        if requests is None:
            raise ImportError("requests library is required but not installed")
        self.session = requests.Session()  # type: ignore
        self.request_count = 0
        self.last_request_time = 0.0

    def make_request(self, url: str, timeout: int = 30) -> Any:
        """Realiza una petición web segura con control de velocidad"""
        if requests is None:
            raise ImportError("requests library is required")
            
        # Control de velocidad - mínimo 2 segundos entre peticiones
        current_time = time.time()
        if current_time - self.last_request_time < 2:
            time.sleep(2 - (current_time - self.last_request_time))

        # Rotar User-Agent
        self.session.headers.update(  # type: ignore
            {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )

        self.request_count += 1
        self.last_request_time = time.time()

        try:
            response = self.session.get(url, timeout=timeout, allow_redirects=True)  # type: ignore
            response.raise_for_status()  # type: ignore
            logger.debug(f"✅ Conexión exitosa a {url[:50]}...")
            return response
        except Exception as e:
            logger.warning(f"❌ Error en conexión a {url}: {e}")
            raise


class MetacortexAdvancedSearch:
    """
    Sistema de búsqueda inteligente avanzado de METACORTEX v2.0
    """

    def __init__(
        self,
        cognitive_agent: Optional[Any] = None,
        enable_cache: bool = True,
        enable_async: bool = True,
        rate_limit_per_minute: int = 30,
    ):
        """
        Inicializar sistema de búsqueda

        Args:
            cognitive_agent: Agente cognitivo METACORTEX (opcional)
            enable_cache: Habilitar caché de resultados (v2.0)
            enable_async: Habilitar fetching asíncrono (v2.0)
            rate_limit_per_minute: Límite de requests por minuto (v2.0)
        """
        self.cognitive_agent = cognitive_agent
        self.connection_manager = SecureConnectionManager()
        self.search_history: List[Dict[str, Any]] = []

        # 🧠 REGISTRARSE EN LA RED NEURONAL SIMBIÓTICA
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            self.neural_network.register_module("internet_search", self)
            logger.info("✅ Módulo 'internet_search' registrado en la red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo registrar en red neuronal: {e}")
            self.neural_network = None  # type: ignore

        # Configuración de fuentes de búsqueda
        self.search_sources: Dict[str, Dict[str, Any]] = {
            "wikipedia": {"enabled": WIKIPEDIA_AVAILABLE, "priority": 0.9},
            "arxiv": {"enabled": ARXIV_AVAILABLE, "priority": 0.95},
            "google": {"enabled": True, "priority": 0.8},
            "academic": {"enabled": True, "priority": 0.85},
            "news": {"enabled": True, "priority": 0.7},
        }

        # 🆕 v2.0: Caché de resultados
        self.enable_cache = enable_cache and CACHE_V2_AVAILABLE
        self.cache = None
        self.embedding_system = None

        if self.enable_cache and get_global_cache is not None:
            try:
                self.cache = get_global_cache()
                if EMBEDDINGS_AVAILABLE and get_embedding_system is not None:
                    self.embedding_system = get_embedding_system()
                logger.info("✅ Caché v2.0 habilitado para búsquedas")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo habilitar caché: {e}")
                self.enable_cache = False

        # 🆕 v2.0: Async fetching
        self.enable_async = enable_async and AIOHTTP_AVAILABLE
        if self.enable_async:
            logger.info("✅ Fetching asíncrono habilitado")
        elif not AIOHTTP_AVAILABLE:
            logger.info("ℹ️  aiohttp no disponible. Usando modo síncrono (v1.0)")

        # 🆕 v2.0: Rate limiting por fuente
        self.rate_limit_per_minute = rate_limit_per_minute
        self.request_timestamps: defaultdict[str, List[float]] = defaultdict(list)

        # 🆕 v2.0: Circuit breakers por fuente
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        for source in self.search_sources.keys():
            self.circuit_breakers[source] = {
                "failures": 0,
                "threshold": 3,
                "open": False,
                "last_failure_time": 0.0,
                "cooldown": 300,  # 5 minutos
            }

        # 🆕 v2.0: Métricas
        self.metrics: Dict[str, Any] = {
            "total_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "async_searches": 0,
            "sync_searches": 0,
            "errors_by_source": defaultdict(int),
            "avg_response_time": 0.0,
        }

        logger.info("🔍 Sistema de búsqueda avanzada METACORTEX v2.0 inicializado")

    # 🆕 v2.0: Métodos de caché y circuit breakers

    def _check_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Verificar si hay resultados en caché para esta búsqueda

        Args:
            query: Query de búsqueda

        Returns:
            Resultados en caché o None
        """
        if not self.enable_cache or not self.cache:
            return None

        try:
            # Búsqueda por hash exacto
            cache_key = f"search:{query.lower().strip()}"
            cached = self.cache.get("internet_search", cache_key)

            if cached:
                self.metrics["cache_hits"] += 1
                logger.info(f"✅ Cache hit para: {query[:50]}")
                return cached

            # Búsqueda semántica (si embeddings disponibles)
            if self.embedding_system:
                embedding = self.embedding_system.encode(query)  # type: ignore
                # search_similar(query_embedding, k, threshold) - NO acepta namespace
                similar = self.cache.search_similar(  # type: ignore
                    embedding, k=1, threshold=0.90
                )

                if similar and len(similar) > 0:
                    self.metrics["cache_hits"] += 1
                    similar_key, similarity = similar[0]
                    logger.info(f"✅ Cache hit semántico para: {query[:50]} (sim={similarity:.2f})")
                    # Obtener valor del caché usando la clave similar
                    return self.cache.get("internet_search", similar_key)

            self.metrics["cache_misses"] += 1
            return None

        except Exception as e:
            logger.warning(f"⚠️  Error verificando caché: {e}")
            return None

    def _save_to_cache(self, query: str, results: Dict[str, Any]):
        """
        Guardar resultados de búsqueda en caché

        Args:
            query: Query de búsqueda
            results: Resultados a cachear
        """
        if not self.enable_cache or not self.cache:
            return

        try:
            cache_key = f"search:{query.lower().strip()}"

            # Guardar con TTL de 2 horas
            self.cache.set("internet_search", cache_key, results, ttl=7200)

            # Índice vectorial (si disponible)
            if self.embedding_system:
                embedding = self.embedding_system.encode(query)  # type: ignore
                # add_vector(cache_key, embedding) - NO acepta results ni namespace
                self.cache.add_vector(cache_key, embedding)  # type: ignore

            logger.debug(f"💾 Resultados cacheados para: {query[:50]}")

        except Exception as e:
            logger.warning(f"⚠️  Error guardando en caché: {e}")

    def _check_circuit_breaker(self, source: str) -> bool:
        """
        Verificar estado del circuit breaker para una fuente

        Args:
            source: Nombre de la fuente

        Returns:
            True si el circuit breaker está abierto (no hacer request)
        """
        breaker = self.circuit_breakers.get(source)
        if not breaker:
            return False

        if breaker["open"]:
            # Verificar si cooldown ha terminado
            time_since_failure = time.time() - breaker["last_failure_time"]
            if time_since_failure > breaker["cooldown"]:
                logger.info(f"♻️  Circuit breaker reiniciado para: {source}")
                breaker["open"] = False
                breaker["failures"] = 0
                return False
            else:
                logger.warning(
                    f"⚠️  Circuit breaker abierto para {source} ({int(breaker['cooldown'] - time_since_failure)}s restantes)"
                )
                return True

        return False

    def _record_failure(self, source: str):
        """
        Registrar fallo en circuit breaker

        Args:
            source: Nombre de la fuente
        """
        breaker = self.circuit_breakers.get(source)
        if not breaker:
            return

        breaker["failures"] += 1
        breaker["last_failure_time"] = time.time()
        self.metrics["errors_by_source"][source] += 1

        if breaker["failures"] >= breaker["threshold"]:
            breaker["open"] = True
            logger.error(
                f"⚡ Circuit breaker activado para {source} ({breaker['failures']} fallos)"
            )

    def _record_success(self, source: str):
        """
        Registrar éxito y resetear circuit breaker

        Args:
            source: Nombre de la fuente
        """
        breaker = self.circuit_breakers.get(source)
        if breaker:
            breaker["failures"] = 0
            breaker["open"] = False

    def _check_rate_limit(self, source: str) -> bool:
        """
        Verificar rate limiting para una fuente

        Args:
            source: Nombre de la fuente

        Returns:
            True si se puede hacer el request, False si excede el límite
        """
        current_time = time.time()
        timestamps = self.request_timestamps[source]

        # Eliminar timestamps antiguos (>1 minuto)
        self.request_timestamps[source] = [
            ts for ts in timestamps if current_time - ts < 60
        ]

        # Verificar límite
        if len(self.request_timestamps[source]) >= self.rate_limit_per_minute:
            logger.warning(f"⚠️  Rate limit alcanzado para {source}")
            return False

        return True

    def _record_request(self, source: str):
        """
        Registrar timestamp de request para rate limiting

        Args:
            source: Nombre de la fuente
        """
        self.request_timestamps[source].append(time.time())

    async def _async_fetch_url(
        self, url: str, source: str, timeout: int = 30
    ) -> Optional[str]:
        """
        🆕 v2.0: Fetch asíncrono de URL con aiohttp

        Args:
            url: URL a fetchear
            source: Nombre de la fuente
            timeout: Timeout en segundos

        Returns:
            Contenido HTML o None si falla
        """
        if not AIOHTTP_AVAILABLE or aiohttp is None:
            return None

        try:
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            }

            timeout_obj = aiohttp.ClientTimeout(total=timeout)  # type: ignore
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:  # type: ignore
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        self._record_success(source)
                        return await response.text()
                    else:
                        logger.warning(f"⚠️  Status {response.status} para {url[:50]}")
                        self._record_failure(source)
                        return None

        except Exception as e:
            logger.warning(f"❌ Error en fetch async: {e}")
            self._record_failure(source)
            return None

    def get_v2_metrics(self) -> Dict[str, Any]:
        """
        🆕 v2.0: Obtener métricas de rendimiento

        Returns:
            Diccionario con métricas detalladas
        """
        cache_hit_rate = 0.0
        total_cache = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        if total_cache > 0:
            cache_hit_rate = (self.metrics["cache_hits"] / total_cache) * 100

        return {
            "version": "2.0",
            "total_searches": self.metrics["total_searches"],
            "cache_enabled": self.enable_cache,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "async_searches": self.metrics["async_searches"],
            "sync_searches": self.metrics["sync_searches"],
            "avg_response_time": f"{self.metrics['avg_response_time']:.2f}s",
            "errors_by_source": dict(self.metrics["errors_by_source"]),
            "circuit_breakers": {
                source: {"open": breaker["open"], "failures": breaker["failures"]}
                for source, breaker in self.circuit_breakers.items()
            },
        }

    def intelligent_search(
        self,
        query: str,
        max_sources: int = 5,
        source_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Realiza una búsqueda inteligente multi-fuente

        Args:
            query: Consulta de búsqueda
            max_sources: Número máximo de fuentes a consultar
            source_filter: Lista de fuentes específicas a usar

        Returns:
            Diccionario con resultados sintetizados
        """
        start_time = time.time()
        self.metrics["total_searches"] += 1
        logger.info(f"🧠 Iniciando búsqueda inteligente: {query}")

        # 🆕 v2.0: Verificar caché primero
        cached_result = self._check_cache(query)
        if cached_result:
            logger.info(f"✅ Retornando resultado en caché para: {query[:50]}")
            return cached_result

        # 1. Análisis del query
        query_analysis = self._analyze_query(query)

        # 2. Selección de fuentes
        selected_sources = self._select_sources(
            query_analysis, source_filter, max_sources
        )

        # 3. Búsqueda paralela
        search_results = self._parallel_search(query, selected_sources, query_analysis)

        # 4. Síntesis de resultados
        synthesized_result = self._synthesize_results(
            query, search_results, query_analysis
        )

        # 5. Almacenar en historial
        search_duration = time.time() - start_time
        search_record: Dict[str, Any] = {
            "query": query,
            "timestamp": time.time(),
            "sources_used": list(selected_sources.keys()),
            "results_count": len(search_results),
            "synthesis_quality": synthesized_result.get("quality", "unknown"),
            "duration": search_duration,
        }
        self.search_history.append(search_record)

        # 🆕 v2.0: Guardar en caché y actualizar métricas
        self._save_to_cache(query, synthesized_result)

        # Actualizar métrica de tiempo de respuesta
        if self.metrics["total_searches"] > 0:
            prev_avg = self.metrics["avg_response_time"]
            n = self.metrics["total_searches"]
            self.metrics["avg_response_time"] = (
                (prev_avg * (n - 1)) + search_duration
            ) / n

        # Contar tipo de búsqueda
        if self.enable_async:
            self.metrics["async_searches"] += 1
        else:
            self.metrics["sync_searches"] += 1

        logger.info(f"✅ Búsqueda completada en {search_record['duration']:.2f}s")
        return synthesized_result

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analiza la consulta para determinar la mejor estrategia de búsqueda"""
        query_lower = query.lower()

        # Detectar tipo de consulta
        query_type = "general"
        if any(
            word in query_lower
            for word in ["qué es", "define", "concepto", "definición"]
        ):
            query_type = "definition"
        elif any(word in query_lower for word in ["cómo", "tutorial", "pasos", "guía"]):
            query_type = "tutorial"
        elif any(
            word in query_lower
            for word in ["investigación", "paper", "estudio", "research"]
        ):
            query_type = "academic"
        elif any(
            word in query_lower
            for word in ["noticias", "actualidad", "reciente", "últimas"]
        ):
            query_type = "news"

        # Extraer keywords
        keywords = self._extract_keywords(query)

        # Evaluar complejidad
        complexity = self._evaluate_complexity(query)

        return {
            "original_query": query,
            "query_type": query_type,
            "keywords": keywords,
            "complexity": complexity,
            "suggested_sources": self._suggest_sources_by_type(query_type),
        }

    def _extract_keywords(self, query: str) -> List[str]:
        """Extrae palabras clave de la consulta"""
        # Limpiar y dividir
        words = re.findall(r"\b[a-záéíóúñ]+\b", query.lower())

        # Filtrar palabras vacías
        stop_words = {
            "es",
            "el",
            "la",
            "de",
            "que",
            "y",
            "a",
            "en",
            "un",
            "ser",
            "se",
            "no",
            "te",
            "lo",
            "le",
            "da",
            "su",
            "por",
            "son",
            "con",
            "para",
            "al",
            "del",
            "los",
            "las",
            "una",
            "como",
            "qué",
            "cómo",
        }

        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return list(set(keywords))[:10]  # Máximo 10 keywords

    def _evaluate_complexity(self, query: str) -> float:
        """Evalúa la complejidad de la consulta (0-1)"""
        complexity_score = 0.0

        # Longitud de la consulta
        if len(query.split()) > 5:
            complexity_score += 0.3

        # Operadores booleanos
        if any(op in query.upper() for op in ["AND", "OR", "NOT"]):
            complexity_score += 0.4

        # Caracteres especiales
        if any(char in query for char in ['"', "(", ")", "+", "-", "*"]):
            complexity_score += 0.2

        # Términos técnicos
        technical_terms = len([word for word in query.split() if len(word) > 8])
        complexity_score += min(technical_terms * 0.1, 0.3)

        return min(complexity_score, 1.0)

    def _suggest_sources_by_type(self, query_type: str) -> List[str]:
        """Sugiere fuentes basándose en el tipo de consulta"""
        suggestions = {
            "definition": ["wikipedia", "academic", "google"],
            "tutorial": ["google", "academic", "wikipedia"],
            "academic": ["arxiv", "academic", "wikipedia", "google"],
            "news": ["news", "google", "wikipedia"],
            "general": ["wikipedia", "google", "arxiv", "academic"],
        }
        return suggestions.get(query_type, suggestions["general"])

    def _select_sources(
        self,
        query_analysis: Dict[str, Any],
        source_filter: Optional[List[str]],
        max_sources: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Selecciona las mejores fuentes para la búsqueda"""
        suggested_sources = query_analysis.get("suggested_sources", [])

        if source_filter:
            # Usar fuentes específicas del filtro
            available_sources = {
                name: config
                for name, config in self.search_sources.items()
                if name in source_filter and config["enabled"]
            }
        else:
            # Priorizar fuentes sugeridas
            available_sources = {}
            for source_name in suggested_sources:
                if (
                    source_name in self.search_sources
                    and self.search_sources[source_name]["enabled"]
                ):
                    available_sources[source_name] = self.search_sources[source_name]

            # Añadir otras fuentes si no se alcanza el máximo
            for name, config in self.search_sources.items():
                if len(available_sources) >= max_sources:
                    break
                if name not in available_sources and config["enabled"]:
                    available_sources[name] = config

        # Limitar al máximo
        selected = dict(list(available_sources.items())[:max_sources])
        logger.debug(f"Fuentes seleccionadas: {list(selected.keys())}")
        return selected

    def _parallel_search(
        self,
        query: str,
        sources: Dict[str, Dict[str, Any]],
        query_analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Ejecuta búsquedas en paralelo en las fuentes seleccionadas"""
        results: List[Dict[str, Any]] = []

        for source_name in sources:
            try:
                if source_name == "wikipedia":
                    results.extend(self._search_wikipedia(query, query_analysis))
                elif source_name == "arxiv":
                    results.extend(self._search_arxiv(query, query_analysis))
                elif source_name == "google":
                    results.extend(self._search_google(query, query_analysis))
                elif source_name == "academic":
                    results.extend(self._search_academic(query, query_analysis))
                elif source_name == "news":
                    results.extend(self._search_news(query, query_analysis))

                # Pequeña pausa entre fuentes
                time.sleep(1)

            except Exception as e:
                logger.warning(f"Error en búsqueda {source_name}: {e}")
                continue

        return results

    def _search_wikipedia(
        self, query: str, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Búsqueda en Wikipedia"""
        if not WIKIPEDIA_AVAILABLE or wikipedia is None:
            return []

        try:
            # Buscar artículos
            search_results = wikipedia.search(query, results=3)  # type: ignore
            articles: List[Dict[str, Any]] = []

            for title in search_results[:2]:  # Máximo 2 artículos
                try:
                    page = wikipedia.page(title, auto_suggest=False)  # type: ignore
                    articles.append(
                        {
                            "source": "Wikipedia",
                            "title": page.title,  # type: ignore
                            "url": page.url,  # type: ignore
                            "content": page.summary[:1500],  # type: ignore
                            "relevance": 0.9,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error: {e}", exc_info=True)
                    continue

            return articles
        except Exception as e:
            logger.warning(f"Error en búsqueda Wikipedia: {e}")
            return []

    def _search_arxiv(
        self, query: str, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Búsqueda en ArXiv"""
        if not ARXIV_AVAILABLE or arxiv is None:
            return []

        try:
            search = arxiv.Search(  # type: ignore
                query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance  # type: ignore
            )

            papers: List[Dict[str, Any]] = []
            for result in search.results():  # type: ignore
                authors = ", ".join([str(a) for a in result.authors[:3]])
                papers.append(
                    {
                        "source": "ArXiv",
                        "title": result.title,
                        "url": result.entry_id,
                        "content": f"Autores: {authors}.\n\n{result.summary[:1200]}",
                        "relevance": 0.95,
                    }
                )

            return papers
        except Exception as e:
            logger.warning(f"Error en búsqueda ArXiv: {e}")
            return []

    def _search_google(
        self, query: str, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Búsqueda simulada en Google (implementación básica)"""
        # NOTA: En implementación real usaríamos Google Custom Search API
        try:
            # Simular resultados de Google
            return [
                {
                    "source": "Google",
                    "title": f"Resultados sobre: {query}",
                    "url": f"https://www.google.com/search?q={quote_plus(query)}",
                    "content": f"Información web general sobre {query}. Esta es una implementación básica que simula resultados de búsqueda web.",
                    "relevance": 0.8,
                }
            ]
        except Exception as e:
            logger.warning(f"Error en búsqueda Google: {e}")
            return []

    def _search_academic(
        self, query: str, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Búsqueda académica simulada"""
        return [
            {
                "source": "Academic",
                "title": f"Fuentes Académicas: {query}",
                "url": "#",
                "content": f"Información académica especializada sobre {query} de fuentes universitarias y journals.",
                "relevance": 0.85,
            }
        ]

    def _search_news(
        self, query: str, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Búsqueda de noticias simulada"""
        return [
            {
                "source": "News",
                "title": f"Noticias Recientes: {query}",
                "url": "#",
                "content": f"Últimas noticias y actualidad sobre {query} de fuentes periodísticas confiables.",
                "relevance": 0.7,
            }
        ]

    def _synthesize_results(
        self, query: str, results: List[Dict[str, Any]], analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sintetiza los resultados de búsqueda en un informe coherente"""
        if not results:
            return {
                "query": query,
                "synthesis": f"No se encontró información relevante sobre '{query}'",
                "sources": [],
                "quality": "low",
                "confidence": 0.0,
            }

        # Generar síntesis
        synthesis_parts = [f"Información encontrada sobre '{query}':\n"]

        for i, result in enumerate(results[:5], 1):
            synthesis_parts.append(
                f"{i}. **{result.get('title', 'Sin título')}** "
                f"({result.get('source', 'Fuente desconocida')})\n"
                f"   {result.get('content', '')[:300]}...\n"
            )

        synthesis = "\n".join(synthesis_parts)

        # Evaluar calidad
        quality = self._evaluate_synthesis_quality(results, analysis)

        # Calcular confianza
        confidence = min(len(results) / 5.0, 1.0) * 0.8 + quality * 0.2

        return {
            "query": query,
            "synthesis": synthesis,
            "sources": [r.get("source", "unknown") for r in results],
            "source_count": len(results),
            "quality": quality,
            "confidence": confidence,
            "analysis": analysis,
        }

    def _evaluate_synthesis_quality(
        self, results: List[Dict[str, Any]], analysis: Dict[str, Any]
    ) -> float:
        """Evalúa la calidad de la síntesis"""
        if not results:
            return 0.0

        # Factores de calidad
        source_count_factor = min(len(results) / 3.0, 1.0) * 0.4
        relevance_factor = (
            sum(r.get("relevance", 0.5) for r in results) / len(results) * 0.4
        )
        diversity_factor = (
            len(set(r.get("source", "") for r in results)) / len(results) * 0.2
        )

        quality = source_count_factor + relevance_factor + diversity_factor
        return min(quality, 1.0)

    def get_search_history(self) -> List[Dict[str, Any]]:
        """Obtiene el historial de búsquedas"""
        return self.search_history.copy()

    def clear_search_history(self):
        """Limpia el historial de búsquedas"""
        self.search_history.clear()
        logger.info("Historial de búsquedas limpiado")


def create_metacortex_search_engine(cognitive_agent: Optional[Any] = None) -> MetacortexAdvancedSearch:
    """
    Función helper para crear una instancia del motor de búsqueda METACORTEX
    """
    return MetacortexAdvancedSearch(cognitive_agent=cognitive_agent)


# Función de compatibilidad para integración con el sistema existente
def search_and_synthesize(query: str, max_sources: int = 5) -> Dict[str, Any]:
    """
    Función de conveniencia para búsqueda y síntesis directa
    """
    search_engine = MetacortexAdvancedSearch()
    return search_engine.intelligent_search(query, max_sources=max_sources)


if __name__ == "__main__":
    # Test del sistema
    print("🧠 METACORTEX - Sistema de Búsqueda Inteligente")
    print("=" * 50)

    search_engine = MetacortexAdvancedSearch()

    test_queries = [
        "machine learning en Python",
        "últimas noticias inteligencia artificial",
        "qué es la física cuántica",
        "cómo programar en JavaScript",
    ]

    for query in test_queries:
        print(f"\n🔍 Buscando: {query}")
        result = search_engine.intelligent_search(query, max_sources=3)
        print(f"📊 Calidad: {result['quality']:.2f}")
        print(f"🎯 Confianza: {result['confidence']:.2f}")
        print(f"📚 Fuentes: {', '.join(result['sources'])}")
        print(f"📄 Síntesis: {result['synthesis'][:200]}...")