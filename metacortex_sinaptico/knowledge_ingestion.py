# -*- coding: utf-8 -*-
"""
🌍 METACORTEX Knowledge Ingestion Engine
=========================================

Sistema de ingestión masiva de conocimiento para METACORTEX.
Accede a TODO el conocimiento humano disponible y lo integra
en el sistema jerárquico de forma exponencial.

Fuentes de conocimiento:
    pass  # TODO: Implementar
- Wikipedia (60+ millones de artículos en todos los idiomas)
- ArXiv (2+ millones de papers científicos)
- Project Gutenberg (70,000+ libros)
- OpenLibrary (millones de libros)
- Internet Archive (800+ billones de páginas)
- PubMed (35+ millones de artículos médicos)
- Stack Overflow (conocimiento técnico)
- GitHub (código y documentación)
- YouTube (transcripciones y conocimiento audiovisual)
- Reddit (conocimiento colectivo)

METACORTEX Project - 2025
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Importaciones de METACORTEX
from .learning import StructuralLearningSystem
from internet_search import MetacortexAdvancedSearch

logger = logging.getLogger(__name__)

# Importaciones condicionales para fuentes de conocimiento
try:
    import wikipedia  # type: ignore

    wikipedia_available = True
except ImportError:
    wikipedia_available = False
    logger.warning("⚠️ Wikipedia no disponible. Instala: pip install wikipedia")

try:
    import arxiv  # type: ignore

    arxiv_available = True
except ImportError:
    arxiv_available = False
    logger.warning("⚠️ ArXiv no disponible. Instala: pip install arxiv")

try:
    import requests  # type: ignore

    requests_available = True
except ImportError:
    requests_available = False
    logger.warning("⚠️ Requests no disponible. Instala: pip install requests")


class KnowledgeSource:
    """Clase base para fuentes de conocimiento"""

    def __init__(self, name: str, priority: float = 0.5):
        self.name = name
        self.priority = priority
        self.stats: Dict[str, Any] = {
            "concepts_extracted": 0,
            "errors": 0,
            "last_fetch": None,
            "total_fetches": 0,
        }

    async def fetch(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener conocimiento de la fuente"""
        raise NotImplementedError("Debe implementarse este método")

    def extract_concepts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraer conceptos estructurados del conocimiento"""
        raise NotImplementedError("Debe implementarse el método extract_concepts")
    
    def get_quality_score(self) -> float:
        """Calcula score de calidad de la fuente basado en éxito vs errores."""
        total = self.stats["concepts_extracted"] + self.stats["errors"]
        if total == 0:
            return 0.5  # Score neutral si no hay datos
        return self.stats["concepts_extracted"] / total


class WikipediaSource(KnowledgeSource):
    """Fuente de conocimiento: Wikipedia"""

    def __init__(self, languages: List[str] = ["es", "en"]):
        super().__init__("Wikipedia", priority=0.9)
        self.languages = languages
        self.current_lang = 0

    async def fetch(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Buscar artículos de Wikipedia"""
        if not wikipedia_available:
            return []

        results: List[Dict[str, Any]] = []
        for lang in self.languages[:2]:  # Máximo 2 idiomas por búsqueda
            try:
                import wikipedia  # type: ignore
                wikipedia.set_lang(lang)
                search_results = wikipedia.search(query, results=limit)

                for title in search_results[:5]:  # Máximo 5 artículos
                    try:
                        page = wikipedia.page(title, auto_suggest=False)
                        results.append(
                            {
                                "source": "wikipedia",
                                "language": lang,
                                "title": page.title,
                                "url": page.url,
                                "content": page.content,
                                "summary": page.summary,
                                "categories": page.categories
                                if hasattr(page, "categories")
                                else [],
                                "links": page.links[:50]
                                if hasattr(page, "links")
                                else [],  # Primeros 50 enlaces
                                "images": page.images[:10]
                                if hasattr(page, "images")
                                else [],
                            }
                        )
                        self.stats["concepts_extracted"] += 1
                        await asyncio.sleep(0.5)  # Rate limiting
                    except Exception as e:
                        logger.error(f"Error al obtener página {title}: {e}", exc_info=True)
                        continue

                self.stats["total_fetches"] += 1
                self.stats["last_fetch"] = datetime.now()

            except Exception as e:
                logger.warning(f"Error en Wikipedia ({lang}): {e}")
                self.stats["errors"] += 1

        return results

    def extract_concepts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraer conceptos de un artículo de Wikipedia"""
        concepts = []

        # Concepto principal del artículo
        main_concept = {
            "name": data["title"],
            "type": "wikipedia_article",
            "domain": self._infer_domain(data),
            "content": data["summary"][:1000],
            "metadata": {
                "url": data["url"],
                "language": data["language"],
                "categories": data["categories"][:10],
            },
        }
        concepts.append(main_concept)

        # Conceptos relacionados (de los enlaces)
        for link in data["links"][:20]:  # Primeros 20 enlaces
            concepts.append(
                {
                    "name": link,
                    "type": "related_concept",
                    "related_to": data["title"],
                    "metadata": {"source": "wikipedia_link"},
                }
            )

        return concepts

    def _infer_domain(self, data: Dict[str, Any]) -> str:
        """Inferir dominio de conocimiento del artículo"""
        categories = " ".join(data.get("categories", [])).lower()
        title = data["title"].lower()
        content = data["summary"].lower()

        text = f"{title} {categories} {content[:500]}"

        # Dominios de conocimiento
        domains = {
            "science": [
                "ciencia",
                "science",
                "física",
                "physics",
                "química",
                "chemistry",
                "biología",
                "biology",
            ],
            "technology": [
                "tecnología",
                "technology",
                "computación",
                "computer",
                "software",
                "hardware",
            ],
            "mathematics": [
                "matemática",
                "mathematics",
                "álgebra",
                "algebra",
                "geometría",
                "geometry",
            ],
            "history": ["historia", "history", "histórico", "historical"],
            "philosophy": ["filosofía", "philosophy", "filosófico", "philosophical"],
            "arts": ["arte", "art", "música", "music", "literatura", "literature"],
            "medicine": [
                "medicina",
                "medicine",
                "salud",
                "health",
                "médico",
                "medical",
            ],
            "geography": [
                "geografía",
                "geography",
                "país",
                "country",
                "ciudad",
                "city",
            ],
            "economics": ["economía", "economics", "económico", "economic"],
            "politics": ["política", "politics", "político", "political"],
        }

        for domain, keywords in domains.items():
            if any(kw in text for kw in keywords):
                return domain

        return "general"


class ArXivSource(KnowledgeSource):
    """Fuente de conocimiento: ArXiv (papers científicos)"""

    def __init__(self):
        super().__init__("ArXiv", priority=0.95)

    async def fetch(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Buscar papers en ArXiv"""
        if not arxiv_available:
            return []

        results: List[Dict[str, Any]] = []
        try:
            import arxiv  # type: ignore
            search = arxiv.Search(
                query=query, max_results=limit, sort_by=arxiv.SortCriterion.Relevance
            )

            for result in search.results():
                results.append(
                    {
                        "source": "arxiv",
                        "title": result.title,
                        "url": result.entry_id,
                        "abstract": result.summary,
                        "authors": [str(a) for a in result.authors],
                        "categories": result.categories,
                        "published": result.published,
                        "updated": result.updated,
                        "pdf_url": result.pdf_url,
                    }
                )
                self.stats["concepts_extracted"] += 1
                await asyncio.sleep(1)  # Rate limiting

            self.stats["total_fetches"] += 1
            self.stats["last_fetch"] = datetime.now()

        except Exception as e:
            logger.warning(f"Error en ArXiv: {e}")
            self.stats["errors"] += 1

        return results

    def extract_concepts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraer conceptos de un paper de ArXiv"""
        concepts = []

        # Concepto principal del paper
        main_concept = {
            "name": data["title"],
            "type": "scientific_paper",
            "domain": "academic",
            "subdomain": data["categories"][0] if data["categories"] else "general",
            "content": data["abstract"][:1000],
            "metadata": {
                "url": data["url"],
                "authors": data["authors"][:5],
                "published": str(data["published"]),
                "categories": data["categories"],
            },
        }
        concepts.append(main_concept)

        # Conceptos de autores
        for author in data["authors"][:3]:
            concepts.append(
                {
                    "name": author,
                    "type": "researcher",
                    "related_to": data["title"],
                    "metadata": {"role": "author"},
                }
            )

        return concepts


class WebCrawlerSource(KnowledgeSource):
    """Fuente de conocimiento: Web Crawler genérico"""

    def __init__(self):
        super().__init__("WebCrawler", priority=0.7)
        self.visited_urls: Set[str] = set()

    async def fetch(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Crawlear páginas web relevantes"""
        # Esta es una implementación básica
        # En producción usaríamos Scrapy, BeautifulSoup, Selenium, etc.
        return []

    def extract_concepts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraer conceptos de páginas web"""
        return []


class KnowledgeIngestionEngine:
    """
    Motor de ingestión masiva de conocimiento para METACORTEX

    Este sistema aprende de forma continua y exponencial de todas
    las fuentes de conocimiento humano disponibles.
    """

    def __init__(
        self,
        learning_system: StructuralLearningSystem,
        search_engine: Optional[MetacortexAdvancedSearch] = None,
        storage_path: str = ".metacortex_knowledge",
    ):
        self.learning_system = learning_system
        self.search_engine = search_engine or MetacortexAdvancedSearch()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Fuentes de conocimiento
        self.sources: Dict[str, KnowledgeSource] = {}
        self._initialize_sources()

        # Estado del motor
        self.running = False
        self.stats = {
            "total_concepts": 0,
            "concepts_per_source": defaultdict(int),
            "start_time": None,
            "last_learning_cycle": None,
            "learning_cycles": 0,
            "errors": 0,
        }

        # Cola de aprendizaje
        self.learning_queue: asyncio.Queue = asyncio.Queue()

        # Checkpoint para guardar progreso
        self.checkpoint_file = self.storage_path / "ingestion_checkpoint.json"
        self._load_checkpoint()

        logger.info("🌍 Knowledge Ingestion Engine inicializado")

    def _initialize_sources(self) -> None:
        """Inicializar todas las fuentes de conocimiento disponibles"""
        if wikipedia_available:
            self.sources["wikipedia"] = WikipediaSource(
                languages=["es", "en", "fr", "de", "it"]
            )
            logger.info("✅ Wikipedia habilitado (5 idiomas)")

        if arxiv_available:
            self.sources["arxiv"] = ArXivSource()
            logger.info("✅ ArXiv habilitado")

        # Más fuentes se pueden agregar aquí
        logger.info(f"📚 {len(self.sources)} fuentes de conocimiento activas")

    def _load_checkpoint(self) -> None:
        """Cargar checkpoint del progreso anterior"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    checkpoint = json.load(f)
                self.stats.update(checkpoint.get("stats", {}))
                logger.info(
                    f"📂 Checkpoint cargado: {self.stats['total_concepts']} conceptos previos"
                )
            except Exception as e:
                logger.warning(f"⚠️ Error cargando checkpoint: {e}")

    def _save_checkpoint(self) -> None:
        """Guardar checkpoint del progreso actual"""
        try:
            checkpoint: Dict[str, Any] = {
                "stats": dict(self.stats),
                "timestamp": datetime.now().isoformat(),
                "sources": {
                    name: source.stats for name, source in self.sources.items()
                },
            }
            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint, f, indent=2, default=str)
            logger.debug("💾 Checkpoint guardado")
        except Exception as e:
            logger.warning(f"⚠️ Error guardando checkpoint: {e}")

    async def ingest_from_source(
        self, source_name: str, query: str, limit: int = 10
    ) -> int:
        """
        Ingerir conocimiento de una fuente específica

        Args:
            source_name: Nombre de la fuente
            query: Query de búsqueda
            limit: Número máximo de resultados

        Returns:
            Número de conceptos aprendidos
        """
        if source_name not in self.sources:
            logger.warning(f"⚠️ Fuente desconocida: {source_name}")
            return 0

        source = self.sources[source_name]
        concepts_learned = 0

        try:
            # Fetch data de la fuente
            logger.info(f"🔍 Buscando en {source_name}: '{query}'")
            data_items = await source.fetch(query, limit)

            # Extraer y aprender conceptos
            for data in data_items:
                concepts = source.extract_concepts(data)

                for concept in concepts:
                    try:
                        # Agregar al sistema de aprendizaje jerárquico
                        concept_name = concept["name"]

                        # Determinar conceptos relacionados
                        related = []
                        if "related_to" in concept:
                            related.append(concept["related_to"])

                        # Propiedades del concepto
                        properties = {
                            "type": concept.get("type", "general"),
                            "domain": concept.get("domain", "general"),
                            "source": source_name,
                            "timestamp": datetime.now().isoformat(),
                            "metadata": concept.get("metadata", {}),
                        }

                        if "content" in concept:
                            properties["content"] = concept["content"]

                        # Aprender el concepto
                        self.learning_system.add_concept(
                            concept_name, properties
                        )

                        concepts_learned += 1
                        if isinstance(self.stats["total_concepts"], int):
                            self.stats["total_concepts"] += 1
                        if isinstance(self.stats["concepts_per_source"], defaultdict):
                            self.stats["concepts_per_source"][source_name] += 1

                    except Exception as e:
                        logger.error(f"Error aprendiendo concepto: {e}", exc_info=True)
                        if isinstance(self.stats["errors"], int):
                            self.stats["errors"] += 1
                        continue

                # Rate limiting entre items
                await asyncio.sleep(0.1)

            logger.info(f"✅ Aprendidos {concepts_learned} conceptos de {source_name}")

        except Exception as e:
            logger.error(f"❌ Error en ingestión desde {source_name}: {e}")
            if isinstance(self.stats["errors"], int):
                self.stats["errors"] += 1

        return concepts_learned

    async def massive_wikipedia_ingestion(
        self, domains: Optional[List[str]] = None, articles_per_domain: int = 100
    ) -> int:
        """
        Ingestión masiva de Wikipedia

        Args:
            domains: Lista de dominios a ingerir (ej: ['science', 'technology'])
            articles_per_domain: Artículos por dominio

        Returns:
            Total de conceptos aprendidos
        """
        if "wikipedia" not in self.sources:
            logger.warning("⚠️ Wikipedia no disponible")
            return 0

        # Dominios predeterminados si no se especifican
        if domains is None:
            domains = [
                "science",
                "technology",
                "mathematics",
                "physics",
                "chemistry",
                "biology",
                "medicine",
                "philosophy",
                "history",
                "geography",
                "economics",
                "politics",
                "arts",
                "literature",
                "music",
                "psychology",
                "sociology",
                "anthropology",
                "linguistics",
                "computer_science",
                "engineering",
                "astronomy",
                "geology",
            ]

        logger.info(f"🌍 INGESTIÓN MASIVA DE WIKIPEDIA: {len(domains)} dominios")
        total_concepts = 0

        for domain in domains:
            try:
                concepts = await self.ingest_from_source(
                    "wikipedia", domain, limit=articles_per_domain
                )
                total_concepts += concepts
                logger.info(f"📚 {domain}: {concepts} conceptos")

                # Guardar checkpoint periódicamente
                if total_concepts % 100 == 0:
                    self._save_checkpoint()

                # Rate limiting entre dominios
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"❌ Error en dominio {domain}: {e}")
                continue

        logger.info(
            f"🎉 INGESTIÓN MASIVA COMPLETADA: {total_concepts} conceptos totales"
        )
        self._save_checkpoint()

        return total_concepts

    async def continuous_learning_cycle(
        self, interval_minutes: int = 60, sources: Optional[List[str]] = None
    ) -> None:
        """
        Ciclo de aprendizaje continuo

        El sistema aprende constantemente de todas las fuentes disponibles

        Args:
            interval_minutes: Intervalo entre ciclos (minutos)
            sources: Fuentes específicas a usar (None = todas)
        """
        logger.info(
            f"🔄 Iniciando aprendizaje continuo (intervalo: {interval_minutes}m)"
        )
        self.running = True
        self.stats["start_time"] = datetime.now()

        # Fuentes a usar
        active_sources = sources if sources else list(self.sources.keys())

        # Queries rotativas para exploración
        exploration_queries = self._generate_exploration_queries()
        query_index = 0

        while self.running:
            try:
                if isinstance(self.stats["learning_cycles"], int):
                    self.stats["learning_cycles"] += 1
                cycle_start = time.time()

                logger.info(f"🧠 CICLO DE APRENDIZAJE #{self.stats['learning_cycles']}")

                # Aprender de cada fuente
                cycle_concepts = 0
                for source_name in active_sources:
                    query = exploration_queries[query_index % len(exploration_queries)]
                    concepts = await self.ingest_from_source(
                        source_name, query, limit=5
                    )
                    cycle_concepts += concepts
                    query_index += 1
                    await asyncio.sleep(5)  # Rate limiting entre fuentes

                cycle_duration = time.time() - cycle_start
                self.stats["last_learning_cycle"] = datetime.now()

                logger.info(
                    f"✅ Ciclo completado: {cycle_concepts} conceptos en {cycle_duration:.1f}s"
                )

                # Guardar checkpoint
                self._save_checkpoint()

                # Esperar hasta el siguiente ciclo
                wait_seconds = interval_minutes * 60
                logger.info(
                    f"⏳ Esperando {interval_minutes} minutos hasta el próximo ciclo..."
                )
                await asyncio.sleep(wait_seconds)

            except Exception as e:
                logger.error(f"❌ Error en ciclo de aprendizaje: {e}")
                if isinstance(self.stats["errors"], int):
                    self.stats["errors"] += 1
                await asyncio.sleep(60)  # Esperar 1 minuto antes de reintentar

    def _generate_exploration_queries(self) -> List[str]:
        """Generar queries de exploración para aprendizaje continuo"""
        # Queries que cubren diferentes áreas de conocimiento
        queries = [
            # Ciencia
            "quantum physics",
            "molecular biology",
            "neuroscience",
            "climate change",
            "artificial intelligence",
            "machine learning",
            "genetics",
            "chemistry",
            # Tecnología
            "computer science",
            "software engineering",
            "robotics",
            "blockchain",
            "cybersecurity",
            "cloud computing",
            "internet of things",
            "5G technology",
            # Matemáticas
            "number theory",
            "topology",
            "differential equations",
            "graph theory",
            "probability theory",
            "linear algebra",
            "calculus",
            "statistics",
            # Historia
            "ancient civilizations",
            "world war",
            "industrial revolution",
            "renaissance",
            "colonialism",
            "cold war",
            "space race",
            "digital revolution",
            # Filosofía
            "epistemology",
            "ethics",
            "metaphysics",
            "logic",
            "existentialism",
            "phenomenology",
            "pragmatism",
            "stoicism",
            "consciousness",
            # Arte y Cultura
            "classical music",
            "modern art",
            "literature",
            "cinema",
            "architecture",
            "photography",
            "sculpture",
            "poetry",
            "theatre",
            "dance",
            # Medicina
            "immunology",
            "cardiology",
            "oncology",
            "psychiatry",
            "epidemiology",
            "pharmacology",
            "surgery",
            "radiology",
            "pathology",
            "anatomy",
            # Economía
            "macroeconomics",
            "microeconomics",
            "game theory",
            "behavioral economics",
            "international trade",
            "monetary policy",
            "stock market",
            "cryptocurrency",
            # Psicología
            "cognitive psychology",
            "developmental psychology",
            "social psychology",
            "neuroscience",
            "psychotherapy",
            "personality",
            "memory",
            "perception",
        ]
        return queries

    def stop(self) -> None:
        """Detener el motor de ingestión"""
        logger.info("🛑 Deteniendo motor de ingestión...")
        self.running = False
        self._save_checkpoint()

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del motor de ingestión"""
        stats = dict(self.stats)

        # Calcular uptime
        start_time = self.stats.get("start_time")
        if start_time and isinstance(start_time, datetime):
            uptime = datetime.now() - start_time
            stats["uptime"] = str(uptime)

            # Conceptos por hora
            hours = uptime.total_seconds() / 3600
            total_concepts = self.stats.get("total_concepts", 0)
            if hours > 0 and isinstance(total_concepts, int):
                stats["concepts_per_hour"] = total_concepts / hours

        # Estadísticas de fuentes
        stats["sources"] = {
            name: {"stats": source.stats, "priority": source.priority}
            for name, source in self.sources.items()
        }

        # Estadísticas del sistema de aprendizaje
        if hasattr(self.learning_system, "hierarchical_graph"):
            learning_stats = self.learning_system.hierarchical_graph.get_stats()
            stats["learning_system"] = learning_stats

        return stats


class DocumentParser:
    """Parser avanzado multi-formato para documentos."""
    
    def __init__(self):
        self.supported_formats = ['txt', 'md', 'html', 'json', 'csv']
        
    def parse_text(self, content: str) -> Dict[str, Any]:
        """Parse texto plano."""
        return {
            "type": "plain_text",
            "content": content,
            "length": len(content),
            "paragraphs": content.split('\n\n')
        }
    
    def parse_markdown(self, content: str) -> Dict[str, Any]:
        """Parse Markdown extrayendo headers, listas, etc."""
        headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        code_blocks = re.findall(r'```[\w]*\n(.*?)```', content, re.DOTALL)
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        
        return {
            "type": "markdown",
            "content": content,
            "headers": headers,
            "code_blocks": code_blocks,
            "links": links,
            "length": len(content)
        }
    
    def parse_html(self, content: str) -> Dict[str, Any]:
        """Parse HTML básico extrayendo texto."""
        # Remover tags HTML simples
        text = re.sub(r'<[^>]+>', '', content)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return {
            "type": "html",
            "content": text,
            "length": len(text)
        }
    
    def parse_json(self, content: str) -> Dict[str, Any]:
        """Parse JSON."""
        try:
            data = json.loads(content)
            return {
                "type": "json",
                "data": data,
                "keys": list(data.keys()) if isinstance(data, dict) else None
            }
        except json.JSONDecodeError:
            return {"type": "json", "error": "Invalid JSON", "content": content}
    
    def parse(self, content: str, format_type: str = 'txt') -> Dict[str, Any]:
        """Parse content según formato."""
        parsers = {
            'txt': self.parse_text,
            'md': self.parse_markdown,
            'html': self.parse_html,
            'json': self.parse_json
        }
        
        parser = parsers.get(format_type, self.parse_text)
        return parser(content)


class SemanticChunker:
    """Sistema de chunking semántico con overlap configurable."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk por oraciones preservando contexto."""
        # Separar por oraciones (simple)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Guardar chunk actual
                chunks.append(' '.join(current_chunk))
                
                # Overlap: mantener últimas oraciones
                overlap_count = max(1, len(current_chunk) // 4)
                current_chunk = current_chunk[-overlap_count:]
                current_size = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Último chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """Chunk por párrafos preservando estructura."""
        paragraphs = text.split('\n\n')
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                
                # Overlap con último párrafo
                if self.overlap > 0:
                    current_chunk = [current_chunk[-1]]
                    current_size = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(para)
            current_size += para_size
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def chunk_fixed_size(self, text: str) -> List[str]:
        """Chunk por tamaño fijo con overlap."""
        chunks: List[str] = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap
        
        return chunks


class ConceptDeduplicator:
    """Sistema de deduplicación de conceptos usando similaridad."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.seen_concepts: Set[str] = set()
        self.concept_fingerprints: Dict[str, str] = {}
    
    def normalize_text(self, text: str) -> str:
        """Normaliza texto para comparación."""
        # Lowercase y remover puntuación
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_fingerprint(self, text: str) -> str:
        """Genera fingerprint simple del texto."""
        normalized = self.normalize_text(text)
        words = sorted(set(normalized.split()))
        return ' '.join(words[:50])  # Primeras 50 palabras únicas
    
    def simple_similarity(self, text1: str, text2: str) -> float:
        """Calcula similaridad simple basada en palabras compartidas."""
        words1 = set(self.normalize_text(text1).split())
        words2 = set(self.normalize_text(text2).split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def is_duplicate(self, concept_name: str, content: str = "") -> bool:
        """Verifica si un concepto es duplicado."""
        # Check exacto
        if concept_name in self.seen_concepts:
            return True
        
        # Check por fingerprint
        fingerprint = self.get_fingerprint(f"{concept_name} {content}")
        
        for existing_name, existing_fp in self.concept_fingerprints.items():
            similarity = self.simple_similarity(fingerprint, existing_fp)
            if similarity >= self.similarity_threshold:
                logger.debug(f"Duplicado detectado: {concept_name} ≈ {existing_name} ({similarity:.2f})")
                return True
        
        # No es duplicado, registrar
        self.seen_concepts.add(concept_name)
        self.concept_fingerprints[concept_name] = fingerprint
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas de deduplicación."""
        return {
            "total_concepts": len(self.seen_concepts),
            "fingerprints": len(self.concept_fingerprints)
        }


class KnowledgeQualityFilter:
    """Filtro de calidad para conocimiento ingerido."""
    
    def __init__(self, min_length: int = 50, min_quality_score: float = 0.3):
        self.min_length = min_length
        self.min_quality_score = min_quality_score
    
    def calculate_quality_score(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calcula score de calidad del contenido."""
        score = 0.5  # Base
        
        # Longitud (más largo = mejor, hasta cierto punto)
        length = len(content)
        if length >= self.min_length:
            score += 0.1
        if length >= 500:
            score += 0.1
        if length >= 2000:
            score += 0.1
        
        # Diversidad de palabras
        words = content.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 0.2
        
        # Metadata presente
        if metadata:
            score += 0.1
            if 'url' in metadata:
                score += 0.05
            if 'categories' in metadata or 'tags' in metadata:
                score += 0.05
        
        return min(1.0, score)
    
    def should_ingest(self, content: str, metadata: Dict[str, Any]) -> Tuple[bool, float]:
        """Determina si el contenido debe ingerirse."""
        if len(content) < self.min_length:
            return False, 0.0
        
        quality = self.calculate_quality_score(content, metadata)
        return quality >= self.min_quality_score, quality


async def start_massive_ingestion(
    learning_system: StructuralLearningSystem, mode: str = "massive"
) -> KnowledgeIngestionEngine:
    """
    Iniciar ingestión masiva de conocimiento

    Args:
        learning_system: Sistema de aprendizaje METACORTEX
        mode: Modo de ingestión
            - "massive": Ingestión masiva única
            - "continuous": Aprendizaje continuo 24/7
            - "hybrid": Masiva + continua

    Returns:
        Motor de ingestión configurado
    """
    engine = KnowledgeIngestionEngine(learning_system)

    if mode == "massive":
        logger.info("🚀 Modo: INGESTIÓN MASIVA")
        concepts = await engine.massive_wikipedia_ingestion(articles_per_domain=50)
        logger.info(f"✅ Ingestión masiva completada: {concepts} conceptos")

    elif mode == "continuous":
        logger.info("🔄 Modo: APRENDIZAJE CONTINUO 24/7")
        await engine.continuous_learning_cycle(interval_minutes=30)

    elif mode == "hybrid":
        logger.info("🌟 Modo: HÍBRIDO (Masiva + Continua)")
        # Primero ingestión masiva
        concepts = await engine.massive_wikipedia_ingestion(articles_per_domain=50)
        logger.info(f"✅ Fase masiva completada: {concepts} conceptos")

        # Luego aprendizaje continuo
        await engine.continuous_learning_cycle(interval_minutes=60)

    return engine


if __name__ == "__main__":
    # Test del sistema

    async def test_ingestion():
        print("🌍 METACORTEX Knowledge Ingestion Engine - TEST")
        print("=" * 60)

        # Crear sistema de aprendizaje
        learning_system = StructuralLearningSystem(use_hierarchical=True)

        # Crear motor de ingestión
        engine = KnowledgeIngestionEngine(learning_system)

        # Test 1: Ingestión desde Wikipedia
        print("\n📚 Test 1: Wikipedia")
        concepts = await engine.ingest_from_source(
            "wikipedia", "artificial intelligence", limit=3
        )
        print(f"✅ Aprendidos {concepts} conceptos")

        # Test 2: Ingestión desde ArXiv
        if "arxiv" in engine.sources:
            print("\n📄 Test 2: ArXiv")
            concepts = await engine.ingest_from_source(
                "arxiv", "machine learning", limit=3
            )
            print(f"✅ Aprendidos {concepts} conceptos")

        # Mostrar estadísticas
        print("\n📊 Estadísticas:")
        stats = engine.get_stats()
        print(f"Total conceptos: {stats['total_concepts']}")
        print(f"Por fuente: {dict(stats['concepts_per_source'])}")

        # Estadísticas del grafo jerárquico
        if "learning_system" in stats:
            ls = stats["learning_system"]
            print("\n🧠 Sistema de aprendizaje:")
            print(f"  Activos: {ls['active_nodes']}")
            print(f"  Archivados: {ls['archived_nodes']}")
            print(f"  Archivos: {ls['archive_count']}")

    asyncio.run(test_ingestion())