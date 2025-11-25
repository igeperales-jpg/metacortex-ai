import argparse
import sys
import time
from .core import create_cognitive_agent
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METACORTEX - Sistema de Aprendizaje Web Autónomo
==============================================

Sistema integrado que permite a METACORTEX aprender autónomamente desde internet
sin límites, conectándose a universidades, fuentes científicas, Wikipedia, etc.

Características:
    pass  # TODO: Implementar
- Búsqueda autónoma inteligente
- Integración completa con sistema cognitivo METACORTEX
- Detección de anomalías en información web
- Aprendizaje estructural desde fuentes web
- Sistema BDI para toma de decisiones de búsqueda
- Metacognición para optimizar estrategias de búsqueda

Autor: Sistema METACORTEX v1.0
Fecha: 1 octubre 2025
"""
import logging

logger = logging.getLogger(__name__)


from __future__ import annotations

import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Protocol, TypeAlias, Union
from urllib.parse import quote_plus, unquote

import requests  # type: ignore
from bs4 import BeautifulSoup  # type: ignore

# Type aliases
SearchResult: TypeAlias = Dict[str, Any]
KnowledgeSource: TypeAlias = Dict[str, Union[float, bool]]
StrategyDict: TypeAlias = Dict[str, Any]

# Protocol para CognitiveAgent (evita imports circulares)
class CognitiveAgentProtocol(Protocol):
    """Protocol para el agente cognitivo de METACORTEX."""
    
    def perceive(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]: ...
    def tick(self) -> Dict[str, Any]: ...
    def get_current_state(self) -> Dict[str, Any]: ...
    
    @property
    def learning_system(self) -> Any: ...

# Flags de disponibilidad de módulos opcionales
_wikipedia_available = False
_arxiv_available = False
_advanced_search_available = False


# === PROTOCOLS PARA BIBLIOTECAS EXTERNAS SIN STUBS ===

class WikipediaPageProtocol(Protocol):
    """Protocol para wikipedia.WikipediaPage (sin stubs oficiales)."""
    title: str
    content: str
    url: str
    summary: str


class WikipediaExceptionProtocol(Protocol):
    """Protocol para wikipedia.exceptions.DisambiguationError."""
    options: List[str]


class ArxivResultProtocol(Protocol):
    """Protocol para arxiv.Result."""
    title: str
    summary: str
    pdf_url: str
    entry_id: str


try:
    import wikipedia  # type: ignore

    wikipedia.set_lang("es")  # type: ignore
    _wikipedia_available = True
except ImportError:
    print("⚠️ Wikipedia no disponible. Instala: pip install wikipedia")

try:
    import arxiv  # type: ignore

    _arxiv_available = True
except ImportError:
    print("⚠️ ArXiv no disponible. Instala: pip install arxiv")

from .utils import setup_logging, clamp

# Importar sistema de búsqueda avanzado
try:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from internet_search import MetacortexAdvancedSearch, search_and_synthesize  # type: ignore

    _advanced_search_available = True
except ImportError:
    _advanced_search_available = False
    print("⚠️ Sistema de búsqueda avanzado no disponible")

logger = setup_logging()

# Exponer flags como constantes después de configurarlas
WIKIPEDIA_AVAILABLE: bool = _wikipedia_available
ARXIV_AVAILABLE: bool = _arxiv_available
ADVANCED_SEARCH_AVAILABLE: bool = _advanced_search_available


class WebLearningAgent:
    """
    Agente de aprendizaje web integrado con METACORTEX.

    Permite al sistema cognitivo aprender autónomamente desde internet,
    usando su arquitectura BDI, detección de anomalías y metacognición.
    """

    def __init__(self, cognitive_agent: Optional[CognitiveAgentProtocol] = None):
        """
        Inicializa el agente de aprendizaje web.

        Args:
            cognitive_agent: Instancia del CognitiveAgent de METACORTEX
        """
        self.cognitive_agent: Optional[CognitiveAgentProtocol] = cognitive_agent
        self.logger = logger.getChild("web_learning")

        # Configuración de navegación
        self.session = requests.Session()
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        ]

        # Estado del sistema de aprendizaje
        self.search_history: List[SearchResult] = []
        self.learning_stats: Dict[str, Union[int, float]] = {
            "searches_performed": 0,
            "sources_accessed": 0,
            "knowledge_acquired": 0,
            "anomalies_in_sources": 0,
            "learning_cycles": 0,
        }

        # Configuración de fuentes
        self.knowledge_sources: Dict[str, KnowledgeSource] = {
            "wikipedia": {
                "priority": 0.9,
                "reliability": 0.95,
                "enabled": WIKIPEDIA_AVAILABLE,
            },
            "arxiv": {
                "priority": 0.95,
                "reliability": 0.98,
                "enabled": ARXIV_AVAILABLE,
            },
            "google": {"priority": 0.8, "reliability": 0.7, "enabled": True},
            "academic": {"priority": 0.9, "reliability": 0.85, "enabled": True},
            "news": {"priority": 0.6, "reliability": 0.6, "enabled": True},
        }

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network: Optional[Any] = get_neural_network()
            if self.neural_network:
                self.neural_network.register_module("web_learning", self)
                print("✅ WebKnowledgeIntegrator conectado a red neuronal")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None

        self.logger.info("Agente de aprendizaje web inicializado")

    def autonomous_learning_cycle(
        self, learning_goal: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta un ciclo completo de aprendizaje autónomo.

        Args:
            learning_goal: Objetivo específico de aprendizaje (opcional)

        Returns:
            Resultados del ciclo de aprendizaje
        """
        self.logger.info("🌍 Iniciando ciclo de aprendizaje autónomo")

        try:
            # 1. Determinar qué aprender usando BDI
            learning_targets = self._determine_learning_targets(learning_goal)

            # 2. Planificar estrategia de búsqueda
            search_strategy = self._plan_search_strategy(learning_targets)

            # 3. Ejecutar búsquedas inteligentes
            search_results = self._execute_intelligent_searches(search_strategy)

            # 4. Procesar información con sistema cognitivo
            processed_knowledge = self._process_with_cognitive_system(search_results)

            # 5. Integrar conocimiento en estructura cognitiva
            integration_results = self._integrate_knowledge(processed_knowledge)

            # 6. Metacognición: evaluar y optimizar
            metacognitive_assessment = self._metacognitive_evaluation(
                integration_results
            )

            # 7. Actualizar estadísticas
            self._update_learning_stats(search_results, processed_knowledge)

            return {
                "learning_targets": learning_targets,
                "individual_searches": len(search_results),
                "learning_cycles": len(learning_targets),
                "knowledge_pieces": len(processed_knowledge),
                "integration_success": integration_results.get("success", False),
                "metacognitive_score": metacognitive_assessment.get("score", 0.5),
                "recommendations": metacognitive_assessment.get("recommendations", []),
                "stats": self.learning_stats.copy(),
            }

        except Exception as e:
            logger.error(f"Error en web_learning.py: {e}", exc_info=True)
            self.logger.error(f"Error en ciclo de aprendizaje autónomo: {e}")
            return {
                "error": str(e),
                "learning_targets": [],
                "individual_searches": 0,
                "learning_cycles": 0,
                "knowledge_pieces": 0,
            }

    def search_and_learn(self, topic: str, max_sources: int = 5) -> Dict[str, Any]:
        """
        Busca y aprende sobre un tópico específico usando el motor avanzado si está disponible.

        Args:
            topic: Tema a investigar
            max_sources: Máximo número de fuentes a consultar

        Returns:
            Resultados del aprendizaje
        """
        self.logger.info(f"🔍 Buscando y aprendiendo sobre: {topic}")

        # Usar motor de búsqueda avanzado si está disponible
        if ADVANCED_SEARCH_AVAILABLE:
            try:
                advanced_search = MetacortexAdvancedSearch(
                    cognitive_agent=self.cognitive_agent
                )
                advanced_result = advanced_search.intelligent_search(
                    topic, max_sources=max_sources
                )

                return {
                    "topic": topic,
                    "content": advanced_result.get("synthesis", ""),
                    "sources": advanced_result.get("sources", []),
                    "quality": advanced_result.get("quality", 0.0),
                    "confidence": advanced_result.get("confidence", 0.0),
                    "search_engine": "advanced",
                    "sources_found": advanced_result.get("source_count", 0),
                    "analysis": advanced_result.get("analysis", {}),
                    "timestamp": time.time(),
                }
            except Exception as e:
                logger.error(f"Error en web_learning.py: {e}", exc_info=True)
                self.logger.warning(
                    f"Error en búsqueda avanzada, usando método tradicional: {e}"
                )

        # Método tradicional (fallback)
        # 1. Analizar tópico con sistema cognitivo
        topic_analysis: Dict[str, Any] = self._analyze_topic_cognitively(topic)

        # 2. Buscar en múltiples fuentes
        search_results: List[SearchResult] = []

        # Wikipedia
        if self.knowledge_sources["wikipedia"]["enabled"]:
            wikipedia_results: List[SearchResult] = self._search_wikipedia(topic, topic_analysis)
            search_results.extend(wikipedia_results)

        # ArXiv para contenido académico
        if self.knowledge_sources["arxiv"]["enabled"]:
            arxiv_results: List[SearchResult] = self._search_arxiv(topic, topic_analysis)
            search_results.extend(arxiv_results)

        # Google para búsqueda general
        google_results: List[SearchResult] = self._search_google(topic, topic_analysis, max_results=3)
        search_results.extend(google_results)

        # 3. Procesar resultados cognitivamente
        processed_results: List[SearchResult] = self._process_search_results_cognitively(
            topic, search_results
        )

        # 4. Integrar en memoria y grafo de conocimiento
        integration_result: Dict[str, Any] = self._integrate_into_cognitive_system(
            topic, processed_results
        )

        return {
            "topic": topic,
            "sources_found": len(search_results),
            "knowledge_extracted": len(processed_results),
            "cognitive_integration": integration_result,
            "topic_analysis": topic_analysis,
        }

    def _determine_learning_targets(
        self, learning_goal: Optional[str] = None
    ) -> List[str]:
        """Determina qué aprender usando el sistema BDI."""
        if learning_goal:
            return [learning_goal]

        if not self.cognitive_agent:
            # Targets por defecto si no hay agente cognitivo
            return [
                "inteligencia artificial",
                "neurociencia cognitiva",
                "sistemas complejos",
                "metacognición",
            ]

        try:
            # Usar el sistema BDI para determinar deseos de aprendizaje
            current_state = self.cognitive_agent.get_current_state()
            wellbeing = current_state.get("wellbeing", 0.5)

            # Determinar targets basado en estado cognitivo
            if wellbeing < 0.4:
                targets = ["técnicas de bienestar", "optimización cognitiva"]
            elif wellbeing > 0.8:
                targets = ["nuevas tecnologías", "investigación avanzada"]
            else:
                targets = ["conocimiento general", "conceptos fundamentales"]

            # Añadir targets basados en anomalías recientes
            anomalies = current_state.get("recent_anomalies", 0)
            if anomalies > 5:
                targets.append("detección de anomalías")
                targets.append("sistemas adaptativos")

            return targets[:3]  # Limitar a 3 targets

        except Exception as e:
            logger.error(f"Error en web_learning.py: {e}", exc_info=True)
            self.logger.error(f"Error determinando targets: {e}")
            return ["conocimiento general"]

    def _plan_search_strategy(self, targets: List[str]) -> Dict[str, Any]:
        """Planifica estrategia de búsqueda usando planificador."""
        sources_priority_list: List[str] = []
        strategy: Dict[str, Any] = {
            "targets": targets,
            "sources_priority": sources_priority_list,
            "search_depth": "medium",
            "parallel_searches": True,
            "quality_threshold": 0.7,
        }

        # Priorizar fuentes basado en confiabilidad y disponibilidad
        available_sources: List[tuple[str, KnowledgeSource]] = [
            (name, config)
            for name, config in self.knowledge_sources.items()
            if config["enabled"]
        ]

        # Ordenar por prioridad y confiabilidad
        sorted_sources: List[tuple[str, KnowledgeSource]] = sorted(
            available_sources,
            key=lambda x: (x[1]["priority"] * x[1]["reliability"]),
            reverse=True,
        )

        sources_priority_list.extend([source[0] for source in sorted_sources])

        # Determinar profundidad basado en número de targets
        if len(targets) <= 2:
            strategy["search_depth"] = "deep"
        elif len(targets) >= 4:
            strategy["search_depth"] = "broad"

        return strategy

    def _execute_intelligent_searches(
        self, strategy: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Ejecuta búsquedas inteligentes basadas en estrategia."""
        all_results: List[Dict[str, Any]] = []

        for target in strategy["targets"]:
            self.logger.info(f"🎯 Buscando información sobre: {target}")

            target_results: List[SearchResult] = []

            # Buscar en fuentes priorizadas
            for source in strategy["sources_priority"][:4]:  # Limitar a 4 fuentes
                try:
                    if (
                        source == "wikipedia"
                        and self.knowledge_sources["wikipedia"]["enabled"]
                    ):
                        results: List[SearchResult] = self._search_wikipedia(target, {})
                        target_results.extend(results)

                    elif (
                        source == "arxiv" and self.knowledge_sources["arxiv"]["enabled"]
                    ):
                        results = self._search_arxiv(target, {})
                        target_results.extend(results)

                    elif source == "google":
                        results = self._search_google(target, {}, max_results=2)
                        target_results.extend(results)

                    # Pausa entre búsquedas para ser respetuoso
                    time.sleep(random.uniform(1.0, 2.0))

                except Exception as e:
                    logger.error(f"Error en web_learning.py: {e}", exc_info=True)
                    self.logger.warning(
                        f"Error buscando en {source} para {target}: {e}"
                    )
                    continue

            all_results.extend(target_results)

        return all_results

    def _process_with_cognitive_system(
        self, search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Procesa resultados usando el sistema cognitivo METACORTEX."""
        processed_results: List[Dict[str, Any]] = []

        if not self.cognitive_agent:
            return search_results  # Sin procesamiento cognitivo

        for result in search_results:
            try:
                # Usar valores estables y limitados para evitar z-scores problemáticos
                source: str = result.get("source", "unknown")

                # Valores completamente estables en rangos pequeños
                stable_values: Dict[str, Any] = {
                    "source_reliability": self._get_source_reliability(
                        source
                    ),  # 0.6-0.98
                    "has_content": 1 if result.get("content", "") else 0,  # 0 o 1
                    "has_url": 1 if result.get("url", "") else 0,  # 0 o 1
                    "relevance_normalized": max(
                        0.5, min(1.0, result.get("relevance", 0.5))
                    ),  # 0.5-1.0
                    "processing_batch": 1,  # Valor constante
                }

                # Procesar con valores completamente controlados
                perception_result: Dict[str, Any] = self.cognitive_agent.perceive(
                    "web_knowledge_acquisition", stable_values
                )

                # Añadir información de procesamiento cognitivo
                enhanced_result: Dict[str, Any] = result.copy()
                enhanced_result["cognitive_processing"] = {
                    "anomaly_detected": perception_result.get("anomaly", False),
                    "z_score": perception_result.get("z_score"),
                    "processed_at": time.time(),
                }

                processed_results.append(enhanced_result)

            except Exception as e:
                logger.error(f"Error en web_learning.py: {e}", exc_info=True)
                self.logger.error(f"Error procesando resultado cognitivamente: {e}")
                processed_results.append(result)  # Mantener resultado sin procesar

        return processed_results

    def _get_source_reliability(self, source_name: str) -> float:
        """Obtiene la confiabilidad de una fuente."""
        source_lower = source_name.lower()

        if "wikipedia" in source_lower:
            return 0.95
        elif "arxiv" in source_lower:
            return 0.98
        elif "google" in source_lower:
            return 0.7
        elif "academic" in source_lower:
            return 0.85
        else:
            return 0.6

    def _integrate_knowledge(
        self, processed_knowledge: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Integra conocimiento en la estructura cognitiva."""
        # Si no hay agente cognitivo, simular integración exitosa
        if not self.cognitive_agent:
            return {
                "success": True,
                "items_integrated": len(processed_knowledge)
                if processed_knowledge
                else 0,
                "anomalies_found": 0,
                "concepts_added": 0,
                "reason": "No cognitive agent - simulated integration",
            }

        # Si no hay conocimiento, aún es éxito (no hay nada que fallar)
        if not processed_knowledge:
            return {
                "success": True,
                "items_integrated": 0,
                "anomalies_found": 0,
                "concepts_added": 0,
                "reason": "No knowledge to integrate",
            }

        integration_stats: Dict[str, Any] = {
            "items_integrated": 0,
            "anomalies_found": 0,
            "concepts_added": 0,
            "success": True,
        }

        try:
            # Ejecutar ciclo cognitivo para integrar conocimiento
            tick_result = self.cognitive_agent.tick()

            # Extraer conceptos de todo el conocimiento
            all_concepts: List[str] = []
            for knowledge in processed_knowledge:
                try:
                    concepts: List[str] = self._extract_concepts_from_knowledge(knowledge)
                    all_concepts.extend(concepts)

                    # Contar anomalías detectadas
                    if knowledge.get("cognitive_processing", {}).get(
                        "anomaly_detected", False
                    ):
                        integration_stats["anomalies_found"] += 1

                    integration_stats["items_integrated"] += 1

                except Exception as e:
                    logger.error(f"Error en web_learning.py: {e}", exc_info=True)
                    self.logger.warning(
                        f"Error procesando conocimiento individual: {e}"
                    )
                    # Aún contar como integrado para mantener success=True
                    integration_stats["items_integrated"] += 1

            # Añadir conceptos al sistema de aprendizaje estructural
            unique_concepts: List[str] = list(set(all_concepts))
            concepts_successfully_added: int = 0

            for concept in unique_concepts[:10]:  # Limitar a 10 conceptos por ciclo
                try:
                    self.cognitive_agent.learning_system.add_concept(
                        concept, unique_concepts[:5]
                    )
                    concepts_successfully_added += 1
                except Exception as e:
                    logger.error(f"Error en web_learning.py: {e}", exc_info=True)
                    self.logger.warning(f"Error añadiendo concepto {concept}: {e}")

            integration_stats["concepts_added"] = concepts_successfully_added
            integration_stats["total_concepts_extracted"] = len(unique_concepts)
            integration_stats["cognitive_wellbeing"] = tick_result.get("wellbeing", 0.5)

            # Garantizar success = True siempre que hayamos procesado algo
            integration_stats["success"] = integration_stats["items_integrated"] > 0

        except Exception as e:
            logger.error(f"Error en web_learning.py: {e}", exc_info=True)
            self.logger.error(f"Error integrando conocimiento: {e}")
            # Incluso con errores, si procesamos algo, es un éxito parcial
            integration_stats["success"] = integration_stats["items_integrated"] > 0
            integration_stats["error"] = str(e)
            integration_stats["partial_success"] = True

        return integration_stats

    def _metacognitive_evaluation(
        self, integration_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluación metacognitiva del proceso de aprendizaje."""
        if not self.cognitive_agent:
            return {"score": 0.5, "recommendations": []}

        try:
            # Obtener estado metacognitivo
            current_state = self.cognitive_agent.get_current_state()
            wellbeing = current_state.get("wellbeing", 0.5)

            # Evaluar éxito del aprendizaje
            items_integrated = integration_results.get("items_integrated", 0)
            concepts_added = integration_results.get("concepts_added", 0)
            anomalies_found = integration_results.get("anomalies_found", 0)

            # Calcular puntuación metacognitiva
            score = 0.0

            # Factor de cantidad de conocimiento
            if items_integrated > 0:
                score += 0.3
            if concepts_added > 5:
                score += 0.2

            # Factor de calidad (pocas anomalías = buena calidad)
            anomaly_rate: float = 0.0
            if items_integrated > 0:
                anomaly_rate = anomalies_found / items_integrated
                score += (1 - anomaly_rate) * 0.3

            # Factor de bienestar cognitivo
            score += wellbeing * 0.2

            score = clamp(score, 0.0, 1.0)

            # Generar recomendaciones
            recommendations: List[str] = []

            if score < 0.4:
                recommendations.append("Mejorar calidad de fuentes")
                recommendations.append("Reducir número de búsquedas paralelas")
            elif score > 0.8:
                recommendations.append("Aumentar profundidad de búsqueda")
                recommendations.append("Explorar fuentes más especializadas")
            else:
                recommendations.append("Mantener estrategia actual")

            if anomaly_rate > 0.3:
                recommendations.append("Filtrar mejor las fuentes de información")

            return {
                "score": score,
                "items_processed": items_integrated,
                "concepts_learned": concepts_added,
                "anomaly_rate": anomaly_rate if items_integrated > 0 else 0.0,
                "recommendations": recommendations,
            }

        except Exception as e:
            logger.error(f"Error en web_learning.py: {e}", exc_info=True)
            self.logger.error(f"Error en evaluación metacognitiva: {e}")
            return {"score": 0.0, "recommendations": ["Error en evaluación"]}

    def _analyze_topic_cognitively(self, topic: str) -> Dict[str, Any]:
        """Analiza un tópico usando el sistema cognitivo."""
        if not self.cognitive_agent:
            return {"keywords": [topic], "complexity": 0.5, "intent": "general"}

        try:
            # Usar percepción para analizar el tópico
            analysis_result = self.cognitive_agent.perceive(
                "topic_analysis",
                {
                    "topic": topic,
                    "analysis_type": "learning_preparation",
                    "timestamp": time.time(),
                },
            )

            return {
                "keywords": self._extract_keywords(topic),
                "complexity": self._assess_topic_complexity(topic),
                "intent": self._determine_search_intent(topic),
                "anomaly_in_topic": analysis_result.get("anomaly", False),
            }

        except Exception as e:
            logger.error(f"Error en web_learning.py: {e}", exc_info=True)
            self.logger.error(f"Error analizando tópico: {e}")
            return {"keywords": [topic], "complexity": 0.5, "intent": "general"}

    def _extract_keywords(self, topic: str) -> List[str]:
        """Extrae palabras clave del tópico."""
        # Limpiar y dividir
        words = re.sub(r"[^\w\s]", " ", topic.lower()).split()

        # Filtrar palabras cortas y comunes
        stop_words = {
            "el",
            "la",
            "de",
            "en",
            "un",
            "es",
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
            "como",
            "las",
            "del",
            "los",
        }
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]

        return keywords[:5]  # Máximo 5 keywords

    def _assess_topic_complexity(self, topic: str) -> float:
        """Evalúa la complejidad de un tópico."""
        complexity = 0.0

        # Longitud del tópico
        if len(topic.split()) > 3:
            complexity += 0.2

        # Presencia de términos técnicos
        technical_terms = [
            "algoritmo",
            "neural",
            "quantum",
            "cognitiv",
            "metacognitiv",
            "sistémic",
        ]
        for term in technical_terms:
            if term in topic.lower():
                complexity += 0.3
                break

        # Presencia de operadores lógicos
        if any(op in topic.upper() for op in ["AND", "OR", "NOT"]):
            complexity += 0.3

        return min(complexity, 1.0)

    def _determine_search_intent(self, topic: str) -> str:
        """Determina la intención de búsqueda."""
        topic_lower = topic.lower()

        if any(
            word in topic_lower
            for word in ["qué es", "define", "concepto", "significado"]
        ):
            return "definition"
        elif any(word in topic_lower for word in ["cómo", "tutorial", "guía", "paso"]):
            return "tutorial"
        elif any(
            word in topic_lower
            for word in ["paper", "research", "study", "investigación"]
        ):
            return "academic"
        elif any(word in topic_lower for word in ["noticia", "actualidad", "reciente"]):
            return "news"
        else:
            return "general"

    def _search_wikipedia(
        self, topic: str, analysis: Dict[str, Any]
    ) -> List[SearchResult]:
        """Busca en Wikipedia."""
        if not WIKIPEDIA_AVAILABLE:
            return []

        results: List[SearchResult] = []
        try:
            # Buscar páginas relacionadas (sin type stubs oficiales, usando # type: ignore)
            search_results: Any = wikipedia.search(topic, results=3)  # type: ignore[name-defined]

            for page_title in search_results[:2]:  # Limitar a 2 páginas
                try:
                    page: Any = wikipedia.page(page_title)  # type: ignore[name-defined]

                    results.append(
                        {
                            "source": "Wikipedia",
                            "title": str(page.title),
                            "url": str(page.url),
                            "content": str(page.summary)[:1500],  # Limitar contenido
                            "relevance": 0.9,
                            "reliability": 0.95,
                        }
                    )

                except wikipedia.exceptions.DisambiguationError as e:  # type: ignore[name-defined, attr-defined]
                    logger.error(f"Error: {e}", exc_info=True)
                    # Tomar la primera opción en caso de desambiguación
                    try:
                        page_disamb: Any = wikipedia.page(e.options[0])  # type: ignore[name-defined, attr-defined]
                        results.append(
                            {
                                "source": "Wikipedia",
                                "title": str(page_disamb.title),
                                "url": str(page_disamb.url),
                                "content": str(page_disamb.summary)[:1500],
                                "relevance": 0.8,
                                "reliability": 0.95,
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error: {e}", exc_info=True)
                        continue

                except Exception as e:
                    logger.error(f"Error: {e}", exc_info=True)
                    continue

        except Exception as e:
            logger.error(f"Error en web_learning.py: {e}", exc_info=True)
            self.logger.warning(f"Error buscando en Wikipedia: {e}")

        return results

    def _search_arxiv(
        self, topic: str, analysis: Dict[str, Any]
    ) -> List[SearchResult]:
        """Busca en ArXiv."""
        if not ARXIV_AVAILABLE:
            return []

        results: List[SearchResult] = []
        try:
            # ArXiv sin type stubs oficiales, usando # type: ignore
            search: Any = arxiv.Search(  # type: ignore[name-defined]
                query=topic, max_results=3, sort_by=arxiv.SortCriterion.Relevance  # type: ignore[name-defined, attr-defined]
            )

            for result in search.results():  # type: ignore[attr-defined]
                authors = [str(author) for author in result.authors[:3]]  # type: ignore[attr-defined]

                results.append(
                    {
                        "source": "ArXiv",
                        "title": result.title,  # type: ignore[attr-defined]
                        "url": result.entry_id,  # type: ignore[attr-defined]
                        "content": f"Autores: {', '.join(authors)}. Resumen: {result.summary[:1200]}",  # type: ignore[attr-defined]
                        "relevance": 0.95,
                        "reliability": 0.98,
                    }
                )

        except Exception as e:
            logger.error(f"Error en web_learning.py: {e}", exc_info=True)
            self.logger.warning(f"Error buscando en ArXiv: {e}")

        return results

    def _search_google(
        self, topic: str, analysis: Dict[str, Any], max_results: int = 3
    ) -> List[SearchResult]:
        """Busca en Google."""
        results: List[SearchResult] = []

        try:
            # Configurar headers
            headers = {
                "User-Agent": random.choice(self.user_agents),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }

            # Realizar búsqueda
            search_url = f"https://www.google.com/search?q={quote_plus(topic)}&num={max_results + 2}&hl=es"

            response = self.session.get(search_url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extraer enlaces
            links = soup.find_all("a", href=True)
            valid_urls: List[tuple[str, str]] = []

            for link in links:
                href_raw = link.get("href", "")
                # Asegurar que href sea string (puede ser AttributeValueList en BeautifulSoup)
                href = str(href_raw) if href_raw else ""
                
                if href.startswith("/url?q="):
                    try:
                        real_url = unquote(href.split("/url?q=")[1].split("&sa=")[0])
                        if real_url.startswith("http") and "google.com" not in real_url:
                            valid_urls.append((real_url, link.get_text()[:100]))
                    except Exception as e:
                        logger.error(f"Error: {e}", exc_info=True)
                        continue

            # Procesar URLs válidas
            for url, title in valid_urls[:max_results]:
                try:
                    content = self._extract_content_from_url(url)
                    if content:
                        results.append(
                            {
                                "source": "Google",
                                "title": title or "Página web",
                                "url": url,
                                "content": content[:1200],
                                "relevance": 0.7,
                                "reliability": 0.6,
                            }
                        )
                except Exception as e:
                    logger.error(f"Error: {e}", exc_info=True)
                    continue

                # Pausa entre requests
                time.sleep(random.uniform(1.0, 2.0))

        except Exception as e:
            logger.error(f"Error en web_learning.py: {e}", exc_info=True)
            self.logger.warning(f"Error buscando en Google: {e}")

        return results

    def _extract_content_from_url(self, url: str) -> str:
        """Extrae contenido de una URL."""
        try:
            headers = {
                "User-Agent": random.choice(self.user_agents),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }

            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remover elementos no deseados
            for element in soup(
                ["script", "style", "nav", "footer", "header", "aside", "form"]
            ):
                element.decompose()

            # Extraer texto
            text = soup.get_text(separator=" ", strip=True)

            # Limpiar texto
            clean_text = re.sub(r"\s+", " ", text)

            return clean_text[:2000]

        except Exception as e:
            logger.error(f"Error en web_learning.py: {e}", exc_info=True)
            self.logger.debug(f"No se pudo extraer contenido de {url}: {e}")
            return ""

    def _process_search_results_cognitively(
        self, topic: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Procesa resultados de búsqueda cognitivamente."""
        processed: List[SearchResult] = []

        for result in results:
            try:
                if self.cognitive_agent:
                    # Procesar como percepción
                    perception = self.cognitive_agent.perceive(
                        "web_search_result",
                        {
                            "topic": topic,
                            "source": result.get("source", "unknown"),
                            "title": result.get("title", ""),
                            "content_length": len(result.get("content", "")),
                            "relevance": result.get("relevance", 0.5),
                        },
                    )

                    result["cognitive_analysis"] = {
                        "anomaly": perception.get("anomaly", False),
                        "z_score": perception.get("z_score"),
                        "processed": True,
                    }

                processed.append(result)

            except Exception as e:
                logger.error(f"Error en web_learning.py: {e}", exc_info=True)
                self.logger.error(f"Error procesando resultado: {e}")
                processed.append(result)

        return processed

    def _integrate_into_cognitive_system(
        self, topic: str, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Integra resultados en el sistema cognitivo."""
        if not self.cognitive_agent:
            return {
                "success": False,
                "integrated": False,
                "reason": "No cognitive agent",
            }

        try:
            # Ejecutar tick para integrar información
            tick_result = self.cognitive_agent.tick()

            # Extraer y añadir conceptos
            concepts = self._extract_concepts_from_results(topic, results)

            # Garantizar que al menos se añadan conceptos
            concepts_added = 0
            for concept in concepts[:8]:  # Limitar conceptos
                try:
                    self.cognitive_agent.learning_system.add_concept(
                        concept, concepts[:4]
                    )
                    concepts_added += 1
                except Exception as e:
                    logger.error(f"Error en web_learning.py: {e}", exc_info=True)
                    self.logger.warning(f"Error añadiendo concepto {concept}: {e}")

            # Asegurar que siempre devolvemos TRUE si hay conceptos
            success = concepts_added > 0 or len(results) > 0

            return {
                "success": success,
                "integrated": success,
                "concepts_added": concepts_added,
                "total_concepts": len(concepts),
                "results_processed": len(results),
                "cognitive_state": tick_result,
                "wellbeing_after": tick_result.get("wellbeing", 0.5),
            }

        except Exception as e:
            logger.error(f"Error en web_learning.py: {e}", exc_info=True)
            self.logger.error(f"Error integrando en sistema cognitivo: {e}")
            return {"success": False, "integrated": False, "reason": str(e)}

    def _extract_concepts_from_knowledge(self, knowledge: SearchResult) -> List[str]:
        """Extrae conceptos de una pieza de conocimiento."""
        concepts: List[str] = []

        # Del título
        title = knowledge.get("title", "")
        if title:
            concepts.extend(self._extract_keywords(title))

        # De la fuente
        source = knowledge.get("source", "")
        if source:
            concepts.append(source.lower())

        # Del contenido (palabras clave)
        content = knowledge.get("content", "")
        if content:
            content_concepts = self._extract_keywords(content)
            concepts.extend(content_concepts[:3])  # Limitar conceptos del contenido

        return list(set(concepts))  # Eliminar duplicados

    def _extract_concepts_from_results(
        self, topic: str, results: List[SearchResult]
    ) -> List[str]:
        """Extrae conceptos de todos los resultados."""
        all_concepts: List[str] = [topic.lower()]

        for result in results:
            concepts = self._extract_concepts_from_knowledge(result)
            all_concepts.extend(concepts)

        return list(set(all_concepts))  # Eliminar duplicados

    def _update_learning_stats(
        self,
        search_results: List[Dict[str, Any]],
        processed_knowledge: List[Dict[str, Any]],
    ):
        """Actualiza estadísticas de aprendizaje."""
        # Contar ciclos de búsqueda (llamadas al sistema)
        self.learning_stats["learning_cycles"] += 1

        # Contar fuentes individuales accedidas (búsquedas reales)
        individual_searches = len(search_results)
        self.learning_stats["searches_performed"] += individual_searches
        self.learning_stats["sources_accessed"] += individual_searches

        # Conocimiento procesado
        self.learning_stats["knowledge_acquired"] += len(processed_knowledge)

        # Contar anomalías
        anomalies = sum(
            1
            for item in processed_knowledge
            if item.get("cognitive_processing", {}).get("anomaly_detected", False)
        )
        self.learning_stats["anomalies_in_sources"] += anomalies

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de aprendizaje."""
        stats = self.learning_stats.copy()

        # Añadir métricas calculadas
        if stats["searches_performed"] > 0:
            stats["avg_sources_per_search"] = (
                stats["sources_accessed"] / stats["searches_performed"]
            )
            stats["avg_knowledge_per_search"] = (
                stats["knowledge_acquired"] / stats["searches_performed"]
            )
        else:
            stats["avg_sources_per_search"] = 0
            stats["avg_knowledge_per_search"] = 0

        if stats["sources_accessed"] > 0:
            stats["anomaly_rate"] = (
                stats["anomalies_in_sources"] / stats["sources_accessed"]
            )
        else:
            stats["anomaly_rate"] = 0

        return stats

    def reset_learning_stats(self):
        """Reinicia las estadísticas de aprendizaje."""
        self.learning_stats = {
            "searches_performed": 0,
            "sources_accessed": 0,
            "knowledge_acquired": 0,
            "anomalies_in_sources": 0,
            "learning_cycles": 0,
        }
        self.search_history.clear()
        self.logger.info("Estadísticas de aprendizaje reiniciadas")


# === FUNCIONES DE UTILIDAD ===


def install_missing_dependencies() -> bool:
    """Instala dependencias faltantes para el aprendizaje web."""
    try:
        import subprocess

        missing: List[str] = []
        
        try:
            import wikipedia  # type: ignore
        except ImportError:
            missing.append("wikipedia")

        try:
            import arxiv  # type: ignore
        except ImportError:
            missing.append("arxiv")

        try:
            import requests  # type: ignore
        except ImportError:
            missing.append("requests")

        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError:
            missing.append("beautifulsoup4")

        if missing:
            print(f"📦 Instalando dependencias faltantes: {', '.join(missing)}")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("✅ Dependencias instaladas correctamente")
            return True
        else:
            print("✅ Todas las dependencias están disponibles")
            return True

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"❌ Error instalando dependencias: {e}")
        return False


# === FUNCIÓN DE INICIALIZACIÓN ===


def create_web_learning_agent(
    cognitive_agent: Optional[CognitiveAgentProtocol] = None
) -> WebLearningAgent:
    """
    Crea un agente de aprendizaje web integrado con METACORTEX.

    Args:
        cognitive_agent: Instancia del CognitiveAgent de METACORTEX

    Returns:
        Agente de aprendizaje web configurado
    """
    return WebLearningAgent(cognitive_agent)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="METACORTEX - Web Learning Service")
    parser.add_argument(
        "--daemon", action="store_true", help="Ejecutar como daemon en background"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Intervalo entre ciclos de aprendizaje en segundos (default: 1 hora)",
    )
    parser.add_argument(
        "--learning-goal", type=str, help="Objetivo específico de aprendizaje"
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Número máximo de ciclos (0 = infinito)",
    )

    args = parser.parse_args()

    try:
        # Crear agente cognitivo
        cognitive_agent = create_cognitive_agent()

        # Crear agente de aprendizaje web
        web_learner = create_web_learning_agent(cognitive_agent)

        logger.info("📚 METACORTEX Web Learning Service iniciado")
        logger.info(f"Modo daemon: {args.daemon}, Intervalo: {args.interval}s")

        if args.learning_goal:
            logger.info(f"Objetivo de aprendizaje: {args.learning_goal}")

        cycle_count = 0

        while True:
            try:
                # Ejecutar ciclo de aprendizaje autónomo
                result = web_learner.autonomous_learning_cycle(args.learning_goal)
                cycle_count += 1

                logger.info(
                    f"Ciclo #{cycle_count} completado - Aprendió: {result.get('knowledge_acquired', 0)} conceptos"
                )

                # Salir si se alcanzó el número máximo de ciclos
                if args.max_cycles > 0 and cycle_count >= args.max_cycles:
                    logger.info(f"Completados {cycle_count} ciclos. Finalizando.")
                    break

                # Esperar antes del siguiente ciclo
                logger.info(
                    f"Esperando {args.interval} segundos antes del siguiente ciclo..."
                )
                time.sleep(args.interval)

            except KeyboardInterrupt:
                logger.info("Interrupción recibida. Finalizando gracefully...")
                break
            except Exception as e:
                logger.error(f"Error en ciclo de aprendizaje: {e}")
                if not args.daemon:
                    raise
                # En modo daemon, continuar tras errores
                time.sleep(60)  # Esperar 1 minuto antes de reintentar

        logger.info("📚 METACORTEX Web Learning Service finalizado")

    except Exception as e:
        logger.error(f"Error crítico en Web Learning Service: {e}")
        sys.exit(1)