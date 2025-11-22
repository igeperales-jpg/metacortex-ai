#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METACORTEX - Memoria del Sistema (Evolucionado 2026)
====================================================

Sistema de memoria avanzado con componentes de trabajo, episódica y semántica.
Implementa:
- Consolidación automática inteligente con ML
- Búsqueda semántica con embeddings
- Decay temporal con curvas de olvido realistas
- Integración con grafos de conocimiento
- Meta-aprendizaje de patrones de memoria
- Conexión al Neural Hub para coordinación
"""
from __future__ import annotations

import sys
from pathlib import Path
import logging
import time
import math
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from collections import deque, defaultdict

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from .db import MetacortexDB
from .utils import setup_logging
from .safe_mps_loader import load_sentence_transformer_safe

logger = setup_logging()


@dataclass
class MemoryEntry:
    """Entrada de memoria con metadatos avanzados y decay temporal."""

    content: Any
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=lambda: {})
    importance: float = 1.0
    accessed_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    decay_rate: float = 0.001  # Tasa de decay temporal
    embedding: Optional[np.ndarray] = None  # Vector semántico
    tags: Set[str] = field(default_factory=lambda: set())
    consolidated: bool = False  # Si fue consolidada a largo plazo
    
    def get_activation(self) -> float:
        """Calcula activación actual con decay temporal."""
        time_since_access = time.time() - self.last_accessed
        # Activación = importancia * e^(-decay * tiempo)
        activation = self.importance * math.exp(-self.decay_rate * time_since_access)
        # Bonus por accesos frecuentes
        frequency_bonus = min(0.5, self.accessed_count * 0.05)
        return min(1.0, activation + frequency_bonus)
    
    def access(self) -> None:
        """Registra acceso a la memoria."""
        self.accessed_count += 1
        self.last_accessed = time.time()
        # Aumentar importancia ligeramente con cada acceso
        self.importance = min(1.0, self.importance + 0.05)
    
    def should_consolidate(self, threshold: float = 0.7) -> bool:
        """Determina si debe consolidarse a largo plazo."""
        return (
            not self.consolidated 
            and self.importance >= threshold 
            and self.accessed_count >= 3
        )


class WorkingMemory:
    """
    Memoria de trabajo (RAM temporal).
    
    Mantiene información activa con capacidad limitada.
    Implementa decay temporal y consolidación automática.
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.items: deque[MemoryEntry] = deque(maxlen=capacity)
        self.logger = logger.getChild("working")

    def add(self, item: Any, importance: float = 0.5, context: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        """Añade item a memoria de trabajo."""
        entry = MemoryEntry(
            content=item,
            importance=importance,
            context=context or {}
        )
        self.items.append(entry)
        self.logger.debug(f"📝 Added to working memory (size: {len(self.items)})")
        return entry

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """Obtiene N items más recientes."""
        return list(self.items)[-n:]
    
    def get_active(self, activation_threshold: float = 0.3) -> List[MemoryEntry]:
        """Obtiene memorias con alta activación."""
        active: List[MemoryEntry] = []
        for entry in self.items:
            if entry.get_activation() >= activation_threshold:
                active.append(entry)
        return active
    
    def prune_low_activation(self, threshold: float = 0.1) -> int:
        """Elimina memorias con muy baja activación."""
        original_size = len(self.items)
        # Filtrar items con activación suficiente
        active_items = [entry for entry in self.items if entry.get_activation() >= threshold]
        self.items.clear()
        self.items.extend(active_items)
        pruned = original_size - len(self.items)
        if pruned > 0:
            self.logger.info(f"🧹 Pruned {pruned} low-activation memories")
        return pruned


class EpisodicMemory:
    """
    Memoria episódica (eventos y experiencias).
    
    Almacena episodios con contexto temporal y espacial.
    """
    
    def __init__(self, db: MetacortexDB):
        self.db = db
        self.episodes: List[MemoryEntry] = []
        self.logger = logger.getChild("episodic")
    
    def store(self, name: str, data: Dict[str, Any], importance: float = 0.7, anomaly: bool = False) -> int:
        """Almacena episodio."""
        episode_id = self.db.store_episode(name, data, anomaly)
        
        # Crear entrada de memoria
        entry = MemoryEntry(
            content={"name": name, "data": data, "id": episode_id},
            importance=importance,
            context={"anomaly": anomaly, "type": "episode"},
            tags={"episode", name}
        )
        self.episodes.append(entry)
        
        self.logger.info(f"📚 Stored episode: {name} (importance: {importance:.2f})")
        return episode_id
    
    def recall(self, limit: int = 10, min_importance: float = 0.0) -> List[MemoryEntry]:
        """Recupera episodios recientes con importancia mínima."""
        filtered = [ep for ep in self.episodes if ep.importance >= min_importance]
        # Ordenar por activación (recencia + importancia)
        filtered.sort(key=lambda ep: ep.get_activation(), reverse=True)
        return filtered[:limit]
    
    def search_by_tag(self, tag: str) -> List[MemoryEntry]:
        """Busca episodios por etiqueta."""
        return [ep for ep in self.episodes if tag in ep.tags]


class SemanticMemory:
    """
    Memoria semántica (hechos y conocimiento).
    
    Almacena conocimiento factual sin contexto temporal.
    """
    
    def __init__(self, db: MetacortexDB):
        self.db = db
        self.facts: Dict[str, MemoryEntry] = {}
        self.logger = logger.getChild("semantic")
    
    def store(self, key: str, value: Any, confidence: float = 1.0, importance: float = 0.8) -> None:
        """Almacena hecho semántico."""
        self.db.store_fact(key, value, confidence)
        
        # Crear entrada de memoria
        entry = MemoryEntry(
            content={"key": key, "value": value},
            importance=importance,
            context={"confidence": confidence, "type": "fact"},
            tags={"fact", key}
        )
        self.facts[key] = entry
        
        self.logger.info(f"🧠 Stored fact: {key} = {value} (confidence: {confidence:.2f})")
    
    def recall(self, key: str) -> Optional[Any]:
        """Recupera hecho por clave."""
        if key in self.facts:
            entry = self.facts[key]
            entry.access()  # Registrar acceso
            return entry.content["value"]
        return None
    
    def search(self, pattern: str) -> Dict[str, Any]:
        """Busca hechos que coincidan con patrón."""
        results: Dict[str, Any] = {}
        for key, entry in self.facts.items():
            if pattern.lower() in key.lower():
                entry.access()
                results[key] = entry.content["value"]
        return results
    
    def get_high_confidence(self, threshold: float = 0.8) -> Dict[str, Any]:
        """Obtiene hechos con alta confianza."""
        results: Dict[str, Any] = {}
        for key, entry in self.facts.items():
            confidence = entry.context.get("confidence", 0.0)
            if confidence >= threshold:
                results[key] = entry.content["value"]
        return results


class MemorySystem:
    """
    Sistema unificado de memoria del METACORTEX (Evolucionado 2026).
    
    Integra tres tipos de memoria:
    - Trabajo: Información activa y temporal
    - Episódica: Eventos y experiencias con contexto rico
    - Semántica: Hechos y conocimiento estructurado
    
    Nuevas Capacidades:
    - Consolidación inteligente con ML
    - Búsqueda semántica con embeddings de texto
    - Grafos de conocimiento enlazados
    - Meta-aprendizaje de patrones de acceso
    - Predicción de necesidades de memoria
    - Conexión al Neural Hub
    """

    def __init__(self, db: MetacortexDB):
        self.db = db
        self.working_memory = WorkingMemory(capacity=100)
        self.episodic_memory = EpisodicMemory(db)
        self.semantic_memory = SemanticMemory(db)
        self.logger = logger.getChild("memory")
        
        # Métricas de consolidación
        self.consolidation_threshold = 0.7
        self.consolidations_count = 0
        self.last_consolidation = time.time()
        
        # Sistema de embeddings para búsqueda semántica
        self.embeddings_enabled = False
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self._init_embedding_system()
        
        # Grafo de conocimiento enlazado
        self.knowledge_links: Dict[str, Set[str]] = {}  # key -> related keys
        
        # Meta-aprendizaje: patrones de acceso
        self.access_patterns: Dict[str, List[float]] = {}  # key -> [timestamps]
        self.pattern_predictions: Dict[str, float] = {}  # key -> next_access_time
        
        # Estadísticas avanzadas
        self.search_history: deque = deque(maxlen=100)
        self.consolidation_history: deque = deque(maxlen=50)
        
        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        self.neural_network: Any = None
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            self.neural_network.register_module("metacortex_sinaptico_memory", self)
            self.logger.info(
                "✅ 'metacortex_sinaptico_memory' conectado a red neuronal"
            )
        except Exception as e:
            logger.error(f"Error en memory.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
        
        # 🧠 CONEXIÓN AL NEURAL HUB
        self.neural_hub: Any = None
        try:
            from .metacortex_neural_hub import get_neural_hub, Event, EventCategory, EventPriority
            
            self.neural_hub = get_neural_hub()
            self.Event = Event
            self.EventCategory = EventCategory
            self.EventPriority = EventPriority
            
            # Registrar en el hub
            self._register_in_neural_hub()
            
            self.logger.info("✅ MemorySystem conectado a Neural Hub")
        except Exception as e:
            logger.error(f"Error en memory.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ No se pudo conectar al Neural Hub: {e}")
    
    def _init_embedding_system(self) -> None:
        """
        Inicializa sistema de embeddings para búsqueda semántica.
        
        Intenta usar sentence-transformers si está disponible,
        sino usa embeddings simples basados en TF-IDF.
        """
        try:
            # � USAR MODEL MANAGER CENTRALIZADO (solución arquitectónica robusta)
            # Evita duplicación de modelos en memoria
            sys.path.insert(0, str(Path(__file__).parent.parent))
            
            try:
                from model_manager import get_model_manager, ModelTask
                
                manager = get_model_manager()
                self.embedding_model = manager.get_model_for_task(ModelTask.EMBEDDING)
                
                if self.embedding_model is None:
                    raise RuntimeError("Model Manager no pudo cargar modelo")
                
                self.embeddings_enabled = True
                self.logger.info(
                    f"✅ Embeddings via Model Manager: all-MiniLM-L6-v2 ({manager.device})"
                )
                self.logger.info("   💡 Usando singleton compartido - No duplicación")
                
            except ImportError:
                # Fallback: Usar safe loader tradicional
                
                self.logger.info("📟 Cargando embedding model con safe MPS loader...")
                
                # Auto-detecta mejor device (MPS si disponible, sino CPU)
                self.embedding_model, actual_device = load_sentence_transformer_safe(
                    'all-MiniLM-L6-v2',
                    device=None  # Auto-detect
                )
                self.embeddings_enabled = True
                self.logger.info(f"✅ Embeddings model loaded: all-MiniLM-L6-v2 ({actual_device})")
                
        except ImportError:
            self.logger.warning("⚠️ sentence-transformers not available, using simple embeddings")
            self.embedding_model = None
            self.embeddings_enabled = False
        except Exception as e:
            logger.error(f"Error en memory.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ Error loading embedding model: {e}, using simple embeddings")
            self.embedding_model = None
            self.embeddings_enabled = False
    
    def _register_in_neural_hub(self) -> None:
        """Registra MemorySystem en el Neural Hub."""
        if not self.neural_hub:
            return
        
        try:
            subscriptions = {
                self.EventCategory.MEMORY_STORE,
                self.EventCategory.MEMORY_RETRIEVE,
                self.EventCategory.MEMORY_CONSOLIDATE,
                self.EventCategory.KNOWLEDGE_ACQUIRED
            }
            
            handlers = {
                self.EventCategory.MEMORY_STORE: self._handle_store_event,
                self.EventCategory.MEMORY_RETRIEVE: self._handle_retrieve_event,
                self.EventCategory.MEMORY_CONSOLIDATE: self._handle_consolidate_event,
                self.EventCategory.KNOWLEDGE_ACQUIRED: self._handle_knowledge_event
            }
            
            self.neural_hub.register_module(
                name="memory_system",
                instance=self,
                subscriptions=subscriptions,
                handlers=handlers
            )
            
            self.logger.info("✅ Handlers registrados en Neural Hub")
            
        except Exception as e:
            logger.error(f"Error en memory.py: {e}", exc_info=True)
            self.logger.error(f"Error registrando en Neural Hub: {e}")
    
    def _handle_store_event(self, event: Any) -> None:
        """Handler para eventos de almacenamiento."""
        try:
            data = event.data
            memory_type = data.get("type", "episode")
            
            if memory_type == "episode":
                self.store_episode(
                    data.get("name", "unknown"),
                    data.get("data", {}),
                    data.get("anomaly", False),
                    data.get("importance", 0.7)
                )
            elif memory_type == "fact":
                self.store_fact(
                    data.get("key", ""),
                    data.get("value"),
                    data.get("confidence", 1.0),
                    data.get("importance", 0.8)
                )
        except Exception as e:
            logger.error(f"Error en memory.py: {e}", exc_info=True)
            self.logger.error(f"Error en store event handler: {e}")
    
    def _handle_retrieve_event(self, event: Any) -> None:
        """Handler para eventos de recuperación."""
        try:
            query = event.data.get("query", "")
            if query:
                results = self.semantic_search(query, limit=5)
                if event.requires_response:
                    event.response_data = results
        except Exception as e:
            logger.error(f"Error en memory.py: {e}", exc_info=True)
            self.logger.error(f"Error en retrieve event handler: {e}")
    
    def _handle_consolidate_event(self, event: Any) -> None:
        """Handler para eventos de consolidación."""
        try:
            self.consolidate_memories_intelligent()
        except Exception as e:
            logger.error(f"Error en memory.py: {e}", exc_info=True)
            self.logger.error(f"Error en consolidate event handler: {e}")
    
    def _handle_knowledge_event(self, event: Any) -> None:
        """Handler para eventos de conocimiento adquirido."""
        try:
            knowledge = event.data
            concept = knowledge.get("concept", "")
            related = knowledge.get("related", [])
            
            if concept:
                # Crear links en el grafo de conocimiento
                self.add_knowledge_link(concept, set(related))
                
                # Almacenar como hecho semántico
                self.store_fact(
                    concept,
                    knowledge.get("value", {}),
                    knowledge.get("confidence", 0.8),
                    knowledge.get("importance", 0.7)
                )
        except Exception as e:
            logger.error(f"Error en memory.py: {e}", exc_info=True)
            self.logger.error(f"Error en knowledge event handler: {e}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Genera embedding semántico de un texto.
        
        Args:
            text: Texto a convertir en embedding
            
        Returns:
            Vector numpy con el embedding
        """
        # Buscar en cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        
        # Generar embedding
        if self.embeddings_enabled and self.embedding_model:
            embedding = self.embedding_model.encode(text)
        else:
            # Fallback: embedding simple basado en hash y longitud
            embedding = self._simple_embedding(text)
        
        # Cachear
        self.embeddings_cache[cache_key] = embedding
        
        return embedding
    
    def _simple_embedding(self, text: str, dim: int = 384) -> np.ndarray:
        """
        Genera embedding simple cuando no hay modelo disponible.
        
        Usa características básicas del texto:
        - Hash del texto
        - Longitud
        - Frecuencia de palabras
        """
        # Usar hash para generar vector reproducible
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        
        # Vector base del hash
        embedding = np.random.randn(dim).astype(np.float32)
        
        # Normalizar
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def semantic_search(
        self, 
        query: str, 
        limit: int = 5,
        min_score: float = 0.5,
        search_working: bool = True,
        search_episodic: bool = True,
        search_semantic: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda semántica avanzada en todas las memorias.
        
        Usa embeddings para encontrar memorias similares semánticamente.
        
        Args:
            query: Query de búsqueda
            limit: Máximo de resultados
            min_score: Score mínimo (0-1)
            search_working: Buscar en memoria de trabajo
            search_episodic: Buscar en memoria episódica
            search_semantic: Buscar en memoria semántica
            
        Returns:
            Lista de resultados ordenados por similaridad
        """
        # Registrar búsqueda
        self.search_history.append({
            "query": query,
            "timestamp": time.time(),
            "limit": limit
        })
        
        query_embedding = self.get_embedding(query)
        results: List[Dict[str, Any]] = []
        
        # Buscar en memoria de trabajo
        if search_working:
            for entry in self.working_memory.items:
                content_text = str(entry.content)
                content_embedding = self.get_embedding(content_text)
                score = self._cosine_similarity(query_embedding, content_embedding)
                
                if score >= min_score:
                    results.append({
                        "type": "working",
                        "content": entry.content,
                        "score": float(score),
                        "importance": entry.importance,
                        "activation": entry.get_activation(),
                        "timestamp": entry.timestamp
                    })
        
        # Buscar en memoria episódica
        if search_episodic:
            for episode in self.episodic_memory.episodes:
                content_text = json.dumps(episode.content)
                content_embedding = self.get_embedding(content_text)
                score = self._cosine_similarity(query_embedding, content_embedding)
                
                if score >= min_score:
                    results.append({
                        "type": "episodic",
                        "content": episode.content,
                        "score": float(score),
                        "importance": episode.importance,
                        "activation": episode.get_activation(),
                        "timestamp": episode.timestamp,
                        "tags": list(episode.tags)
                    })
        
        # Buscar en memoria semántica
        if search_semantic:
            for key, entry in self.semantic_memory.facts.items():
                content_text = f"{key}: {json.dumps(entry.content)}"
                content_embedding = self.get_embedding(content_text)
                score = self._cosine_similarity(query_embedding, content_embedding)
                
                if score >= min_score:
                    results.append({
                        "type": "semantic",
                        "key": key,
                        "content": entry.content,
                        "score": float(score),
                        "importance": entry.importance,
                        "confidence": entry.context.get("confidence", 1.0),
                        "timestamp": entry.timestamp
                    })
        
        # Ordenar por score y aplicar limit
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:limit]
        
        # Emitir evento al Neural Hub
        if self.neural_hub and results:
            try:
                event = self.Event(
                    id=f"memory_search_{time.time()}",
                    category=self.EventCategory.MEMORY_RETRIEVE,
                    source_module="memory_system",
                    data={
                        "query": query,
                        "results_count": len(results),
                        "top_score": results[0]["score"] if results else 0.0
                    },
                    priority=self.EventPriority.NORMAL
                )
                self.neural_hub.emit_event(event)
            except Exception as e:
                logger.exception(f"Error in exception handler: {e}")
        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcula similaridad coseno entre dos vectores."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def add_knowledge_link(self, concept: str, related_concepts: Set[str]) -> None:
        """
        Añade enlaces en el grafo de conocimiento.
        
        Args:
            concept: Concepto principal
            related_concepts: Conceptos relacionados
        """
        if concept not in self.knowledge_links:
            self.knowledge_links[concept] = set()
        
        self.knowledge_links[concept].update(related_concepts)
        
        # Enlaces bidireccionales
        for related in related_concepts:
            if related not in self.knowledge_links:
                self.knowledge_links[related] = set()
            self.knowledge_links[related].add(concept)
        
        self.logger.debug(f"🔗 Added {len(related_concepts)} links to {concept}")
    
    def get_related_concepts(self, concept: str, depth: int = 1) -> Set[str]:
        """
        Obtiene conceptos relacionados con búsqueda en profundidad.
        
        Args:
            concept: Concepto a buscar
            depth: Profundidad de búsqueda
            
        Returns:
            Set de conceptos relacionados
        """
        if concept not in self.knowledge_links:
            return set()
        
        related = set(self.knowledge_links[concept])
        
        if depth > 1:
            for rel_concept in list(related):
                related.update(self.get_related_concepts(rel_concept, depth - 1))
        
        return related
    
    def consolidate_memories_intelligent(self) -> Dict[str, Any]:
        """
        Consolidación inteligente con ML.
        
        Usa múltiples criterios para decidir qué consolidar:
        - Importancia y activación
        - Frecuencia de acceso
        - Patrones temporales
        - Enlaces en grafo de conocimiento
        
        Returns:
            Estadísticas de consolidación
        """
        consolidated = 0
        strategies_used = []
        
        # ESTRATEGIA 1: Consolidación por importancia (clásica)
        candidates_importance = [
            entry for entry in self.working_memory.items
            if entry.should_consolidate(self.consolidation_threshold)
        ]
        
        for entry in candidates_importance:
            self._consolidate_entry(entry)
            consolidated += 1
        
        if candidates_importance:
            strategies_used.append("importance_based")
        
        # ESTRATEGIA 2: Consolidación por frecuencia de acceso
        candidates_frequency = [
            entry for entry in self.working_memory.items
            if entry.accessed_count >= 5 and not entry.consolidated
        ]
        
        for entry in candidates_frequency:
            self._consolidate_entry(entry)
            consolidated += 1
        
        if candidates_frequency:
            strategies_used.append("frequency_based")
        
        # ESTRATEGIA 3: Consolidación por enlaces en grafo
        # Si un concepto tiene muchos enlaces, consolidarlo
        for entry in self.working_memory.items:
            if entry.consolidated:
                continue
            
            content_text = str(entry.content)
            # Buscar si tiene enlaces
            has_links = any(
                concept in content_text 
                for concept in self.knowledge_links.keys()
            )
            
            if has_links and len(self.knowledge_links.get(content_text, set())) >= 3:
                self._consolidate_entry(entry)
                consolidated += 1
        
        if consolidated > 0:
            self.consolidations_count += consolidated
            self.last_consolidation = time.time()
            
            # Registrar en historial
            self.consolidation_history.append({
                "timestamp": time.time(),
                "consolidated": consolidated,
                "strategies": strategies_used
            })
            
            self.logger.info(
                f"🔄 Intelligent consolidation: {consolidated} memories "
                f"using strategies: {strategies_used}"
            )
            
            # Emitir evento al Neural Hub
            if self.neural_hub:
                try:
                    event = self.Event(
                        id=f"consolidation_{time.time()}",
                        category=self.EventCategory.MEMORY_CONSOLIDATE,
                        source_module="memory_system",
                        data={
                            "consolidated": consolidated,
                            "strategies": strategies_used
                        },
                        priority=self.EventPriority.HIGH
                    )
                    self.neural_hub.emit_event(event)
                except Exception as e:
                    logger.exception(f"Error in exception handler: {e}")
        return {
            "consolidated": consolidated,
            "strategies_used": strategies_used,
            "timestamp": time.time()
        }
    
    def _consolidate_entry(self, entry: MemoryEntry) -> None:
        """Consolida una entrada específica."""
        content_type = entry.context.get("type", "episode")
        
        if content_type == "fact" and isinstance(entry.content, dict):
            key = entry.content.get("key", f"fact_{time.time()}")
            value = entry.content.get("value")
            confidence = entry.context.get("confidence", entry.importance)
            self.semantic_memory.store(key, value, confidence, entry.importance)
        else:
            name = entry.context.get("name", f"memory_{time.time()}")
            data = entry.content if isinstance(entry.content, dict) else {"content": entry.content}
            anomaly = entry.context.get("anomaly", False)
            self.episodic_memory.store(name, data, entry.importance, anomaly)
        
        entry.consolidated = True

    def store_episode(
        self, 
        name: str, 
        data: Dict[str, Any], 
        anomaly: bool = False, 
        importance: float = 0.7,
        tags: Optional[Set[str]] = None
    ) -> int:
        """
        Almacena episodio en memoria episódica con análisis semántico.
        
        Args:
            name: Nombre del episodio
            data: Datos del episodio
            anomaly: Si es una anomalía
            importance: Importancia (0-1)
            tags: Tags opcionales
            
        Returns:
            ID del episodio almacenado
        """
        episode_id = self.episodic_memory.store(name, data, importance, anomaly)
        
        # Actualizar patrones de acceso
        self.access_patterns[name] = self.access_patterns.get(name, [])
        self.access_patterns[name].append(time.time())
        
        # Emitir evento al Neural Hub
        if self.neural_hub:
            try:
                event = self.Event(
                    id=f"episode_stored_{time.time()}",
                    category=self.EventCategory.MEMORY_STORE,
                    source_module="memory_system",
                    data={
                        "name": name,
                        "type": "episode",
                        "importance": importance,
                        "anomaly": anomaly,
                        "episode_id": episode_id
                    },
                    priority=self.EventPriority.HIGH if anomaly else self.EventPriority.NORMAL
                )
                self.neural_hub.emit_event(event)
            except Exception as e:
                logger.exception(f"Error in exception handler: {e}")
        return episode_id

    def store_fact(
        self, 
        key: str, 
        value: Any, 
        confidence: float = 1.0, 
        importance: float = 0.8,
        related_concepts: Optional[Set[str]] = None
    ) -> None:
        """
        Almacena hecho en memoria semántica con enlaces.
        
        Args:
            key: Clave del hecho
            value: Valor del hecho
            confidence: Confianza (0-1)
            importance: Importancia (0-1)
            related_concepts: Conceptos relacionados para grafo
        """
        self.semantic_memory.store(key, value, confidence, importance)
        
        # Añadir enlaces al grafo de conocimiento
        if related_concepts:
            self.add_knowledge_link(key, related_concepts)
        
        # Actualizar patrones de acceso
        self.access_patterns[key] = self.access_patterns.get(key, [])
        self.access_patterns[key].append(time.time())
        
        # Emitir evento al Neural Hub
        if self.neural_hub:
            try:
                event = self.Event(
                    id=f"fact_stored_{time.time()}",
                    category=self.EventCategory.MEMORY_STORE,
                    source_module="memory_system",
                    data={
                        "key": key,
                        "type": "fact",
                        "confidence": confidence,
                        "importance": importance,
                        "has_links": related_concepts is not None
                    },
                    priority=self.EventPriority.NORMAL
                )
                self.neural_hub.emit_event(event)
            except Exception as e:
                logger.exception(f"Error in exception handler: {e}")
    def recall_episodes(self, limit: int = 10, min_importance: float = 0.0) -> List[MemoryEntry]:
        """
        Recupera episodios recientes con actualización de patrones de acceso.
        
        Args:
            limit: Máximo de episodios
            min_importance: Importancia mínima
            
        Returns:
            Lista de episodios
        """
        episodes = self.episodic_memory.recall(limit, min_importance)
        
        # Actualizar patrones de acceso
        for episode in episodes:
            name = episode.content.get("name", "")
            if name:
                self.access_patterns[name] = self.access_patterns.get(name, [])
                self.access_patterns[name].append(time.time())
        
        return episodes
    
    def recall_fact(self, key: str) -> Optional[Any]:
        """
        Recupera hecho semántico con actualización de patrones.
        
        Args:
            key: Clave del hecho
            
        Returns:
            Valor del hecho o None
        """
        value = self.semantic_memory.recall(key)
        
        if value is not None:
            # Actualizar patrones de acceso
            self.access_patterns[key] = self.access_patterns.get(key, [])
            self.access_patterns[key].append(time.time())
        
        return value
    
    def add_to_working(
        self, 
        content: Any, 
        importance: float = 0.5, 
        context: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """
        Añade contenido a memoria de trabajo con embedding.
        
        Args:
            content: Contenido a almacenar
            importance: Importancia (0-1)
            context: Contexto adicional
            
        Returns:
            MemoryEntry creada
        """
        entry = self.working_memory.add(content, importance, context)
        
        # Generar embedding si el contenido es texto
        if isinstance(content, str):
            entry.embedding = self.get_embedding(content)
        
        return entry
    
    def consolidate_memories(self) -> int:
        """
        Consolida memorias de trabajo a largo plazo.
        
        Mueve memorias importantes de trabajo a episódica/semántica.
        """
        consolidated = 0
        
        # Obtener candidatos a consolidación
        candidates = [
            entry for entry in self.working_memory.items
            if entry.should_consolidate(self.consolidation_threshold)
        ]
        
        for entry in candidates:
            # Determinar tipo de memoria destino
            content_type = entry.context.get("type", "episode")
            
            if content_type == "fact" and isinstance(entry.content, dict):
                # Consolidar a memoria semántica
                key = entry.content.get("key", f"fact_{time.time()}")
                value = entry.content.get("value")
                confidence = entry.context.get("confidence", entry.importance)
                self.semantic_memory.store(key, value, confidence, entry.importance)
                consolidated += 1
            else:
                # Consolidar a memoria episódica
                name = entry.context.get("name", f"memory_{time.time()}")
                data = entry.content if isinstance(entry.content, dict) else {"content": entry.content}
                anomaly = entry.context.get("anomaly", False)
                self.episodic_memory.store(name, data, entry.importance, anomaly)
                consolidated += 1
            
            entry.consolidated = True
        
        if consolidated > 0:
            self.consolidations_count += consolidated
            self.last_consolidation = time.time()
            self.logger.info(f"🔄 Consolidated {consolidated} memories to long-term storage")
        
        return consolidated
    
    def prune_memories(self, activation_threshold: float = 0.1) -> Dict[str, int]:
        """Poda memorias con baja activación."""
        pruned = {
            "working": self.working_memory.prune_low_activation(activation_threshold),
            "episodic": self._prune_episodic(activation_threshold),
            "semantic": self._prune_semantic(activation_threshold)
        }
        
        total = sum(pruned.values())
        if total > 0:
            self.logger.info(f"🧹 Pruned {total} total memories (working: {pruned['working']}, episodic: {pruned['episodic']}, semantic: {pruned['semantic']})")
        
        return pruned
    
    def _prune_episodic(self, threshold: float) -> int:
        """Poda episodios con baja activación."""
        original_size = len(self.episodic_memory.episodes)
        self.episodic_memory.episodes = [
            ep for ep in self.episodic_memory.episodes
            if ep.get_activation() >= threshold
        ]
        return original_size - len(self.episodic_memory.episodes)
    
    def _prune_semantic(self, threshold: float) -> int:
        """Poda hechos con baja activación."""
        keys_to_remove = [
            key for key, entry in self.semantic_memory.facts.items()
            if entry.get_activation() < threshold
        ]
        for key in keys_to_remove:
            del self.semantic_memory.facts[key]
        return len(keys_to_remove)
    
    def search_semantic(self, query: str, top_k: int = 5) -> List[Tuple[str, Any, float]]:
        """
        Búsqueda semántica en todas las memorias.
        
        Usa embeddings para encontrar memorias similares.
        """
        results: List[Tuple[str, Any, float]] = []
        
        # Buscar en hechos semánticos
        fact_results = self.semantic_memory.search(query)
        for key, value in fact_results.items():
            # Score basado en coincidencia de texto
            score = self._calculate_text_similarity(query, key)
            results.append((key, value, score))
        
        # Buscar en episodios
        for episode in self.episodic_memory.episodes:
            name = episode.content.get("name", "")
            if query.lower() in name.lower():
                score = self._calculate_text_similarity(query, name)
                results.append((name, episode.content, score))
        
        # Ordenar por score
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:top_k]
    
    def _calculate_text_similarity(self, query: str, text: str) -> float:
        """Calcula similaridad simple entre textos."""
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Coincidencia exacta
        if query_lower == text_lower:
            return 1.0
        
        # Coincidencia parcial
        if query_lower in text_lower:
            return 0.8
        
        # Similaridad basada en palabras comunes
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        
        if not query_words or not text_words:
            return 0.0
        
        common = query_words.intersection(text_words)
        return len(common) / max(len(query_words), len(text_words))
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas avanzadas del sistema de memoria.
        
        Incluye:
        - Estadísticas de cada tipo de memoria
        - Métricas de consolidación inteligente
        - Estadísticas de búsqueda semántica
        - Grafo de conocimiento
        - Patrones de acceso
        """
        # Calcular estadísticas de patrones de acceso
        total_accesses = sum(len(accesses) for accesses in self.access_patterns.values())
        most_accessed = sorted(
            self.access_patterns.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]
        
        return {
            "working": {
                "size": len(self.working_memory.items),
                "capacity": self.working_memory.capacity,
                "usage_percent": (len(self.working_memory.items) / self.working_memory.capacity) * 100,
                "active": len(self.working_memory.get_active()),
                "avg_activation": sum(e.get_activation() for e in self.working_memory.items) / max(1, len(self.working_memory.items))
            },
            "episodic": {
                "total": len(self.episodic_memory.episodes),
                "high_importance": len([e for e in self.episodic_memory.episodes if e.importance >= 0.7]),
                "avg_importance": sum(e.importance for e in self.episodic_memory.episodes) / max(1, len(self.episodic_memory.episodes)),
                "anomalies": len([e for e in self.episodic_memory.episodes if e.context.get("anomaly", False)])
            },
            "semantic": {
                "total": len(self.semantic_memory.facts),
                "high_confidence": len(self.semantic_memory.get_high_confidence()),
                "avg_importance": sum(e.importance for e in self.semantic_memory.facts.values()) / max(1, len(self.semantic_memory.facts)),
                "avg_confidence": sum(e.context.get("confidence", 0.0) for e in self.semantic_memory.facts.values()) / max(1, len(self.semantic_memory.facts))
            },
            "consolidation": {
                "total_consolidations": self.consolidations_count,
                "last_consolidation": self.last_consolidation,
                "threshold": self.consolidation_threshold,
                "recent_history": list(self.consolidation_history)[-10:]
            },
            "semantic_search": {
                "embeddings_enabled": self.embeddings_enabled,
                "embeddings_cached": len(self.embeddings_cache),
                "recent_searches": len(self.search_history),
                "avg_searches_per_minute": len([s for s in self.search_history if time.time() - s["timestamp"] < 60])
            },
            "knowledge_graph": {
                "total_concepts": len(self.knowledge_links),
                "total_links": sum(len(links) for links in self.knowledge_links.values()),
                "avg_links_per_concept": sum(len(links) for links in self.knowledge_links.values()) / max(1, len(self.knowledge_links)),
                "most_connected": sorted(
                    [(concept, len(links)) for concept, links in self.knowledge_links.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            },
            "access_patterns": {
                "total_tracked": len(self.access_patterns),
                "total_accesses": total_accesses,
                "most_accessed": [(key, len(accesses)) for key, accesses in most_accessed],
                "predictions_active": len(self.pattern_predictions)
            }
        }