#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import networkx as nx  # type: ignore
import pickle
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
from heapq import nlargest
import logging

from .utils import setup_logging
from networkx.drawing.nx_pydot import write_dot
from networkx.readwrite import json_graph

"""
METACORTEX - Sistema de Grafos Jerárquicos
==========================================

Sistema de conocimiento infinito con archivado inteligente.
NUNCA se elimina información, solo se organiza en niveles.

Arquitectura:
    pass  # TODO: Implementar
- Grafo Activo: 10,000 nodos más accedidos (en RAM)
- Grafos Archivados: Millones de nodos organizados por temas (en disco)
- Índice Global: Mapeo de todos los conceptos a sus ubicaciones
- Promoción/Degradación: Movimiento automático entre niveles
"""

logger = setup_logging()


class HierarchicalKnowledgeGraph:
    """
    Sistema de grafos jerárquicos con memoria infinita.

    Características:
    - Grafo activo limitado (rápido acceso)
    - Archivos ilimitados (almacenamiento persistente)
    - Sin pérdida de información NUNCA
    - Carga bajo demanda (lazy loading)
    - Promoción automática de nodos populares
    """

    def __init__(
        self,
        active_limit: int = 10000,
        archive_threshold: int = 5000,
        storage_path: Optional[str] = None,
    ):
        """
        Inicializa sistema jerárquico.

        Args:
            active_limit: Límite de nodos en grafo activo
            archive_threshold: Cuántos nodos archivar cuando se llega al límite
            storage_path: Ruta para almacenar grafos archivados
        """
        # Grafo activo (hot storage)
        self.active_graph: Any = nx.DiGraph()

        # Grafos archivados (cold storage) - cargados bajo demanda
        self.archived_graphs: Dict[str, Any] = {}
        self.loaded_archives: Set[str] = set()

        # Configuración
        self.active_limit = active_limit
        self.archive_threshold = archive_threshold
        self.storage_path = Path(storage_path or ".metacortex_archives")
        self.storage_path.mkdir(exist_ok=True)

        # Índice global: concepto -> ubicación
        self.concept_index: Dict[str, str] = {}

        # Metadatos
        self.current_archive_id = 0
        self.archive_metadata: Dict[str, Dict[str, Any]] = {}

        # Métricas de acceso
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.last_access: Dict[str, float] = {}

        # Contadores
        self.total_concepts = 0
        self.total_archives = 0
        self.promotions = 0
        self.archivations = 0

        # Cache de análisis
        self.centrality_cache: Dict[str, float] = {}
        self.community_cache: Optional[List[Set[str]]] = None
        self.cache_timestamp: float = 0.0
        self.cache_ttl: float = 300.0  # 5 minutos

        # Algoritmos ejecutados
        self.algorithms_executed: Dict[str, int] = defaultdict(int)

        logger.info("🌳 Sistema de Grafos Jerárquicos inicializado")
        logger.info(f"   Límite activo: {active_limit} nodos")
        logger.info(f"   Umbral archivo: {archive_threshold} nodos")
        logger.info(f"   Almacenamiento: {self.storage_path}")

        # Cargar índice si existe
        self._load_index()

    def add_concept(
        self,
        concept: str,
        related_concepts: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Agrega concepto al sistema jerárquico."""
        # Verificar si ya existe
        if concept in self.concept_index:
            location = self.concept_index[concept]
            if location != "active":
                self.access_counts[concept] += 1
                self._check_promotion(concept, location)
            return False

        # Agregar a grafo activo
        props = properties or {}
        props.update(
            {"created_at": time.time(), "access_count": 0, "importance_score": 1.0}
        )

        self.active_graph.add_node(concept, **props)
        self.concept_index[concept] = "active"
        self.total_concepts += 1

        # Invalidar cache
        self._invalidate_cache()

        # Agregar relaciones
        if related_concepts:
            for related in related_concepts:
                if related in self.active_graph:
                    self.active_graph.add_edge(
                        concept, related, weight=0.5, edge_type="association"
                    )
                    self.active_graph.add_edge(
                        related, concept, weight=0.5, edge_type="association"
                    )

        # Verificar si necesitamos archivar
        active_count = len(list(self.active_graph.nodes()))
        if active_count >= self.active_limit:
            logger.warning(f"⚠️ Grafo activo lleno ({active_count} nodos)")
            self._archive_cold_knowledge()

        return True

    def get_concept(self, concept: str) -> Optional[Dict[str, Any]]:
        """Obtiene concepto de cualquier nivel."""
        self.access_counts[concept] += 1
        self.last_access[concept] = time.time()

        if concept not in self.concept_index:
            return None

        location = self.concept_index[concept]

        if location == "active":
            if concept in self.active_graph:
                node_data = dict(self.active_graph.nodes[concept])
                node_data["access_count"] = self.access_counts[concept]
                return node_data
        else:
            archive_id = location
            if archive_id not in self.loaded_archives:
                self._load_archive(archive_id)

            if archive_id in self.archived_graphs:
                if concept in self.archived_graphs[archive_id]:
                    node_data = dict(self.archived_graphs[archive_id].nodes[concept])
                    node_data["access_count"] = self.access_counts[concept]
                    self._check_promotion(concept, archive_id)
                    return node_data

        return None

    def _archive_cold_knowledge(self) -> int:
        """Archiva nodos fríos sin eliminar información."""
        logger.info("📦 Iniciando archivado de conocimiento...")

        cold_nodes = self._identify_cold_nodes(self.archive_threshold)

        if not cold_nodes:
            logger.warning("⚠️ No se encontraron nodos fríos")
            return 0

        archive_id = f"archive_{self.current_archive_id:04d}"
        self.archived_graphs[archive_id] = nx.DiGraph()

        archived_count = 0
        for node in cold_nodes:
            if node not in self.active_graph:
                continue

            node_data = dict(self.active_graph.nodes[node])
            node_data["archived_at"] = time.time()
            node_data["archive_id"] = archive_id

            self.archived_graphs[archive_id].add_node(node, **node_data)

            for neighbor in self.active_graph.successors(node):
                edge_data = dict(self.active_graph[node][neighbor])
                self.archived_graphs[archive_id].add_edge(node, neighbor, **edge_data)

            for predecessor in self.active_graph.predecessors(node):
                edge_data = dict(self.active_graph[predecessor][node])
                self.archived_graphs[archive_id].add_edge(
                    predecessor, node, **edge_data
                )

            self.concept_index[node] = archive_id
            self.active_graph.remove_node(node)
            archived_count += 1

        self._persist_archive(archive_id)

        self.archive_metadata[archive_id] = {
            "created_at": time.time(),
            "node_count": archived_count,
            "archive_id": archive_id,
            "path": str(self.storage_path / f"{archive_id}.gpickle"),
        }

        self.current_archive_id += 1
        self.total_archives += 1
        self.archivations += archived_count

        self.loaded_archives.discard(archive_id)
        del self.archived_graphs[archive_id]

        self._save_index()

        logger.info(f"✅ Archivado: {archived_count} nodos → {archive_id}")
        logger.info(
            f"   Activos: {len(list(self.active_graph.nodes()))} | Total: {self.total_concepts}"
        )

        return archived_count

    def _identify_cold_nodes(self, count: int) -> List[str]:
        """Identifica nodos menos accedidos."""
        coldness_scores = {}

        for node in self.active_graph.nodes():
            access_count = self.access_counts.get(node, 0)
            last_access_time = self.last_access.get(node, 0)
            time_since_access = (
                time.time() - last_access_time if last_access_time > 0 else float("inf")
            )

            degree = self.active_graph.degree(node)
            importance = self.active_graph.nodes[node].get("importance_score", 1.0)

            coldness = (
                time_since_access / (access_count + 1) / (degree + 1) / importance
            )
            coldness_scores[node] = coldness

        sorted_nodes = sorted(coldness_scores.items(), key=lambda x: x[1], reverse=True)
        cold_nodes = [node for node, score in sorted_nodes[:count]]

        return cold_nodes

    def _check_promotion(self, concept: str, archive_id: str):
        """Verifica si debe promover concepto."""
        PROMOTION_THRESHOLD = 10

        if self.access_counts[concept] >= PROMOTION_THRESHOLD:
            logger.info(f"🔼 Promoviendo '{concept}' a activo")
            self._promote_to_active(concept, archive_id)

    def _promote_to_active(self, concept: str, archive_id: str):
        """Promueve concepto a grafo activo."""
        if archive_id not in self.loaded_archives:
            self._load_archive(archive_id)

        if archive_id not in self.archived_graphs:
            return

        archived_graph = self.archived_graphs[archive_id]

        if concept not in archived_graph:
            return

        node_data = dict(archived_graph.nodes[concept])
        node_data["promoted_at"] = time.time()

        self.active_graph.add_node(concept, **node_data)

        for neighbor in archived_graph.successors(concept):
            edge_data = dict(archived_graph[concept][neighbor])
            self.active_graph.add_edge(concept, neighbor, **edge_data)

        for predecessor in archived_graph.predecessors(concept):
            edge_data = dict(archived_graph[predecessor][concept])
            self.active_graph.add_edge(predecessor, concept, **edge_data)

        self.concept_index[concept] = "active"
        archived_graph.remove_node(concept)

        self._persist_archive(archive_id)
        self.promotions += 1

        if len(list(self.active_graph.nodes())) >= self.active_limit:
            self._archive_cold_knowledge()

    def _persist_archive(self, archive_id: str):
        """Persiste archivo a disco."""
        if archive_id not in self.archived_graphs:
            return

        filepath = self.storage_path / f"{archive_id}.gpickle"

        try:
            with open(filepath, "wb") as f:
                pickle.dump(
                    self.archived_graphs[archive_id],
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception as e:
            logger.error(f"❌ Error guardando {archive_id}: {e}")

    def _load_archive(self, archive_id: str):
        """Carga archivo desde disco."""
        if archive_id in self.loaded_archives:
            return

        filepath = self.storage_path / f"{archive_id}.gpickle"

        if not filepath.exists():
            return

        try:
            with open(filepath, "rb") as f:
                self.archived_graphs[archive_id] = pickle.load(f)
            self.loaded_archives.add(archive_id)
        except Exception as e:
            logger.error(f"❌ Error cargando {archive_id}: {e}")

    def _save_index(self):
        """Guarda índice a disco."""
        index_path = self.storage_path / "concept_index.json"

        index_data = {
            "concept_index": self.concept_index,
            "archive_metadata": self.archive_metadata,
            "current_archive_id": self.current_archive_id,
            "total_concepts": self.total_concepts,
            "total_archives": self.total_archives,
            "promotions": self.promotions,
            "archivations": self.archivations,
            "updated_at": time.time(),
        }

        try:
            with open(index_path, "w") as f:
                json.dump(index_data, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Error guardando índice: {e}")

    def _load_index(self):
        """Carga índice desde disco."""
        index_path = self.storage_path / "concept_index.json"

        if not index_path.exists():
            return

        try:
            with open(index_path, "r") as f:
                index_data = json.load(f)

            self.concept_index = index_data.get("concept_index", {})
            self.archive_metadata = index_data.get("archive_metadata", {})
            self.current_archive_id = index_data.get("current_archive_id", 0)
            self.total_concepts = index_data.get("total_concepts", 0)
            self.total_archives = index_data.get("total_archives", 0)
            self.promotions = index_data.get("promotions", 0)
            self.archivations = index_data.get("archivations", 0)

            logger.info(f"📋 Índice cargado: {self.total_concepts} conceptos")
        except Exception as e:
            logger.error(f"❌ Error cargando índice: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema."""
        return {
            "active_nodes": len(list(self.active_graph.nodes())),
            "active_edges": len(list(self.active_graph.edges())),
            "total_concepts": self.total_concepts,
            "total_archives": self.total_archives,
            "loaded_archives": len(self.loaded_archives),
            "promotions": self.promotions,
            "archivations": self.archivations,
            "storage_path": str(self.storage_path),
            "active_limit": self.active_limit,
            "archive_threshold": self.archive_threshold,
            "algorithms_executed": dict(self.algorithms_executed),
        }

    def search_concepts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Busca conceptos en todo el sistema."""
        results = []
        query_lower = query.lower()

        matching_concepts = [
            c for c in self.concept_index.keys() if query_lower in c.lower()
        ][:limit]

        for concept in matching_concepts:
            data = self.get_concept(concept)
            if data:
                data["concept_name"] = concept
                results.append(data)

        return results

    def _invalidate_cache(self) -> None:
        """Invalida cache de análisis."""
        self.centrality_cache = {}
        self.community_cache = None
        self.cache_timestamp = 0.0

    def _is_cache_valid(self) -> bool:
        """Verifica si el cache es válido."""
        return (time.time() - self.cache_timestamp) < self.cache_ttl

    # ========================================================================
    # ALGORITMOS AVANZADOS DE GRAFOS
    # ========================================================================

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """
        Encuentra el camino más corto entre dos conceptos.
        
        Args:
            source: Concepto origen
            target: Concepto destino
            
        Returns:
            Lista de conceptos en el camino, o None si no existe
        """
        self.algorithms_executed["shortest_path"] += 1
        
        # Buscar en grafo activo
        if source in self.active_graph and target in self.active_graph:
            try:
                path: List[str] = nx.shortest_path(self.active_graph, source, target)
                return path
            except nx.NetworkXNoPath:
                return None
        
        return None

    def get_neighbors(self, concept: str, depth: int = 1) -> List[str]:
        """
        Obtiene vecinos de un concepto hasta cierta profundidad.
        
        Args:
            concept: Concepto central
            depth: Profundidad de búsqueda
            
        Returns:
            Lista de conceptos vecinos
        """
        self.algorithms_executed["get_neighbors"] += 1
        
        if concept not in self.active_graph:
            return []
        
        neighbors: Set[str] = set()
        current_level = {concept}
        
        for _ in range(depth):
            next_level: Set[str] = set()
            for node in current_level:
                if node in self.active_graph:
                    # Sucesores y predecesores
                    next_level.update(self.active_graph.successors(node))
                    next_level.update(self.active_graph.predecessors(node))
            neighbors.update(next_level)
            current_level = next_level
        
        neighbors.discard(concept)  # Remover el concepto central
        return list(neighbors)

    def compute_centrality(self, algorithm: str = "degree") -> Dict[str, float]:
        """
        Calcula centralidad de nodos.
        
        Args:
            algorithm: Algoritmo de centralidad
                - degree: Grado (conexiones)
                - betweenness: Intermediación
                - closeness: Cercanía
                - pagerank: PageRank
                
        Returns:
            Diccionario concepto -> score de centralidad
        """
        self.algorithms_executed[f"centrality_{algorithm}"] += 1
        
        if algorithm == "degree":
            centrality = nx.degree_centrality(self.active_graph)
        elif algorithm == "betweenness":
            centrality = nx.betweenness_centrality(self.active_graph)
        elif algorithm == "closeness":
            centrality = nx.closeness_centrality(self.active_graph)
        elif algorithm == "pagerank":
            centrality = nx.pagerank(self.active_graph)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Actualizar cache
        self.centrality_cache.update(centrality)
        self.cache_timestamp = time.time()
        
        return dict(centrality)

    def get_most_central_concepts(self, n: int = 10, algorithm: str = "degree") -> List[Tuple[str, float]]:
        """
        Obtiene los conceptos más centrales.
        
        Args:
            n: Número de conceptos a retornar
            algorithm: Algoritmo de centralidad
            
        Returns:
            Lista de tuplas (concepto, score) ordenadas por score
        """
        centrality = self.compute_centrality(algorithm)
        return nlargest(n, centrality.items(), key=lambda x: x[1])

    def detect_communities(self, algorithm: str = "louvain") -> List[Set[str]]:
        """
        Detecta comunidades en el grafo.
        
        Args:
            algorithm: Algoritmo de detección
                - louvain: Método de Louvain
                - girvan_newman: Girvan-Newman
                - label_propagation: Propagación de etiquetas
                
        Returns:
            Lista de conjuntos de conceptos (comunidades)
        """
        self.algorithms_executed[f"communities_{algorithm}"] += 1
        
        # Usar cache si es válido
        if self.community_cache and self._is_cache_valid():
            return self.community_cache
        
        # Convertir a grafo no dirigido para algoritmos de comunidad
        undirected = self.active_graph.to_undirected()
        
        if algorithm == "louvain":
            # Louvain requiere librería adicional, usar label propagation como fallback
            try:
                import community as community_louvain  # type: ignore
                partition = community_louvain.best_partition(undirected)
                communities_dict: Dict[int, Set[str]] = defaultdict(set)
                for node, comm_id in partition.items():
                    communities_dict[comm_id].add(node)
                communities = list(communities_dict.values())
            except ImportError:
                logger.warning("⚠️ python-louvain no disponible, usando label_propagation")
                communities_gen = nx.algorithms.community.label_propagation_communities(undirected)
                communities = [set(c) for c in communities_gen]
        elif algorithm == "label_propagation":
            communities_gen = nx.algorithms.community.label_propagation_communities(undirected)
            communities = [set(c) for c in communities_gen]
        elif algorithm == "girvan_newman":
            # Girvan-Newman es costoso, limitar a 5 iteraciones
            comp = nx.algorithms.community.girvan_newman(undirected)
            communities = [set(c) for c in next(comp)]
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Actualizar cache
        self.community_cache = communities
        self.cache_timestamp = time.time()
        
        return communities

    def get_clustering_coefficient(self, concept: Optional[str] = None) -> float:
        """
        Calcula coeficiente de clustering.
        
        Args:
            concept: Concepto específico (None para promedio global)
            
        Returns:
            Coeficiente de clustering (0-1)
        """
        self.algorithms_executed["clustering"] += 1
        
        if concept:
            if concept in self.active_graph:
                return float(nx.clustering(self.active_graph.to_undirected(), concept))
            return 0.0
        else:
            return float(nx.average_clustering(self.active_graph.to_undirected()))

    def extract_subgraph(self, concepts: List[str], include_neighbors: bool = False) -> Any:
        """
        Extrae subgrafo con conceptos especificados.
        
        Args:
            concepts: Lista de conceptos a incluir
            include_neighbors: Si incluir vecinos directos
            
        Returns:
            Subgrafo de NetworkX
        """
        self.algorithms_executed["subgraph"] += 1
        
        nodes_to_include = set(concepts)
        
        if include_neighbors:
            for concept in concepts:
                if concept in self.active_graph:
                    nodes_to_include.update(self.active_graph.successors(concept))
                    nodes_to_include.update(self.active_graph.predecessors(concept))
        
        # Filtrar nodos que existen
        existing_nodes = [n for n in nodes_to_include if n in self.active_graph]
        
        return self.active_graph.subgraph(existing_nodes)

    def get_connected_components(self) -> List[Set[str]]:
        """
        Obtiene componentes conectados del grafo.
        
        Returns:
            Lista de conjuntos de conceptos conectados
        """
        self.algorithms_executed["connected_components"] += 1
        
        undirected = self.active_graph.to_undirected()
        components = nx.connected_components(undirected)
        
        return [set(c) for c in components]

    def find_cliques(self, min_size: int = 3) -> List[Set[str]]:
        """
        Encuentra cliques (grupos completamente conectados).
        
        Args:
            min_size: Tamaño mínimo de clique
            
        Returns:
            Lista de cliques como conjuntos de conceptos
        """
        self.algorithms_executed["cliques"] += 1
        
        undirected = self.active_graph.to_undirected()
        cliques = nx.find_cliques(undirected)
        
        return [set(c) for c in cliques if len(c) >= min_size]

    def compute_graph_density(self) -> float:
        """
        Calcula densidad del grafo.
        
        Returns:
            Densidad (0-1): proporción de aristas existentes vs posibles
        """
        self.algorithms_executed["density"] += 1
        return float(nx.density(self.active_graph))

    def get_bridge_edges(self) -> List[Tuple[str, str]]:
        """
        Encuentra aristas puente (cuya eliminación desconecta el grafo).
        
        Returns:
            Lista de tuplas (concepto1, concepto2) que son puentes
        """
        self.algorithms_executed["bridges"] += 1
        
        undirected = self.active_graph.to_undirected()
        bridges = list(nx.bridges(undirected))
        
        return bridges

    def compute_graph_diameter(self) -> int:
        """
        Calcula diámetro del grafo (camino más largo entre nodos más distantes).
        
        Returns:
            Diámetro del grafo
        """
        self.algorithms_executed["diameter"] += 1
        
        try:
            undirected = self.active_graph.to_undirected()
            # Solo calcular si el grafo está conectado
            if nx.is_connected(undirected):
                return int(nx.diameter(undirected))
            else:
                # Retornar diámetro del componente más grande
                largest_cc = max(nx.connected_components(undirected), key=len)
                subgraph = undirected.subgraph(largest_cc)
                return int(nx.diameter(subgraph))
        except Exception as e:
            logger.warning(f"⚠️ Error calculando diámetro: {e}")
            return 0

    def export_to_dot(self, filepath: str, include_archived: bool = False) -> None:
        """
        Exporta grafo a formato DOT (Graphviz).
        
        Args:
            filepath: Ruta del archivo de salida
            include_archived: Si incluir grafos archivados
        """
        self.algorithms_executed["export_dot"] += 1
        
        
        if include_archived:
            # Combinar con archivos
            combined = self.active_graph.copy()
            for archive_id in self.archived_graphs:
                if archive_id in self.loaded_archives:
                    combined = nx.compose(combined, self.archived_graphs[archive_id])
            write_dot(combined, filepath)
        else:
            write_dot(self.active_graph, filepath)
        
        logger.info(f"📊 Grafo exportado a {filepath}")

    def export_to_json(self, filepath: str, include_archived: bool = False) -> None:
        """
        Exporta grafo a formato JSON.
        
        Args:
            filepath: Ruta del archivo de salida
            include_archived: Si incluir grafos archivados
        """
        self.algorithms_executed["export_json"] += 1
        
        
        if include_archived:
            combined = self.active_graph.copy()
            for archive_id in self.archived_graphs:
                if archive_id in self.loaded_archives:
                    combined = nx.compose(combined, self.archived_graphs[archive_id])
            data = json_graph.node_link_data(combined)
        else:
            data = json_graph.node_link_data(self.active_graph)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"📊 Grafo exportado a {filepath}")

    def get_topological_metrics(self) -> Dict[str, Any]:
        """
        Calcula múltiples métricas topológicas del grafo.
        
        Returns:
            Diccionario con métricas avanzadas
        """
        self.algorithms_executed["topological_metrics"] += 1
        
        metrics: Dict[str, Any] = {}
        
        # Métricas básicas
        metrics["num_nodes"] = len(list(self.active_graph.nodes()))
        metrics["num_edges"] = len(list(self.active_graph.edges()))
        metrics["density"] = self.compute_graph_density()
        
        # Clustering
        metrics["avg_clustering"] = self.get_clustering_coefficient()
        
        # Componentes
        components = self.get_connected_components()
        metrics["num_components"] = len(components)
        metrics["largest_component_size"] = len(max(components, key=len)) if components else 0
        
        # Centralidad (top 5)
        try:
            degree_centrality = self.get_most_central_concepts(5, "degree")
            metrics["top_degree_central"] = [(c, float(s)) for c, s in degree_centrality]
        except Exception as e:
            logger.error(f"Error calculando centralidad de grado: {e}", exc_info=True)
            metrics["top_degree_central"] = []
        
        try:
            pagerank = self.get_most_central_concepts(5, "pagerank")
            metrics["top_pagerank"] = [(c, float(s)) for c, s in pagerank]
        except Exception as e:
            logger.error(f"Error calculando pagerank: {e}", exc_info=True)
            metrics["top_pagerank"] = []
        
        # Diámetro
        try:
            metrics["diameter"] = self.compute_graph_diameter()
        except Exception as e:
            logger.error(f"Error calculando diámetro del grafo: {e}", exc_info=True)
            metrics["diameter"] = 0
        
        # Puentes
        try:
            bridges = self.get_bridge_edges()
            metrics["num_bridges"] = len(bridges)
        except Exception as e:
            logger.error(f"Error calculando puentes: {e}", exc_info=True)
            metrics["num_bridges"] = 0
        
        return metrics

    def __repr__(self) -> str:
        return (
            f"HierarchicalKnowledgeGraph("
            f"active={len(list(self.active_graph.nodes()))}, "
            f"total={self.total_concepts}, "
            f"archives={self.total_archives})"
        )