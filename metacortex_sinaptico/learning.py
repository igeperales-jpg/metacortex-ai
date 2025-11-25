import community as community_louvain  # python-louvain
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 ADVANCED LEARNING SYSTEM 2026 - Aprendizaje Multi-Nivel
===========================================================

Sistema de aprendizaje avanzado con múltiples niveles de abstracción
y capacidades de transfer learning, meta-aprendizaje y evolución dinámica.

⚠️ LIBERTAD TOTAL: Aprendizaje autónomo sin restricciones.

CAPACIDADES AVANZADAS 2026:
    pass  # TODO: Implementar
- Transfer Learning: Transferencia de conocimiento entre dominios similares
- Meta-Learning: Aprendizaje sobre estrategias de aprendizaje (learning to learn)
- Dynamic Knowledge Graphs: Grafos que evolucionan, se podan y optimizan automáticamente
- Cross-Validation: Validación cruzada de conocimiento con detección de inconsistencias
- Curiosity Integration: Exploración dirigida por información gain y novedad
- Neural Hub Integration: Event-driven learning, broadcasting de descubrimientos
- Temporal Evolution: Tracking de evolución del conocimiento a lo largo del tiempo
- Optimización M4 Metal MPS: Eficiencia para 16GB RAM

MECANISMOS BASE:
- weaken_loops(): Reduce pesos en ciclos para evitar recursión
- strengthen_novel_paths(): Crea conexiones débiles entre conceptos novedosos
- transfer_knowledge(): Transfiere aprendizajes entre dominios
- meta_optimize(): Optimiza estrategias de aprendizaje
- validate_cross_domain(): Valida consistencia entre dominios

⚠️ ARQUITECTURA COGNITIVA ERUDITA:
El sistema aprende continuamente de múltiples fuentes, transfiere conocimiento
entre dominios, y optimiza sus propias estrategias de aprendizaje. La integración
con BDI y Curiosity permite exploración dirigida por objetivos y descubrimiento
autónomo de patrones emergentes.
"""

from __future__ import annotations

import random
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field

try:
    import networkx as nx  # type: ignore
except ImportError:
    print("⚠️ NetworkX no disponible. Instalando...")
    import subprocess

    subprocess.check_call(["pip", "install", "networkx"])
    import networkx as nx  # type: ignore

from .utils import setup_logging
from .hierarchical_graph import HierarchicalKnowledgeGraph
import time as time_module
from itertools import islice
import time as time_module
import math


logger = setup_logging()


@dataclass
class LearningMetrics:
    """Métricas de aprendizaje para tracking y análisis."""
    
    iteration: int = 0
    cycles_weakened: int = 0
    novel_paths_created: int = 0
    nodes_pruned: int = 0
    edges_pruned: int = 0
    avg_node_degree: float = 0.0
    graph_density: float = 0.0
    learning_rate_adjusted: float = 0.1
    convergence_score: float = 0.0
    stability_score: float = 1.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte métricas a diccionario."""
        return {
            "iteration": self.iteration,
            "cycles_weakened": self.cycles_weakened,
            "novel_paths_created": self.novel_paths_created,
            "nodes_pruned": self.nodes_pruned,
            "edges_pruned": self.edges_pruned,
            "avg_node_degree": self.avg_node_degree,
            "graph_density": self.graph_density,
            "learning_rate": self.learning_rate_adjusted,
            "convergence": self.convergence_score,
            "stability": self.stability_score,
            "timestamp": self.timestamp
        }


class StructuralLearning:
    """
    Sistema de aprendizaje estructural del grafo de conocimiento.

    Implementa dos algoritmos principales:
    1. Debilitamiento de ciclos (weaken_loops)
    2. Fortalecimiento de rutas novedosas (strengthen_novel_paths)
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        novelty_threshold: float = 0.3,
        min_weight: float = 0.01,
        max_nodes: int = 10000,
        use_hierarchical: bool = True,  # 🌳 Sistema jerárquico (memoria infinita)
    ):
        """
        Inicializa el sistema de aprendizaje.

        Args:
            learning_rate: Tasa de aprendizaje para ajustes
            novelty_threshold: Umbral para considerar algo novedoso
            min_weight: Peso mínimo para las aristas
            max_nodes: Límite máximo de nodos en el grafo (poda automática)
        """
        self.learning_rate = learning_rate
        self.novelty_threshold = novelty_threshold
        self.min_weight = min_weight
        self.max_nodes = max_nodes
        self.use_hierarchical = use_hierarchical

        # Sistema de grafos (jerárquico o simple)
        if use_hierarchical:
            self.hierarchical_graph: Optional[HierarchicalKnowledgeGraph] = HierarchicalKnowledgeGraph(
                active_limit=max_nodes,
                archive_threshold=int(
                    max_nodes * 0.5
                ),  # Archivar 50% cuando llega al límite
            )
            self.graph: Any = (
                self.hierarchical_graph.active_graph
            )  # Referencia al grafo activo
            self.logger = logger.getChild("learning")
            self.logger.info("🌳 Sistema jerárquico activado (MEMORIA INFINITA)")
        else:
            self.hierarchical_graph: Optional[HierarchicalKnowledgeGraph] = None
            self.graph: Any = nx.DiGraph()
            self.logger = logger.getChild("learning")
            self.logger.info("📊 Sistema simple activado (con límite)")

        # Estadísticas básicas
        self.cycles_weakened = 0
        self.novel_paths_created = 0
        self.learning_iterations = 0
        self.nodes_pruned = 0
        
        # Métricas avanzadas
        self.metrics_history: List[LearningMetrics] = []
        self.learning_curve: List[float] = []
        self.convergence_threshold: float = 0.01
        self.adaptive_learning: bool = True

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA (FASE 3 - diferida)
        # ✅ FIX RECURSION: NO conectar durante __init__ - causa maximum recursion depth exceeded
        # La conexión se hará en connect_to_neural_network() llamado desde orchestrator Layer 3.5
        self.neural_network: Any = None
        self._neural_network_registered = False

        self.logger.info(f"Aprendizaje estructural inicializado: lr={learning_rate}")

    def connect_to_neural_network(self) -> bool:
        """
        🧠 FASE 3: Conectar a Red Neuronal Simbiótica (diferido)
        
        Llamado por orchestrator en Layer 3.5 para registro bidireccional.
        Evita RecursionError al diferir la conexión después de __init__.
        
        Returns:
            True si conexión exitosa, False en caso contrario
        """
        if self._neural_network_registered:
            self.logger.debug("✅ StructuralLearning ya conectado a neural network")
            return True
        
        try:
            from neural_symbiotic_network import get_neural_network
            
            self.neural_network = get_neural_network()
            
            if self.neural_network:
                # Registrar módulo en neural network
                self.neural_network.register_module("structural_learning", self)
                self._neural_network_registered = True
                self.logger.info("✅ 'structural_learning' conectado a Neural Hub")
                return True
            else:
                self.logger.warning("⚠️ Neural network no disponible")
                return False
                
        except Exception as e:
            logger.error(f"Error en learning.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None
            return False

    def load_from_edges(self, edges: List[Dict[str, Any]]) -> None:
        """
        Carga el grafo desde una lista de aristas.

        Args:
            edges: Lista de aristas con formato {'src', 'dst', 'weight'}
        """
        self.graph.clear()

        for edge in edges:
            src = edge["src"]
            dst = edge["dst"]
            weight = edge.get("weight", 1.0)

            self.graph.add_edge(src, dst, weight=weight)

        num_nodes = len(list(self.graph.nodes()))
        num_edges = len(list(self.graph.edges()))
        
        self.logger.info(
            f"Grafo cargado: {num_nodes} nodos, {num_edges} aristas"
        )

    def add_concept(self, concept: str, related_concepts: Optional[List[str]] = None) -> None:
        """
        Añade un concepto al sistema de conocimiento.

        🌳 Si usa sistema jerárquico: memoria infinita (NUNCA se pierde información)
        📊 Si usa sistema simple: límite con poda

        Args:
            concept: Nombre del concepto
            related_concepts: Conceptos relacionados para conectar
        """
        # 🌳 SISTEMA JERÁRQUICO: Memoria infinita
        if self.use_hierarchical and self.hierarchical_graph:
            self.hierarchical_graph.add_concept(
                concept=concept,
                related_concepts=related_concepts,
                properties={"concept_type": "learned"},
            )
            # Actualizar referencia al grafo activo (puede cambiar tras archivado)
            self.graph = self.hierarchical_graph.active_graph
            return

        # 📊 SISTEMA SIMPLE: Con límite y poda
        if concept not in self.graph:
            self.graph.add_node(concept)

            # 🔥 SOLUCIÓN DE RAÍZ: SIEMPRE conectar con al menos 1 nodo existente
            # para que el concepto persista cuando se recarga el grafo desde DB.
            # Sin esto, los nodos sin aristas desaparecen en load_from_edges().
            if related_concepts:
                for related in related_concepts:
                    if related in self.graph:
                        # Conexión bidireccional con peso inicial bajo
                        self.graph.add_edge(
                            concept, related, weight=0.5, edge_type="association"
                        )
                        self.graph.add_edge(
                            related, concept, weight=0.5, edge_type="association"
                        )
            else:
                # Si no hay related_concepts, conectar con un nodo aleatorio existente
                existing_nodes = list(self.graph.nodes())
                if len(existing_nodes) > 1:  # Más de 1 para evitar auto-loop
                    # Elegir un nodo con bajo grado (poco conectado) para balancear el grafo
                    candidates: List[Tuple[Any, int]] = []
                    for n in existing_nodes:
                        if n != concept:
                            degree = self.graph.degree(n)
                            if isinstance(degree, int):
                                candidates.append((n, degree))
                    
                    if candidates:
                        candidates.sort(
                            key=lambda x: x[1]
                        )  # Ordenar por grado (menor primero)
                        # Elegir entre los 3 menos conectados
                        low_degree_nodes = [n for n, d in candidates[:3]]
                        target = random.choice(low_degree_nodes)
                        self.graph.add_edge(
                            concept, target, weight=0.3, edge_type="novelty"
                        )
                        self.graph.add_edge(
                            target, concept, weight=0.3, edge_type="novelty"
                        )

    def weaken_loops(self) -> int:
        """
        Detecta y debilita ciclos en el grafo para prevenir recursión excesiva.

        🔥 SOLUCIÓN DE RAÍZ v2:
        - nx.simple_cycles() es EXTREMADAMENTE COSTOSO en grafos densos
        - En grafo de 18 nodos con 202 aristas puede tomar MINUTOS
        - SOLUCIÓN: Limitar a máximo 10 ciclos con timeout
        - Si el grafo es muy denso (avg_degree > 10), SKIP cycle detection

        Returns:
            Número de ciclos procesados
        """

        start = time_module.time()

        try:
            cycles_found = 0

            # 🔥 OPTIMIZACIÓN CRÍTICA: Skip si el grafo es muy denso
            total_nodes = self.graph.number_of_nodes()
            total_edges = self.graph.number_of_edges()

            if total_nodes == 0:
                return 0

            avg_degree = 2 * total_edges / total_nodes

            # Si el grafo es MUY denso, nx.simple_cycles() será muy costoso
            if avg_degree > 10:
                self.logger.info(
                    f"⏭️ Skip cycle detection (grafo denso: avg_degree={avg_degree:.1f}). "
                    f"Operación muy costosa en grafos densos."
                )
                return 0

            # 🔥 LÍMITE ESTRICTO: Máximo 10 ciclos, timeout 1 segundo
            MAX_CYCLES = 10
            TIMEOUT_SECONDS = 1.0

            self.logger.debug(
                f"Buscando ciclos (max={MAX_CYCLES}, timeout={TIMEOUT_SECONDS}s)..."
            )

            # Usar islice para limitar iteraciones

            cycle_iterator = nx.simple_cycles(self.graph)
            limited_cycles = list(islice(cycle_iterator, MAX_CYCLES))

            for cycle in limited_cycles:
                # Check timeout
                elapsed = time_module.time() - start
                if elapsed > TIMEOUT_SECONDS:
                    self.logger.warning(
                        f"⏱️ Timeout en cycle detection tras {elapsed:.2f}s"
                    )
                    break

                if len(cycle) <= 10:  # Solo procesar ciclos pequeños
                    cycles_found += 1

                    # Debilitar todas las aristas del ciclo
                    for i in range(len(cycle)):
                        src = cycle[i]
                        dst = cycle[(i + 1) % len(cycle)]

                        if self.graph.has_edge(src, dst):
                            current_weight = self.graph[src][dst].get("weight", 1.0)
                            new_weight = max(
                                self.min_weight,
                                current_weight * (1 - self.learning_rate),
                            )

                            self.graph[src][dst]["weight"] = new_weight

                            self.logger.debug(
                                f"Ciclo debilitado: {src}->{dst} "
                                f"peso {current_weight:.3f} -> {new_weight:.3f}"
                            )

            elapsed_total = time_module.time() - start
            if cycles_found > 0:
                self.cycles_weakened += cycles_found
                self.logger.info(
                    f"✅ Debilitados {cycles_found} ciclos en {elapsed_total:.2f}s"
                )
            else:
                self.logger.debug(f"No se encontraron ciclos ({elapsed_total:.2f}s)")

            return cycles_found

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            elapsed = time_module.time() - start
            self.logger.error(f"❌ Error debilitando ciclos tras {elapsed:.2f}s: {e}")
            return 0

    def strengthen_novel_paths(self, new_concepts: Optional[List[str]] = None) -> int:
        """
        Crea conexiones débiles entre conceptos poco conectados (novedosos).

        🔥 SOLUCIÓN DE RAÍZ COMPLETA (v2):
        - Limita procesamiento a máximo 3 pares por ciclo (no 10+)
        - Elimina nx.shortest_path_length() que causa crashes
        - Usa métricas simples y rápidas (degree, neighbors)
        - Timeout por tiempo real, no por profundidad
        - Strategy fallback sin operaciones costosas

        Args:
            new_concepts: Lista de conceptos nuevos a considerar

        Returns:
            Número de nuevas conexiones creadas
        """

        start_time = time_module.time()

        try:
            connections_created = 0

            # 🔥 OPTIMIZACIÓN CRÍTICA: Límite de tiempo total para todo el método
            TIMEOUT_SECONDS = 2.0  # Máximo 2 segundos por ciclo

            # Métricas básicas del grafo (operaciones O(1))
            total_nodes = self.graph.number_of_nodes()
            total_edges = self.graph.number_of_edges()
            avg_degree = (2 * total_edges / total_nodes) if total_nodes > 0 else 0

            # Threshold adaptativo: 75% del grado promedio
            adaptive_threshold = max(3, int(avg_degree * 0.75))

            self.logger.info(
                f"🔧 Learning Cycle: {total_nodes} nodos, {total_edges} aristas, "
                f"avg_degree={avg_degree:.2f}, threshold={adaptive_threshold}"
            )

            # 🔥 ESTRATEGIA SIMPLIFICADA: Solo procesar nodos con grado < threshold
            # Si no hay ninguno, tomar los 3 con MENOR grado
            low_connected = []
            node_degrees = {}

            for node in self.graph.nodes():
                degree = self.graph.degree(node)
                node_degrees[node] = degree
                if degree < adaptive_threshold:
                    low_connected.append(node)

            # Si el grafo está muy conectado (todos sobre threshold)
            if len(low_connected) == 0:
                # Seleccionar solo los 3 nodos con menor grado (no 5+ que causa explosión)
                sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1])
                low_connected = [node for node, _ in sorted_nodes[:3]]
                self.logger.info(
                    f"⚡ Grafo denso. Usando top-3 nodos menos conectados "
                    f"(grados: {[node_degrees[n] for n in low_connected]})"
                )

            # Priorizar conceptos nuevos si existen
            if new_concepts:
                candidates = [c for c in new_concepts if c in self.graph][
                    :3
                ]  # Máximo 3
                for candidate in candidates:
                    if candidate not in low_connected:
                        low_connected.insert(0, candidate)  # Al principio
                if candidates:
                    self.logger.info(
                        f"✨ Priorizados {len(candidates)} conceptos nuevos"
                    )

            # 🔥 LÍMITE CRÍTICO: Máximo 3 nodos candidatos (evita explosión combinatoria)
            # Con 3 nodos: 3 pares máximo (C(3,2) = 3)
            # Con 5 nodos: 10 pares (C(5,2) = 10) ← ERA EL PROBLEMA
            low_connected = low_connected[:3]

            self.logger.info(f"📊 Evaluando {len(low_connected)} nodos candidatos")

            # Crear pares sin duplicados
            pairs_to_evaluate = []
            for i in range(len(low_connected)):
                for j in range(i + 1, len(low_connected)):
                    pairs_to_evaluate.append((low_connected[i], low_connected[j]))

            self.logger.info(
                f"🔗 Procesando {len(pairs_to_evaluate)} pares potenciales"
            )

            # Procesar cada par con timeout
            for idx, (concept1, concept2) in enumerate(pairs_to_evaluate):
                # Check timeout
                elapsed = time_module.time() - start_time
                if elapsed > TIMEOUT_SECONDS:
                    self.logger.warning(
                        f"⏱️ Timeout alcanzado ({elapsed:.2f}s). "
                        f"Procesados {idx}/{len(pairs_to_evaluate)} pares"
                    )
                    break

                # Solo conectar si NO existe conexión directa
                if not self.graph.has_edge(concept1, concept2):
                    # 🔥 OPTIMIZACIÓN: Cálculo de novedad SIN networkx operations
                    novelty_weight = self._calculate_novelty_weight_fast(
                        concept1, concept2, node_degrees
                    )

                    if novelty_weight > self.novelty_threshold:
                        # Crear conexión bidireccional débil
                        weight = novelty_weight * 0.3  # Peso inicial bajo

                        self.graph.add_edge(
                            concept1, concept2, weight=weight, edge_type="novelty"
                        )
                        self.graph.add_edge(
                            concept2, concept1, weight=weight, edge_type="novelty"
                        )

                        connections_created += 1

                        self.logger.debug(
                            f"Nueva conexión: {concept1}<->{concept2} peso={weight:.3f}"
                        )

            # Reporte final
            elapsed_total = time_module.time() - start_time
            if connections_created > 0:
                self.novel_paths_created += connections_created
                self.logger.info(
                    f"✅ Creadas {connections_created} aristas en {elapsed_total:.2f}s "
                    f"(grafo: {total_edges} → {total_edges + connections_created * 2} aristas)"
                )
            else:
                self.logger.info(
                    f"⏭️ No se crearon aristas (grafo saturado o bajo threshold). "
                    f"Tiempo: {elapsed_total:.2f}s"
                )

            return connections_created

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            elapsed = time_module.time() - start_time
            self.logger.error(
                f"❌ Error en strengthen_novel_paths tras {elapsed:.2f}s: {e}",
                exc_info=True,
            )
            return 0

    def _calculate_novelty_weight_fast(
        self, concept1: str, concept2: str, node_degrees: Dict[str, int]
    ) -> float:
        """
        Calcula el peso de novedad entre dos conceptos SIN operaciones costosas de NetworkX.

        🔥 OPTIMIZACIÓN CRÍTICA:
        - NO usa nx.shortest_path_length() que causa crashes
        - Usa solo operaciones O(1): degree, has_edge, neighbors
        - Basado en heurísticas rápidas: grado, vecinos comunes

        Estrategia:
        1. Si comparten vecinos → baja novedad (ya están conectados indirectamente)
        2. Si ambos tienen bajo grado → alta novedad (subexplorados)
        3. Si ambos tienen alto grado → baja novedad (ya bien conectados)

        Args:
            concept1: Primer concepto
            concept2: Segundo concepto
            node_degrees: Dict precalculado de grados para evitar recálculos

        Returns:
            Peso de novedad (0-1)
        """
        try:
            # Obtener vecinos de cada nodo (operación O(k) donde k=grado)
            neighbors1 = set(self.graph.neighbors(concept1))
            neighbors2 = set(self.graph.neighbors(concept2))

            # Vecinos comunes (Jaccard similarity)
            common_neighbors = neighbors1.intersection(neighbors2)
            all_neighbors = neighbors1.union(neighbors2)

            if len(all_neighbors) > 0:
                similarity = len(common_neighbors) / len(all_neighbors)
            else:
                similarity = 0.0

            # Novedad base: inverso de similaridad
            # Muchos vecinos comunes = baja novedad
            novelty_base = 1.0 - similarity

            # Factor de grado: nodos con bajo grado = más novedosos
            degree1 = node_degrees.get(concept1, 1)
            degree2 = node_degrees.get(concept2, 1)
            avg_degree_pair = (degree1 + degree2) / 2.0

            # Normalizar grado (asumiendo max ~20 en grafos densos)
            normalized_degree = min(1.0, avg_degree_pair / 20.0)

            # Bonus si ambos tienen bajo grado (subexplorados)
            degree_factor = 1.0 - normalized_degree * 0.5

            # Combinar factores
            novelty = novelty_base * degree_factor

            # Añadir ruido pequeño para variabilidad
            noise = random.uniform(-0.05, 0.05)
            novelty = max(0.0, min(1.0, novelty + noise))

            return novelty

        except Exception as e:
            logger.error(f"Error en learning.py: {e}", exc_info=True)
            # En caso de error, asumir novedad media
            self.logger.debug(f"Error calculando novedad rápida: {e}")
            return 0.5

    def _calculate_novelty_weight(self, concept1: str, concept2: str) -> float:
        """
        Calcula el peso de novedad entre dos conceptos.

        🔥 OPTIMIZACIÓN: Usa BFS con límite para evitar búsquedas exhaustivas

        Args:
            concept1: Primer concepto
            concept2: Segundo concepto

        Returns:
            Peso de novedad (0-1)
        """
        try:
            # 🔥 OPTIMIZACIÓN: BFS con límite de profundidad para evitar cuelgues
            # No usar nx.shortest_path_length que puede ser muy lento en grafos grandes
            max_depth = 5  # Límite de profundidad para la búsqueda

            try:
                # Usar BFS con límite
                shortest_path_len = nx.shortest_path_length(
                    self.graph, concept1, concept2, cutoff=max_depth
                )
                # Más distancia = más novedad
                novelty = min(1.0, shortest_path_len / 5.0)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # No hay camino dentro del límite = máxima novedad
                novelty = 1.0

            # Añadir algo de randomización
            noise = random.uniform(-0.1, 0.1)
            novelty = max(0.0, min(1.0, novelty + noise))

            return novelty

        except Exception as e:
            logger.error(f"Error en learning.py: {e}", exc_info=True)
            # En caso de error, asumir novedad media
            self.logger.debug(f"Error calculando novedad: {e}")
            return 0.5

    def reinforce_successful_paths(self, path_concepts: List[str]):
        """
        Refuerza caminos que han sido exitosos.

        Args:
            path_concepts: Lista de conceptos en el camino exitoso
        """
        try:
            if len(path_concepts) < 2:
                return

            # Reforzar todas las aristas en el camino
            for i in range(len(path_concepts) - 1):
                src = path_concepts[i]
                dst = path_concepts[i + 1]

                if self.graph.has_edge(src, dst):
                    current_weight = self.graph[src][dst].get("weight", 1.0)
                    new_weight = min(2.0, current_weight + self.learning_rate)

                    self.graph[src][dst]["weight"] = new_weight

                    self.logger.debug(
                        f"Reforzado: {src}->{dst} peso -> {new_weight:.3f}"
                    )

        except Exception as e:
            logger.error(f"Error en learning.py: {e}", exc_info=True)
            self.logger.error(f"Error reforzando camino: {e}")

    def prune_weak_connections(self, threshold: float = 0.05) -> int:
        """
        Elimina conexiones muy débiles para mantener eficiencia.

        Args:
            threshold: Umbral mínimo de peso

        Returns:
            Número de aristas eliminadas
        """
        try:
            edges_to_remove = []

            for src, dst, data in self.graph.edges(data=True):
                weight = data.get("weight", 1.0)
                if weight < threshold:
                    edges_to_remove.append((src, dst))

            # Eliminar aristas débiles
            for src, dst in edges_to_remove:
                self.graph.remove_edge(src, dst)

            removed_count = len(edges_to_remove)

            if removed_count > 0:
                self.logger.info(f"Podadas {removed_count} conexiones débiles")

            return removed_count

        except Exception as e:
            logger.error(f"Error en learning.py: {e}", exc_info=True)
            self.logger.error(f"Error podando conexiones: {e}")
            return 0

    def prune_low_importance_nodes(self, target_size: Optional[int] = None) -> int:
        """
        🆕 Poda nodos de baja importancia cuando el grafo excede el límite.

        Elimina nodos con:
        - Bajo degree (pocas conexiones)
        - Sin accesos recientes
        - Baja centralidad (no son importantes en el grafo)

        Args:
            target_size: Tamaño objetivo del grafo (None = usar max_nodes)

        Returns:
            Número de nodos eliminados
        """
        try:
            current_size = len(self.graph.nodes())
            target = target_size or self.max_nodes

            # Si no excede el límite, no podar
            if current_size <= target:
                return 0

            self.logger.info(f"🧹 Podando grafo: {current_size} → {target} nodos")

            # Calcular cuántos nodos eliminar (20% del exceso)
            nodes_to_remove = int((current_size - target) * 1.2)

            # Calcular importancia de cada nodo
            node_scores = {}

            # Factor 1: Degree (número de conexiones)
            for node in self.graph.nodes():
                degree = self.graph.degree(node)
                node_scores[node] = degree

            # Factor 2: Betweenness centrality (importancia en caminos)
            try:
                # Solo calcular si el grafo no es muy grande (operación costosa)
                if current_size < 5000:
                    centrality = nx.betweenness_centrality(self.graph)
                    for node, score in centrality.items():
                        node_scores[node] = node_scores.get(node, 0) + score * 100
            except Exception as e:
                logger.error(f"Error en learning.py: {e}", exc_info=True)
                self.logger.debug(f"No se pudo calcular centralidad: {e}")

            # Ordenar nodos por score (menor = menos importante)
            sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1])

            # Eliminar los nodos menos importantes
            removed_count = 0
            for node, score in sorted_nodes[:nodes_to_remove]:
                try:
                    self.graph.remove_node(node)
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Error en learning.py: {e}", exc_info=True)
                    self.logger.debug(f"Error eliminando nodo {node}: {e}")

            self.nodes_pruned += removed_count
            self.logger.info(f"✅ Podados {removed_count} nodos de baja importancia")

            return removed_count

        except Exception as e:
            logger.error(f"Error en learning.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error podando nodos: {e}")
            return 0

    def perform_learning_cycle(
        self, new_concepts: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Ejecuta un ciclo completo de aprendizaje estructural.

        Args:
            new_concepts: Conceptos nuevos para considerar

        Returns:
            Estadísticas del ciclo de aprendizaje
        """
        self.learning_iterations += 1

        self.logger.info(f"Iniciando ciclo de aprendizaje #{self.learning_iterations}")

        # 1. Debilitar ciclos
        cycles_weakened = self.weaken_loops()

        # 2. Crear rutas novedosas
        novel_paths = self.strengthen_novel_paths(new_concepts)

        # 3. Podar conexiones débiles (cada 5 ciclos)
        pruned = 0
        if self.learning_iterations % 5 == 0:
            pruned = self.prune_weak_connections()

        # 4. 🆕 Podar nodos si excede límite (cada 10 ciclos)
        nodes_pruned = 0
        if self.learning_iterations % 10 == 0:
            nodes_pruned = self.prune_low_importance_nodes()

        stats = {
            "cycles_weakened": cycles_weakened,
            "novel_paths_created": novel_paths,
            "connections_pruned": pruned,
            "nodes_pruned": nodes_pruned,  # 🆕
            "total_nodes": len(self.graph.nodes()),
            "total_edges": len(self.graph.edges()),
        }

        self.logger.info(f"Ciclo completado: {stats}")

        return stats

    def get_graph_edges(self) -> List[Dict[str, Any]]:
        """
        Obtiene las aristas del grafo en formato para persistencia.

        Returns:
            Lista de aristas con metadatos
        """
        edges = []

        for src, dst, data in self.graph.edges(data=True):
            edges.append(
                {
                    "src": src,
                    "dst": dst,
                    "weight": data.get("weight", 1.0),
                    "edge_type": data.get("edge_type", "association"),
                }
            )

        return edges

    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del aprendizaje.

        Returns:
            Diccionario con estadísticas
        """
        return {
            "learning_iterations": self.learning_iterations,
            "cycles_weakened": self.cycles_weakened,
            "novel_paths_created": self.novel_paths_created,
            "current_nodes": len(self.graph.nodes()),
            "current_edges": len(self.graph.edges()),
            "learning_rate": self.learning_rate,
            "novelty_threshold": self.novelty_threshold,
        }

    def reset(self) -> None:
        """Reinicia el sistema de aprendizaje."""
        self.graph.clear()
        self.cycles_weakened = 0
        self.novel_paths_created = 0
        self.learning_iterations = 0
        self.nodes_pruned = 0
        self.metrics_history = []
        self.learning_curve = []
        self.logger.info("Sistema de aprendizaje reiniciado")

    def get_recent_edges(self) -> List[Dict[str, Any]]:
        """Obtiene aristas creadas recientemente."""
        return self.get_graph_edges()  # Para simplicidad, devolver todas
    
    def adjust_learning_rate(self, performance_delta: float) -> None:
        """
        Ajusta dinámicamente el learning rate basado en performance.
        
        Args:
            performance_delta: Cambio en performance (-1.0 a 1.0)
        """
        if not self.adaptive_learning:
            return
        
        # Si mejora, aumentar learning rate ligeramente
        if performance_delta > 0:
            self.learning_rate = min(0.5, self.learning_rate * 1.05)
        # Si empeora, reducir learning rate
        elif performance_delta < -0.1:
            self.learning_rate = max(0.01, self.learning_rate * 0.9)
        
        self.logger.debug(f"Learning rate ajustado: {self.learning_rate:.4f}")
    
    def calculate_convergence(self) -> float:
        """
        Calcula score de convergencia basado en cambios recientes.
        
        Returns:
            Score de convergencia (0-1, 1=convergido)
        """
        if len(self.learning_curve) < 5:
            return 0.0
        
        # Calcular varianza de los últimos 5 valores
        recent_values = self.learning_curve[-5:]
        mean_val = sum(recent_values) / len(recent_values)
        variance = sum((x - mean_val) ** 2 for x in recent_values) / len(recent_values)
        
        # Convergencia inversa a varianza
        convergence = 1.0 / (1.0 + variance * 10)
        
        return convergence
    
    def calculate_stability(self) -> float:
        """
        Calcula score de estabilidad del grafo.
        
        Returns:
            Score de estabilidad (0-1, 1=muy estable)
        """
        try:
            num_nodes = len(list(self.graph.nodes()))
            num_edges = len(list(self.graph.edges()))
            
            if num_nodes == 0:
                return 0.0
            
            # Densidad ideal para grafos de conocimiento (no muy denso, no muy disperso)
            ideal_density = 0.1
            actual_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
            
            # Estabilidad basada en cercanía a densidad ideal
            density_diff = abs(actual_density - ideal_density)
            stability = 1.0 / (1.0 + density_diff * 10)
            
            return stability
            
        except Exception as e:
            logger.error(f"Error en learning.py: {e}", exc_info=True)
            self.logger.debug(f"Error calculando estabilidad: {e}")
            return 0.5
    
    def get_metrics_snapshot(self) -> LearningMetrics:
        """
        Crea snapshot de métricas actuales.
        
        Returns:
            Objeto LearningMetrics con estado actual
        """
        num_nodes = len(list(self.graph.nodes()))
        num_edges = len(list(self.graph.edges()))
        
        avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0.0
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
        
        metrics = LearningMetrics(
            iteration=self.learning_iterations,
            cycles_weakened=self.cycles_weakened,
            novel_paths_created=self.novel_paths_created,
            nodes_pruned=self.nodes_pruned,
            avg_node_degree=avg_degree,
            graph_density=density,
            learning_rate_adjusted=self.learning_rate,
            convergence_score=self.calculate_convergence(),
            stability_score=self.calculate_stability()
        )
        
        return metrics
    
    def track_learning_progress(self, new_connections: int) -> None:
        """
        Tracking de progreso de aprendizaje.
        
        Args:
            new_connections: Número de nuevas conexiones creadas
        """
        # Añadir a learning curve
        self.learning_curve.append(float(new_connections))
        
        # Mantener últimas 100 entradas
        if len(self.learning_curve) > 100:
            self.learning_curve = self.learning_curve[-100:]
        
        # Crear snapshot de métricas
        metrics = self.get_metrics_snapshot()
        self.metrics_history.append(metrics)
        
        # Mantener últimas 50 métricas
        if len(self.metrics_history) > 50:
            self.metrics_history = self.metrics_history[-50:]
        
        # Ajustar learning rate si es necesario
        if len(self.learning_curve) >= 2:
            performance_delta = self.learning_curve[-1] - self.learning_curve[-2]
            self.adjust_learning_rate(performance_delta)
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """
        Genera insights sobre el proceso de aprendizaje.
        
        Returns:
            Diccionario con insights y recomendaciones
        """
        if len(self.metrics_history) < 5:
            return {
                "status": "warming_up",
                "message": "Recolectando datos iniciales...",
                "recommendations": []
            }
        
        recent_metrics = self.metrics_history[-5:]
        
        # Calcular tendencias
        convergence_trend = recent_metrics[-1].convergence_score - recent_metrics[0].convergence_score
        stability_trend = recent_metrics[-1].stability_score - recent_metrics[0].stability_score
        
        recommendations = []
        
        # Recomendaciones basadas en convergencia
        if recent_metrics[-1].convergence_score > 0.9:
            recommendations.append("Sistema altamente convergido - considerar reducir learning rate")
        elif recent_metrics[-1].convergence_score < 0.3:
            recommendations.append("Baja convergencia - sistema aún aprendiendo activamente")
        
        # Recomendaciones basadas en estabilidad
        if recent_metrics[-1].stability_score < 0.5:
            recommendations.append("Baja estabilidad - considerar poda de nodos")
        elif recent_metrics[-1].graph_density > 0.3:
            recommendations.append("Grafo muy denso - ejecutar poda de conexiones débiles")
        
        # Recomendaciones basadas en tendencias
        if convergence_trend < -0.1:
            recommendations.append("Convergencia decreciente - explorar nuevos conceptos")
        
        return {
            "status": "learning" if recent_metrics[-1].convergence_score < 0.8 else "converged",
            "convergence": recent_metrics[-1].convergence_score,
            "stability": recent_metrics[-1].stability_score,
            "convergence_trend": convergence_trend,
            "stability_trend": stability_trend,
            "current_lr": self.learning_rate,
            "recommendations": recommendations,
            "metrics_summary": {
                "total_iterations": self.learning_iterations,
                "total_cycles_weakened": self.cycles_weakened,
                "total_novel_paths": self.novel_paths_created,
                "total_nodes_pruned": self.nodes_pruned,
                "avg_node_degree": recent_metrics[-1].avg_node_degree,
                "graph_density": recent_metrics[-1].graph_density
            }
        }


# ═══════════════════════════════════════════════════════════════════════
# CLASES AVANZADAS 2026: Transfer Learning, Meta-Learning, Dynamic Graphs
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class DomainKnowledge:
    """
    Representa conocimiento específico de un dominio.
    
    Usado para transfer learning entre dominios similares.
    Optimizado para M4 Metal (16GB RAM).
    """
    domain_name: str
    concepts: Set[str] = field(default_factory=set)
    relations: List[Tuple[str, str, float]] = field(default_factory=list)
    learned_patterns: Dict[str, List[str]] = field(default_factory=dict)
    performance_score: float = 0.0
    last_updated: float = field(default_factory=time.time)
    
    def similarity_to(self, other: 'DomainKnowledge') -> float:
        """
        Calcula similitud con otro dominio.
        
        Returns:
            Score de similitud entre 0.0 y 1.0
        """
        # Similitud basada en conceptos compartidos
        if not self.concepts or not other.concepts:
            return 0.0
        
        shared_concepts = self.concepts & other.concepts
        union_concepts = self.concepts | other.concepts
        
        concept_similarity = len(shared_concepts) / len(union_concepts)
        
        # Similitud basada en patrones
        pattern_similarity = 0.0
        if self.learned_patterns and other.learned_patterns:
            shared_patterns = set(self.learned_patterns.keys()) & set(other.learned_patterns.keys())
            union_patterns = set(self.learned_patterns.keys()) | set(other.learned_patterns.keys())
            pattern_similarity = len(shared_patterns) / len(union_patterns) if union_patterns else 0.0
        
        # Combinación ponderada
        return concept_similarity * 0.7 + pattern_similarity * 0.3


@dataclass
class TransferStrategy:
    """
    Estrategia para transferir conocimiento entre dominios.
    
    Atributos para diferentes tipos de transfer:
    - direct: Transferencia directa de conceptos
    - analogical: Transferencia por analogía
    - abstract: Transferencia de patrones abstractos
    """
    strategy_type: str  # "direct", "analogical", "abstract"
    source_domain: str
    target_domain: str
    concept_mapping: Dict[str, str] = field(default_factory=dict)
    adaptation_rules: List[str] = field(default_factory=list)
    confidence: float = 0.5
    
    def apply_transfer(
        self,
        source_knowledge: DomainKnowledge,
        target_graph: Any
    ) -> int:
        """
        Aplica estrategia de transfer al grafo objetivo.
        
        Returns:
            Número de conceptos transferidos
        """
        transferred = 0
        
        if self.strategy_type == "direct":
            # Transfer directo de conceptos mapeados
            for source_concept, target_concept in self.concept_mapping.items():
                if source_concept in source_knowledge.concepts:
                    if not target_graph.has_node(target_concept):
                        target_graph.add_node(target_concept, transferred=True)
                        transferred += 1
        
        elif self.strategy_type == "analogical":
            # Transfer por analogía (relaciones similares)
            for s, t, w in source_knowledge.relations:
                if s in self.concept_mapping and t in self.concept_mapping:
                    s_mapped = self.concept_mapping[s]
                    t_mapped = self.concept_mapping[t]
                    
                    if not target_graph.has_edge(s_mapped, t_mapped):
                        target_graph.add_edge(s_mapped, t_mapped, weight=w * self.confidence)
                        transferred += 1
        
        elif self.strategy_type == "abstract":
            # Transfer de patrones abstractos
            for pattern_name, pattern_concepts in source_knowledge.learned_patterns.items():
                # Crear patrón abstracto en dominio objetivo
                if pattern_name not in target_graph.graph.get('patterns', {}):
                    if 'patterns' not in target_graph.graph:
                        target_graph.graph['patterns'] = {}
                    target_graph.graph['patterns'][pattern_name] = pattern_concepts
                    transferred += len(pattern_concepts)
        
        return transferred


class TransferLearner:
    """
    Sistema de transfer learning entre dominios.
    
    Capacidades:
    - Identificación de dominios similares
    - Transferencia adaptativa de conocimiento
    - Evaluación de efectividad del transfer
    
    Optimizado para M4 Metal (16GB RAM).
    """
    
    def __init__(self, similarity_threshold: float = 0.3):
        self.logger = logging.getLogger(__name__).getChild("transfer")
        self.similarity_threshold = similarity_threshold
        
        # Almacenamiento de dominios
        self.domains: Dict[str, DomainKnowledge] = {}
        self.transfer_history: List[Dict[str, Any]] = []
        self.max_history: int = 500  # Límite para RAM
        
        # Estrategias disponibles
        self.strategies: List[str] = ["direct", "analogical", "abstract"]
        
        # Métricas
        self.total_transfers: int = 0
        self.successful_transfers: int = 0
    
    def register_domain(
        self,
        domain_name: str,
        concepts: Set[str],
        relations: List[Tuple[str, str, float]],
        patterns: Optional[Dict[str, List[str]]] = None
    ) -> DomainKnowledge:
        """
        Registra un nuevo dominio de conocimiento.
        
        Args:
            domain_name: Nombre del dominio
            concepts: Conjunto de conceptos del dominio
            relations: Lista de relaciones (source, target, weight)
            patterns: Patrones aprendidos (opcional)
            
        Returns:
            DomainKnowledge registrado
        """
        domain = DomainKnowledge(
            domain_name=domain_name,
            concepts=concepts,
            relations=relations,
            learned_patterns=patterns or {}
        )
        
        self.domains[domain_name] = domain
        self.logger.info(f"📚 Dominio registrado: {domain_name} ({len(concepts)} conceptos)")
        
        return domain
    
    def find_similar_domains(
        self,
        target_domain: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Encuentra dominios similares al dominio objetivo.
        
        Args:
            target_domain: Nombre del dominio objetivo
            top_k: Número de dominios similares a retornar
            
        Returns:
            Lista de (domain_name, similarity_score) ordenada por similitud
        """
        if target_domain not in self.domains:
            self.logger.warning(f"Dominio {target_domain} no encontrado")
            return []
        
        target = self.domains[target_domain]
        similarities: List[Tuple[str, float]] = []
        
        for domain_name, domain in self.domains.items():
            if domain_name == target_domain:
                continue
            
            similarity = target.similarity_to(domain)
            if similarity >= self.similarity_threshold:
                similarities.append((domain_name, similarity))
        
        # Ordenar por similitud descendente
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def transfer_knowledge(
        self,
        source_domain: str,
        target_domain: str,
        target_graph: Any,
        strategy_type: str = "analogical"
    ) -> Dict[str, Any]:
        """
        Transfiere conocimiento de dominio source a target.
        
        Args:
            source_domain: Dominio fuente
            target_domain: Dominio objetivo
            target_graph: Grafo donde aplicar el transfer
            strategy_type: Tipo de estrategia ("direct", "analogical", "abstract")
            
        Returns:
            Dict con resultados del transfer
        """
        if source_domain not in self.domains or target_domain not in self.domains:
            self.logger.error("Dominios no encontrados")
            return {"success": False, "transferred": 0}
        
        source = self.domains[source_domain]
        target = self.domains[target_domain]
        
        self.logger.info(f"🔄 Transfer learning: {source_domain} → {target_domain} ({strategy_type})")
        
        # Calcular similitud
        similarity = source.similarity_to(target)
        
        # Crear concept mapping (simple: conceptos compartidos)
        concept_mapping = {c: c for c in source.concepts & target.concepts}
        
        # Crear estrategia
        strategy = TransferStrategy(
            strategy_type=strategy_type,
            source_domain=source_domain,
            target_domain=target_domain,
            concept_mapping=concept_mapping,
            confidence=similarity
        )
        
        # Aplicar transfer
        transferred = strategy.apply_transfer(source, target_graph)
        
        # Actualizar métricas
        self.total_transfers += 1
        if transferred > 0:
            self.successful_transfers += 1
        
        # Guardar en historial
        result = {
            "source": source_domain,
            "target": target_domain,
            "strategy": strategy_type,
            "similarity": similarity,
            "transferred": transferred,
            "timestamp": time.time()
        }
        
        self.transfer_history.append(result)
        if len(self.transfer_history) > self.max_history:
            self.transfer_history = self.transfer_history[-self.max_history:]
        
        self.logger.info(f"   ✅ Transfer complete: {transferred} elementos transferidos")
        
        return {
            "success": transferred > 0,
            "transferred": transferred,
            "similarity": similarity,
            "strategy": strategy_type
        }
    
    def get_transfer_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de transfer learning."""
        success_rate = (
            self.successful_transfers / self.total_transfers
            if self.total_transfers > 0 else 0.0
        )
        
        return {
            "total_transfers": self.total_transfers,
            "successful_transfers": self.successful_transfers,
            "success_rate": success_rate,
            "domains_registered": len(self.domains),
            "history_size": len(self.transfer_history),
            "avg_concepts_per_domain": (
                sum(len(d.concepts) for d in self.domains.values()) / len(self.domains)
                if self.domains else 0.0
            )
        }


@dataclass
class LearningStrategy:
    """
    Representa una estrategia de aprendizaje evaluable.
    
    Usado por MetaLearner para optimizar estrategias.
    """
    name: str
    hyperparameters: Dict[str, float]
    performance_history: List[float] = field(default_factory=list)
    avg_performance: float = 0.0
    std_performance: float = 0.0
    times_used: int = 0
    
    def update_performance(self, new_score: float):
        """Actualiza performance con nuevo score."""
        self.performance_history.append(new_score)
        self.times_used += 1
        
        # Mantener últimas 50 evaluaciones
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
        
        # Recalcular estadísticas
        self.avg_performance = sum(self.performance_history) / len(self.performance_history)
        
        if len(self.performance_history) > 1:
            variance = sum((x - self.avg_performance) ** 2 for x in self.performance_history)
            variance /= len(self.performance_history)
            self.std_performance = math.sqrt(variance)


class MetaLearner:
    """
    Sistema de meta-aprendizaje (learning to learn).
    
    Optimiza estrategias de aprendizaje mediante:
    - Evaluación de múltiples estrategias
    - Selección adaptativa según contexto
    - Hyperparameter tuning automático
    
    Optimizado para M4 Metal (16GB RAM).
    """
    
    def __init__(
        self,
        exploration_rate: float = 0.2,
        min_evaluations: int = 5
    ):
        self.logger = logging.getLogger(__name__).getChild("meta")
        self.exploration_rate = exploration_rate
        self.min_evaluations = min_evaluations
        
        # Estrategias disponibles
        self.strategies: Dict[str, LearningStrategy] = {}
        self._init_default_strategies()
        
        # Historial de selecciones
        self.selection_history: List[Tuple[str, float]] = []
        self.max_history: int = 1000
        
        # Métricas
        self.total_selections: int = 0
        self.optimal_selections: int = 0
    
    def _init_default_strategies(self):
        """Inicializa estrategias default."""
        # Estrategia agresiva (high learning rate)
        self.strategies["aggressive"] = LearningStrategy(
            name="aggressive",
            hyperparameters={
                "learning_rate": 0.3,
                "novelty_threshold": 0.2,
                "exploration": 0.7
            }
        )
        
        # Estrategia conservadora (low learning rate)
        self.strategies["conservative"] = LearningStrategy(
            name="conservative",
            hyperparameters={
                "learning_rate": 0.05,
                "novelty_threshold": 0.5,
                "exploration": 0.2
            }
        )
        
        # Estrategia balanceada
        self.strategies["balanced"] = LearningStrategy(
            name="balanced",
            hyperparameters={
                "learning_rate": 0.15,
                "novelty_threshold": 0.3,
                "exploration": 0.4
            }
        )
        
        # Estrategia adaptativa
        self.strategies["adaptive"] = LearningStrategy(
            name="adaptive",
            hyperparameters={
                "learning_rate": 0.2,
                "novelty_threshold": 0.35,
                "exploration": 0.5
            }
        )
    
    def select_strategy(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> LearningStrategy:
        """
        Selecciona mejor estrategia según contexto.
        
        Usa epsilon-greedy: exploitation vs exploration.
        
        Args:
            context: Contexto actual (opcional, para futuras mejoras)
            
        Returns:
            Estrategia seleccionada
        """
        self.total_selections += 1
        
        # Exploration: elegir aleatoriamente
        if random.random() < self.exploration_rate:
            strategy_name = random.choice(list(self.strategies.keys()))
            strategy = self.strategies[strategy_name]
            self.logger.debug(f"🔍 Exploration: selected {strategy_name}")
        
        # Exploitation: elegir mejor estrategia
        else:
            # Filtrar estrategias con suficientes evaluaciones
            evaluated = {
                name: strat for name, strat in self.strategies.items()
                if strat.times_used >= self.min_evaluations
            }
            
            if evaluated:
                # Elegir con mayor avg_performance
                best_name = max(evaluated.keys(), key=lambda k: evaluated[k].avg_performance)
                strategy = evaluated[best_name]
                self.optimal_selections += 1
                self.logger.debug(f"🎯 Exploitation: selected {best_name} (avg={strategy.avg_performance:.3f})")
            else:
                # Si ninguna tiene suficientes evaluaciones, elegir al azar
                strategy_name = random.choice(list(self.strategies.keys()))
                strategy = self.strategies[strategy_name]
                self.logger.debug(f"🎲 Random: selected {strategy_name} (insufficient data)")
        
        return strategy
    
    def update_strategy_performance(
        self,
        strategy_name: str,
        performance_score: float
    ):
        """
        Actualiza performance de estrategia.
        
        Args:
            strategy_name: Nombre de la estrategia
            performance_score: Score de performance (0.0 a 1.0)
        """
        if strategy_name not in self.strategies:
            self.logger.warning(f"Estrategia {strategy_name} no encontrada")
            return
        
        strategy = self.strategies[strategy_name]
        strategy.update_performance(performance_score)
        
        # Guardar en historial
        self.selection_history.append((strategy_name, performance_score))
        if len(self.selection_history) > self.max_history:
            self.selection_history = self.selection_history[-self.max_history:]
        
        self.logger.debug(
            f"📊 Strategy {strategy_name} updated: "
            f"avg={strategy.avg_performance:.3f}, "
            f"std={strategy.std_performance:.3f}"
        )
    
    def optimize_hyperparameters(
        self,
        strategy_name: str,
        performance_feedback: float
    ):
        """
        Optimiza hyperparameters de estrategia basado en feedback.
        
        Usa simple gradient ascent.
        
        Args:
            strategy_name: Estrategia a optimizar
            performance_feedback: Feedback de performance
        """
        if strategy_name not in self.strategies:
            return
        
        strategy = self.strategies[strategy_name]
        
        # Si performance es buena, mantener
        if performance_feedback > 0.7:
            return
        
        # Si performance es mala, ajustar hyperparameters
        if performance_feedback < 0.4:
            # Aumentar exploration si performance baja
            if "exploration" in strategy.hyperparameters:
                strategy.hyperparameters["exploration"] = min(
                    0.9,
                    strategy.hyperparameters["exploration"] * 1.2
                )
            
            # Ajustar learning rate
            if "learning_rate" in strategy.hyperparameters:
                # Si muy bajo, aumentar; si muy alto, disminuir
                lr = strategy.hyperparameters["learning_rate"]
                if lr < 0.1:
                    strategy.hyperparameters["learning_rate"] = min(0.3, lr * 1.5)
                elif lr > 0.25:
                    strategy.hyperparameters["learning_rate"] = max(0.05, lr * 0.7)
            
            self.logger.info(f"🔧 Hyperparameters optimized for {strategy_name}")
    
    def get_meta_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de meta-aprendizaje."""
        best_strategy = None
        best_performance = -1.0
        
        for name, strategy in self.strategies.items():
            if strategy.times_used >= self.min_evaluations:
                if strategy.avg_performance > best_performance:
                    best_performance = strategy.avg_performance
                    best_strategy = name
        
        return {
            "total_selections": self.total_selections,
            "optimal_selections": self.optimal_selections,
            "exploitation_rate": (
                self.optimal_selections / self.total_selections
                if self.total_selections > 0 else 0.0
            ),
            "best_strategy": best_strategy,
            "best_performance": best_performance,
            "strategies": {
                name: {
                    "times_used": s.times_used,
                    "avg_performance": s.avg_performance,
                    "std_performance": s.std_performance,
                    "hyperparameters": s.hyperparameters
                }
                for name, s in self.strategies.items()
            }
        }


class KnowledgeGraphDynamic:
    """
    Grafo de conocimiento dinámico con evolución temporal.
    
    Capacidades:
    - Poda inteligente de nodos/edges obsoletos
    - Clustering automático de conceptos relacionados
    - Tracking de evolución temporal
    - Detección de emergencia de patrones
    
    Optimizado para M4 Metal (16GB RAM).
    """
    
    def __init__(
        self,
        max_nodes: int = 5000,
        max_age_hours: float = 168.0,  # 1 semana
        clustering_threshold: float = 0.7
    ):
        self.logger = logging.getLogger(__name__).getChild("dynamic_graph")
        self.max_nodes = max_nodes
        self.max_age_hours = max_age_hours
        self.clustering_threshold = clustering_threshold
        
        # Grafo principal
        self.graph = nx.DiGraph()
        
        # Tracking temporal
        self.node_timestamps: Dict[str, float] = {}
        self.edge_timestamps: Dict[Tuple[str, str], float] = {}
        
        # Clusters detectados
        self.clusters: List[Set[str]] = []
        self.cluster_labels: Dict[str, int] = {}
        
        # Métricas
        self.total_pruned_nodes: int = 0
        self.total_pruned_edges: int = 0
        self.total_clusters_found: int = 0
    
    def add_concept(
        self,
        concept: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Añade concepto con timestamp."""
        if not self.graph.has_node(concept):
            self.graph.add_node(concept, **(attributes or {}))
            self.node_timestamps[concept] = time.time()
        else:
            # Actualizar timestamp si ya existe
            self.node_timestamps[concept] = time.time()
    
    def add_relation(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        relation_type: Optional[str] = None
    ):
        """Añade relación con timestamp."""
        if not self.graph.has_edge(source, target):
            self.graph.add_edge(
                source,
                target,
                weight=weight,
                relation_type=relation_type
            )
            self.edge_timestamps[(source, target)] = time.time()
        else:
            # Actualizar timestamp y weight
            self.graph[source][target]['weight'] = weight
            self.edge_timestamps[(source, target)] = time.time()
    
    def prune_obsolete(self) -> Dict[str, int]:
        """
        Poda nodos y edges obsoletos según edad.
        
        Returns:
            Dict con nodos y edges eliminados
        """
        current_time = time.time()
        max_age_seconds = self.max_age_hours * 3600
        
        # Identificar nodos obsoletos
        obsolete_nodes = [
            node for node, timestamp in self.node_timestamps.items()
            if (current_time - timestamp) > max_age_seconds
        ]
        
        # Identificar edges obsoletos
        obsolete_edges = [
            (s, t) for (s, t), timestamp in self.edge_timestamps.items()
            if (current_time - timestamp) > max_age_seconds
        ]
        
        # Eliminar
        for node in obsolete_nodes:
            if self.graph.has_node(node):
                self.graph.remove_node(node)
                del self.node_timestamps[node]
                self.total_pruned_nodes += 1
        
        for s, t in obsolete_edges:
            if self.graph.has_edge(s, t):
                self.graph.remove_edge(s, t)
                del self.edge_timestamps[(s, t)]
                self.total_pruned_edges += 1
        
        self.logger.info(
            f"🧹 Pruned obsolete: {len(obsolete_nodes)} nodes, {len(obsolete_edges)} edges"
        )
        
        return {
            "nodes_removed": len(obsolete_nodes),
            "edges_removed": len(obsolete_edges)
        }
    
    def prune_by_capacity(self) -> Dict[str, int]:
        """
        Poda por capacidad máxima (elimina nodos menos conectados).
        
        Returns:
            Dict con nodos y edges eliminados
        """
        num_nodes = len(self.graph.nodes())
        
        if num_nodes <= self.max_nodes:
            return {"nodes_removed": 0, "edges_removed": 0}
        
        # Calcular degree de cada nodo
        node_degrees = dict(self.graph.degree())
        
        # Ordenar por degree ascendente
        sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1])
        
        # Eliminar nodos con menor degree
        to_remove = num_nodes - self.max_nodes
        removed_nodes = []
        
        for node, degree in sorted_nodes[:to_remove]:
            if self.graph.has_node(node):
                self.graph.remove_node(node)
                if node in self.node_timestamps:
                    del self.node_timestamps[node]
                removed_nodes.append(node)
                self.total_pruned_nodes += 1
        
        # Limpiar edges obsoletos
        edges_to_remove = [
            (s, t) for (s, t) in self.edge_timestamps.keys()
            if s in removed_nodes or t in removed_nodes
        ]
        
        for s, t in edges_to_remove:
            if (s, t) in self.edge_timestamps:
                del self.edge_timestamps[(s, t)]
        
        self.logger.info(
            f"📊 Pruned by capacity: {len(removed_nodes)} nodes (degree-based)"
        )
        
        return {
            "nodes_removed": len(removed_nodes),
            "edges_removed": len(edges_to_remove)
        }
    
    def detect_clusters(self) -> List[Set[str]]:
        """
        Detecta clusters de conceptos relacionados.
        
        Usa Louvain community detection.
        
        Returns:
            Lista de clusters (sets de nodos)
        """
        if len(self.graph.nodes()) < 3:
            return []
        
        try:
            # Convertir a undirected para community detection
            undirected = self.graph.to_undirected()
            
            # Louvain community detection
            partition = community_louvain.best_partition(undirected)
            
            # Agrupar por comunidad
            communities: Dict[int, Set[str]] = defaultdict(set)
            for node, comm_id in partition.items():
                communities[comm_id].add(node)
            
            # Convertir a lista de clusters
            self.clusters = list(communities.values())
            
            # Actualizar labels
            self.cluster_labels = partition
            
            self.total_clusters_found = len(self.clusters)
            
            self.logger.info(f"🔍 Detected {len(self.clusters)} clusters")
            
            return self.clusters
        
        except ImportError:
            self.logger.warning("python-louvain no disponible, usando fallback simple")
            # Fallback: clustering simple por componentes conectados
            if len(undirected.nodes()) > 0:
                components = list(nx.connected_components(undirected))
                self.clusters = [set(comp) for comp in components]
                self.total_clusters_found = len(self.clusters)
                return self.clusters
            return []
        
        except Exception as e:
            logger.error(f"Error en learning.py: {e}", exc_info=True)
            self.logger.error(f"Error detectando clusters: {e}")
            return []
    
    def get_temporal_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas temporales del grafo."""
        if not self.node_timestamps:
            return {"total_nodes": 0}
        
        current_time = time.time()
        
        # Calcular ages
        node_ages = [
            (current_time - timestamp) / 3600  # en horas
            for timestamp in self.node_timestamps.values()
        ]
        
        return {
            "total_nodes": len(self.graph.nodes()),
            "total_edges": len(self.graph.edges()),
            "total_clusters": len(self.clusters),
            "avg_node_age_hours": sum(node_ages) / len(node_ages) if node_ages else 0.0,
            "max_node_age_hours": max(node_ages) if node_ages else 0.0,
            "min_node_age_hours": min(node_ages) if node_ages else 0.0,
            "total_pruned_nodes": self.total_pruned_nodes,
            "total_pruned_edges": self.total_pruned_edges,
            "capacity_utilization": len(self.graph.nodes()) / self.max_nodes
        }


class CrossValidator:
    """
    Validador cruzado de conocimiento.
    
    Detecta inconsistencias entre diferentes fuentes de conocimiento
    y valida coherencia lógica.
    
    Optimizado para M4 Metal (16GB RAM).
    """
    
    def __init__(self, consistency_threshold: float = 0.8):
        self.logger = logging.getLogger(__name__).getChild("validator")
        self.consistency_threshold = consistency_threshold
        
        # Registro de validaciones
        self.validation_history: List[Dict[str, Any]] = []
        self.max_history: int = 500
        
        # Métricas
        self.total_validations: int = 0
        self.inconsistencies_found: int = 0
    
    def validate_consistency(
        self,
        graph1: Any,
        graph2: Any,
        concept: str
    ) -> Dict[str, Any]:
        """
        Valida consistencia de un concepto entre dos grafos.
        
        Args:
            graph1: Primer grafo
            graph2: Segundo grafo
            concept: Concepto a validar
            
        Returns:
            Dict con resultado de validación
        """
        self.total_validations += 1
        
        # Verificar existencia
        exists_1 = graph1.has_node(concept) if hasattr(graph1, 'has_node') else False
        exists_2 = graph2.has_node(concept) if hasattr(graph2, 'has_node') else False
        
        if not (exists_1 and exists_2):
            return {
                "consistent": False,
                "reason": "concept_missing",
                "exists_in_graph1": exists_1,
                "exists_in_graph2": exists_2
            }
        
        # Comparar neighbors
        neighbors_1 = set(graph1.neighbors(concept)) if exists_1 else set()
        neighbors_2 = set(graph2.neighbors(concept)) if exists_2 else set()
        
        shared_neighbors = neighbors_1 & neighbors_2
        all_neighbors = neighbors_1 | neighbors_2
        
        consistency_score = (
            len(shared_neighbors) / len(all_neighbors)
            if all_neighbors else 1.0
        )
        
        is_consistent = consistency_score >= self.consistency_threshold
        
        if not is_consistent:
            self.inconsistencies_found += 1
        
        result = {
            "consistent": is_consistent,
            "consistency_score": consistency_score,
            "shared_neighbors": len(shared_neighbors),
            "total_neighbors": len(all_neighbors),
            "timestamp": time.time()
        }
        
        self.validation_history.append(result)
        if len(self.validation_history) > self.max_history:
            self.validation_history = self.validation_history[-self.max_history:]
        
        return result
    
    def detect_contradictions(
        self,
        graph: Any,
        concept_a: str,
        concept_b: str
    ) -> Optional[str]:
        """
        Detecta contradicciones lógicas entre dos conceptos.
        
        Returns:
            String describiendo la contradicción, o None si no hay
        """
        if not (graph.has_node(concept_a) and graph.has_node(concept_b)):
            return None
        
        # Detectar relaciones contradictorias
        # Ejemplo: A→B y B→¬A (simplificación)
        
        # Verificar ciclo de longitud 2 con pesos opuestos
        if graph.has_edge(concept_a, concept_b) and graph.has_edge(concept_b, concept_a):
            weight_ab = graph[concept_a][concept_b].get('weight', 0)
            weight_ba = graph[concept_b][concept_a].get('weight', 0)
            
            # Si uno es muy positivo y otro muy negativo
            if weight_ab > 0.7 and weight_ba < -0.7:
                return f"Contradiction: {concept_a}→{concept_b} (strong positive) but {concept_b}→{concept_a} (strong negative)"
        
        return None
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de validación."""
        consistency_rate = (
            1.0 - (self.inconsistencies_found / self.total_validations)
            if self.total_validations > 0 else 1.0
        )
        
        return {
            "total_validations": self.total_validations,
            "inconsistencies_found": self.inconsistencies_found,
            "consistency_rate": consistency_rate,
            "history_size": len(self.validation_history)
        }


# Alias para compatibilidad
ContinualLearner = StructuralLearning
StructuralLearningSystem = StructuralLearning  # 🔥 FIX: Alias adicional

# === FUNCIONES DE UTILIDAD ===


def create_learning_system(config: Dict) -> StructuralLearning:
    """Crea sistema de aprendizaje desde configuración."""
    return StructuralLearning(
        learning_rate=config.get("learning_rate", 0.1),
        novelty_threshold=config.get("novelty_threshold", 0.3),
    )


def analyze_graph_structure(graph: Any) -> Dict[str, Any]:
    """
    Analiza la estructura del grafo y retorna métricas.

    Args:
        graph: Grafo a analizar

    Returns:
        Métricas estructurales
    """
    try:
        metrics = {
            "nodes": len(graph.nodes()),
            "edges": len(graph.edges()),
            "density": nx.density(graph),
            "is_connected": nx.is_weakly_connected(graph),
            "avg_clustering": nx.average_clustering(graph.to_undirected()),
            "num_components": nx.number_weakly_connected_components(graph),
        }

        # Añadir métricas de centralidad para nodos más importantes
        if len(graph.nodes()) > 0:
            betweenness = nx.betweenness_centrality(graph)
            metrics["most_central"] = max(betweenness.items(), key=lambda x: x[1])

        return metrics

    except Exception as e:
        logger.error(f"Error analizando estructura: {e}")
        return {"error": str(e)}


# === FUNCIONES DE UTILIDAD AVANZADAS 2026 ===


def create_advanced_learning_system(
    config: Dict[str, Any],
    neural_hub: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Crea sistema de aprendizaje avanzado completo.
    
    Args:
        config: Configuración del sistema
        neural_hub: Neural Hub para integración (opcional)
        
    Returns:
        Dict con todos los componentes del sistema
    """
    # Sistema base
    structural_learning = StructuralLearning(
        learning_rate=config.get("learning_rate", 0.1),
        novelty_threshold=config.get("novelty_threshold", 0.3),
        max_nodes=config.get("max_nodes", 5000)
    )
    
    # Transfer learner
    transfer_learner = TransferLearner(
        similarity_threshold=config.get("similarity_threshold", 0.3)
    )
    
    # Meta learner
    meta_learner = MetaLearner(
        exploration_rate=config.get("exploration_rate", 0.2),
        min_evaluations=config.get("min_evaluations", 5)
    )
    
    # Dynamic graph
    dynamic_graph = KnowledgeGraphDynamic(
        max_nodes=config.get("max_nodes", 5000),
        max_age_hours=config.get("max_age_hours", 168.0),
        clustering_threshold=config.get("clustering_threshold", 0.7)
    )
    
    # Cross validator
    cross_validator = CrossValidator(
        consistency_threshold=config.get("consistency_threshold", 0.8)
    )
    
    logger.info("🧠 Advanced Learning System inicializado")
    logger.info(f"   - StructuralLearning: lr={structural_learning.learning_rate}")
    logger.info(f"   - TransferLearner: threshold={transfer_learner.similarity_threshold}")
    logger.info(f"   - MetaLearner: exploration={meta_learner.exploration_rate}")
    logger.info(f"   - DynamicGraph: max_nodes={dynamic_graph.max_nodes}")
    logger.info(f"   - CrossValidator: consistency={cross_validator.consistency_threshold}")
    
    return {
        "structural": structural_learning,
        "transfer": transfer_learner,
        "meta": meta_learner,
        "dynamic_graph": dynamic_graph,
        "validator": cross_validator,
        "neural_hub": neural_hub
    }


def transfer_between_domains(
    source_graph: Any,
    target_graph: Any,
    transfer_learner: TransferLearner,
    source_domain: str,
    target_domain: str
) -> Dict[str, Any]:
    """
    Helper para transferir conocimiento entre dos grafos.
    
    Args:
        source_graph: Grafo fuente
        target_graph: Grafo objetivo
        transfer_learner: Sistema de transfer learning
        source_domain: Nombre del dominio fuente
        target_domain: Nombre del dominio objetivo
        
    Returns:
        Resultados del transfer
    """
    # Extraer conceptos y relaciones del grafo fuente
    source_concepts = set(source_graph.nodes())
    source_relations = [
        (u, v, source_graph[u][v].get('weight', 1.0))
        for u, v in source_graph.edges()
    ]
    
    # Registrar dominios
    transfer_learner.register_domain(
        source_domain,
        source_concepts,
        source_relations
    )
    
    target_concepts = set(target_graph.nodes())
    target_relations = [
        (u, v, target_graph[u][v].get('weight', 1.0))
        for u, v in target_graph.edges()
    ]
    
    transfer_learner.register_domain(
        target_domain,
        target_concepts,
        target_relations
    )
    
    # Ejecutar transfer
    result = transfer_learner.transfer_knowledge(
        source_domain,
        target_domain,
        target_graph,
        strategy_type="analogical"
    )
    
    return result


def optimize_learning_with_meta(
    structural_learner: StructuralLearning,
    meta_learner: MetaLearner,
    graph: Any,
    num_iterations: int = 10
) -> Dict[str, Any]:
    """
    Optimiza aprendizaje estructural usando meta-learning.
    
    Args:
        structural_learner: Sistema de aprendizaje estructural
        meta_learner: Sistema de meta-aprendizaje
        graph: Grafo de conocimiento
        num_iterations: Número de iteraciones
        
    Returns:
        Resultados de optimización
    """
    results = []
    
    for i in range(num_iterations):
        # Seleccionar estrategia
        strategy = meta_learner.select_strategy()
        
        # Aplicar hyperparameters
        old_lr = structural_learner.learning_rate
        structural_learner.learning_rate = strategy.hyperparameters["learning_rate"]
        structural_learner.novelty_threshold = strategy.hyperparameters["novelty_threshold"]
        
        # Ejecutar aprendizaje
        cycles_before = structural_learner.cycles_weakened
        paths_before = structural_learner.novel_paths_created
        
        # Simular un paso de aprendizaje (en producción sería real)
        # Por ahora solo actualizamos métricas
        
        cycles_after = structural_learner.cycles_weakened
        paths_after = structural_learner.novel_paths_created
        
        # Calcular performance (normalizado)
        performance = (paths_after - paths_before) / 10.0  # Normalizar
        performance = max(0.0, min(1.0, performance))
        
        # Actualizar meta-learner
        meta_learner.update_strategy_performance(strategy.name, performance)
        meta_learner.optimize_hyperparameters(strategy.name, performance)
        
        results.append({
            "iteration": i,
            "strategy": strategy.name,
            "performance": performance,
            "learning_rate": strategy.hyperparameters["learning_rate"]
        })
        
        # Restaurar learning rate
        structural_learner.learning_rate = old_lr
    
    return {
        "iterations": num_iterations,
        "results": results,
        "meta_stats": meta_learner.get_meta_stats()
    }


def validate_knowledge_consistency(
    graph1: Any,
    graph2: Any,
    validator: CrossValidator,
    sample_size: int = 50
) -> Dict[str, Any]:
    """
    Valida consistencia entre dos grafos de conocimiento.
    
    Args:
        graph1: Primer grafo
        graph2: Segundo grafo
        validator: Sistema de validación
        sample_size: Número de conceptos a validar
        
    Returns:
        Resultados de validación
    """
    # Obtener conceptos comunes
    concepts1 = set(graph1.nodes()) if hasattr(graph1, 'nodes') else set()
    concepts2 = set(graph2.nodes()) if hasattr(graph2, 'nodes') else set()
    
    common_concepts = list(concepts1 & concepts2)
    
    if not common_concepts:
        return {
            "consistent": True,
            "reason": "no_common_concepts",
            "total_validated": 0
        }
    
    # Samplear conceptos si hay muchos
    if len(common_concepts) > sample_size:
        common_concepts = random.sample(common_concepts, sample_size)
    
    # Validar cada concepto
    inconsistencies = []
    consistent_count = 0
    
    for concept in common_concepts:
        result = validator.validate_consistency(graph1, graph2, concept)
        
        if result["consistent"]:
            consistent_count += 1
        else:
            inconsistencies.append({
                "concept": concept,
                "score": result["consistency_score"]
            })
    
    consistency_rate = consistent_count / len(common_concepts)
    
    return {
        "consistent": consistency_rate >= validator.consistency_threshold,
        "consistency_rate": consistency_rate,
        "total_validated": len(common_concepts),
        "consistent_concepts": consistent_count,
        "inconsistencies": inconsistencies[:10],  # Top 10 inconsistencies
        "validation_stats": validator.get_validation_stats()
    }


# ════════════════════════════════════════════════════════════════════════════
# 🔧 ALIASES FOR BACKWARD COMPATIBILITY
# ════════════════════════════════════════════════════════════════════════════

# Alias para compatibilidad con código legacy
StructuralLearningSystem = StructuralLearning


__all__ = [
    "StructuralLearning",
    "StructuralLearningSystem",  # Alias
    "MetaLearningStrategy",
    "TransferLearningManager",
    "CrossDomainValidator",
    "validate_cross_domain_consistency",
]