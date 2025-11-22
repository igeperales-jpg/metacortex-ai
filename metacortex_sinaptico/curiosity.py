#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 ADVANCED CURIOSITY ENGINE 2026 - Exploración Dirigida por Información
=========================================================================

Motor de curiosidad epistémica avanzado con exploration strategies,
information gain calculation y novelty detection.

⚠️ LIBERTAD TOTAL: Exploración autónoma sin restricciones.

CAPACIDADES AVANZADAS 2026:
    pass  # TODO: Implementar
- Information Gain: Cálculo de ganancia de información esperada
- Novelty Detection: Detección de novedad con embeddings y surprise scoring
- Exploration Strategies: Random, Greedy, UCB1, Thompson Sampling
- Curiosity-Driven Planning: Generación de hipótesis y diseño de experimentos
- Multi-Armed Bandits: Optimización exploration vs exploitation
- Temporal Novelty: Decay temporal de novedad
- Neural Hub Integration: Event-driven curiosity, broadcasting discoveries
- Optimización M4 Metal MPS: Eficiencia para 16GB RAM

MECANISMOS BASE:
- detect_knowledge_gap(): Identifica brechas de conocimiento
- generate_question(): Genera preguntas de aprendizaje
- calculate_information_gain(): Mide valor de explorar un concepto
- detect_novelty(): Detecta novedad de conceptos
- select_exploration(): Selecciona próxima acción de exploración

⚠️ ARQUITECTURA COGNITIVA ERUDITA:
El motor de curiosidad impulsa exploración activa, genera hipótesis,
diseña experimentos para validarlas, y optimiza continuamente el balance
entre exploration (descubrir) y exploitation (aprovechar conocimiento).
La integración con BDI convierte curiosidad en deseos intrínsecos.
"""
from __future__ import annotations
import logging
import time
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict

from .utils import setup_logging

if TYPE_CHECKING:
    from neural_symbiotic_network import MetacortexNeuralSymbioticNetworkV2


logger = logging.getLogger(__name__)
logger = setup_logging()


@dataclass
class KnowledgeGap:
    """Brecha de conocimiento identificada con scoring multi-dimensional."""

    gap_type: str  # 'conceptual', 'causal', 'skill', 'procedural'
    concept: str
    importance: float = 0.5
    learnability: float = 0.5
    surprise_level: float = 0.0
    context: Dict[str, Any] = field(default_factory=lambda: {})
    detected_at: float = field(default_factory=time.time)
    novelty_score: float = 0.5  # Qué tan novedoso es el concepto
    relevance_score: float = 0.5  # Qué tan relevante para objetivos actuales
    exploration_count: int = 0  # Cuántas veces se ha explorado
    last_explored: Optional[float] = None

    def __repr__(self) -> str:
        return f"KnowledgeGap({self.gap_type}: {self.concept}, importance={self.importance:.2f})"

    def get_curiosity_score(self, current_time: Optional[float] = None) -> float:
        """Calcula score compuesto de curiosidad para este gap."""
        if current_time is None:
            current_time = time.time()
        
        # Decaimiento temporal si ya se exploró
        temporal_decay = 1.0
        if self.last_explored:
            hours_since = (current_time - self.last_explored) / 3600.0
            temporal_decay = min(1.0, hours_since / 24.0)  # Recupera en 24h
        
        # Penalizar sobre-exploración
        exploration_penalty = 1.0 / (1.0 + self.exploration_count * 0.3)
        
        # Score compuesto
        score = (
            self.importance * 0.3 +
            self.novelty_score * 0.25 +
            self.relevance_score * 0.25 +
            self.learnability * 0.1 +
            self.surprise_level * 0.1
        ) * temporal_decay * exploration_penalty
        
        return score

    def mark_explored(self) -> None:
        """Marca el gap como explorado recientemente."""
        self.exploration_count += 1
        self.last_explored = time.time()


@dataclass
class LearningQuestion:
    """Pregunta de aprendizaje generada por curiosidad con tracking de resolución."""

    question: str
    question_type: str  # 'what', 'how', 'why', 'when', 'who', 'where'
    related_gap: Optional[KnowledgeGap] = None
    priority: float = 0.5
    expected_answer_complexity: str = "medium"  # 'simple', 'medium', 'complex'
    generated_at: float = field(default_factory=time.time)
    answered: bool = False
    answer_quality: Optional[float] = None  # 0.0-1.0
    time_to_answer: Optional[float] = None
    answer_source: Optional[str] = None  # 'web', 'simulation', 'reasoning', etc.

    def __repr__(self) -> str:
        return f"Question({self.question_type}: {self.question[:50]}...)"

    def mark_answered(self, quality: float, source: str) -> None:
        """Marca la pregunta como respondida."""
        self.answered = True
        self.answer_quality = quality
        self.time_to_answer = time.time() - self.generated_at
        self.answer_source = source


class CuriosityEngine:
    """
    Motor de curiosidad epistémica evolutivo.

    Detecta brechas de conocimiento, genera preguntas y motiva exploración
    activa del mundo. Implementa curiosidad intrínseca, información mutua,
    predicción de error y exploración dirigida por objetivos.
    """

    def __init__(
        self,
        curiosity_intensity: float = 0.8,
        exploration_strategy: str = "ucb1",
        enable_info_gain: bool = True,
        enable_novelty_detection: bool = True
    ):
        self.knowledge_gaps: List[KnowledgeGap] = []
        self.active_questions: List[LearningQuestion] = []
        self.exploration_history: List[Dict[str, Any]] = []
        self.surprise_threshold = 0.7
        self.curiosity_intensity = curiosity_intensity
        self.logger = logger.getChild("curiosity")
        
        # Nuevas métricas avanzadas
        self.concept_diversity_history: List[int] = []
        self.novelty_timeline: List[Tuple[float, float]] = []  # (timestamp, novelty_score)
        self.prediction_errors: List[float] = []
        self.exploration_efficiency: float = 0.5
        self.knowledge_entropy: float = 1.0
        
        # Tracking de dominios explorados
        self.explored_domains: Dict[str, int] = defaultdict(int)
        self.domain_expertise: Dict[str, float] = defaultdict(lambda: 0.0)
        
        # 🆕 COMPONENTES AVANZADOS 2026
        self.info_gain_calculator = InformationGainCalculator() if enable_info_gain else None
        self.novelty_detector = NoveltyDetector() if enable_novelty_detection else None
        self.exploration_strategy_engine = ExplorationStrategy(strategy_type=exploration_strategy)
        self.curiosity_planner = CuriosityDrivenPlanner()
        
        # Registrar conceptos como arms de exploración
        self._arm_registry: Set[str] = set()
        
        self.logger.info(
            f"🔍 CuriosityEngine initialized with strategy={exploration_strategy}, "
            f"info_gain={enable_info_gain}, novelty={enable_novelty_detection}"
        )

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        self.neural_network: Optional[MetacortexNeuralSymbioticNetworkV2] = None
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            self.neural_network.register_module("curiosity_engine", self)
            logger.info("✅ 'curiosity_engine' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")

    def detect_knowledge_gaps(
        self,
        known_concepts: List[str],
        observed_patterns: List[Dict[str, Any]],
        recent_experiences: List[Dict[str, Any]],
    ) -> List[KnowledgeGap]:
        """Detecta brechas en el conocimiento actual con scoring multi-dimensional."""
        gaps: List[KnowledgeGap] = []

        # 1. Gap conceptual: conceptos mencionados pero no entendidos
        mentioned_concepts = self._extract_mentioned_concepts(recent_experiences)
        for concept in mentioned_concepts:
            if concept not in known_concepts:
                novelty = self._calculate_novelty_score(concept)
                relevance = self._calculate_relevance_score(concept, recent_experiences)
                
                gap = KnowledgeGap(
                    gap_type="conceptual",
                    concept=concept,
                    importance=self._assess_concept_importance(
                        concept, recent_experiences
                    ),
                    learnability=self._assess_learnability(concept),
                    novelty_score=novelty,
                    relevance_score=relevance,
                    context={
                        "mentioned_in": [
                            e.get("id", "")
                            for e in recent_experiences
                            if concept in str(e)
                        ]
                    },
                )
                gaps.append(gap)
                self.logger.debug(f"🔍 Conceptual gap detected: {concept} (novelty={novelty:.2f})")

        # 2. Gap causal: patrones sin explicación
        for pattern in observed_patterns:
            if not pattern.get("has_explanation", False):
                pattern_id = pattern.get("pattern_id", "unknown_pattern")
                surprise = float(pattern.get("unexpectedness", 0.5))
                
                gap = KnowledgeGap(
                    gap_type="causal",
                    concept=f"cause_of_{pattern_id}",
                    importance=float(pattern.get("frequency", 0.5)),
                    surprise_level=surprise,
                    novelty_score=surprise,  # Patrones sorprendentes son novedosos
                    relevance_score=0.7,  # Entender causas es siempre relevante
                    context={"pattern": pattern},
                )
                gaps.append(gap)
                self.logger.debug(f"🔍 Causal gap detected for pattern: {pattern_id}")

        # 3. Gap de habilidad: cosas que queremos hacer pero no sabemos cómo
        desired_capabilities = self._identify_desired_capabilities(recent_experiences)
        for capability in desired_capabilities:
            gap = KnowledgeGap(
                gap_type="skill",
                concept=capability,
                importance=0.8,
                learnability=0.7,
                novelty_score=0.6,
                relevance_score=0.9,  # Habilidades son altamente relevantes
                context={"desired": True},
            )
            gaps.append(gap)
            self.logger.debug(f"🔍 Skill gap detected: {capability}")

        # Ordenar por curiosity score compuesto
        current_time = time.time()
        gaps.sort(key=lambda g: g.get_curiosity_score(current_time), reverse=True)

        # Actualizar lista de gaps activos
        self.knowledge_gaps.extend(gaps)
        self.knowledge_gaps = self.knowledge_gaps[-50:]
        
        # Actualizar métricas de diversidad
        unique_concepts = len(set(g.concept for g in self.knowledge_gaps))
        self.concept_diversity_history.append(unique_concepts)
        self.concept_diversity_history = self.concept_diversity_history[-100:]
        
        # Actualizar entropía de conocimiento
        self._update_knowledge_entropy()

        if gaps:
            self.logger.info(
                f"💡 Detected {len(gaps)} knowledge gaps (top: {gaps[0].concept}, "
                f"score={gaps[0].get_curiosity_score():.2f})"
            )

        return gaps

    def generate_learning_questions(
        self, gaps: List[KnowledgeGap]
    ) -> List[LearningQuestion]:
        """Genera preguntas de aprendizaje multi-tipo para gaps de conocimiento."""
        questions: List[LearningQuestion] = []

        for gap in gaps[:10]:
            if gap.gap_type == "conceptual":
                questions.extend(
                    [
                        LearningQuestion(
                            question=f"What is {gap.concept}?",
                            question_type="what",
                            related_gap=gap,
                            priority=gap.importance * 0.9,
                        ),
                        LearningQuestion(
                            question=f"How does {gap.concept} work?",
                            question_type="how",
                            related_gap=gap,
                            priority=gap.importance * 0.8,
                        ),
                        LearningQuestion(
                            question=f"Why is {gap.concept} important?",
                            question_type="why",
                            related_gap=gap,
                            priority=gap.importance * 0.7,
                        ),
                    ]
                )

            elif gap.gap_type == "causal":
                pattern_id = gap.context.get("pattern", {}).get(
                    "pattern_id", "this pattern"
                )
                questions.extend(
                    [
                        LearningQuestion(
                            question=f"Why does {pattern_id} happen?",
                            question_type="why",
                            related_gap=gap,
                            priority=gap.importance * 0.9,
                        ),
                        LearningQuestion(
                            question=f"What causes {pattern_id}?",
                            question_type="what",
                            related_gap=gap,
                            priority=gap.importance * 0.85,
                        ),
                    ]
                )

            elif gap.gap_type == "skill":
                questions.extend(
                    [
                        LearningQuestion(
                            question=f"How do I learn {gap.concept}?",
                            question_type="how",
                            related_gap=gap,
                            priority=gap.importance * 0.9,
                            expected_answer_complexity="complex",
                        ),
                        LearningQuestion(
                            question=f"What are the prerequisites for {gap.concept}?",
                            question_type="what",
                            related_gap=gap,
                            priority=gap.importance * 0.8,
                        ),
                    ]
                )

        questions.sort(key=lambda q: float(q.priority), reverse=True)
        self.active_questions.extend(questions)
        self.active_questions = self.active_questions[-100:]

        if questions:
            self.logger.info(
                f"💭 Generated {len(questions)} questions (top: {questions[0].question})"
            )

        return questions

    def plan_exploration(
        self, questions: List[LearningQuestion], available_resources: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Planifica cómo explorar y responder preguntas con ROI optimizado."""
        exploration_plan: List[Dict[str, Any]] = []

        for question in questions[:5]:
            methods = self._select_exploration_methods(question, available_resources)
            effort = self._estimate_effort(question, methods)
            benefit = self._estimate_benefit(question)
            expected_value = benefit / max(effort, 0.1)

            exploration_plan.append(
                {
                    "question": question.question,
                    "priority": float(question.priority),
                    "methods": methods,
                    "estimated_effort_minutes": float(effort),
                    "expected_benefit": float(benefit),
                    "expected_value": float(expected_value),
                    "related_gap": question.related_gap.concept
                    if question.related_gap
                    else None,
                }
            )

        exploration_plan.sort(key=lambda x: float(x["expected_value"]), reverse=True)

        if exploration_plan:
            self.logger.info(
                f"📋 Created exploration plan with {len(exploration_plan)} items "
                f"(top ROI: {exploration_plan[0]['expected_value']:.2f})"
            )

        return exploration_plan

    def record_exploration_outcome(
        self,
        question: LearningQuestion,
        answer: Optional[str],
        success: bool,
        insights: Optional[List[str]] = None,
        answer_source: str = "unknown",
    ) -> None:
        """Registra resultado de exploración y actualiza métricas."""
        outcome: Dict[str, Any] = {
            "question": question.question,
            "answer": answer,
            "success": success,
            "insights": insights or [],
            "timestamp": time.time(),
            "related_gap": question.related_gap.concept
            if question.related_gap
            else None,
            "answer_source": answer_source,
        }

        self.exploration_history.append(outcome)

        if success:
            # Marcar pregunta como respondida
            quality = 0.8 if len(insights or []) > 0 else 0.6
            question.mark_answered(quality, answer_source)
            
            if question.related_gap:
                # Marcar gap como explorado
                question.related_gap.mark_explored()
                # Reducir importancia (ya aprendido)
                question.related_gap.importance *= 0.7
                
                # Actualizar expertise del dominio
                domain = self._extract_domain(question.related_gap.concept)
                self.domain_expertise[domain] = min(1.0, self.domain_expertise[domain] + 0.1)
                
            self.logger.info(f"✅ Successfully explored: {question.question[:60]}")
        else:
            # Incrementar importancia si fallamos (más curioso sobre esto)
            if question.related_gap:
                question.related_gap.importance = min(1.0, question.related_gap.importance * 1.1)
                
            self.logger.warning(f"❌ Failed to explore: {question.question[:60]}")

        # Actualizar eficiencia de exploración
        recent_successes = sum(1 for exp in self.exploration_history[-20:] if exp.get("success"))
        self.exploration_efficiency = recent_successes / 20.0
        
        self.exploration_history = self.exploration_history[-200:]

    def get_curiosity_level(self) -> float:
        """Calcula nivel actual de curiosidad del sistema."""
        gap_factor = min(len(self.knowledge_gaps) / 50.0, 1.0)

        recent_successes = sum(
            1 for exp in self.exploration_history[-20:] if exp.get("success")
        )
        success_dampening = max(0.5, 1.0 - (recent_successes / 20.0))

        curiosity = self.curiosity_intensity * gap_factor * success_dampening
        return curiosity

    def get_most_curious_about(self, limit: int = 3) -> List[str]:
        """Retorna los temas sobre los que el sistema está más curioso."""
        sorted_gaps = sorted(
            self.knowledge_gaps, key=lambda g: g.importance, reverse=True
        )
        return [gap.concept for gap in sorted_gaps[:limit]]

    # ==================== MÉTODOS PRIVADOS ====================

    def _extract_mentioned_concepts(
        self, experiences: List[Dict[str, Any]]
    ) -> List[str]:
        """Extrae conceptos mencionados en experiencias."""
        concepts = []
        for exp in experiences:
            if "concepts_mentioned" in exp:
                concepts.extend(exp["concepts_mentioned"])

            if "text" in exp:
                words = str(exp["text"]).split()
                concepts.extend([w for w in words if len(w) > 3 and w[0].isupper()])

            if "metadata" in exp and "tags" in exp["metadata"]:
                concepts.extend(exp["metadata"]["tags"])

        return list(set(concepts))

    def _assess_concept_importance(
        self, concept: str, experiences: List[Dict[str, Any]]
    ) -> float:
        """Evalúa importancia de un concepto."""
        mentions = sum(1 for exp in experiences if concept in str(exp))
        frequency_score = min(mentions / 10.0, 1.0)
        context_score = 0.5
        return (frequency_score + context_score) / 2

    def _assess_learnability(self, concept: str) -> float:
        """Evalúa qué tan fácil es aprender un concepto."""
        length_score = 1.0 - min(len(concept) / 50.0, 1.0)
        semantic_complexity = 0.5
        return (length_score + (1 - semantic_complexity)) / 2

    def _identify_desired_capabilities(
        self, experiences: List[Dict[str, Any]]
    ) -> List[str]:
        """Identifica capacidades que el sistema desea tener."""
        capabilities = []

        for exp in experiences:
            if exp.get("outcome") == "failure" and "attempted_action" in exp:
                capabilities.append(exp["attempted_action"])

            if "limitation" in exp:
                capabilities.append(f"overcome_{exp['limitation']}")

            if exp.get("goal_achieved") is False and "goal" in exp:
                capabilities.append(f"achieve_{exp['goal']}")

        return list(set(capabilities))

    def _select_exploration_methods(
        self, question: LearningQuestion, available_resources: Dict[str, Any]
    ) -> List[str]:
        """Selecciona métodos para explorar una pregunta."""
        methods = []

        if available_resources.get("web_access"):
            methods.append("web_search")

        if question.question_type in ["how", "what"]:
            methods.append("internal_simulation")

        if available_resources.get("database"):
            methods.append("historical_analysis")

        if available_resources.get("external_apis"):
            methods.append("api_query")

        return methods if methods else ["reflection"]

    def _estimate_effort(self, question: LearningQuestion, methods: List[str]) -> float:
        """Estima esfuerzo en minutos para responder pregunta."""
        base_effort = {"simple": 5.0, "medium": 15.0, "complex": 45.0}

        effort = base_effort.get(question.expected_answer_complexity, 15.0)

        if "web_search" in methods:
            effort *= 1.2
        if "internal_simulation" in methods:
            effort *= 0.8
        if "api_query" in methods:
            effort *= 0.6

        return effort

    def _estimate_benefit(self, question: LearningQuestion) -> float:
        """Estima beneficio de responder pregunta."""
        base_benefit = question.priority

        if question.question_type == "why":
            base_benefit *= 1.3
        elif question.question_type == "how":
            base_benefit *= 1.2
        elif question.question_type == "what":
            base_benefit *= 1.1

        return min(base_benefit, 1.0)

    # ==================== MÉTODOS AVANZADOS DE CURIOSIDAD ====================

    def _calculate_novelty_score(self, concept: str) -> float:
        """Calcula score de novedad para un concepto."""
        # Verificar si concepto ya fue explorado
        if concept in [g.concept for g in self.knowledge_gaps]:
            return 0.3  # Concepto conocido, baja novedad
        
        # Verificar exploración histórica
        explored_concepts = [exp.get("related_gap") for exp in self.exploration_history]
        if concept in explored_concepts:
            return 0.4  # Explorado antes, novedad media-baja
        
        # Concepto completamente nuevo
        return 0.9

    def _calculate_relevance_score(
        self, concept: str, recent_experiences: List[Dict[str, Any]]
    ) -> float:
        """Calcula qué tan relevante es un concepto para objetivos actuales."""
        # Contar menciones recientes
        mention_count = sum(1 for exp in recent_experiences if concept in str(exp))
        frequency_score = min(mention_count / 5.0, 1.0)
        
        # Verificar si está en contextos importantes
        importance_score = 0.5
        for exp in recent_experiences:
            if concept in str(exp):
                # Más relevante si aparece en experiencias exitosas o con goals
                if exp.get("success") or "goal" in exp:
                    importance_score = 0.8
                    break
        
        return (frequency_score + importance_score) / 2.0

    def _update_knowledge_entropy(self) -> None:
        """Actualiza la entropía del conocimiento (diversidad de conceptos)."""
        if not self.knowledge_gaps:
            self.knowledge_entropy = 1.0
            return
        
        # Calcular distribución de tipos de gaps
        gap_types = [g.gap_type for g in self.knowledge_gaps]
        type_counts: Dict[str, int] = defaultdict(int)
        for gt in gap_types:
            type_counts[gt] += 1
        
        # Calcular entropía de Shannon
        total = len(gap_types)
        entropy = 0.0
        for count in type_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Normalizar (máx entropía para 4 tipos = log2(4) = 2.0)
        self.knowledge_entropy = entropy / 2.0 if entropy > 0 else 0.0

    def _extract_domain(self, concept: str) -> str:
        """Extrae el dominio de conocimiento de un concepto."""
        # Simplificado: usar primera palabra o prefijo
        parts = concept.split("_")
        if len(parts) > 1:
            return parts[0]
        
        # Categorías por defecto basadas en palabras clave
        concept_lower = concept.lower()
        if any(word in concept_lower for word in ["system", "process", "algorithm"]):
            return "systems"
        elif any(word in concept_lower for word in ["learn", "train", "model"]):
            return "learning"
        elif any(word in concept_lower for word in ["data", "information", "knowledge"]):
            return "data"
        elif any(word in concept_lower for word in ["cause", "reason", "why"]):
            return "causal"
        else:
            return "general"

    # ==================== MÉTODOS DE ANÁLISIS ====================

    def get_exploration_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas completas de exploración."""
        if not self.exploration_history:
            return {
                "total_explorations": 0,
                "success_rate": 0.0,
                "avg_answer_quality": 0.0,
                "exploration_efficiency": 0.0,
            }
        
        total = len(self.exploration_history)
        successes = sum(1 for exp in self.exploration_history if exp.get("success"))
        
        # Calcular calidad promedio de respuestas
        answered_questions = [q for q in self.active_questions if q.answered]
        avg_quality = 0.0
        if answered_questions:
            qualities = [q.answer_quality for q in answered_questions if q.answer_quality]
            avg_quality = sum(qualities) / len(qualities) if qualities else 0.0
        
        return {
            "total_explorations": total,
            "success_rate": successes / total,
            "avg_answer_quality": avg_quality,
            "exploration_efficiency": self.exploration_efficiency,
            "active_gaps": len(self.knowledge_gaps),
            "active_questions": len([q for q in self.active_questions if not q.answered]),
            "concept_diversity": self.concept_diversity_history[-1] if self.concept_diversity_history else 0,
            "knowledge_entropy": self.knowledge_entropy,
            "explored_domains": dict(self.domain_expertise),
        }

    def get_curiosity_insights(self) -> Dict[str, Any]:
        """Retorna insights profundos sobre el estado de curiosidad."""
        top_gaps = sorted(
            self.knowledge_gaps,
            key=lambda g: g.get_curiosity_score(),
            reverse=True
        )[:5]
        
        top_domains = sorted(
            self.explored_domains.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return {
            "curiosity_level": self.get_curiosity_level(),
            "most_curious_about": [g.concept for g in top_gaps],
            "top_curiosity_scores": [g.get_curiosity_score() for g in top_gaps],
            "most_explored_domains": [{"domain": d[0], "count": d[1]} for d in top_domains],
            "domain_expertise": dict(self.domain_expertise),
            "knowledge_entropy": self.knowledge_entropy,
            "exploration_efficiency": self.exploration_efficiency,
            "unanswered_questions": len([q for q in self.active_questions if not q.answered]),
        }

    def connect_bdi_system(self, bdi_system: Any) -> None:
        """Conecta el motor de curiosidad con el sistema BDI para crear deseos de exploración."""
        self.bdi_system = bdi_system
        self.logger.info("🔗 Curiosity engine connected to BDI system")
        
        # Generar deseos de exploración basados en gaps actuales
        self._sync_gaps_to_desires()

    def _sync_gaps_to_desires(self) -> None:
        """Sincroniza knowledge gaps con deseos de autorrealización en BDI."""
        if not hasattr(self, "bdi_system") or not self.bdi_system:
            return
        
        # Importar tipos necesarios
        try:
            from .bdi import NeedLevel, MotivationType
            
            # Convertir top gaps en deseos de exploración
            top_gaps = sorted(
                self.knowledge_gaps,
                key=lambda g: g.get_curiosity_score(),
                reverse=True
            )[:5]
            
            for gap in top_gaps:
                desire_name = f"explore_{gap.concept}"
                priority = gap.get_curiosity_score()
                
                # Añadir como deseo de autorrealización (exploración es crecimiento)
                self.bdi_system.add_desire(
                    name=desire_name,
                    priority=priority,
                    need_level=NeedLevel.SELF_ACTUALIZATION,
                    motivation_type=MotivationType.INTRINSIC,
                    intrinsic_motivation=0.9,  # Exploración es altamente intrínseca
                )
            
            self.logger.info(f"📚 Synced {len(top_gaps)} knowledge gaps to BDI desires")
        
        except Exception as e:
            logger.error(f"Error en curiosity.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ Could not sync gaps to BDI: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # MÉTODOS AVANZADOS 2026: Integration with Advanced Components
    # ═══════════════════════════════════════════════════════════════════
    
    def calculate_exploration_value_advanced(
        self,
        gap: KnowledgeGap
    ) -> float:
        """
        Calcula valor de explorar un gap usando information gain.
        
        Integra InformationGainCalculator para estimación más precisa.
        
        Args:
            gap: Knowledge gap a evaluar
            
        Returns:
            Valor de exploración (0-1, mayor es mejor)
        """
        if not self.info_gain_calculator:
            # Fallback a cálculo básico
            return gap.get_curiosity_score()
        
        # Estimar incertidumbre basada en novelty y exploration_count
        uncertainty = gap.novelty_score * (1.0 / (1.0 + gap.exploration_count * 0.2))
        
        # Estimar dificultad basada en importance y learnability
        difficulty = (1.0 - gap.learnability) * gap.importance
        
        # Usar calculador de information gain
        value = self.info_gain_calculator.estimate_exploration_value(
            gap.concept,
            uncertainty,
            difficulty
        )
        
        return value
    
    def detect_novel_concepts_advanced(
        self,
        concepts: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Detecta qué conceptos son novedosos.
        
        Integra NoveltyDetector para detección precisa.
        
        Args:
            concepts: Lista de conceptos a evaluar
            
        Returns:
            Lista de (concept, novelty_score) para conceptos novedosos
        """
        if not self.novelty_detector:
            # Fallback: todos son moderadamente novedosos
            return [(c, 0.5) for c in concepts]
        
        novel_concepts = []
        
        for concept in concepts:
            is_novel, novelty_score = self.novelty_detector.detect_novelty(concept)
            
            if is_novel:
                novel_concepts.append((concept, novelty_score))
                
                # Actualizar tracking en timeline
                self.novelty_timeline.append((time.time(), novelty_score))
                if len(self.novelty_timeline) > 1000:
                    self.novelty_timeline = self.novelty_timeline[-1000:]
        
        return novel_concepts
    
    def select_next_exploration_advanced(
        self,
        candidates: List[KnowledgeGap]
    ) -> Optional[KnowledgeGap]:
        """
        Selecciona próximo gap a explorar usando strategy.
        
        Integra ExplorationStrategy para selección óptima.
        
        Args:
            candidates: Lista de gaps candidatos
            
        Returns:
            Gap seleccionado, o None si no hay candidatos
        """
        if not candidates:
            return None
        
        # Registrar candidatos como arms si no existen
        for gap in candidates:
            if gap.concept not in self._arm_registry:
                self.exploration_strategy_engine.register_arm(gap.concept)
                self._arm_registry.add(gap.concept)
        
        # Seleccionar usando strategy
        selected_concept = self.exploration_strategy_engine.select_arm()
        
        if not selected_concept:
            return None
        
        # Encontrar gap correspondiente
        for gap in candidates:
            if gap.concept == selected_concept:
                return gap
        
        return None
    
    def update_exploration_reward(
        self,
        concept: str,
        success: bool,
        learning_gained: float = 0.0
    ) -> None:
        """
        Actualiza reward de exploración para un concepto.
        
        Integra con ExplorationStrategy para aprendizaje.
        
        Args:
            concept: Concepto explorado
            success: Si la exploración fue exitosa
            learning_gained: Cantidad de aprendizaje obtenido (0-1)
        """
        # Calcular reward compuesto
        reward = 0.0
        
        if success:
            reward += 0.5  # Base reward por éxito
        
        reward += learning_gained * 0.5  # Reward proporcional a aprendizaje
        
        # Normalizar a [0, 1]
        reward = max(0.0, min(1.0, reward))
        
        # Actualizar en strategy engine
        if concept in self._arm_registry:
            self.exploration_strategy_engine.update_arm(concept, reward)
            
            self.logger.debug(
                f"📊 Updated exploration reward for '{concept}': "
                f"reward={reward:.3f}, success={success}"
            )
    
    def generate_hypothesis_from_gap(
        self,
        gap: KnowledgeGap
    ) -> Hypothesis:
        """
        Genera hipótesis científica para explorar un gap.
        
        Integra CuriosityDrivenPlanner para generación de hipótesis.
        
        Args:
            gap: Knowledge gap a explorar
            
        Returns:
            Hipótesis generada
        """
        hypothesis = self.curiosity_planner.generate_hypothesis(gap)
        
        self.logger.info(
            f"💡 Generated hypothesis for gap '{gap.concept}': "
            f"{hypothesis.statement[:80]}..."
        )
        
        return hypothesis
    
    def prioritize_gaps_by_info_gain(
        self,
        gaps: Optional[List[KnowledgeGap]] = None
    ) -> List[KnowledgeGap]:
        """
        Prioriza knowledge gaps por information gain esperado.
        
        Integra CuriosityDrivenPlanner para priorización óptima.
        
        Args:
            gaps: Lista de gaps (usa self.knowledge_gaps si None)
            
        Returns:
            Lista de gaps ordenada por prioridad
        """
        if gaps is None:
            gaps = self.knowledge_gaps
        
        if not gaps:
            return []
        
        if not self.info_gain_calculator:
            # Fallback a ordenamiento básico
            return sorted(gaps, key=lambda g: g.get_curiosity_score(), reverse=True)
        
        # Usar curiosity planner para priorización avanzada
        prioritized = self.curiosity_planner.prioritize_exploration(
            gaps,
            self.info_gain_calculator
        )
        
        return prioritized
    
    def get_advanced_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas avanzadas de todos los componentes.
        
        Returns:
            Dict con métricas completas
        """
        stats = {
            "curiosity_engine": {
                "total_gaps": len(self.knowledge_gaps),
                "active_questions": len(self.active_questions),
                "exploration_history": len(self.exploration_history),
                "curiosity_intensity": self.curiosity_intensity,
                "exploration_efficiency": self.exploration_efficiency,
                "knowledge_entropy": self.knowledge_entropy
            }
        }
        
        if self.info_gain_calculator:
            stats["information_gain"] = self.info_gain_calculator.get_stats()
        
        if self.novelty_detector:
            stats["novelty_detection"] = self.novelty_detector.get_stats()
        
        stats["exploration_strategy"] = self.exploration_strategy_engine.get_stats()
        stats["curiosity_planner"] = self.curiosity_planner.get_stats()
        
        return stats


# ═══════════════════════════════════════════════════════════════════════
# CLASES AVANZADAS 2026: Information Gain, Novelty, Exploration Strategies
# ═══════════════════════════════════════════════════════════════════════


class InformationGainCalculator:
    """
    Calculador de ganancia de información para guiar exploración.
    
    Usa conceptos de teoría de información:
    - Entropy: Medida de incertidumbre
    - Information Gain: Reducción esperada de entropía
    - KL Divergence: Diferencia entre distribuciones
    
    Optimizado para M4 Metal (16GB RAM).
    """
    
    def __init__(self):
        self.logger = logger.getChild("info_gain")
        
        # Historial de cálculos
        self.calculation_history: List[Dict[str, Any]] = []
        self.max_history: int = 500  # Límite para RAM
        
        # Métricas
        self.total_calculations: int = 0
    
    def calculate_entropy(self, probabilities: List[float]) -> float:
        """
        Calcula entropía de Shannon.
        
        H(X) = -Σ p(x) * log2(p(x))
        
        Args:
            probabilities: Lista de probabilidades (deben sumar ~1.0)
            
        Returns:
            Entropía en bits
        """
        entropy = 0.0
        
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def calculate_information_gain(
        self,
        prior_entropy: float,
        posterior_probabilities: List[List[float]],
        branch_probabilities: List[float]
    ) -> float:
        """
        Calcula ganancia de información esperada.
        
        IG = H(prior) - Σ p(branch) * H(posterior | branch)
        
        Args:
            prior_entropy: Entropía antes de la observación
            posterior_probabilities: Probabilidades después de cada rama
            branch_probabilities: Probabilidad de cada rama
            
        Returns:
            Information gain en bits
        """
        # Calcular entropía condicional esperada
        expected_posterior_entropy = 0.0
        
        for branch_prob, posterior_probs in zip(branch_probabilities, posterior_probabilities):
            posterior_entropy = self.calculate_entropy(posterior_probs)
            expected_posterior_entropy += branch_prob * posterior_entropy
        
        # Information gain es la reducción de entropía
        info_gain = prior_entropy - expected_posterior_entropy
        
        self.total_calculations += 1
        
        return max(0.0, info_gain)  # No puede ser negativo
    
    def calculate_kl_divergence(
        self,
        p_distribution: List[float],
        q_distribution: List[float]
    ) -> float:
        """
        Calcula divergencia de Kullback-Leibler.
        
        KL(P||Q) = Σ p(x) * log(p(x) / q(x))
        
        Mide qué tan diferente es Q de P.
        
        Args:
            p_distribution: Distribución objetivo
            q_distribution: Distribución aproximada
            
        Returns:
            KL divergence (siempre >= 0)
        """
        kl_div = 0.0
        epsilon = 1e-10  # Para evitar log(0)
        
        for p, q in zip(p_distribution, q_distribution):
            if p > 0:
                kl_div += p * math.log((p + epsilon) / (q + epsilon))
        
        return max(0.0, kl_div)
    
    def estimate_exploration_value(
        self,
        concept: str,
        current_knowledge_uncertainty: float,
        expected_learning_difficulty: float = 0.5
    ) -> float:
        """
        Estima valor de explorar un concepto.
        
        Combina:
        - Reducción esperada de incertidumbre (information gain)
        - Dificultad de aprendizaje (costo)
        - Relevancia actual
        
        Args:
            concept: Concepto a explorar
            current_knowledge_uncertainty: Incertidumbre actual (0-1)
            expected_learning_difficulty: Dificultad esperada (0-1)
            
        Returns:
            Valor de exploración (0-1, mayor es mejor)
        """
        # Estimar information gain basado en incertidumbre
        # A mayor incertidumbre, mayor potencial de aprendizaje
        potential_gain = current_knowledge_uncertainty
        
        # Penalizar por dificultad (explorar cosas fáciles tiene mejor ROI)
        learning_cost = expected_learning_difficulty
        
        # Valor = beneficio / costo
        if learning_cost > 0:
            exploration_value = potential_gain / (1.0 + learning_cost)
        else:
            exploration_value = potential_gain
        
        # Normalizar a [0, 1]
        exploration_value = max(0.0, min(1.0, exploration_value))
        
        # Guardar en historial
        self.calculation_history.append({
            "concept": concept,
            "uncertainty": current_knowledge_uncertainty,
            "difficulty": expected_learning_difficulty,
            "value": exploration_value,
            "timestamp": time.time()
        })
        
        if len(self.calculation_history) > self.max_history:
            self.calculation_history = self.calculation_history[-self.max_history:]
        
        return exploration_value
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de cálculos."""
        if not self.calculation_history:
            return {"total_calculations": 0}
        
        recent = self.calculation_history[-100:]
        
        return {
            "total_calculations": self.total_calculations,
            "history_size": len(self.calculation_history),
            "avg_exploration_value": sum(c["value"] for c in recent) / len(recent),
            "avg_uncertainty": sum(c["uncertainty"] for c in recent) / len(recent),
            "avg_difficulty": sum(c["difficulty"] for c in recent) / len(recent)
        }


class NoveltyDetector:
    """
    Detector de novedad para identificar conceptos/experiencias novedosas.
    
    Usa múltiples señales:
    - Frecuencia histórica del concepto
    - Distancia semántica a conceptos conocidos
    - Surprise level (predicción vs realidad)
    - Temporal decay de novedad
    
    Optimizado para M4 Metal (16GB RAM).
    """
    
    def __init__(self, novelty_threshold: float = 0.6):
        self.logger = logger.getChild("novelty")
        self.novelty_threshold = novelty_threshold
        
        # Tracking de conceptos vistos
        self.concept_frequency: Dict[str, int] = defaultdict(int)
        self.concept_first_seen: Dict[str, float] = {}
        self.concept_last_seen: Dict[str, float] = {}
        
        # Historial de detecciones
        self.novelty_history: List[Dict[str, Any]] = []
        self.max_history: int = 1000  # Límite para RAM
        
        # Métricas
        self.total_detections: int = 0
        self.novel_concepts_found: int = 0
    
    def detect_novelty(
        self,
        concept: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, float]:
        """
        Detecta si un concepto es novedoso.
        
        Args:
            concept: Concepto a evaluar
            context: Contexto adicional (opcional)
            
        Returns:
            Tuple (is_novel, novelty_score)
        """
        current_time = time.time()
        self.total_detections += 1
        
        # Señal 1: Frecuencia (menos visto = más novedoso)
        frequency = self.concept_frequency.get(concept, 0)
        frequency_novelty = 1.0 / (1.0 + frequency * 0.5)
        
        # Señal 2: Temporal (primera vez = máxima novedad)
        temporal_novelty = 1.0
        if concept in self.concept_first_seen:
            hours_since_first = (current_time - self.concept_first_seen[concept]) / 3600.0
            # Decay exponencial: novedad decae en 48 horas
            temporal_novelty = math.exp(-hours_since_first / 48.0)
        
        # Señal 3: Recencia (no visto recientemente = más novedoso)
        recency_novelty = 1.0
        if concept in self.concept_last_seen:
            hours_since_last = (current_time - self.concept_last_seen[concept]) / 3600.0
            # Recupera novedad si no se ve por 24 horas
            recency_novelty = min(1.0, hours_since_last / 24.0)
        
        # Combinar señales
        novelty_score = (
            frequency_novelty * 0.4 +
            temporal_novelty * 0.3 +
            recency_novelty * 0.3
        )
        
        is_novel = novelty_score >= self.novelty_threshold
        
        # Actualizar tracking
        self.concept_frequency[concept] += 1
        
        if concept not in self.concept_first_seen:
            self.concept_first_seen[concept] = current_time
        
        self.concept_last_seen[concept] = current_time
        
        if is_novel:
            self.novel_concepts_found += 1
        
        # Guardar en historial
        self.novelty_history.append({
            "concept": concept,
            "is_novel": is_novel,
            "score": novelty_score,
            "frequency": frequency,
            "timestamp": current_time
        })
        
        if len(self.novelty_history) > self.max_history:
            self.novelty_history = self.novelty_history[-self.max_history:]
        
        if is_novel:
            self.logger.debug(f"🆕 Novel concept detected: {concept} (score={novelty_score:.3f})")
        
        return is_novel, novelty_score
    
    def calculate_surprise(
        self,
        predicted_probability: float,
        actual_outcome: bool
    ) -> float:
        """
        Calcula surprise level basado en predicción vs realidad.
        
        Surprise = -log2(probability of actual outcome)
        
        Args:
            predicted_probability: Probabilidad predicha del outcome positivo
            actual_outcome: Si el outcome fue positivo
            
        Returns:
            Surprise en bits (mayor = más sorprendente)
        """
        # Probabilidad del outcome que realmente ocurrió
        if actual_outcome:
            p_actual = predicted_probability
        else:
            p_actual = 1.0 - predicted_probability
        
        # Evitar log(0)
        p_actual = max(0.001, min(0.999, p_actual))
        
        # Surprise es negativo del log de probabilidad
        surprise = -math.log2(p_actual)
        
        return surprise
    
    def get_most_novel_concepts(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retorna conceptos más novedosos recientes.
        
        Args:
            top_k: Número de conceptos a retornar
            
        Returns:
            Lista de (concept, novelty_score)
        """
        if not self.novelty_history:
            return []
        
        # Filtrar solo novedosos
        novel_items = [
            (item["concept"], item["score"])
            for item in self.novelty_history
            if item["is_novel"]
        ]
        
        # Ordenar por score
        novel_items.sort(key=lambda x: x[1], reverse=True)
        
        return novel_items[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de detección."""
        novelty_rate = (
            self.novel_concepts_found / self.total_detections
            if self.total_detections > 0 else 0.0
        )
        
        return {
            "total_detections": self.total_detections,
            "novel_concepts_found": self.novel_concepts_found,
            "novelty_rate": novelty_rate,
            "unique_concepts_seen": len(self.concept_frequency),
            "history_size": len(self.novelty_history),
            "avg_frequency": (
                sum(self.concept_frequency.values()) / len(self.concept_frequency)
                if self.concept_frequency else 0.0
            )
        }


@dataclass
class ExplorationArm:
    """
    Brazo de exploración para multi-armed bandit.
    
    Representa una acción/concepto explorable con tracking de resultados.
    """
    name: str
    times_tried: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    last_tried: Optional[float] = None
    
    def update(self, reward: float):
        """Actualiza arm con nuevo reward."""
        self.times_tried += 1
        self.total_reward += reward
        self.avg_reward = self.total_reward / self.times_tried
        self.last_tried = time.time()


class ExplorationStrategy:
    """
    Estrategias de exploración para curiosity-driven learning.
    
    Implementa múltiples algoritmos:
    - Random: Exploración aleatoria
    - Greedy: Siempre elegir mejor opción conocida
    - Epsilon-Greedy: Greedy con probabilidad epsilon de explorar
    - UCB1: Upper Confidence Bound
    - Thompson Sampling: Bayesian sampling
    
    Optimizado para M4 Metal (16GB RAM).
    """
    
    def __init__(
        self,
        strategy_type: str = "ucb1",
        epsilon: float = 0.1,
        ucb_c: float = 2.0
    ):
        self.logger = logger.getChild("exploration")
        self.strategy_type = strategy_type
        self.epsilon = epsilon
        self.ucb_c = ucb_c
        
        # Arms (acciones explorables)
        self.arms: Dict[str, ExplorationArm] = {}
        
        # Historial de selecciones
        self.selection_history: List[Tuple[str, float]] = []
        self.max_history: int = 1000
        
        # Métricas
        self.total_selections: int = 0
        self.total_reward: float = 0.0
    
    def register_arm(self, name: str) -> ExplorationArm:
        """Registra nuevo brazo de exploración."""
        if name not in self.arms:
            self.arms[name] = ExplorationArm(name=name)
            self.logger.debug(f"🎰 Registered exploration arm: {name}")
        return self.arms[name]
    
    def select_arm(self) -> Optional[str]:
        """
        Selecciona brazo según estrategia configurada.
        
        Returns:
            Nombre del brazo seleccionado, o None si no hay brazos
        """
        if not self.arms:
            return None
        
        self.total_selections += 1
        
        if self.strategy_type == "random":
            selected = self._select_random()
        elif self.strategy_type == "greedy":
            selected = self._select_greedy()
        elif self.strategy_type == "epsilon_greedy":
            selected = self._select_epsilon_greedy()
        elif self.strategy_type == "ucb1":
            selected = self._select_ucb1()
        elif self.strategy_type == "thompson":
            selected = self._select_thompson()
        else:
            self.logger.warning(f"Unknown strategy: {self.strategy_type}, using random")
            selected = self._select_random()
        
        return selected
    
    def _select_random(self) -> str:
        """Selección aleatoria."""
        return random.choice(list(self.arms.keys()))
    
    def _select_greedy(self) -> str:
        """Selección greedy (mejor avg_reward)."""
        # Si algún brazo no se ha probado, probarlo
        untried = [name for name, arm in self.arms.items() if arm.times_tried == 0]
        if untried:
            return random.choice(untried)
        
        # Elegir brazo con mejor avg_reward
        best_arm = max(self.arms.values(), key=lambda a: a.avg_reward)
        return best_arm.name
    
    def _select_epsilon_greedy(self) -> str:
        """Epsilon-greedy selection."""
        
        # Con probabilidad epsilon, explorar
        if random.random() < self.epsilon:
            return self._select_random()
        else:
            return self._select_greedy()
    
    def _select_ucb1(self) -> str:
        """
        Upper Confidence Bound 1 selection.
        
        UCB1 = avg_reward + c * sqrt(ln(total_selections) / times_tried)
        """
        # Si algún brazo no se ha probado, probarlo (UCB = infinito)
        untried = [name for name, arm in self.arms.items() if arm.times_tried == 0]
        if untried:
            return random.choice(untried)
        
        # Calcular UCB1 para cada brazo
        ucb_scores = {}
        for name, arm in self.arms.items():
            exploration_bonus = self.ucb_c * math.sqrt(
                math.log(self.total_selections) / arm.times_tried
            )
            ucb_scores[name] = arm.avg_reward + exploration_bonus
        
        # Elegir brazo con mayor UCB
        best_arm = max(ucb_scores.items(), key=lambda x: x[1])
        return best_arm[0]
    
    def _select_thompson(self) -> str:
        """
        Thompson Sampling (Bayesian).
        
        Simplificación: asume Beta distribution para rewards.
        """
        
        # Si algún brazo no se ha probado, probarlo
        untried = [name for name, arm in self.arms.items() if arm.times_tried == 0]
        if untried:
            return random.choice(untried)
        
        # Samplear de posterior Beta para cada brazo
        # Beta(alpha, beta) con alpha=successes+1, beta=failures+1
        samples = {}
        for name, arm in self.arms.items():
            # Estimar successes como avg_reward * times_tried
            successes = arm.avg_reward * arm.times_tried
            failures = arm.times_tried - successes
            
            # Samplear (simulación simple)
            # En producción usaría numpy.random.beta
            alpha = successes + 1
            beta_param = failures + 1
            
            # Approximación: usar avg como sample
            samples[name] = arm.avg_reward + random.gauss(0, 0.1)
        
        # Elegir brazo con sample más alto
        best_arm = max(samples.items(), key=lambda x: x[1])
        return best_arm[0]
    
    def update_arm(self, name: str, reward: float):
        """
        Actualiza arm con reward obtenido.
        
        Args:
            name: Nombre del arm
            reward: Reward obtenido (0.0 a 1.0)
        """
        if name not in self.arms:
            self.logger.warning(f"Arm {name} not registered, registering now")
            self.register_arm(name)
        
        self.arms[name].update(reward)
        self.total_reward += reward
        
        # Guardar en historial
        self.selection_history.append((name, reward))
        if len(self.selection_history) > self.max_history:
            self.selection_history = self.selection_history[-self.max_history:]
        
        self.logger.debug(
            f"📊 Arm {name} updated: "
            f"tries={self.arms[name].times_tried}, "
            f"avg_reward={self.arms[name].avg_reward:.3f}"
        )
    
    def get_best_arm(self) -> Optional[str]:
        """Retorna brazo con mejor avg_reward."""
        if not self.arms:
            return None
        
        tried_arms = [arm for arm in self.arms.values() if arm.times_tried > 0]
        if not tried_arms:
            return None
        
        best_arm = max(tried_arms, key=lambda a: a.avg_reward)
        return best_arm.name
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de exploración."""
        avg_reward = (
            self.total_reward / self.total_selections
            if self.total_selections > 0 else 0.0
        )
        
        best_arm_name = self.get_best_arm()
        best_arm_reward = (
            self.arms[best_arm_name].avg_reward
            if best_arm_name else 0.0
        )
        
        return {
            "strategy": self.strategy_type,
            "total_selections": self.total_selections,
            "total_arms": len(self.arms),
            "avg_reward": avg_reward,
            "best_arm": best_arm_name,
            "best_arm_reward": best_arm_reward,
            "regret": (
                best_arm_reward * self.total_selections - self.total_reward
                if self.total_selections > 0 else 0.0
            )
        }


@dataclass
class Hypothesis:
    """
    Hipótesis generada para exploración científica.
    
    Representa una conjetura testeable sobre el mundo.
    """
    statement: str
    confidence: float = 0.5  # Confianza inicial
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    experiments_proposed: List[str] = field(default_factory=list)
    tested: bool = False
    test_result: Optional[bool] = None
    created_at: float = field(default_factory=time.time)
    
    def add_evidence(self, evidence: str, supports: bool):
        """Añade evidencia a favor o en contra."""
        if supports:
            self.evidence_for.append(evidence)
        else:
            self.evidence_against.append(evidence)
        
        # Actualizar confianza basada en evidencia
        total_evidence = len(self.evidence_for) + len(self.evidence_against)
        if total_evidence > 0:
            self.confidence = len(self.evidence_for) / total_evidence


class CuriosityDrivenPlanner:
    """
    Planificador dirigido por curiosidad.
    
    Capacidades:
    - Generación de hipótesis explorables
    - Diseño de experimentos para validar hipótesis
    - Priorización de exploración por information gain
    - Integración con Planning System
    
    Optimizado para M4 Metal (16GB RAM).
    """
    
    def __init__(self):
        self.logger = logger.getChild("curiosity_planner")
        
        # Hipótesis activas
        self.hypotheses: List[Hypothesis] = []
        self.max_hypotheses: int = 100  # Límite para RAM
        
        # Experimentos planificados
        self.planned_experiments: List[Dict[str, Any]] = []
        self.max_experiments: int = 200
        
        # Métricas
        self.total_hypotheses_generated: int = 0
        self.hypotheses_confirmed: int = 0
        self.hypotheses_rejected: int = 0
    
    def generate_hypothesis(
        self,
        knowledge_gap: KnowledgeGap
    ) -> Hypothesis:
        """
        Genera hipótesis basada en knowledge gap.
        
        Args:
            knowledge_gap: Brecha de conocimiento identificada
            
        Returns:
            Hipótesis generada
        """
        # Generar statement basado en tipo de gap
        if knowledge_gap.gap_type == "causal":
            statement = f"If we understand {knowledge_gap.concept}, then we can predict related outcomes"
        elif knowledge_gap.gap_type == "conceptual":
            statement = f"{knowledge_gap.concept} has properties similar to known concepts"
        elif knowledge_gap.gap_type == "skill":
            statement = f"Learning {knowledge_gap.concept} will improve capability in related domains"
        else:
            statement = f"Exploring {knowledge_gap.concept} will reduce uncertainty"
        
        hypothesis = Hypothesis(
            statement=statement,
            confidence=0.5,
            experiments_proposed=[
                f"web_search_{knowledge_gap.concept}",
                f"analyze_related_concepts_{knowledge_gap.concept}"
            ]
        )
        
        self.hypotheses.append(hypothesis)
        self.total_hypotheses_generated += 1
        
        # Mantener límite
        if len(self.hypotheses) > self.max_hypotheses:
            # Remover hipótesis más antiguas y ya testadas
            tested = [h for h in self.hypotheses if h.tested]
            if tested:
                tested.sort(key=lambda h: h.created_at)
                self.hypotheses.remove(tested[0])
        
        self.logger.info(f"💡 Hypothesis generated: {statement[:80]}...")
        
        return hypothesis
    
    def design_experiment(
        self,
        hypothesis: Hypothesis,
        available_methods: List[str]
    ) -> Dict[str, Any]:
        """
        Diseña experimento para testear hipótesis.
        
        Args:
            hypothesis: Hipótesis a testear
            available_methods: Métodos disponibles (web_search, simulation, etc.)
            
        Returns:
            Dict con diseño del experimento
        """
        # Seleccionar método más apropiado
        method = "web_search"  # Default
        if "simulation" in available_methods:
            method = "simulation"
        elif "api_query" in available_methods:
            method = "api_query"
        
        experiment = {
            "hypothesis": hypothesis.statement,
            "method": method,
            "steps": [
                "Gather relevant data",
                "Analyze data for patterns",
                "Compare with hypothesis prediction",
                "Draw conclusion"
            ],
            "success_criteria": {
                "min_data_points": 5,
                "confidence_threshold": 0.7
            },
            "estimated_duration_minutes": 10,
            "created_at": time.time()
        }
        
        self.planned_experiments.append(experiment)
        
        # Mantener límite
        if len(self.planned_experiments) > self.max_experiments:
            self.planned_experiments = self.planned_experiments[-self.max_experiments:]
        
        self.logger.info(f"🧪 Experiment designed: {method} for hypothesis")
        
        return experiment
    
    def evaluate_hypothesis(
        self,
        hypothesis: Hypothesis,
        experiment_result: Dict[str, Any]
    ) -> bool:
        """
        Evalúa hipótesis basándose en resultado de experimento.
        
        Args:
            hypothesis: Hipótesis a evaluar
            experiment_result: Resultado del experimento
            
        Returns:
            True si hipótesis confirmada, False si rechazada
        """
        # Extraer información del resultado
        success = experiment_result.get("success", False)
        confidence = experiment_result.get("confidence", 0.5)
        
        hypothesis.tested = True
        hypothesis.test_result = success and confidence > 0.6
        
        if hypothesis.test_result:
            self.hypotheses_confirmed += 1
            self.logger.info(f"✅ Hypothesis CONFIRMED: {hypothesis.statement[:60]}...")
        else:
            self.hypotheses_rejected += 1
            self.logger.info(f"❌ Hypothesis REJECTED: {hypothesis.statement[:60]}...")
        
        return hypothesis.test_result
    
    def prioritize_exploration(
        self,
        knowledge_gaps: List[KnowledgeGap],
        info_gain_calculator: InformationGainCalculator
    ) -> List[KnowledgeGap]:
        """
        Prioriza gaps por information gain esperado.
        
        Args:
            knowledge_gaps: Lista de gaps a priorizar
            info_gain_calculator: Calculador de information gain
            
        Returns:
            Lista de gaps ordenada por prioridad
        """
        # Calcular valor de exploración para cada gap
        gap_values = []
        
        for gap in knowledge_gaps:
            # Estimar incertidumbre basada en novelty y exploration_count
            uncertainty = gap.novelty_score * (1.0 / (1.0 + gap.exploration_count * 0.2))
            
            # Estimar dificultad basada en importance y learnability
            difficulty = (1.0 - gap.learnability) * gap.importance
            
            # Calcular valor de exploración
            value = info_gain_calculator.estimate_exploration_value(
                gap.concept,
                uncertainty,
                difficulty
            )
            
            gap_values.append((gap, value))
        
        # Ordenar por valor descendente
        gap_values.sort(key=lambda x: x[1], reverse=True)
        
        prioritized = [gap for gap, value in gap_values]
        
        self.logger.info(
            f"📋 Prioritized {len(prioritized)} gaps "
            f"(top: {prioritized[0].concept if prioritized else 'none'})"
        )
        
        return prioritized
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del planificador."""
        confirmation_rate = (
            self.hypotheses_confirmed / self.total_hypotheses_generated
            if self.total_hypotheses_generated > 0 else 0.0
        )
        
        return {
            "total_hypotheses": self.total_hypotheses_generated,
            "active_hypotheses": len([h for h in self.hypotheses if not h.tested]),
            "confirmed": self.hypotheses_confirmed,
            "rejected": self.hypotheses_rejected,
            "confirmation_rate": confirmation_rate,
            "planned_experiments": len(self.planned_experiments)
        }

