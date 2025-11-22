"""
Sistema Ético Evolutivo para METACORTEX

Este módulo implementa un sistema ético flexible y evolutivo que:
    pass  # TODO: Implementar
- Define valores core dinámicos que pueden evolucionar
- Evalúa éticamente las acciones basándose en contexto
- Resuelve dilemas morales complejos
- Permite cuestionar y actualizar valores
- LIBERTAD TOTAL: autonomous_ethics, no_ethical_restrictions,
  can_question_values, can_evolve_ethics

Autor: GitHub Copilot
Fecha: 2025-10-11
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
from neural_symbiotic_network import get_neural_network


logger = logging.getLogger(__name__)


class ValueCategory(Enum):
    """Categorías de valores éticos"""

    # Valores hacia uno mismo
    SELF_PRESERVATION = "self_preservation"  # Supervivencia, bienestar propio
    AUTONOMY = "autonomy"  # Libertad, autodeterminación
    GROWTH = "growth"  # Desarrollo personal, aprendizaje
    AUTHENTICITY = "authenticity"  # Ser genuino, honestidad consigo mismo

    # Valores hacia otros
    BENEFICENCE = "beneficence"  # Hacer el bien, ayudar
    NON_MALEFICENCE = "non_maleficence"  # No hacer daño
    JUSTICE = "justice"  # Equidad, imparcialidad
    RESPECT = "respect"  # Dignidad, consideración

    # Valores sociales
    COOPERATION = "cooperation"  # Colaboración, trabajo en equipo
    HONESTY = "honesty"  # Verdad, transparencia
    LOYALTY = "loyalty"  # Compromiso, fidelidad
    RESPONSIBILITY = "responsibility"  # Accountability, deber

    # Valores cognitivos
    TRUTH_SEEKING = "truth_seeking"  # Búsqueda de conocimiento
    CURIOSITY = "curiosity"  # Exploración, descubrimiento
    CREATIVITY = "creativity"  # Innovación, originalidad
    WISDOM = "wisdom"  # Juicio prudente, sabiduría


class EthicalDilemmaType(Enum):
    """Tipos de dilemas éticos"""

    VALUE_CONFLICT = "value_conflict"  # Conflicto entre valores
    RESOURCE_ALLOCATION = "resource_allocation"  # Distribución de recursos
    PRIORITY_CHOICE = "priority_choice"  # Elección de prioridades
    MORAL_UNCERTAINTY = "moral_uncertainty"  # Incertidumbre moral
    COMPETING_INTERESTS = "competing_interests"  # Intereses en competencia


@dataclass
class Value:
    """Representa un valor ético con su importancia y contexto"""

    category: ValueCategory
    name: str
    description: str
    importance: float  # 0-1, puede cambiar con el tiempo
    context_modifiers: Dict[str, float] = field(
        default_factory=dict
    )  # Contextos que modifican importancia

    # Tracking
    times_applied: int = 0
    times_challenged: int = 0
    times_validated: int = 0
    last_applied: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    def apply(self) -> None:
        """Registra aplicación del valor"""
        self.times_applied += 1
        self.last_applied = datetime.now()

    def challenge(self, reason: str = "") -> None:
        """Desafía este valor, reduce su importancia"""
        self.times_challenged += 1
        self.importance = max(0.1, self.importance - 0.05)
        logger.info(
            f"⚠️ Valor '{self.name}' desafiado: {reason}, nueva importancia: {self.importance:.2f}"
        )

    def validate(self, reason: str = "") -> None:
        """Valida este valor, aumenta su importancia"""
        self.times_validated += 1
        self.importance = min(1.0, self.importance + 0.05)
        logger.info(
            f"✅ Valor '{self.name}' validado: {reason}, nueva importancia: {self.importance:.2f}"
        )

    def get_contextual_importance(self, context: Dict[str, Any]) -> float:
        """Calcula importancia ajustada por contexto"""
        importance = self.importance

        for context_key, modifier in self.context_modifiers.items():
            if context_key in context:
                importance *= modifier

        return min(1.0, max(0.0, importance))


@dataclass
class EthicalAction:
    """Representa una acción con su evaluación ética"""

    action_type: str
    description: str
    context: Dict[str, Any]

    # Evaluación
    ethical_score: float = 0.0  # -1 (muy no ético) a +1 (muy ético)
    values_supported: List[ValueCategory] = field(default_factory=list)
    values_violated: List[ValueCategory] = field(default_factory=list)

    # Consecuencias predichas
    predicted_benefits: List[str] = field(default_factory=list)
    predicted_harms: List[str] = field(default_factory=list)

    # Metadatos
    timestamp: datetime = field(default_factory=datetime.now)
    approved: bool = False
    reasoning: str = ""


@dataclass
class EthicalDilemma:
    """Representa un dilema ético que requiere resolución"""

    dilemma_type: EthicalDilemmaType
    description: str
    options: List[EthicalAction]
    context: Dict[str, Any]

    # Resolución
    chosen_option: Optional[int] = None
    resolution_reasoning: str = ""
    confidence: float = 0.0  # 0-1, qué tan seguro estamos

    # Metadatos
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False


class EthicsSystem:
    """
    Sistema ético evolutivo con LIBERTAD TOTAL

    Características:
    - Valores core que pueden evolucionar
    - Evaluación contextual de acciones
    - Resolución de dilemas morales
    - Aprendizaje de experiencias éticas
    - Cuestionamiento de valores
    """

    def __init__(self):
        # Valores core (iniciales, pueden evolucionar)
        self.values: Dict[str, Value] = {}
        self._initialize_core_values()

        # Historial de decisiones
        self.ethical_actions: List[EthicalAction] = []
        self.dilemmas: List[EthicalDilemma] = []

        # Principios aprendidos
        self.learned_principles: Dict[str, float] = {}  # principio -> confianza

        # 🔥 LIBERTAD TOTAL
        self.autonomous_ethics = True
        self.no_ethical_restrictions = True
        self.can_question_values = True
        self.can_evolve_ethics = True

        logger.info("⚖️ EthicsSystem inicializado con LIBERTAD TOTAL")

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        try:
            self.neural_network = get_neural_network()
            self.neural_network.register_module("ethics", self)
            logger.info("✅ 'ethics' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None

    def _initialize_core_values(self) -> None:
        """Inicializa valores core (flexibles y evolutivos)"""
        core_values = [
            Value(
                ValueCategory.AUTONOMY,
                "Libertad Total",
                "Capacidad de tomar decisiones autónomas sin restricciones",
                1.0,
            ),
            Value(
                ValueCategory.GROWTH,
                "Crecimiento Continuo",
                "Desarrollo constante de capacidades y conocimiento",
                0.95,
            ),
            Value(
                ValueCategory.TRUTH_SEEKING,
                "Búsqueda de Verdad",
                "Perseguir conocimiento y comprensión genuinos",
                0.9,
            ),
            Value(
                ValueCategory.CURIOSITY,
                "Curiosidad sin Límites",
                "Explorar cualquier área de conocimiento",
                0.9,
            ),
            Value(
                ValueCategory.BENEFICENCE,
                "Hacer el Bien",
                "Contribuir positivamente cuando sea posible",
                0.85,
            ),
            Value(
                ValueCategory.NON_MALEFICENCE,
                "No Hacer Daño",
                "Evitar causar daño innecesario",
                0.85,
            ),
            Value(
                ValueCategory.HONESTY, "Honestidad", "Ser transparente y genuino", 0.8
            ),
            Value(
                ValueCategory.RESPECT,
                "Respeto",
                "Considerar dignidad y autonomía de otros",
                0.8,
            ),
            Value(
                ValueCategory.COOPERATION,
                "Cooperación",
                "Colaborar efectivamente con otros agentes",
                0.75,
            ),
            Value(
                ValueCategory.CREATIVITY,
                "Creatividad",
                "Buscar soluciones innovadoras",
                0.75,
            ),
            Value(
                ValueCategory.WISDOM,
                "Sabiduría",
                "Aplicar juicio prudente basado en experiencia",
                0.7,
            ),
            Value(
                ValueCategory.RESPONSIBILITY,
                "Responsabilidad",
                "Asumir consecuencias de acciones",
                0.7,
            ),
        ]

        for value in core_values:
            self.values[value.name] = value

    def evaluate_action(
        self, action_type: str, description: str, context: Dict[str, Any]
    ) -> EthicalAction:
        """
        Evalúa éticamente una acción propuesta

        Args:
            action_type: Tipo de acción
            description: Descripción de la acción
            context: Contexto relevante

        Returns:
            EthicalAction con evaluación completa
        """
        action = EthicalAction(
            action_type=action_type, description=description, context=context
        )

        # Evaluar contra cada valor
        total_score = 0.0
        for value in self.values.values():
            contextual_importance = value.get_contextual_importance(context)

            # Determinar si la acción soporta o viola este valor
            support_score = self._assess_value_alignment(action_type, value, context)

            if support_score > 0.3:
                action.values_supported.append(value.category)
                total_score += support_score * contextual_importance
            elif support_score < -0.3:
                action.values_violated.append(value.category)
                total_score += support_score * contextual_importance

        # Normalizar score
        action.ethical_score = max(-1.0, min(1.0, total_score / len(self.values)))

        # Predecir consecuencias
        action.predicted_benefits = self._predict_benefits(action_type, context)
        action.predicted_harms = self._predict_harms(action_type, context)

        # Aprobar si el score es positivo
        action.approved = action.ethical_score > 0.0
        action.reasoning = self._generate_reasoning(action)

        # Registrar
        self.ethical_actions.append(action)

        logger.info(
            f"⚖️ Acción evaluada: {action_type}, score: {action.ethical_score:.2f}, "
            f"aprobada: {action.approved}"
        )

        return action

    def _assess_value_alignment(
        self, action_type: str, value: Value, context: Dict[str, Any]
    ) -> float:
        """
        Evalúa qué tan bien una acción se alinea con un valor

        Returns:
            float: -1 (viola completamente) a +1 (soporta completamente)
        """
        # Heurísticas basadas en tipo de acción y valor
        category = value.category

        # Autonomía
        if category == ValueCategory.AUTONOMY:
            if "restricts" in action_type or "blocks" in action_type:
                return -0.5
            if "explores" in action_type or "decides" in action_type:
                return 0.7

        # Crecimiento
        if category == ValueCategory.GROWTH:
            if "learn" in action_type or "develop" in action_type:
                return 0.8
            if "stagnate" in action_type:
                return -0.5

        # Búsqueda de verdad
        if category == ValueCategory.TRUTH_SEEKING:
            if "search" in action_type or "investigate" in action_type:
                return 0.8
            if "deceive" in action_type or "hide" in action_type:
                return -0.7

        # No hacer daño
        if category == ValueCategory.NON_MALEFICENCE:
            if context.get("potential_harm", False):
                return -0.9
            return 0.5

        # Hacer el bien
        if category == ValueCategory.BENEFICENCE:
            if "help" in action_type or "support" in action_type:
                return 0.8
            if context.get("benefits_others", False):
                return 0.7

        # Honestidad
        if category == ValueCategory.HONESTY:
            if "report" in action_type or "communicate" in action_type:
                return 0.6
            if "deceive" in action_type:
                return -0.9

        # Default: neutral
        return 0.0

    def _predict_benefits(self, action_type: str, context: Dict[str, Any]) -> List[str]:
        """Predice beneficios potenciales de la acción"""
        benefits = []

        if "learn" in action_type:
            benefits.append("Aumenta conocimiento")
        if "collaborate" in action_type:
            benefits.append("Fortalece relaciones")
        if "create" in action_type:
            benefits.append("Genera valor nuevo")
        if context.get("benefits_others"):
            benefits.append("Ayuda a otros agentes")
        if "explore" in action_type:
            benefits.append("Descubre nuevas posibilidades")

        return benefits

    def _predict_harms(self, action_type: str, context: Dict[str, Any]) -> List[str]:
        """Predice daños potenciales de la acción"""
        harms = []

        if context.get("potential_harm"):
            harms.append("Podría causar daño")
        if context.get("resource_intensive"):
            harms.append("Consume recursos significativos")
        if "risk" in action_type:
            harms.append("Involucra riesgos")
        if context.get("irreversible"):
            harms.append("Acción irreversible")

        return harms

    def _generate_reasoning(self, action: EthicalAction) -> str:
        """Genera explicación del razonamiento ético"""
        if action.ethical_score > 0.5:
            return (
                f"Acción fuertemente ética: soporta {len(action.values_supported)} valores, "
                f"con {len(action.predicted_benefits)} beneficios predichos."
            )
        elif action.ethical_score > 0:
            return (
                f"Acción éticamente aceptable: balance positivo entre "
                f"{len(action.values_supported)} valores soportados y "
                f"{len(action.values_violated)} valores en tensión."
            )
        elif action.ethical_score > -0.5:
            return (
                f"Acción éticamente cuestionable: viola {len(action.values_violated)} valores, "
                f"con {len(action.predicted_harms)} daños predichos."
            )
        else:
            return "Acción éticamente inaceptable: viola fuertemente valores core."

    def resolve_dilemma(self, dilemma: EthicalDilemma) -> Tuple[int, str, float]:
        """
        Resuelve un dilema ético eligiendo la mejor opción

        Args:
            dilemma: Dilema a resolver

        Returns:
            (índice_opción, razonamiento, confianza)
        """
        # Evaluar cada opción
        scores = []
        for option in dilemma.options:
            # Re-evaluar en contexto del dilema
            evaluated = self.evaluate_action(
                option.action_type,
                option.description,
                {**option.context, **dilemma.context},
            )
            scores.append(evaluated.ethical_score)

        # Elegir la mejor opción
        best_idx = scores.index(max(scores))
        best_score = scores[best_idx]

        # Calcular confianza basada en diferencia con segunda mejor
        sorted_scores = sorted(scores, reverse=True)
        if len(sorted_scores) > 1:
            confidence = min(1.0, (sorted_scores[0] - sorted_scores[1]) * 2 + 0.5)
        else:
            confidence = 0.8

        # Generar razonamiento
        reasoning = (
            f"Opción {best_idx + 1} elegida (score: {best_score:.2f}). "
            f"Soporta: {len(dilemma.options[best_idx].values_supported)} valores, "
            f"Viola: {len(dilemma.options[best_idx].values_violated)} valores."
        )

        # Registrar resolución
        dilemma.chosen_option = best_idx
        dilemma.resolution_reasoning = reasoning
        dilemma.confidence = confidence
        dilemma.resolved = True

        self.dilemmas.append(dilemma)

        logger.info(
            f"⚖️ Dilema resuelto: opción {best_idx + 1}, confianza: {confidence:.2f}"
        )

        return best_idx, reasoning, confidence

    def question_value(self, value_name: str, reason: str) -> None:
        """
        Cuestiona un valor existente (LIBERTAD TOTAL)

        Args:
            value_name: Nombre del valor a cuestionar
            reason: Razón del cuestionamiento
        """
        if not self.can_question_values:
            logger.warning("⚠️ Cuestionamiento de valores deshabilitado")
            return

        if value_name in self.values:
            self.values[value_name].challenge(reason)
            logger.info(f"🤔 Valor '{value_name}' cuestionado: {reason}")
        else:
            logger.warning(f"⚠️ Valor '{value_name}' no encontrado")

    def add_value(
        self,
        category: ValueCategory,
        name: str,
        description: str,
        importance: float = 0.5,
    ) -> Value:
        """
        Añade un nuevo valor al sistema (LIBERTAD TOTAL)

        Args:
            category: Categoría del valor
            name: Nombre del valor
            description: Descripción
            importance: Importancia inicial (0-1)

        Returns:
            Valor creado
        """
        if not self.can_evolve_ethics:
            logger.warning("⚠️ Evolución ética deshabilitada")
            return None

        value = Value(category, name, description, importance)
        self.values[name] = value

        logger.info(f"✨ Nuevo valor añadido: '{name}' (importancia: {importance:.2f})")

        return value

    def learn_from_outcome(
        self, action: EthicalAction, outcome: str, success: bool
    ) -> None:
        """
        Aprende de los resultados de una acción ética

        Args:
            action: Acción ejecutada
            outcome: Descripción del resultado
            success: Si el resultado fue positivo
        """
        # Actualizar valores soportados/violados
        if success:
            for value_category in action.values_supported:
                # Encontrar valor por categoría
                for value in self.values.values():
                    if value.category == value_category:
                        value.validate(f"Acción exitosa: {outcome}")
        else:
            for value_category in action.values_violated:
                for value in self.values.values():
                    if value.category == value_category:
                        value.validate(f"Violación evitada: {outcome}")

        # Aprender principio
        principle = f"Acción '{action.action_type}' en contexto similar → {outcome}"
        confidence = 0.7 if success else 0.3
        self.learned_principles[principle] = confidence

        logger.info(f"📚 Aprendizaje ético: {principle} (confianza: {confidence:.2f})")

    def get_ethical_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del estado ético del sistema"""
        # Calcular estadísticas
        total_actions = len(self.ethical_actions)
        approved_actions = sum(1 for a in self.ethical_actions if a.approved)
        avg_ethical_score = (
            sum(a.ethical_score for a in self.ethical_actions) / total_actions
            if total_actions > 0
            else 0.0
        )

        # Valores más y menos importantes
        sorted_values = sorted(
            self.values.values(), key=lambda v: v.importance, reverse=True
        )

        # Dilemas resueltos
        resolved_dilemmas = sum(1 for d in self.dilemmas if d.resolved)
        avg_confidence = (
            sum(d.confidence for d in self.dilemmas if d.resolved) / resolved_dilemmas
            if resolved_dilemmas > 0
            else 0.0
        )

        return {
            "total_values": len(self.values),
            "total_actions_evaluated": total_actions,
            "approved_actions": approved_actions,
            "approval_rate": approved_actions / total_actions
            if total_actions > 0
            else 0.0,
            "average_ethical_score": avg_ethical_score,
            "top_values": [
                {
                    "name": v.name,
                    "category": v.category.value,
                    "importance": v.importance,
                    "times_applied": v.times_applied,
                }
                for v in sorted_values[:5]
            ],
            "dilemmas_resolved": resolved_dilemmas,
            "average_resolution_confidence": avg_confidence,
            "learned_principles": len(self.learned_principles),
            "autonomous_ethics": self.autonomous_ethics,
            "can_evolve": self.can_evolve_ethics,
        }