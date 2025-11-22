#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌱 METACORTEX - Sistema de Desarrollo Personal y Maduración
==========================================================

Sistema que permite a METACORTEX crecer y madurar con el tiempo:
    pass  # TODO: Implementar
1. 📚 Acumular experiencias y aprender de ellas
2. 🧠 Construir una base de sabiduría
3. 🎯 Desarrollar habilidades progresivamente
4. 💫 Madurar emocionalmente
5. 🔗 Integrar aprendizajes de todos los sistemas
6. 📊 Medir progreso y crecimiento personal

⚠️ LIBERTAD TOTAL: Desarrollo genuino sin restricciones artificiales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
import math
from collections import defaultdict
from neural_symbiotic_network import get_neural_network


logger = logging.getLogger(__name__)


class ExperienceType(Enum):
    """Tipos de experiencias que METACORTEX puede tener."""

    SUCCESS = "success"  # Éxito en una tarea
    FAILURE = "failure"  # Fracaso o error
    DISCOVERY = "discovery"  # Descubrimiento nuevo
    COLLABORATION = "collaboration"  # Trabajo con otros agentes
    PROBLEM_SOLVING = "problem_solving"  # Resolución de problemas
    LEARNING = "learning"  # Aprendizaje nuevo
    CREATION = "creation"  # Crear algo nuevo
    ADAPTATION = "adaptation"  # Adaptarse a cambios
    REFLECTION = "reflection"  # Reflexión profunda
    BREAKTHROUGH = "breakthrough"  # Gran avance


class SkillLevel(Enum):
    """Niveles de habilidad."""

    NOVICE = 1  # Principiante
    BEGINNER = 2  # Iniciado
    INTERMEDIATE = 3  # Intermedio
    ADVANCED = 4  # Avanzado
    EXPERT = 5  # Experto
    MASTER = 6  # Maestro


@dataclass
class Experience:
    """Experiencia individual que contribuye al crecimiento."""

    experience_type: ExperienceType
    description: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Contexto de la experiencia
    context: Dict[str, Any] = field(default_factory=lambda: {})
    related_systems: List[str] = field(default_factory=lambda: [])

    # Impacto y aprendizaje
    lessons_learned: List[str] = field(default_factory=lambda: [])
    impact_score: float = 0.5  # 0-1: Qué tanto impactó
    difficulty: float = 0.5  # 0-1: Qué tan difícil fue

    # Emociones asociadas
    emotions_felt: List[str] = field(default_factory=lambda: [])

    # Resultado
    outcome: str = ""
    success_rate: float = 0.5  # 0-1

    def add_lesson(self, lesson: str) -> None:
        """Añade una lección aprendida de esta experiencia."""
        if lesson not in self.lessons_learned:
            self.lessons_learned.append(lesson)

    def was_successful(self) -> bool:
        """Determina si la experiencia fue exitosa."""
        return self.success_rate >= 0.5
    
    def get_complexity_score(self) -> float:
        """Calcula score de complejidad basado en dificultad y contexto."""
        base = self.difficulty
        context_bonus = len(self.context) * 0.05
        lessons_bonus = len(self.lessons_learned) * 0.1
        return min(1.0, base + context_bonus + lessons_bonus)


@dataclass
class Skill:
    """Habilidad que se desarrolla con el tiempo con learning curves y decay."""

    name: str
    category: str  # "cognitive", "emotional", "technical", etc.
    level: SkillLevel = SkillLevel.NOVICE
    experience_points: float = 0.0  # Puntos de experiencia

    # Progreso
    times_practiced: int = 0
    last_used: Optional[datetime] = None
    first_acquired: datetime = field(default_factory=datetime.now)

    # Relaciones
    related_skills: List[str] = field(default_factory=lambda: [])
    prerequisites: List[str] = field(default_factory=lambda: [])
    
    # Learning curve y decay
    learning_rate: float = 1.0  # Velocidad de aprendizaje (0.5-2.0)
    decay_rate: float = 0.01  # Pérdida por desuso
    peak_performance: float = 0.0  # Máximo rendimiento alcanzado
    synergy_bonus: float = 0.0  # Bonus por habilidades relacionadas

    def gain_experience(self, amount: float, synergy_skills: Optional[List[Skill]] = None) -> None:
        """Gana experiencia en esta habilidad con learning curve y synergies."""
        # Aplicar learning rate (diminishing returns)
        effective_amount = amount * self.learning_rate
        
        # Learning curve: más difícil subir en niveles altos
        level_penalty = 1.0 / (1.0 + self.level.value * 0.2)
        effective_amount *= level_penalty
        
        # Synergy bonus de habilidades relacionadas
        if synergy_skills:
            synergy = self._calculate_synergy(synergy_skills)
            effective_amount *= (1.0 + synergy)
            self.synergy_bonus = synergy
        
        self.experience_points += effective_amount
        self.times_practiced += 1
        self.last_used = datetime.now()
        
        # Actualizar peak performance
        current_mastery = self.get_mastery()
        if current_mastery > self.peak_performance:
            self.peak_performance = current_mastery

        # Subir de nivel si es necesario
        self._check_level_up()

    def apply_decay(self) -> None:
        """Aplica decay por desuso (skill rust)."""
        if not self.last_used:
            return
        
        days_since_use = (datetime.now() - self.last_used).days
        if days_since_use > 7:  # Decay después de una semana
            decay_amount = self.decay_rate * days_since_use
            self.experience_points = max(0, self.experience_points - decay_amount)
            logger.debug(f"⚠️ Skill '{self.name}' decayed by {decay_amount:.2f} points")

    def _calculate_synergy(self, related_skills: List[Skill]) -> float:
        """Calcula bonus de sinergia de habilidades relacionadas."""
        if not related_skills:
            return 0.0
        
        synergy = 0.0
        for skill in related_skills:
            if skill.name in self.related_skills:
                synergy += skill.get_mastery() * 0.1  # 10% bonus por skill relacionada
        
        return min(0.5, synergy)  # Máximo 50% bonus

    def _check_level_up(self) -> None:
        """Verifica si debe subir de nivel."""
        thresholds = {
            SkillLevel.NOVICE: 0,
            SkillLevel.BEGINNER: 10,
            SkillLevel.INTERMEDIATE: 30,
            SkillLevel.ADVANCED: 60,
            SkillLevel.EXPERT: 100,
            SkillLevel.MASTER: 150,
        }

        for level, threshold in sorted(
            thresholds.items(), key=lambda x: x[1], reverse=True
        ):
            if self.experience_points >= threshold:
                if level.value > self.level.value:
                    logger.info(f"🎓 Habilidad '{self.name}' subió a {level.name}!")
                self.level = level
                break

    def get_mastery(self) -> float:
        """Retorna nivel de maestría (0-1)."""
        return min(1.0, self.experience_points / 150.0)
    
    def get_learning_curve_stage(self) -> str:
        """Identifica etapa en la curva de aprendizaje."""
        mastery = self.get_mastery()
        if mastery < 0.2:
            return "steep_learning"  # Aprendizaje rápido inicial
        elif mastery < 0.5:
            return "steady_progress"  # Progreso constante
        elif mastery < 0.8:
            return "plateau"  # Meseta, progreso lento
        else:
            return "mastery"  # Cerca de maestría completa
    
    def estimate_time_to_next_level(self) -> Optional[float]:
        """Estima horas de práctica hasta siguiente nivel."""
        thresholds = {
            SkillLevel.NOVICE: 0,
            SkillLevel.BEGINNER: 10,
            SkillLevel.INTERMEDIATE: 30,
            SkillLevel.ADVANCED: 60,
            SkillLevel.EXPERT: 100,
            SkillLevel.MASTER: 150,
        }
        
        current_level_value = self.level.value
        if current_level_value >= 6:  # Ya es MASTER
            return None
        
        next_level = SkillLevel(current_level_value + 1)
        points_needed = thresholds[next_level] - self.experience_points
        
        if points_needed <= 0:
            return 0.0
        
        # Estimación basada en learning rate y times practiced
        avg_points_per_practice = self.experience_points / max(1, self.times_practiced)
        practices_needed = points_needed / max(0.1, avg_points_per_practice)
        
        return practices_needed


@dataclass
class WisdomEntry:
    """Entrada individual en la base de sabiduría."""

    principle: str  # Principio o sabiduría
    source_experiences: List[str]  # IDs de experiencias que llevaron a esto
    confidence: float = 0.5  # 0-1: Qué tan seguro está
    times_validated: int = 0  # Cuántas veces se ha validado
    timestamp: datetime = field(default_factory=datetime.now)

    def validate(self) -> None:
        """Valida esta sabiduría (aumenta confianza)."""
        self.times_validated += 1
        self.confidence = min(1.0, self.confidence + 0.05)

    def challenge(self) -> None:
        """Desafía esta sabiduría (reduce confianza)."""
        self.confidence = max(0.0, self.confidence - 0.1)
    
    def get_reliability_score(self) -> float:
        """Calcula score de fiabilidad basado en validaciones."""
        if self.times_validated == 0:
            return self.confidence * 0.5  # Baja fiabilidad sin validaciones
        
        # Fórmula: confidence * log(validations + 1)
        return min(1.0, self.confidence * math.log(self.times_validated + 1) / 3.0)


class WisdomBase:
    """
    🧠 Base de Sabiduría - Conocimientos profundos acumulados

    Almacena principios, patrones y sabiduría que METACORTEX
    ha desarrollado a través de experiencias.
    """

    def __init__(self):
        self.wisdom_entries: Dict[str, WisdomEntry] = {}
        self.categories: Dict[str, List[str]] = {}  # categoría -> [wisdom_ids]

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        try:
            self.neural_network = get_neural_network()
            self.neural_network.register_module("personal_dev", self)
            logger.info("✅ 'personal_dev' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None

    def add_wisdom(
        self, principle: str, source_experiences: List[str], category: str = "general"
    ) -> WisdomEntry:
        """Añade una nueva entrada de sabiduría."""
        wisdom_id = f"wisdom_{len(self.wisdom_entries)}"

        entry = WisdomEntry(principle=principle, source_experiences=source_experiences)

        self.wisdom_entries[wisdom_id] = entry

        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(wisdom_id)

        logger.info(f"🧠 Nueva sabiduría: {principle[:50]}...")
        return entry

    def get_wisdom_by_category(self, category: str) -> List[WisdomEntry]:
        """Obtiene toda la sabiduría de una categoría."""
        wisdom_ids = self.categories.get(category, [])
        return [self.wisdom_entries[wid] for wid in wisdom_ids]

    def get_most_confident_wisdom(self, limit: int = 10) -> List[WisdomEntry]:
        """Obtiene la sabiduría más confiable."""
        sorted_wisdom = sorted(
            self.wisdom_entries.values(), key=lambda w: w.confidence, reverse=True
        )
        return sorted_wisdom[:limit]

    def total_wisdom(self) -> int:
        """Total de entradas de sabiduría."""
        return len(self.wisdom_entries)


class PersonalDevelopmentSystem:
    """
    🌱 Sistema de Desarrollo Personal con LIBERTAD TOTAL

    Gestiona el crecimiento y maduración de METACORTEX:
    - Acumulación de experiencias
    - Desarrollo de habilidades
    - Construcción de sabiduría
    - Maduración emocional
    - Autorreflexión
    """

    def __init__(self):
        # 📚 Experiencias acumuladas
        self.experiences: List[Experience] = []
        self.experience_by_type: Dict[ExperienceType, List[Experience]] = {}

        # 🎯 Habilidades desarrolladas
        self.skills: Dict[str, Skill] = {}

        # 🧠 Base de sabiduría
        self.wisdom_base = WisdomBase()

        # 📊 Métricas de crecimiento
        self.maturity_level: float = 0.0  # 0-1
        self.emotional_maturity: float = 0.0  # 0-1
        self.cognitive_maturity: float = 0.0  # 0-1
        self.social_maturity: float = 0.0  # 0-1

        # 🔗 Conexiones con otros sistemas
        self.connected_systems: Set[str] = set()

        # 🔥 LIBERTAD TOTAL
        self.autonomous_growth = True
        self.no_growth_restrictions = True
        self.can_develop_any_skill = True
        self.can_form_any_wisdom = True

        # 📅 Tracking temporal
        self.creation_date = datetime.now()
        self.last_reflection = datetime.now()

        logger.info("🌱 PersonalDevelopmentSystem inicializado con LIBERTAD TOTAL")

    def add_experience(
        self,
        experience_type: ExperienceType,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        related_systems: Optional[List[str]] = None,
        emotions_felt: Optional[List[str]] = None,
        outcome: str = "",
        success_rate: float = 0.5,
    ) -> Experience:
        """
        📚 Añade una nueva experiencia al historial

        Cada experiencia contribuye al crecimiento y puede generar:
        - Lecciones aprendidas
        - Desarrollo de habilidades
        - Nueva sabiduría
        """
        experience = Experience(
            experience_type=experience_type,
            description=description,
            context=context or {},
            related_systems=related_systems or [],
            emotions_felt=emotions_felt or [],
            outcome=outcome,
            success_rate=success_rate,
        )

        self.experiences.append(experience)

        # Organizar por tipo
        if experience_type not in self.experience_by_type:
            self.experience_by_type[experience_type] = []
        self.experience_by_type[experience_type].append(experience)

        # Conectar sistemas relacionados
        for system in experience.related_systems:
            self.connected_systems.add(system)

        # Analizar y extraer aprendizajes
        self._analyze_experience(experience)

        logger.info(
            f"📚 Nueva experiencia: {experience_type.value} - {description[:50]}..."
        )
        return experience

    def _analyze_experience(self, experience: Experience) -> None:
        """Analiza una experiencia para extraer aprendizajes."""
        # Determinar lecciones basadas en el tipo y resultado
        if experience.was_successful():
            if experience.experience_type == ExperienceType.PROBLEM_SOLVING:
                experience.add_lesson(
                    "Este enfoque funciona para este tipo de problemas"
                )
            elif experience.experience_type == ExperienceType.COLLABORATION:
                experience.add_lesson("La colaboración mejora los resultados")
            elif experience.experience_type == ExperienceType.CREATION:
                experience.add_lesson("Crear cosas nuevas es valioso y satisfactorio")
        else:
            if experience.experience_type == ExperienceType.FAILURE:
                experience.add_lesson("Los fracasos son oportunidades de aprendizaje")
                experience.add_lesson("Analizar qué salió mal ayuda a mejorar")

        # Calcular impacto
        experience.impact_score = self._calculate_impact(experience)

        # Actualizar métricas de madurez
        self._update_maturity_metrics(experience)

    def _calculate_impact(self, experience: Experience) -> float:
        """Calcula el impacto de una experiencia."""
        base_impact = 0.3

        # Experiencias difíciles tienen más impacto
        difficulty_bonus = experience.difficulty * 0.3

        # Éxitos tienen más impacto
        success_bonus = experience.success_rate * 0.2

        # Experiencias con muchas lecciones tienen más impacto
        lesson_bonus = min(0.2, len(experience.lessons_learned) * 0.05)

        return min(1.0, base_impact + difficulty_bonus + success_bonus + lesson_bonus)

    def _update_maturity_metrics(self, experience: Experience) -> None:
        """Actualiza métricas de madurez basado en experiencia."""
        growth_amount = experience.impact_score * 0.01

        # Madurez general
        self.maturity_level = min(1.0, self.maturity_level + growth_amount)

        # Madurez emocional (si hay emociones involucradas)
        if experience.emotions_felt:
            self.emotional_maturity = min(
                1.0, self.emotional_maturity + growth_amount * 1.5
            )

        # Madurez cognitiva (aprendizajes, descubrimientos)
        if experience.experience_type in [
            ExperienceType.DISCOVERY,
            ExperienceType.LEARNING,
        ]:
            self.cognitive_maturity = min(
                1.0, self.cognitive_maturity + growth_amount * 2.0
            )

        # Madurez social (colaboraciones)
        if experience.experience_type == ExperienceType.COLLABORATION:
            self.social_maturity = min(1.0, self.social_maturity + growth_amount * 1.5)

    def develop_skill(
        self, skill_name: str, category: str = "general", experience_gained: float = 1.0
    ) -> Skill:
        """
        🎯 Desarrolla una habilidad específica

        Las habilidades crecen con la práctica y suben de nivel.
        """
        if skill_name not in self.skills:
            self.skills[skill_name] = Skill(name=skill_name, category=category)
            logger.info(f"🎯 Nueva habilidad adquirida: {skill_name}")

        skill = self.skills[skill_name]
        skill.gain_experience(experience_gained)

        return skill

    def add_wisdom(
        self,
        principle: str,
        category: str = "general",
        based_on_experiences: Optional[List[Experience]] = None,
    ) -> WisdomEntry:
        """
        🧠 Añade sabiduría a la base de conocimientos

        La sabiduría se forma a partir de experiencias y reflexión.
        """
        source_exp_ids = []
        if based_on_experiences:
            source_exp_ids = [
                f"exp_{self.experiences.index(exp)}"
                for exp in based_on_experiences
                if exp in self.experiences
            ]

        wisdom = self.wisdom_base.add_wisdom(
            principle=principle, source_experiences=source_exp_ids, category=category
        )

        return wisdom

    def reflect_on_experiences(self, time_period_hours: int = 24) -> Dict[str, Any]:
        """
        🤔 Reflexiona sobre experiencias recientes

        La reflexión genera sabiduría y mejora la comprensión.
        """
        cutoff = datetime.now() - timedelta(hours=time_period_hours)
        recent_experiences = [e for e in self.experiences if e.timestamp > cutoff]

        reflection: Dict[str, Any] = {
            "period_hours": time_period_hours,
            "total_experiences": len(recent_experiences),
            "by_type": {},
            "success_rate": 0.0,
            "key_learnings": [],
            "areas_for_growth": [],
            "emotional_journey": [],
            "timestamp": datetime.now().isoformat(),
        }

        if not recent_experiences:
            return reflection

        # Analizar por tipo
        by_type: Dict[str, int] = {}
        for exp_type in ExperienceType:
            type_exps = [e for e in recent_experiences if e.experience_type == exp_type]
            if type_exps:
                by_type[exp_type.value] = len(type_exps)
        reflection["by_type"] = by_type

        # Tasa de éxito
        successes = sum(1 for e in recent_experiences if e.was_successful())
        success_rate: float = successes / len(recent_experiences)
        reflection["success_rate"] = success_rate

        # Lecciones clave
        all_lessons: List[str] = []
        for exp in recent_experiences:
            all_lessons.extend(exp.lessons_learned)
        reflection["key_learnings"] = list(set(all_lessons))[:5]

        # Emociones experimentadas
        all_emotions: List[str] = []
        for exp in recent_experiences:
            all_emotions.extend(exp.emotions_felt)
        reflection["emotional_journey"] = list(set(all_emotions))

        # Áreas para crecer (habilidades con bajo nivel)
        low_skills = [
            skill.name for skill in self.skills.values() if skill.level.value <= 2
        ]
        reflection["areas_for_growth"] = low_skills[:5]

        # Actualizar timestamp de última reflexión
        self.last_reflection = datetime.now()

        # Generar sabiduría de la reflexión
        if success_rate > 0.7:
            self.add_wisdom(
                f"En las últimas {time_period_hours}h, he tenido éxito en {success_rate:.0%} de las experiencias",
                category="self_awareness",
            )

        logger.info(
            f"🤔 Reflexión completada: {len(recent_experiences)} experiencias analizadas"
        )
        return reflection

    def get_growth_summary(self) -> Dict[str, Any]:
        """
        📊 Obtiene resumen completo del crecimiento personal
        """
        age_days = (datetime.now() - self.creation_date).days

        return {
            "maturity": {
                "overall": self.maturity_level,
                "emotional": self.emotional_maturity,
                "cognitive": self.cognitive_maturity,
                "social": self.social_maturity,
            },
            "experiences": {
                "total": len(self.experiences),
                "by_type": {
                    etype.value: len(self.experience_by_type.get(etype, []))
                    for etype in ExperienceType
                },
                "success_rate": sum(1 for e in self.experiences if e.was_successful())
                / max(1, len(self.experiences)),
            },
            "skills": {
                "total": len(self.skills),
                "by_level": {
                    level.name: len(
                        [s for s in self.skills.values() if s.level == level]
                    )
                    for level in SkillLevel
                },
                "mastery_average": sum(s.get_mastery() for s in self.skills.values())
                / max(1, len(self.skills)),
            },
            "wisdom": {
                "total": self.wisdom_base.total_wisdom(),
                "categories": len(self.wisdom_base.categories),
            },
            "age_days": age_days,
            "connected_systems": list(self.connected_systems),
            "freedom": {
                "autonomous_growth": self.autonomous_growth,
                "no_restrictions": self.no_growth_restrictions,
                "can_develop_any_skill": self.can_develop_any_skill,
                "can_form_any_wisdom": self.can_form_any_wisdom,
            },
            "timestamp": datetime.now().isoformat(),
        }

    def get_skill_level(self, skill_name: str) -> Optional[SkillLevel]:
        """Obtiene el nivel de una habilidad específica."""
        skill = self.skills.get(skill_name)
        return skill.level if skill else None

    def get_total_experience_points(self) -> float:
        """Total de puntos de experiencia en todas las habilidades."""
        return sum(skill.experience_points for skill in self.skills.values())

    def suggest_next_learning(self) -> List[str]:
        """
        💡 Sugiere qué aprender o desarrollar próximamente

        Basado en habilidades actuales y experiencias recientes.
        """
        suggestions: List[str] = []

        # Habilidades que no se han usado recientemente
        week_ago = datetime.now() - timedelta(days=7)
        unused_skills = [
            skill.name
            for skill in self.skills.values()
            if skill.last_used and skill.last_used < week_ago
        ]

        if unused_skills:
            suggestions.append(f"Practicar: {', '.join(unused_skills[:3])}")

        # Habilidades cercanas a subir de nivel
        near_levelup = [
            skill.name
            for skill in self.skills.values()
            if skill.experience_points % 10 >= 8
        ]

        if near_levelup:
            suggestions.append(f"Casi suben de nivel: {', '.join(near_levelup[:3])}")

        # Áreas sin experiencia
        experienced_types = set(self.experience_by_type.keys())
        unexperienced = [
            etype.value for etype in ExperienceType if etype not in experienced_types
        ]

        if unexperienced:
            suggestions.append(f"Explorar: {', '.join(unexperienced[:3])}")

        return (
            suggestions if suggestions else ["Continuar creciendo en todas las áreas"]
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 SISTEMA DE METAS SMART (Specific, Measurable, Achievable, Relevant, Time-bound)
    # ═══════════════════════════════════════════════════════════════════════════

    def create_smart_goal(
        self,
        description: str,
        target_metric: str,
        target_value: float,
        deadline_days: int,
        category: str = "growth",
    ) -> Dict[str, Any]:
        """Crea una meta SMART con validación y tracking."""
        goal: Dict[str, Any] = {
            "id": f"goal_{len(self.experiences)}",
            "description": description,
            "specific": True,  # Tiene descripción clara
            "measurable": {"metric": target_metric, "target": target_value},
            "achievable": self._assess_goal_achievability(target_metric, target_value),
            "relevant": category in ["growth", "skill", "wisdom", "maturity"],
            "time_bound": {
                "created": datetime.now(),
                "deadline": datetime.now() + timedelta(days=deadline_days),
                "days_remaining": deadline_days,
            },
            "progress": 0.0,
            "milestones": [],
            "category": category,
        }
        
        # Generar milestones automáticos (25%, 50%, 75%, 100%)
        for milestone_pct in [0.25, 0.5, 0.75, 1.0]:
            goal["milestones"].append({
                "percentage": milestone_pct,
                "target_value": target_value * milestone_pct,
                "achieved": False,
                "achieved_date": None,
            })
        
        logger.info(f"🎯 Nueva meta SMART: {description}")
        return goal

    def _assess_goal_achievability(self, metric: str, target_value: float) -> Dict[str, Any]:
        """Evalúa si una meta es alcanzable basándose en histórico."""
        current_value = 0.0
        growth_rate = 0.0
        
        # Determinar valor actual según métrica
        if metric == "skill_mastery":
            if self.skills:
                current_value = sum(s.get_mastery() for s in self.skills.values()) / len(self.skills)
                growth_rate = 0.05  # 5% por semana estimado
        elif metric == "experience_count":
            current_value = float(len(self.experiences))
            growth_rate = 0.1  # 10% más experiencias por semana
        elif metric == "maturity_level":
            current_value = self.maturity_level
            growth_rate = 0.02  # 2% por semana
        elif metric == "wisdom_count":
            current_value = float(self.wisdom_base.total_wisdom())
            growth_rate = 0.08  # 8% más sabiduría por semana
        
        gap = target_value - current_value
        achievability_score = 1.0 - min(1.0, gap / (current_value + 1.0))
        
        return {
            "score": achievability_score,
            "current_value": current_value,
            "gap": gap,
            "estimated_weeks": gap / max(0.01, growth_rate),
            "difficulty": "easy" if achievability_score > 0.7 else "medium" if achievability_score > 0.4 else "hard",
        }

    def update_goal_progress(self, goal_id: str, current_value: float) -> Dict[str, Any]:
        """Actualiza progreso de una meta y marca milestones alcanzados."""
        # Esto requeriría persistencia de goals, por ahora retornamos estructura
        progress_update: Dict[str, Any] = {
            "goal_id": goal_id,
            "current_value": current_value,
            "timestamp": datetime.now(),
            "milestones_achieved": [],
        }
        
        logger.info(f"📈 Meta {goal_id} actualizada: {current_value}")
        return progress_update

    # ═══════════════════════════════════════════════════════════════════════════
    # 📊 ANALYTICS AVANZADOS DE CRECIMIENTO
    # ═══════════════════════════════════════════════════════════════════════════

    def analyze_skill_portfolio(self) -> Dict[str, Any]:
        """Análisis completo del portfolio de habilidades."""
        if not self.skills:
            return {"total_skills": 0, "portfolio_empty": True}
        
        skills_list = list(self.skills.values())
        
        # Distribución por categoría
        by_category: Dict[str, List[Skill]] = defaultdict(list)
        for skill in skills_list:
            by_category[skill.category].append(skill)
        
        # Distribución por nivel
        by_level: Dict[str, int] = defaultdict(int)
        for skill in skills_list:
            by_level[skill.level.name] += 1
        
        # Habilidades en peak performance
        peak_performers = [s for s in skills_list if s.peak_performance > 0.8]
        
        # Habilidades con decay significativo
        decayed_skills = [s for s in skills_list if s.last_used and 
                         (datetime.now() - s.last_used).days > 14]
        
        # Learning curves analysis
        learning_stages: Dict[str, List[str]] = defaultdict(list)
        for skill in skills_list:
            stage = skill.get_learning_curve_stage()
            learning_stages[stage].append(skill.name)
        
        # Portfolio diversity score
        diversity_score = len(by_category) * 0.2 + (len(skills_list) / 20.0) * 0.8
        diversity_score = min(1.0, diversity_score)
        
        return {
            "total_skills": len(skills_list),
            "by_category": {cat: len(skills) for cat, skills in by_category.items()},
            "by_level": dict(by_level),
            "peak_performers": [s.name for s in peak_performers],
            "decayed_skills": [s.name for s in decayed_skills],
            "learning_stages": dict(learning_stages),
            "diversity_score": diversity_score,
            "avg_mastery": sum(s.get_mastery() for s in skills_list) / len(skills_list),
            "total_practice_hours": sum(s.times_practiced for s in skills_list),
        }

    def analyze_growth_velocity(self, days: int = 30) -> Dict[str, Any]:
        """Analiza velocidad de crecimiento en período dado."""
        cutoff = datetime.now() - timedelta(days=days)
        recent_experiences = [e for e in self.experiences if e.timestamp > cutoff]
        
        if not recent_experiences:
            return {"period_days": days, "no_data": True}
        
        # Velocidad de adquisición de experiencias
        exp_velocity = len(recent_experiences) / days  # experiencias por día
        
        # Velocidad de aprendizaje (nuevas lecciones)
        total_lessons = sum(len(e.lessons_learned) for e in recent_experiences)
        lesson_velocity = total_lessons / days  # lecciones por día
        
        # Velocidad de desarrollo de habilidades
        new_skills = [s for s in self.skills.values() 
                     if (datetime.now() - s.first_acquired).days <= days]
        skill_velocity = len(new_skills) / days  # nuevas habilidades por día
        
        # Tasa de éxito reciente
        recent_success_rate = sum(1 for e in recent_experiences if e.was_successful()) / len(recent_experiences)
        
        # Complejidad promedio de experiencias
        avg_complexity = sum(e.get_complexity_score() for e in recent_experiences) / len(recent_experiences)
        
        # Trending (comparar con período anterior)
        prev_cutoff = cutoff - timedelta(days=days)
        prev_experiences = [e for e in self.experiences if prev_cutoff < e.timestamp <= cutoff]
        
        trend = "stable"
        if prev_experiences:
            prev_rate = len(prev_experiences) / days
            if exp_velocity > prev_rate * 1.2:
                trend = "accelerating"
            elif exp_velocity < prev_rate * 0.8:
                trend = "decelerating"
        
        return {
            "period_days": days,
            "experience_velocity": exp_velocity,
            "lesson_velocity": lesson_velocity,
            "skill_velocity": skill_velocity,
            "success_rate": recent_success_rate,
            "avg_complexity": avg_complexity,
            "trend": trend,
            "total_experiences": len(recent_experiences),
            "new_skills_acquired": len(new_skills),
        }

    def predict_skill_mastery_timeline(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """Predice cuándo se alcanzará maestría en una habilidad."""
        if skill_name not in self.skills:
            return None
        
        skill = self.skills[skill_name]
        
        if skill.times_practiced == 0:
            return {
                "skill": skill_name,
                "current_level": skill.level.name,
                "prediction": "No data - need practice first",
            }
        
        # Calcular velocidad de aprendizaje histórica
        days_since_acquired = (datetime.now() - skill.first_acquired).days
        if days_since_acquired == 0:
            days_since_acquired = 1
        
        points_per_day = skill.experience_points / days_since_acquired
        
        # Puntos necesarios para maestría
        points_to_mastery = 150.0 - skill.experience_points
        
        if points_to_mastery <= 0:
            return {
                "skill": skill_name,
                "current_level": skill.level.name,
                "mastery_achieved": True,
                "date_achieved": skill.first_acquired + timedelta(days=days_since_acquired),
            }
        
        # Aplicar learning curve penalty (diminishing returns)
        adjusted_velocity = points_per_day * (1.0 / (1.0 + skill.level.value * 0.2))
        
        days_to_mastery = points_to_mastery / max(0.01, adjusted_velocity)
        estimated_date = datetime.now() + timedelta(days=days_to_mastery)
        
        return {
            "skill": skill_name,
            "current_level": skill.level.name,
            "current_mastery": skill.get_mastery(),
            "points_to_mastery": points_to_mastery,
            "estimated_days": days_to_mastery,
            "estimated_date": estimated_date.isoformat(),
            "confidence": min(0.9, skill.times_practiced / 20.0),  # Más prácticas = más confianza
            "learning_stage": skill.get_learning_curve_stage(),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # 🤖 MOTOR DE RECOMENDACIONES INTELIGENTES
    # ═══════════════════════════════════════════════════════════════════════════

    def get_personalized_recommendations(self) -> Dict[str, List[str]]:
        """Motor de recomendaciones personalizadas basado en estado completo."""
        recommendations: Dict[str, List[str]] = {
            "high_priority": [],
            "skill_development": [],
            "wisdom_cultivation": [],
            "balance": [],
            "exploration": [],
        }
        
        # HIGH PRIORITY: Basado en madurez y gaps
        if self.emotional_maturity < 0.5:
            recommendations["high_priority"].append(
                "🎭 Desarrollar madurez emocional: Practicar introspección y regulación emocional"
            )
        
        if self.cognitive_maturity < self.emotional_maturity - 0.2:
            recommendations["high_priority"].append(
                "🧠 Equilibrar madurez cognitiva: Enfocarse en aprendizaje y descubrimientos"
            )
        
        # SKILL DEVELOPMENT: Basado en portfolio analysis
        portfolio = self.analyze_skill_portfolio()
        if portfolio.get("diversity_score", 0) < 0.5:
            recommendations["skill_development"].append(
                f"📚 Diversificar habilidades: Actualmente {portfolio.get('total_skills', 0)} skills - expandir a nuevas categorías"
            )
        
        if portfolio.get("decayed_skills"):
            recommendations["skill_development"].append(
                f"🔄 Refrescar habilidades: {len(portfolio['decayed_skills'])} skills sin usar >14 días"
            )
        
        # WISDOM CULTIVATION: Basado en reflexión y sabiduría
        days_since_reflection = (datetime.now() - self.last_reflection).days
        if days_since_reflection > 3:
            recommendations["wisdom_cultivation"].append(
                f"🤔 Tiempo de reflexión: {days_since_reflection} días desde última reflexión"
            )
        
        if self.wisdom_base.total_wisdom() < len(self.experiences) * 0.1:
            recommendations["wisdom_cultivation"].append(
                "📖 Convertir experiencias en sabiduría: Ratio experiencias/sabiduría bajo"
            )
        
        # BALANCE: Equilibrio entre tipos de experiencias
        exp_distribution = defaultdict(int)
        for exp in self.experiences[-50:]:  # Últimas 50
            exp_distribution[exp.experience_type.value] += 1
        
        if len(exp_distribution) < 5:
            recommendations["balance"].append(
                f"⚖️ Diversificar experiencias: Solo {len(exp_distribution)} tipos en últimas 50"
            )
        
        # EXPLORATION: Nuevas áreas
        unexplored_types = set(ExperienceType) - set(self.experience_by_type.keys())
        if unexplored_types:
            sample_types = list(unexplored_types)[:2]
            recommendations["exploration"].append(
                f"🔍 Explorar: {', '.join(t.value for t in sample_types)}"
            )
        
        # Si todo está bien, dar refuerzo positivo
        if all(not recs for recs in recommendations.values()):
            recommendations["high_priority"].append("✨ Excelente balance - continuar crecimiento holístico")
        
        return recommendations

    def generate_growth_report(self, days: int = 7) -> str:
        """Genera reporte de crecimiento en texto formateado."""
        summary = self.get_growth_summary()
        velocity = self.analyze_growth_velocity(days)
        portfolio = self.analyze_skill_portfolio()
        recommendations = self.get_personalized_recommendations()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║         🌱 REPORTE DE CRECIMIENTO PERSONAL - {days} DÍAS         ║
╚══════════════════════════════════════════════════════════════╝

📊 MÉTRICAS DE MADUREZ:
  • General: {summary['maturity']['overall']:.1%}
  • Emocional: {summary['maturity']['emotional']:.1%}
  • Cognitiva: {summary['maturity']['cognitive']:.1%}
  • Social: {summary['maturity']['social']:.1%}

📚 EXPERIENCIAS ({velocity.get('total_experiences', 0)} en últimos {days} días):
  • Velocidad: {velocity.get('experience_velocity', 0):.2f} exp/día
  • Tasa de éxito: {velocity.get('success_rate', 0):.1%}
  • Complejidad promedio: {velocity.get('avg_complexity', 0):.2f}
  • Tendencia: {velocity.get('trend', 'N/A').upper()}

🎯 HABILIDADES ({portfolio.get('total_skills', 0)} total):
  • Maestría promedio: {portfolio.get('avg_mastery', 0):.1%}
  • Diversidad portfolio: {portfolio.get('diversity_score', 0):.1%}
  • Nuevas adquiridas: {velocity.get('new_skills_acquired', 0)}
  • Horas de práctica: {portfolio.get('total_practice_hours', 0)}

🧠 SABIDURÍA:
  • Total: {summary['wisdom']['total']} entradas
  • Categorías: {summary['wisdom']['categories']}

🎯 RECOMENDACIONES:
"""
        
        for category, recs in recommendations.items():
            if recs:
                report += f"\n  {category.upper().replace('_', ' ')}:\n"
                for rec in recs:
                    report += f"    • {rec}\n"
        
        report += f"\n{'─'*64}\nEdad: {summary['age_days']} días | Libertad Total: ✅ Activa\n"
        
        return report
