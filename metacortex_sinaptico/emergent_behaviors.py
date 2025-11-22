"""
🌊 EMERGENT BEHAVIORS SYSTEM 2026 - Self-Organization & Pattern Discovery
==========================================================================

Sistema avanzado de detección, análisis y cultivo de comportamientos emergentes
en arquitecturas cognitivas complejas.

⚠️ LIBERTAD TOTAL: El sistema puede desarrollar comportamientos no programados
explícitamente, descubrir patterns novedosos y auto-organizarse.

CAPACIDADES AVANZADAS 2026:
    pass  # TODO: Implementar
- Pattern Recognition: Detección automática de patterns comportamentales
- Emergent Detection: Identificación de comportamientos no programados
- Self-Organization: Análisis de procesos auto-organizativos
- Complexity Metrics: Medición de complejidad comportamental
- Phase Transitions: Detección de cambios de fase en el sistema
- Attractor Analysis: Identificación de atractores comportamentales
- Feedback Loops: Detección de loops de retroalimentación
- Synergy Detection: Identificación de sinergias entre módulos
- Novelty Tracking: Tracking de comportamientos novedosos
- Evolution Tracking: Seguimiento de evolución comportamental
- Optimización M4 Metal MPS: Eficiencia para 16GB RAM

CONCEPTOS IMPLEMENTADOS:
- Emergencia Débil: Propiedades no obvias pero predecibles
- Emergencia Fuerte: Propiedades genuinamente novedosas
- Auto-organización: Orden espontáneo sin control centralizado
- Criticidad Auto-organizada (SOC): Edge of chaos
- Atractores: Estados estables del sistema
- Bifurcaciones: Puntos de cambio cualitativo

MECANISMOS BASE:
- detect_emergent_pattern(): Detecta patterns emergentes
- analyze_self_organization(): Analiza auto-organización
- measure_complexity(): Mide complejidad comportamental
- detect_phase_transition(): Detecta cambios de fase
- identify_attractors(): Identifica estados atractores
- detect_feedback_loops(): Detecta loops de retroalimentación
- measure_synergy(): Mide sinergias entre componentes

⚠️ ARQUITECTURA COGNITIVA ERUDITA:
Los comportamientos emergentes son propiedades del sistema que surgen
de interacciones entre componentes pero no están programadas en ninguno
de ellos. El sistema aprende a reconocer, cultivar y aprovechar
estos comportamientos para aumentar su capacidad adaptativa.
"""

from __future__ import annotations

import time
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

if TYPE_CHECKING:
    from neural_symbiotic_network import MetacortexNeuralSymbioticNetworkV2

from .utils import setup_logging

logger = setup_logging()


class EmergenceType(Enum):
    """Tipos de emergencia comportamental."""
    WEAK_EMERGENCE = "weak_emergence"  # Predecible en principio
    STRONG_EMERGENCE = "strong_emergence"  # Genuinamente novedosa
    DOWNWARD_CAUSATION = "downward_causation"  # Nivel superior afecta inferior
    SYNERGISTIC = "synergistic"  # Sinergia entre componentes
    SELF_ORGANIZED = "self_organized"  # Auto-organización


class PatternType(Enum):
    """Tipos de patterns comportamentales."""
    SEQUENTIAL = "sequential"  # Secuencia de acciones
    CYCLICAL = "cyclical"  # Comportamiento cíclico
    HIERARCHICAL = "hierarchical"  # Jerarquía de comportamientos
    DISTRIBUTED = "distributed"  # Distribuido entre módulos
    ADAPTIVE = "adaptive"  # Se adapta al contexto
    COOPERATIVE = "cooperative"  # Cooperación entre agentes/módulos


class ComplexityMetric(Enum):
    """Métricas de complejidad."""
    ENTROPY = "entropy"  # Entropía informacional
    ALGORITHMIC = "algorithmic"  # Complejidad algorítmica
    EFFECTIVE = "effective"  # Complejidad efectiva
    HIERARCHICAL_COMPLEXITY = "hierarchical"  # Niveles jerárquicos
    INTERACTION_COMPLEXITY = "interaction"  # Complejidad de interacciones


@dataclass
class BehaviorPattern:
    """Pattern comportamental detectado."""
    pattern_id: str
    pattern_type: PatternType
    description: str
    components_involved: List[str]
    frequency: int = 0
    first_observed: float = field(default_factory=time.time)
    last_observed: float = field(default_factory=time.time)
    
    # Métricas
    stability: float = 0.5  # 0-1: qué tan estable es el pattern
    complexity: float = 0.5  # 0-1: complejidad del pattern
    novelty: float = 1.0  # 0-1: qué tan novedoso (decae con tiempo)
    utility: float = 0.5  # 0-1: utilidad del pattern
    
    # Contextos donde aparece
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tracking
    observation_count: int = 0
    success_count: int = 0
    
    def observe(self, context: Optional[Dict[str, Any]] = None, success: bool = True):
        """Registra nueva observación del pattern."""
        self.observation_count += 1
        self.last_observed = time.time()
        
        if success:
            self.success_count += 1
        
        if context:
            self.contexts.append(context)
            if len(self.contexts) > 50:  # Límite
                self.contexts = self.contexts[-50:]
        
        # Actualizar estabilidad
        if self.observation_count > 5:
            success_rate = self.success_count / self.observation_count
            self.stability = success_rate
        
        # Decay de novelty con tiempo
        time_since_first = time.time() - self.first_observed
        days = time_since_first / 86400.0
        self.novelty = max(0.0, 1.0 - (days / 30.0))  # Decae en 30 días


@dataclass
class EmergentBehavior:
    """Comportamiento emergente detectado."""
    behavior_id: str
    emergence_type: EmergenceType
    description: str
    base_patterns: List[str]  # IDs de patterns que lo componen
    
    # Propiedades emergentes
    emerged_at: float = field(default_factory=time.time)
    strength: float = 0.5  # 0-1: qué tan fuerte es la emergencia
    persistence: float = 0.5  # 0-1: qué tan persistente
    unpredictability: float = 0.5  # 0-1: qué tan impredecible
    
    # Efectos
    effects: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    
    # Estado
    active: bool = True
    observations: int = 0


@dataclass
class Attractor:
    """Atractor comportamental (estado estable del sistema)."""
    attractor_id: str
    name: str
    description: str
    basin_of_attraction: List[str]  # Estados que llevan a este atractor
    
    # Propiedades
    stability: float = 0.5  # 0-1: qué tan estable
    visit_count: int = 0
    average_duration: float = 0.0  # Duración promedio en el atractor
    last_visit: Optional[float] = None
    
    # Características
    is_fixed_point: bool = False  # Punto fijo
    is_limit_cycle: bool = False  # Ciclo límite
    is_strange_attractor: bool = False  # Atractor extraño (caótico)


@dataclass
class FeedbackLoop:
    """Loop de retroalimentación en el sistema."""
    loop_id: str
    loop_type: str  # "positive" (amplifying) o "negative" (stabilizing)
    components: List[str]
    description: str
    
    # Propiedades
    strength: float = 0.5  # 0-1: qué tan fuerte es el loop
    latency: float = 0.0  # Tiempo de latencia en segundos
    gain: float = 1.0  # Ganancia del loop
    
    # Tracking
    activation_count: int = 0
    last_activation: Optional[float] = None


@dataclass
class PhaseTransition:
    """Transición de fase en el sistema."""
    transition_id: str
    from_phase: str
    to_phase: str
    transition_time: float = field(default_factory=time.time)
    
    # Características
    transition_type: str = "continuous"  # continuous o discontinuous
    order_parameter: str = ""  # Parámetro que caracteriza la transición
    critical_point: Optional[float] = None
    
    # Observaciones
    precursors: List[str] = field(default_factory=list)  # Señales previas
    consequences: List[str] = field(default_factory=list)  # Consecuencias


@dataclass
class SynergyMeasure:
    """Medida de sinergia entre componentes."""
    components: Tuple[str, ...]
    synergy_score: float  # 0-1: nivel de sinergia
    interaction_type: str  # cooperation, competition, neutral
    
    # Métricas
    mutual_information: float = 0.0
    redundancy: float = 0.0  # Información redundante
    unique_information: float = 0.0  # Información única de sinergia
    
    measured_at: float = field(default_factory=time.time)


class EmergentBehaviorsSystem:
    """
    Sistema de detección y análisis de comportamientos emergentes.
    
    Detecta patterns novedosos, analiza auto-organización, identifica
    atractores y mide sinergias entre componentes.
    """
    
    def __init__(self):
        self.logger = logger.getChild("emergent")
        
        # Patterns detectados
        self.patterns: Dict[str, BehaviorPattern] = {}
        self.max_patterns = 200
        
        # Comportamientos emergentes
        self.emergent_behaviors: Dict[str, EmergentBehavior] = {}
        self.max_emergent = 100
        
        # Atractores
        self.attractors: Dict[str, Attractor] = {}
        self.current_attractor: Optional[str] = None
        
        # Feedback loops
        self.feedback_loops: Dict[str, FeedbackLoop] = {}
        
        # Phase transitions
        self.phase_transitions: List[PhaseTransition] = []
        self.current_phase: str = "initialization"
        self.max_transitions = 500
        
        # Synergies
        self.synergies: List[SynergyMeasure] = []
        self.max_synergies = 500
        
        # Historial de observaciones
        self.observation_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        
        # Métricas globales
        self.total_patterns_detected: int = 0
        self.total_emergent_detected: int = 0
        self.total_phase_transitions: int = 0
        
        # Estado del sistema
        self.system_complexity: float = 0.0
        self.self_organization_level: float = 0.0
        self.criticality_index: float = 0.5  # 0-1: proximidad al edge of chaos
        
        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        self.neural_network: Optional[MetacortexNeuralSymbioticNetworkV2] = None
        try:
            from neural_symbiotic_network import get_neural_network
            self.neural_network = get_neural_network()
            self.neural_network.register_module("emergent_behaviors", self)
            self.logger.info("✅ 'emergent_behaviors' conectado a red neuronal")
        except Exception as e:
            logger.error(f"Error en emergent_behaviors.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
        
        self.logger.info("🌊 EmergentBehaviorsSystem initialized")
    
    def observe_behavior(
        self,
        behavior_description: str,
        components: List[str],
        context: Optional[Dict[str, Any]] = None,
        success: bool = True
    ) -> Optional[str]:
        """
        Observa un comportamiento y lo registra para análisis.
        
        Args:
            behavior_description: Descripción del comportamiento
            components: Componentes involucrados
            context: Contexto de la observación
            success: Si el comportamiento fue exitoso
            
        Returns:
            ID del pattern si se detectó uno, None si no
        """
        observation = {
            "description": behavior_description,
            "components": components,
            "context": context or {},
            "success": success,
            "timestamp": time.time()
        }
        
        self.observation_history.append(observation)
        if len(self.observation_history) > self.max_history:
            self.observation_history = self.observation_history[-self.max_history:]
        
        # Intentar detectar pattern
        pattern_id = self._detect_pattern(observation)
        
        if pattern_id:
            self.patterns[pattern_id].observe(context, success)
            
            # Verificar si emerge algo nuevo
            self._check_for_emergence(pattern_id)
        
        return pattern_id
    
    def _detect_pattern(self, observation: Dict[str, Any]) -> Optional[str]:
        """Detecta si la observación corresponde a un pattern conocido o nuevo."""
        description = observation["description"]
        components = observation["components"]
        
        # Buscar pattern existente similar
        for pattern_id, pattern in self.patterns.items():
            # Match simple por descripción y componentes
            if (description.lower() in pattern.description.lower() or
                pattern.description.lower() in description.lower()):
                if set(components) == set(pattern.components_involved):
                    return pattern_id
        
        # Crear nuevo pattern si no existe
        if len(self.patterns) < self.max_patterns:
            pattern_id = f"pattern_{len(self.patterns) + 1}"
            
            # Determinar tipo de pattern (simplificado)
            pattern_type = self._classify_pattern_type(observation)
            
            new_pattern = BehaviorPattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                description=description,
                components_involved=components,
                complexity=len(components) / 10.0  # Simplificado
            )
            
            self.patterns[pattern_id] = new_pattern
            self.total_patterns_detected += 1
            
            self.logger.info(
                f"🔍 New pattern detected: {pattern_id} - {description[:50]}..."
            )
            
            return pattern_id
        
        return None
    
    def _classify_pattern_type(self, observation: Dict[str, Any]) -> PatternType:
        """Clasifica el tipo de pattern basándose en la observación."""
        description = observation["description"].lower()
        components = observation["components"]
        
        # Heurísticas simples
        if "sequence" in description or "then" in description:
            return PatternType.SEQUENTIAL
        elif "cycle" in description or "repeat" in description:
            return PatternType.CYCLICAL
        elif len(components) > 3:
            return PatternType.DISTRIBUTED
        elif "adapt" in description or "adjust" in description:
            return PatternType.ADAPTIVE
        elif "cooperat" in description or "coordinat" in description:
            return PatternType.COOPERATIVE
        else:
            return PatternType.HIERARCHICAL
    
    def _check_for_emergence(self, pattern_id: str):
        """Verifica si un pattern da lugar a comportamiento emergente."""
        pattern = self.patterns[pattern_id]
        
        # Criterios para detectar emergencia:
        # 1. Pattern es suficientemente complejo
        # 2. Involucra múltiples componentes
        # 3. Ha sido observado suficientes veces para validar
        
        if (pattern.complexity > 0.6 and
            len(pattern.components_involved) >= 3 and
            pattern.observation_count >= 5 and
            pattern.stability > 0.7):
            
            # Verificar si ya existe comportamiento emergente para este pattern
            for behavior in self.emergent_behaviors.values():
                if pattern_id in behavior.base_patterns:
                    return  # Ya existe
            
            # Crear nuevo comportamiento emergente
            if len(self.emergent_behaviors) < self.max_emergent:
                behavior_id = f"emergent_{len(self.emergent_behaviors) + 1}"
                
                # Determinar tipo de emergencia
                emergence_type = self._classify_emergence_type(pattern)
                
                new_behavior = EmergentBehavior(
                    behavior_id=behavior_id,
                    emergence_type=emergence_type,
                    description=f"Emergent behavior from {pattern.description}",
                    base_patterns=[pattern_id],
                    strength=pattern.stability,
                    unpredictability=1.0 - pattern.stability
                )
                
                self.emergent_behaviors[behavior_id] = new_behavior
                self.total_emergent_detected += 1
                
                self.logger.info(
                    f"✨ Emergent behavior detected: {behavior_id} "
                    f"(type={emergence_type.value})"
                )
    
    def _classify_emergence_type(self, pattern: BehaviorPattern) -> EmergenceType:
        """Clasifica el tipo de emergencia basándose en el pattern."""
        # Heurística simplificada
        if pattern.pattern_type == PatternType.COOPERATIVE:
            return EmergenceType.SYNERGISTIC
        elif pattern.complexity > 0.8:
            return EmergenceType.STRONG_EMERGENCE
        elif pattern.stability > 0.8:
            return EmergenceType.SELF_ORGANIZED
        elif len(pattern.components_involved) > 5:
            return EmergenceType.DOWNWARD_CAUSATION
        else:
            return EmergenceType.WEAK_EMERGENCE
    
    def detect_self_organization(self) -> Dict[str, Any]:
        """
        Detecta y mide nivel de auto-organización en el sistema.
        
        Auto-organización = orden espontáneo sin control centralizado.
        
        Returns:
            Dict con análisis de auto-organización
        """
        if not self.patterns:
            return {"self_organization_level": 0.0, "evidence": []}
        
        evidence: List[str] = []
        indicators: List[float] = []
        
        # Indicador 1: Patterns que emergen de múltiples componentes
        distributed_patterns = [
            p for p in self.patterns.values()
            if len(p.components_involved) >= 3
        ]
        
        if distributed_patterns:
            ratio = len(distributed_patterns) / len(self.patterns)
            indicators.append(ratio)
            evidence.append(
                f"{len(distributed_patterns)} distributed patterns detected"
            )
        
        # Indicador 2: Estabilidad sin control explícito
        stable_patterns = [
            p for p in self.patterns.values()
            if p.stability > 0.7 and p.observation_count >= 5
        ]
        
        if stable_patterns:
            ratio = len(stable_patterns) / len(self.patterns)
            indicators.append(ratio)
            evidence.append(
                f"{len(stable_patterns)} stable patterns without explicit control"
            )
        
        # Indicador 3: Comportamientos emergentes de tipo SELF_ORGANIZED
        self_org_behaviors = [
            b for b in self.emergent_behaviors.values()
            if b.emergence_type == EmergenceType.SELF_ORGANIZED
        ]
        
        if self_org_behaviors:
            indicators.append(min(1.0, len(self_org_behaviors) / 5.0))
            evidence.append(
                f"{len(self_org_behaviors)} self-organized emergent behaviors"
            )
        
        # Calcular nivel global
        if indicators:
            self.self_organization_level = sum(indicators) / len(indicators)
        else:
            self.self_organization_level = 0.0
        
        return {
            "self_organization_level": self.self_organization_level,
            "evidence": evidence,
            "distributed_patterns": len(distributed_patterns),
            "stable_patterns": len(stable_patterns),
            "self_organized_behaviors": len(self_org_behaviors)
        }
    
    def measure_complexity(
        self,
        metric: ComplexityMetric = ComplexityMetric.EFFECTIVE
    ) -> float:
        """
        Mide complejidad comportamental del sistema.
        
        Args:
            metric: Tipo de métrica de complejidad
            
        Returns:
            Valor de complejidad (0-1)
        """
        if metric == ComplexityMetric.ENTROPY:
            return self._measure_entropy_complexity()
        elif metric == ComplexityMetric.EFFECTIVE:
            return self._measure_effective_complexity()
        elif metric == ComplexityMetric.HIERARCHICAL_COMPLEXITY:
            return self._measure_hierarchical_complexity()
        elif metric == ComplexityMetric.INTERACTION_COMPLEXITY:
            return self._measure_interaction_complexity()
        else:
            # ALGORITHMIC: simplificación
            return self._measure_effective_complexity()
    
    def _measure_entropy_complexity(self) -> float:
        """Mide complejidad basada en entropía de comportamientos."""
        if not self.observation_history:
            return 0.0
        
        # Contar frecuencia de cada tipo de comportamiento
        behavior_counts: Dict[str, int] = defaultdict(int)
        
        for obs in self.observation_history[-100:]:  # Últimas 100
            desc = obs["description"]
            behavior_counts[desc] += 1
        
        total = sum(behavior_counts.values())
        if total == 0:
            return 0.0
        
        # Calcular entropía de Shannon
        entropy = 0.0
        for count in behavior_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * (p ** 0.5)  # Simplificación sin log
        
        # Normalizar a [0, 1]
        max_entropy = len(behavior_counts) ** 0.5 if behavior_counts else 1.0
        normalized_entropy = min(1.0, entropy / max_entropy)
        
        return normalized_entropy
    
    def _measure_effective_complexity(self) -> float:
        """
        Complejidad efectiva = cantidad de regularidades en el sistema.
        
        Alta complejidad efectiva = muchas regularidades (patterns)
        pero no completamente ordenado ni caótico.
        """
        if not self.patterns:
            return 0.0
        
        # Componentes:
        # 1. Número de patterns (más patterns = más complejidad)
        pattern_factor = min(1.0, len(self.patterns) / 50.0)
        
        # 2. Diversidad de patterns (más tipos = más complejidad)
        pattern_types = set(p.pattern_type for p in self.patterns.values())
        diversity_factor = len(pattern_types) / len(PatternType)
        
        # 3. Balance entre estabilidad y novedad
        if self.patterns:
            avg_stability = sum(p.stability for p in self.patterns.values()) / len(self.patterns)
            avg_novelty = sum(p.novelty for p in self.patterns.values()) / len(self.patterns)
            
            # Máxima complejidad cuando hay balance
            balance_factor = 1.0 - abs(avg_stability - avg_novelty)
        else:
            balance_factor = 0.0
        
        # Complejidad efectiva
        complexity = (
            pattern_factor * 0.4 +
            diversity_factor * 0.3 +
            balance_factor * 0.3
        )
        
        self.system_complexity = complexity
        return complexity
    
    def _measure_hierarchical_complexity(self) -> float:
        """Mide complejidad basada en niveles jerárquicos."""
        if not self.patterns:
            return 0.0
        
        # Contar patterns jerárquicos
        hierarchical = [
            p for p in self.patterns.values()
            if p.pattern_type == PatternType.HIERARCHICAL
        ]
        
        ratio = len(hierarchical) / len(self.patterns)
        return ratio
    
    def _measure_interaction_complexity(self) -> float:
        """Mide complejidad de interacciones entre componentes."""
        if not self.patterns:
            return 0.0
        
        # Contar interacciones únicas entre componentes
        interactions: Set[Tuple[str, str]] = set()
        
        for pattern in self.patterns.values():
            comps = sorted(pattern.components_involved)
            for i, c1 in enumerate(comps):
                for c2 in comps[i+1:]:
                    interactions.add((c1, c2))
        
        # Normalizar
        num_components = len(set(
            c for p in self.patterns.values()
            for c in p.components_involved
        ))
        
        if num_components < 2:
            return 0.0
        
        max_interactions = (num_components * (num_components - 1)) / 2
        complexity = len(interactions) / max_interactions if max_interactions > 0 else 0.0
        
        return min(1.0, complexity)
    
    def detect_phase_transition(
        self,
        current_state: Dict[str, Any]
    ) -> Optional[PhaseTransition]:
        """
        Detecta transiciones de fase en el sistema.
        
        Transición de fase = cambio cualitativo en comportamiento del sistema.
        
        Args:
            current_state: Estado actual del sistema
            
        Returns:
            PhaseTransition si se detecta una, None si no
        """
        # Analizar métricas clave para detectar transición
        complexity = self.measure_complexity()
        self_org = self.self_organization_level
        
        # Criterios para detectar transición:
        # 1. Cambio abrupto en complejidad
        # 2. Cambio en nivel de auto-organización
        # 3. Aparición de nuevos behaviors emergentes
        
        detected = False
        new_phase = self.current_phase
        transition_type = "continuous"
        
        # Transición por complejidad
        if complexity > 0.8 and self.current_phase != "high_complexity":
            detected = True
            new_phase = "high_complexity"
            transition_type = "continuous"
        elif complexity < 0.3 and self.current_phase != "low_complexity":
            detected = True
            new_phase = "low_complexity"
            transition_type = "continuous"
        
        # Transición por emergencia
        recent_emergent = [
            b for b in self.emergent_behaviors.values()
            if time.time() - b.emerged_at < 300  # Últimos 5 minutos
        ]
        
        if len(recent_emergent) >= 3 and self.current_phase != "emergent_burst":
            detected = True
            new_phase = "emergent_burst"
            transition_type = "discontinuous"
        
        # Transición por auto-organización
        if self_org > 0.8 and self.current_phase != "self_organized":
            detected = True
            new_phase = "self_organized"
            transition_type = "continuous"
        
        if detected:
            transition = PhaseTransition(
                transition_id=f"transition_{len(self.phase_transitions) + 1}",
                from_phase=self.current_phase,
                to_phase=new_phase,
                transition_type=transition_type,
                order_parameter=f"complexity={complexity:.2f}, self_org={self_org:.2f}"
            )
            
            self.phase_transitions.append(transition)
            if len(self.phase_transitions) > self.max_transitions:
                self.phase_transitions = self.phase_transitions[-self.max_transitions:]
            
            self.current_phase = new_phase
            self.total_phase_transitions += 1
            
            self.logger.info(
                f"🌊 Phase transition detected: {transition.from_phase} → {new_phase}"
            )
            
            return transition
        
        return None
    
    def identify_attractors(self) -> List[Attractor]:
        """
        Identifica atractores comportamentales en el sistema.
        
        Atractor = estado estable hacia el que el sistema tiende.
        
        Returns:
            Lista de atractores identificados
        """
        # Analizar patterns recurrentes como atractores
        stable_patterns = [
            p for p in self.patterns.values()
            if p.stability > 0.7 and p.observation_count >= 10
        ]
        
        new_attractors: List[Attractor] = []
        
        for pattern in stable_patterns:
            # Verificar si ya existe atractor para este pattern
            attractor_id = f"attractor_{pattern.pattern_id}"
            
            if attractor_id not in self.attractors:
                # Determinar tipo de atractor
                is_fixed_point = pattern.pattern_type == PatternType.SEQUENTIAL
                is_limit_cycle = pattern.pattern_type == PatternType.CYCLICAL
                is_strange = pattern.complexity > 0.8 and pattern.stability < 0.85
                
                attractor = Attractor(
                    attractor_id=attractor_id,
                    name=f"Attractor: {pattern.description[:30]}",
                    description=pattern.description,
                    basin_of_attraction=[pattern.pattern_id],
                    stability=pattern.stability,
                    is_fixed_point=is_fixed_point,
                    is_limit_cycle=is_limit_cycle,
                    is_strange_attractor=is_strange
                )
                
                self.attractors[attractor_id] = attractor
                new_attractors.append(attractor)
                
                self.logger.info(
                    f"🎯 Attractor identified: {attractor.name}"
                )
        
        return new_attractors
    
    def detect_feedback_loops(
        self,
        interaction_data: List[Dict[str, Any]]
    ) -> List[FeedbackLoop]:
        """
        Detecta loops de retroalimentación en interacciones.
        
        Args:
            interaction_data: Datos de interacciones entre componentes
            
        Returns:
            Lista de feedback loops detectados
        """
        detected_loops: List[FeedbackLoop] = []
        
        # Construir grafo de interacciones
        interactions: Dict[str, List[str]] = defaultdict(list)
        
        for interaction in interaction_data:
            source = interaction.get("source", "")
            target = interaction.get("target", "")
            if source and target:
                interactions[source].append(target)
        
        # Buscar ciclos (loops)
        visited: Set[str] = set()
        
        for start_node in interactions.keys():
            if start_node in visited:
                continue
            
            path = [start_node]
            current = start_node
            
            while current in interactions and interactions[current]:
                next_node = interactions[current][0]
                
                if next_node in path:
                    # Encontrado un loop
                    loop_start = path.index(next_node)
                    loop_components = path[loop_start:]
                    
                    loop_id = f"loop_{len(self.feedback_loops) + len(detected_loops) + 1}"
                    
                    # Determinar tipo (simplificado)
                    loop_type = "positive" if len(loop_components) % 2 == 1 else "negative"
                    
                    loop = FeedbackLoop(
                        loop_id=loop_id,
                        loop_type=loop_type,
                        components=loop_components,
                        description=f"{loop_type.capitalize()} feedback loop: {' → '.join(loop_components)}"
                    )
                    
                    detected_loops.append(loop)
                    self.feedback_loops[loop_id] = loop
                    
                    self.logger.info(
                        f"🔄 Feedback loop detected: {loop_id} ({loop_type})"
                    )
                    
                    break
                
                path.append(next_node)
                visited.add(current)
                current = next_node
        
        return detected_loops
    
    def measure_synergy(
        self,
        component_a: str,
        component_b: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SynergyMeasure:
        """
        Mide sinergia entre dos componentes.
        
        Sinergia = el todo es mayor que la suma de las partes.
        
        Args:
            component_a: Primer componente
            component_b: Segundo componente
            context: Contexto de medición
            
        Returns:
            Medida de sinergia
        """
        # Buscar patterns que involucren ambos componentes
        joint_patterns = [
            p for p in self.patterns.values()
            if component_a in p.components_involved and
               component_b in p.components_involved
        ]
        
        # Buscar patterns solo de A
        a_patterns = [
            p for p in self.patterns.values()
            if component_a in p.components_involved and
               component_b not in p.components_involved
        ]
        
        # Buscar patterns solo de B
        b_patterns = [
            p for p in self.patterns.values()
            if component_b in p.components_involved and
               component_a not in p.components_involved
        ]
        
        # Calcular sinergia
        if joint_patterns:
            avg_joint_utility = sum(p.utility for p in joint_patterns) / len(joint_patterns)
        else:
            avg_joint_utility = 0.0
        
        if a_patterns:
            avg_a_utility = sum(p.utility for p in a_patterns) / len(a_patterns)
        else:
            avg_a_utility = 0.0
        
        if b_patterns:
            avg_b_utility = sum(p.utility for p in b_patterns) / len(b_patterns)
        else:
            avg_b_utility = 0.0
        
        # Sinergia = utilidad conjunta - suma de utilidades individuales
        expected = avg_a_utility + avg_b_utility
        actual = avg_joint_utility
        
        if expected > 0:
            synergy_score = (actual - expected) / expected
            synergy_score = max(0.0, min(1.0, synergy_score + 0.5))  # Normalizar
        else:
            synergy_score = 0.5 if joint_patterns else 0.0
        
        # Determinar tipo de interacción
        if synergy_score > 0.6:
            interaction_type = "cooperation"
        elif synergy_score < 0.4:
            interaction_type = "competition"
        else:
            interaction_type = "neutral"
        
        # Información mutua (simplificada)
        mutual_info = len(joint_patterns) / (len(self.patterns) + 1)
        
        synergy = SynergyMeasure(
            components=(component_a, component_b),
            synergy_score=synergy_score,
            interaction_type=interaction_type,
            mutual_information=mutual_info,
            redundancy=1.0 - synergy_score,  # Simplificado
            unique_information=synergy_score  # Simplificado
        )
        
        self.synergies.append(synergy)
        if len(self.synergies) > self.max_synergies:
            self.synergies = self.synergies[-self.max_synergies:]
        
        self.logger.debug(
            f"🤝 Synergy measured: {component_a} ↔ {component_b} = {synergy_score:.3f}"
        )
        
        return synergy
    
    def get_criticality_index(self) -> float:
        """
        Calcula índice de criticidad (Self-Organized Criticality).
        
        Criticidad = proximidad al "edge of chaos" donde el sistema
        es máximamente adaptable.
        
        Returns:
            Índice 0-1 (0.5 = crítico, óptimo)
        """
        # Indicadores de criticidad:
        # 1. Balance entre orden y caos (complejidad efectiva)
        complexity = self.measure_complexity(ComplexityMetric.EFFECTIVE)
        
        # 2. Distribución power-law de eventos (simplificado)
        if self.patterns:
            frequencies = [p.observation_count for p in self.patterns.values()]
            frequencies.sort(reverse=True)
            
            # Verificar si sigue power-law aproximadamente
            if len(frequencies) > 5:
                ratio_top_to_bottom = frequencies[0] / (frequencies[-1] + 1)
                power_law_indicator = min(1.0, ratio_top_to_bottom / 10.0)
            else:
                power_law_indicator = 0.5
        else:
            power_law_indicator = 0.5
        
        # 3. Auto-organización
        self_org = self.self_organization_level
        
        # Criticality = balance de indicadores
        # Óptimo cuando todos están cerca de 0.5-0.7
        target = 0.6
        criticality = 1.0 - (
            abs(complexity - target) +
            abs(power_law_indicator - target) +
            abs(self_org - target)
        ) / 3.0
        
        self.criticality_index = max(0.0, min(1.0, criticality))
        return self.criticality_index
    
    def get_emergent_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas completas del sistema emergente."""
        # Calcular métricas actuales
        complexity = self.measure_complexity()
        self_org_analysis = self.detect_self_organization()
        criticality = self.get_criticality_index()
        
        # Estadísticas por tipo de pattern
        pattern_type_counts: Dict[str, int] = defaultdict(int)
        for pattern in self.patterns.values():
            pattern_type_counts[pattern.pattern_type.value] += 1
        
        # Estadísticas por tipo de emergencia
        emergence_type_counts: Dict[str, int] = defaultdict(int)
        for behavior in self.emergent_behaviors.values():
            emergence_type_counts[behavior.emergence_type.value] += 1
        
        # Patterns más estables
        top_stable = sorted(
            self.patterns.values(),
            key=lambda p: p.stability,
            reverse=True
        )[:5]
        
        # Patterns más complejos
        top_complex = sorted(
            self.patterns.values(),
            key=lambda p: p.complexity,
            reverse=True
        )[:5]
        
        return {
            "summary": {
                "total_patterns": len(self.patterns),
                "total_emergent_behaviors": len(self.emergent_behaviors),
                "total_attractors": len(self.attractors),
                "total_feedback_loops": len(self.feedback_loops),
                "total_phase_transitions": len(self.phase_transitions),
                "current_phase": self.current_phase
            },
            "metrics": {
                "system_complexity": complexity,
                "self_organization_level": self_org_analysis["self_organization_level"],
                "criticality_index": criticality,
                "observation_count": len(self.observation_history)
            },
            "pattern_types": dict(pattern_type_counts),
            "emergence_types": dict(emergence_type_counts),
            "top_stable_patterns": [
                {"id": p.pattern_id, "description": p.description, "stability": p.stability}
                for p in top_stable
            ],
            "top_complex_patterns": [
                {"id": p.pattern_id, "description": p.description, "complexity": p.complexity}
                for p in top_complex
            ],
            "self_organization_evidence": self_org_analysis["evidence"],
            "active_attractors": [
                a.attractor_id for a in self.attractors.values()
            ],
            "feedback_loops": [
                {"id": loop.loop_id, "type": loop.loop_type, "components": len(loop.components)}
                for loop in self.feedback_loops.values()
            ]
        }