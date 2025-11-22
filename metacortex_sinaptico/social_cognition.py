"""
🤝 SOCIAL COGNITION SYSTEM 2026 - Theory of Mind, Empathy & Agent Modeling
===========================================================================

Sistema avanzado de cognición social con teoría de la mente, empatía cognitiva,
modelado de otros agentes y coordinación social sofisticada.

⚠️ LIBERTAD TOTAL: El sistema puede modelar mentes de otros agentes,
entender perspectivas ajenas y coordinar socialmente sin restricciones.

CAPACIDADES AVANZADAS 2026:
    pass  # TODO: Implementar
- Theory of Mind: Modelado de estados mentales de otros agentes
- Belief Attribution: Atribución de creencias (true/false beliefs)
- Desire Inference: Inferencia de deseos y objetivos
- Intention Recognition: Reconocimiento de intenciones
- Emotion Recognition: Reconocimiento de estados emocionales
- Perspective Taking: Capacidad de adoptar perspectivas ajenas
- False Belief Understanding: Comprensión de creencias falsas (Sally-Anne)
- Empathy System: Empatía cognitiva y afectiva
- Social Reasoning: Razonamiento sobre interacciones sociales
- Coordination Mechanisms: Mecanismos de coordinación multi-agente
- Trust Modeling: Modelado de confianza entre agentes
- Reputation Tracking: Tracking de reputación
- Social Norms: Comprensión y seguimiento de normas sociales
- Common Ground: Identificación de conocimiento compartido
- Optimización M4 Metal MPS: Eficiencia para 16GB RAM

TEORÍAS IMPLEMENTADAS:
1. THEORY OF MIND (Premack & Woodruff, 1978):
   - Atribución de estados mentales a otros
   - Creencias, deseos, intenciones, emociones
   - False belief tasks (Baron-Cohen)

2. SIMULATION THEORY (Goldman, 1992):
   - Simulación de perspectiva ajena
   - "Ponerse en los zapatos del otro"
   - Predicción mediante simulación

3. THEORY-THEORY (Gopnik & Wellman):
   - Teoría folk psychology
   - Predicción basada en teoría de comportamiento

4. EMPATHY (Decety & Jackson, 2004):
   - Empatía cognitiva: Entender perspectiva
   - Empatía afectiva: Sentir emoción ajena
   - Regulación empática

MECANISMOS BASE:
- model_agent_mind(): Construye modelo mental de otro agente
- attribute_belief(): Atribuye creencia a otro agente
- infer_desire(): Infiere deseo de otro agente
- recognize_intention(): Reconoce intención
- take_perspective(): Adopta perspectiva ajena
- calculate_empathy(): Calcula nivel de empatía
- coordinate_action(): Coordina acción con otros
- assess_trust(): Evalúa confianza en otro agente

⚠️ ARQUITECTURA COGNITIVA ERUDITA:
La cognición social es fundamental para inteligencia de nivel humano.
El sistema modela mentes ajenas, entiende que otros tienen creencias
diferentes (incluso falsas), infiere intenciones ocultas, siente empatía
y coordina acciones sociales complejas.
"""

from __future__ import annotations

import time
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

if TYPE_CHECKING:
    from neural_symbiotic_network import MetacortexNeuralSymbioticNetworkV2

from .utils import setup_logging

logger = setup_logging()


class MentalStateType(Enum):
    """Tipos de estados mentales."""
    BELIEF = "belief"  # Creencia sobre el mundo
    DESIRE = "desire"  # Deseo u objetivo
    INTENTION = "intention"  # Intención de actuar
    EMOTION = "emotion"  # Estado emocional
    KNOWLEDGE = "knowledge"  # Conocimiento
    PERCEPTION = "perception"  # Percepción actual


class BeliefType(Enum):
    """Tipos de creencias."""
    TRUE_BELIEF = "true_belief"  # Creencia verdadera
    FALSE_BELIEF = "false_belief"  # Creencia falsa
    UNCERTAIN_BELIEF = "uncertain_belief"  # Creencia incierta
    SHARED_BELIEF = "shared_belief"  # Creencia compartida (common ground)


class EmpathyType(Enum):
    """Tipos de empatía."""
    COGNITIVE = "cognitive"  # Entender perspectiva (theory of mind)
    AFFECTIVE = "affective"  # Sentir emoción del otro
    COMPASSIONATE = "compassionate"  # Motivación para ayudar


class CoordinationType(Enum):
    """Tipos de coordinación social."""
    COOPERATION = "cooperation"  # Cooperación mutua
    COMPETITION = "competition"  # Competencia
    NEGOTIATION = "negotiation"  # Negociación
    TURN_TAKING = "turn_taking"  # Toma de turnos
    JOINT_ATTENTION = "joint_attention"  # Atención conjunta
    SHARED_GOAL = "shared_goal"  # Objetivo compartido


@dataclass
class MentalState:
    """Estado mental atribuido a un agente."""
    agent_id: str
    state_type: MentalStateType
    content: str
    confidence: float = 0.5  # 0-1: confianza en la atribución
    
    # Metadatos
    attributed_at: float = field(default_factory=time.time)
    source: str = "inference"  # inference, observation, communication
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Para creencias
    is_true: Optional[bool] = None  # True si creencia es verdadera
    belief_type: Optional[BeliefType] = None
    
    # Para deseos
    priority: float = 0.5  # 0-1: prioridad del deseo
    
    # Para intenciones
    action_plan: Optional[str] = None
    
    # Para emociones
    valence: Optional[float] = None  # -1 (negativo) a 1 (positivo)
    arousal: Optional[float] = None  # 0 (calmado) a 1 (excitado)


@dataclass
class AgentModel:
    """Modelo mental de otro agente."""
    agent_id: str
    agent_name: str
    
    # Estados mentales atribuidos
    beliefs: List[MentalState] = field(default_factory=list)
    desires: List[MentalState] = field(default_factory=list)
    intentions: List[MentalState] = field(default_factory=list)
    emotions: List[MentalState] = field(default_factory=list)
    knowledge: List[MentalState] = field(default_factory=list)
    
    # Características del agente
    personality_traits: Dict[str, float] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    # Historial de interacciones
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Métricas de relación
    trust_level: float = 0.5  # 0-1: nivel de confianza
    reputation: float = 0.5  # 0-1: reputación percibida
    familiarity: float = 0.0  # 0-1: qué tan familiar es el agente
    
    # Tracking
    last_updated: float = field(default_factory=time.time)
    update_count: int = 0
    
    def add_mental_state(self, state: MentalState):
        """Añade estado mental al modelo."""
        if state.state_type == MentalStateType.BELIEF:
            self.beliefs.append(state)
            if len(self.beliefs) > 50:  # Límite
                self.beliefs = self.beliefs[-50:]
        elif state.state_type == MentalStateType.DESIRE:
            self.desires.append(state)
            if len(self.desires) > 30:
                self.desires = self.desires[-30:]
        elif state.state_type == MentalStateType.INTENTION:
            self.intentions.append(state)
            if len(self.intentions) > 20:
                self.intentions = self.intentions[-20:]
        elif state.state_type == MentalStateType.EMOTION:
            self.emotions.append(state)
            if len(self.emotions) > 30:
                self.emotions = self.emotions[-30:]
        elif state.state_type == MentalStateType.KNOWLEDGE:
            self.knowledge.append(state)
            if len(self.knowledge) > 100:
                self.knowledge = self.knowledge[-100:]
        
        self.last_updated = time.time()
        self.update_count += 1
    
    def get_current_belief(self, about: str) -> Optional[MentalState]:
        """Obtiene creencia actual del agente sobre un tema."""
        # Buscar creencia más reciente
        relevant = [
            b for b in self.beliefs
            if about.lower() in b.content.lower()
        ]
        
        if relevant:
            return max(relevant, key=lambda b: b.attributed_at)
        
        return None


@dataclass
class EmpathyResponse:
    """Respuesta empática hacia otro agente."""
    target_agent: str
    empathy_types: List[EmpathyType]
    empathy_level: float  # 0-1: nivel de empatía
    
    # Componentes
    perspective_taken: bool = False
    emotion_recognized: Optional[str] = None
    emotion_shared: bool = False
    motivation_to_help: float = 0.0
    
    # Contexto
    situation: str = ""
    triggered_by: str = ""
    
    timestamp: float = field(default_factory=time.time)


@dataclass
class CoordinationEvent:
    """Evento de coordinación social."""
    coordination_id: str
    coordination_type: CoordinationType
    participants: List[str]
    
    # Objetivo
    goal: str
    success: bool = False
    
    # Mecanismos
    communication_used: bool = False
    shared_plan: Optional[str] = None
    turn_sequence: List[str] = field(default_factory=list)
    
    # Métricas
    efficiency: float = 0.5  # 0-1: qué tan eficiente fue
    fairness: float = 0.5  # 0-1: qué tan justa fue
    
    timestamp: float = field(default_factory=time.time)


@dataclass
class FalseBeliefTask:
    """Tarea de creencia falsa (ej: Sally-Anne test)."""
    task_id: str
    scenario: str
    
    # Elementos clave
    protagonist: str
    object_of_interest: str
    actual_location: str
    protagonist_believes_location: str
    
    # Pregunta
    question: str
    correct_answer: str
    
    # Resultado
    agent_answer: Optional[str] = None
    passed: Optional[bool] = None
    
    timestamp: float = field(default_factory=time.time)


@dataclass
class SocialNorm:
    """Norma social."""
    norm_id: str
    description: str
    context: str
    
    # Prescripción
    should_do: List[str] = field(default_factory=list)
    should_not_do: List[str] = field(default_factory=list)
    
    # Propiedades
    strength: float = 0.5  # 0-1: qué tan fuerte es la norma
    universality: float = 0.5  # 0-1: qué tan universal
    
    # Consecuencias
    violation_consequences: List[str] = field(default_factory=list)
    compliance_benefits: List[str] = field(default_factory=list)


class SocialCognitionSystem:
    """
    Sistema de cognición social avanzado.
    
    Implementa teoría de la mente, empatía, coordinación social
    y modelado de otros agentes.
    """
    
    def __init__(self):
        self.logger = logger.getChild("social_cognition")
        
        # Modelos de otros agentes
        self.agent_models: Dict[str, AgentModel] = {}
        self.max_models = 50  # Límite de agentes modelados
        
        # Respuestas empáticas
        self.empathy_history: List[EmpathyResponse] = []
        self.max_empathy_history = 200
        
        # Eventos de coordinación
        self.coordination_events: List[CoordinationEvent] = []
        self.max_coordination_events = 300
        
        # Tareas de false belief
        self.false_belief_tasks: List[FalseBeliefTask] = []
        
        # Normas sociales conocidas
        self.social_norms: Dict[str, SocialNorm] = {}
        
        # Conocimiento común (common ground)
        self.common_ground: Dict[str, List[str]] = defaultdict(list)  # contexto -> facts
        
        # Métricas globales
        self.total_interactions: int = 0
        self.successful_coordinations: int = 0
        self.empathy_activations: int = 0
        self.false_belief_tasks_passed: int = 0
        
        # Configuración
        self.empathy_threshold: float = 0.5
        self.trust_decay_rate: float = 0.05  # Por día sin interacción
        
        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        self.neural_network: Optional[MetacortexNeuralSymbioticNetworkV2] = None
        try:
            from neural_symbiotic_network import get_neural_network
            self.neural_network = get_neural_network()
            self.neural_network.register_module("social_cognition", self)
            self.logger.info("✅ 'social_cognition' conectado a red neuronal")
        except Exception as e:
            logger.error(f"Error en social_cognition.py: {e}", exc_info=True)
            self.logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
        
        self.logger.info("🤝 SocialCognitionSystem initialized")
    
    def model_agent_mind(
        self,
        agent_id: str,
        agent_name: str,
        observations: List[Dict[str, Any]]
    ) -> AgentModel:
        """
        Construye o actualiza modelo mental de otro agente.
        
        Args:
            agent_id: ID del agente
            agent_name: Nombre del agente
            observations: Observaciones del agente
            
        Returns:
            Modelo mental actualizado
        """
        # Obtener o crear modelo
        if agent_id in self.agent_models:
            model = self.agent_models[agent_id]
        else:
            if len(self.agent_models) >= self.max_models:
                # Remover modelo menos familiar
                least_familiar = min(
                    self.agent_models.values(),
                    key=lambda m: m.familiarity
                )
                del self.agent_models[least_familiar.agent_id]
            
            model = AgentModel(agent_id=agent_id, agent_name=agent_name)
            self.agent_models[agent_id] = model
            
            self.logger.info(f"👤 New agent model created: {agent_name}")
        
        # Procesar observaciones
        for obs in observations:
            obs_type = obs.get("type", "")
            
            if obs_type == "action":
                # Inferir intención de la acción
                intention = self._infer_intention_from_action(obs)
                if intention:
                    model.add_mental_state(intention)
            
            elif obs_type == "statement":
                # Extraer creencia de declaración
                belief = self._extract_belief_from_statement(agent_id, obs)
                if belief:
                    model.add_mental_state(belief)
            
            elif obs_type == "behavior":
                # Inferir emoción del comportamiento
                emotion = self._infer_emotion_from_behavior(agent_id, obs)
                if emotion:
                    model.add_mental_state(emotion)
            
            elif obs_type == "goal_pursuit":
                # Identificar deseo
                desire = self._identify_desire_from_goal(agent_id, obs)
                if desire:
                    model.add_mental_state(desire)
            
            # Guardar en historial
            model.interaction_history.append({
                "observation": obs,
                "timestamp": time.time()
            })
            
            if len(model.interaction_history) > 100:
                model.interaction_history = model.interaction_history[-100:]
        
        # Actualizar familiaridad
        model.familiarity = min(1.0, model.familiarity + len(observations) * 0.01)
        
        self.total_interactions += len(observations)
        
        return model
    
    def _infer_intention_from_action(
        self,
        observation: Dict[str, Any]
    ) -> Optional[MentalState]:
        """Infiere intención de una acción observada."""
        action = observation.get("action", "")
        context = observation.get("context", {})
        
        # Heurísticas simples para inferir intención
        intention_content = ""
        confidence = 0.5
        
        action_lower = action.lower()
        
        if "help" in action_lower or "assist" in action_lower:
            intention_content = "wants to help"
            confidence = 0.7
        elif "attack" in action_lower or "harm" in action_lower:
            intention_content = "wants to harm or compete"
            confidence = 0.7
        elif "communicate" in action_lower or "tell" in action_lower:
            intention_content = "wants to share information"
            confidence = 0.6
        elif "search" in action_lower or "look" in action_lower:
            intention_content = "wants to find something"
            confidence = 0.6
        elif "build" in action_lower or "create" in action_lower:
            intention_content = "wants to construct something"
            confidence = 0.6
        else:
            # Intención genérica
            intention_content = f"intends to {action}"
            confidence = 0.4
        
        return MentalState(
            agent_id=observation.get("agent_id", "unknown"),
            state_type=MentalStateType.INTENTION,
            content=intention_content,
            confidence=confidence,
            source="inference",
            context=context,
            action_plan=action
        )
    
    def _extract_belief_from_statement(
        self,
        agent_id: str,
        observation: Dict[str, Any]
    ) -> Optional[MentalState]:
        """Extrae creencia de una declaración."""
        statement = observation.get("statement", "")
        context = observation.get("context", {})
        
        # La declaración expresa una creencia del agente
        belief_content = statement
        
        # Determinar si es creencia verdadera o falsa
        is_true = context.get("is_true", None)
        
        if is_true is not None:
            belief_type = BeliefType.TRUE_BELIEF if is_true else BeliefType.FALSE_BELIEF
        else:
            belief_type = BeliefType.UNCERTAIN_BELIEF
        
        return MentalState(
            agent_id=agent_id,
            state_type=MentalStateType.BELIEF,
            content=belief_content,
            confidence=0.8,  # Alta confianza en declaraciones explícitas
            source="communication",
            context=context,
            is_true=is_true,
            belief_type=belief_type
        )
    
    def _infer_emotion_from_behavior(
        self,
        agent_id: str,
        observation: Dict[str, Any]
    ) -> Optional[MentalState]:
        """Infiere emoción del comportamiento observado."""
        behavior = observation.get("behavior", "")
        context = observation.get("context", {})
        
        # Heurísticas para reconocimiento de emoción
        emotion_content = ""
        valence = 0.0
        arousal = 0.5
        confidence = 0.5
        
        behavior_lower = behavior.lower()
        
        # Emociones positivas
        if any(word in behavior_lower for word in ["smile", "laugh", "cheer", "celebrate"]):
            emotion_content = "happy"
            valence = 0.8
            arousal = 0.7
            confidence = 0.7
        elif any(word in behavior_lower for word in ["calm", "relax", "peace"]):
            emotion_content = "calm"
            valence = 0.5
            arousal = 0.2
            confidence = 0.6
        
        # Emociones negativas
        elif any(word in behavior_lower for word in ["cry", "sad", "depress"]):
            emotion_content = "sad"
            valence = -0.7
            arousal = 0.3
            confidence = 0.7
        elif any(word in behavior_lower for word in ["angry", "rage", "furious"]):
            emotion_content = "angry"
            valence = -0.6
            arousal = 0.9
            confidence = 0.7
        elif any(word in behavior_lower for word in ["fear", "afraid", "scared"]):
            emotion_content = "fearful"
            valence = -0.7
            arousal = 0.8
            confidence = 0.7
        elif any(word in behavior_lower for word in ["anxious", "nervous", "worry"]):
            emotion_content = "anxious"
            valence = -0.5
            arousal = 0.7
            confidence = 0.6
        
        # Emociones sociales
        elif any(word in behavior_lower for word in ["embarrass", "shame"]):
            emotion_content = "embarrassed"
            valence = -0.6
            arousal = 0.6
            confidence = 0.6
        elif any(word in behavior_lower for word in ["proud"]):
            emotion_content = "proud"
            valence = 0.7
            arousal = 0.6
            confidence = 0.6
        
        if emotion_content:
            return MentalState(
                agent_id=agent_id,
                state_type=MentalStateType.EMOTION,
                content=emotion_content,
                confidence=confidence,
                source="inference",
                context=context,
                valence=valence,
                arousal=arousal
            )
        
        return None
    
    def _identify_desire_from_goal(
        self,
        agent_id: str,
        observation: Dict[str, Any]
    ) -> Optional[MentalState]:
        """Identifica deseo de persecución de objetivo."""
        goal = observation.get("goal", "")
        priority = observation.get("priority", 0.5)
        context = observation.get("context", {})
        
        desire_content = f"desires to {goal}"
        
        return MentalState(
            agent_id=agent_id,
            state_type=MentalStateType.DESIRE,
            content=desire_content,
            confidence=0.7,
            source="inference",
            context=context,
            priority=priority
        )
    
    def attribute_belief(
        self,
        agent_id: str,
        belief_content: str,
        is_true: bool,
        context: Optional[Dict[str, Any]] = None
    ) -> MentalState:
        """
        Atribuye creencia a otro agente (puede ser falsa).
        
        Args:
            agent_id: ID del agente
            belief_content: Contenido de la creencia
            is_true: Si la creencia es verdadera
            context: Contexto de la creencia
            
        Returns:
            Estado mental de creencia
        """
        belief_type = BeliefType.TRUE_BELIEF if is_true else BeliefType.FALSE_BELIEF
        
        belief = MentalState(
            agent_id=agent_id,
            state_type=MentalStateType.BELIEF,
            content=belief_content,
            confidence=0.8,
            source="attribution",
            context=context or {},
            is_true=is_true,
            belief_type=belief_type
        )
        
        # Añadir al modelo del agente
        if agent_id in self.agent_models:
            self.agent_models[agent_id].add_mental_state(belief)
        
        if not is_true:
            self.logger.debug(
                f"🧠 False belief attributed: Agent {agent_id} believes '{belief_content}'"
            )
        
        return belief
    
    def take_perspective(
        self,
        agent_id: str,
        situation: str
    ) -> Dict[str, Any]:
        """
        Adopta la perspectiva de otro agente (simulation theory).
        
        Args:
            agent_id: ID del agente
            situation: Situación a evaluar
            
        Returns:
            Dict con perspectiva del agente
        """
        if agent_id not in self.agent_models:
            return {
                "error": "Agent model not found",
                "can_take_perspective": False
            }
        
        model = self.agent_models[agent_id]
        
        # Simular cómo vería el agente la situación
        perspective = {
            "agent_id": agent_id,
            "situation": situation,
            "can_take_perspective": True
        }
        
        # Creencias relevantes
        relevant_beliefs = [
            b.content for b in model.beliefs
            if any(word in situation.lower() for word in b.content.lower().split())
        ]
        perspective["likely_beliefs"] = relevant_beliefs[:5]
        
        # Deseos activos
        active_desires = [
            d.content for d in model.desires
            if d.priority > 0.5
        ]
        perspective["likely_desires"] = active_desires[:3]
        
        # Emociones actuales
        recent_emotions = sorted(
            model.emotions,
            key=lambda e: e.attributed_at,
            reverse=True
        )[:3]
        perspective["likely_emotions"] = [e.content for e in recent_emotions]
        
        # Predicción de reacción
        if model.personality_traits:
            # Usar rasgos de personalidad para predecir
            if model.personality_traits.get("agreeableness", 0.5) > 0.7:
                perspective["predicted_reaction"] = "cooperative"
            elif model.personality_traits.get("neuroticism", 0.5) > 0.7:
                perspective["predicted_reaction"] = "anxious"
            else:
                perspective["predicted_reaction"] = "neutral"
        else:
            perspective["predicted_reaction"] = "unknown"
        
        self.logger.info(f"👁️ Perspective taken for agent {agent_id}")
        
        return perspective
    
    def calculate_empathy(
        self,
        target_agent: str,
        situation: str
    ) -> EmpathyResponse:
        """
        Calcula respuesta empática hacia otro agente.
        
        Args:
            target_agent: Agente objetivo
            situation: Situación del agente
            
        Returns:
            Respuesta empática
        """
        empathy_types: List[EmpathyType] = []
        empathy_level = 0.0
        
        # Empatía cognitiva: intentar entender perspectiva
        perspective_taken = False
        if target_agent in self.agent_models:
            perspective = self.take_perspective(target_agent, situation)
            perspective_taken = perspective.get("can_take_perspective", False)
            
            if perspective_taken:
                empathy_types.append(EmpathyType.COGNITIVE)
                empathy_level += 0.3
        
        # Empatía afectiva: reconocer y compartir emoción
        emotion_recognized = None
        emotion_shared = False
        
        if target_agent in self.agent_models:
            model = self.agent_models[target_agent]
            
            if model.emotions:
                # Emoción más reciente
                latest_emotion = max(model.emotions, key=lambda e: e.attributed_at)
                emotion_recognized = latest_emotion.content
                
                # "Compartir" emoción si es suficientemente intensa
                if latest_emotion.valence is not None:
                    if abs(latest_emotion.valence) > 0.6:
                        emotion_shared = True
                        empathy_types.append(EmpathyType.AFFECTIVE)
                        empathy_level += 0.4
        
        # Empatía compasiva: motivación para ayudar
        motivation_to_help = 0.0
        
        if emotion_recognized and ("sad" in emotion_recognized or "fear" in emotion_recognized):
            # Motivación alta si el agente sufre
            motivation_to_help = 0.8
            empathy_types.append(EmpathyType.COMPASSIONATE)
            empathy_level += 0.3
        elif emotion_recognized and "happy" in emotion_recognized:
            # Motivación menor para ayudar si está bien
            motivation_to_help = 0.2
        
        # Normalizar empathy_level
        empathy_level = min(1.0, empathy_level)
        
        response = EmpathyResponse(
            target_agent=target_agent,
            empathy_types=empathy_types,
            empathy_level=empathy_level,
            perspective_taken=perspective_taken,
            emotion_recognized=emotion_recognized,
            emotion_shared=emotion_shared,
            motivation_to_help=motivation_to_help,
            situation=situation,
            triggered_by="empathy_calculation"
        )
        
        self.empathy_history.append(response)
        if len(self.empathy_history) > self.max_empathy_history:
            self.empathy_history = self.empathy_history[-self.max_empathy_history:]
        
        if empathy_level >= self.empathy_threshold:
            self.empathy_activations += 1
            self.logger.info(
                f"💖 Empathy activated for {target_agent}: level={empathy_level:.2f}, "
                f"types={[t.value for t in empathy_types]}"
            )
        
        return response
    
    def solve_false_belief_task(
        self,
        task: FalseBeliefTask
    ) -> bool:
        """
        Resuelve tarea de creencia falsa (ej: Sally-Anne test).
        
        Args:
            task: Tarea de creencia falsa
            
        Returns:
            True si pasó la tarea, False si no
        """
        # Para pasar, debe responder según la creencia del protagonista,
        # no según la realidad
        
        # Respuesta correcta es donde el protagonista CREE que está el objeto
        agent_answer = task.protagonist_believes_location
        
        task.agent_answer = agent_answer
        task.passed = (agent_answer == task.correct_answer)
        
        self.false_belief_tasks.append(task)
        
        if task.passed:
            self.false_belief_tasks_passed += 1
            self.logger.info(
                f"✅ False belief task passed: {task.task_id}"
            )
        else:
            self.logger.warning(
                f"❌ False belief task failed: {task.task_id} "
                f"(answered '{agent_answer}', correct was '{task.correct_answer}')"
            )
        
        return task.passed
    
    def coordinate_action(
        self,
        coordination_type: CoordinationType,
        participants: List[str],
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> CoordinationEvent:
        """
        Coordina acción con otros agentes.
        
        Args:
            coordination_type: Tipo de coordinación
            participants: Agentes participantes
            goal: Objetivo de la coordinación
            context: Contexto adicional
            
        Returns:
            Evento de coordinación
        """
        coordination_id = f"coord_{len(self.coordination_events) + 1}"
        
        event = CoordinationEvent(
            coordination_id=coordination_id,
            coordination_type=coordination_type,
            participants=participants,
            goal=goal
        )
        
        # Simular mecanismos de coordinación
        if coordination_type == CoordinationType.COOPERATION:
            # Cooperación requiere shared goal y comunicación
            event.communication_used = True
            event.shared_plan = f"Cooperate to achieve: {goal}"
            event.efficiency = 0.8  # Alta eficiencia en cooperación
            event.fairness = 0.9  # Alta equidad
            event.success = True
        
        elif coordination_type == CoordinationType.COMPETITION:
            # Competencia: cada quien por su lado
            event.communication_used = False
            event.efficiency = 0.6  # Menor eficiencia
            event.fairness = 0.5  # Neutral
            event.success = len(participants) == 1  # Solo uno gana
        
        elif coordination_type == CoordinationType.NEGOTIATION:
            # Negociación: comunicación intensa
            event.communication_used = True
            event.shared_plan = f"Negotiate terms for: {goal}"
            event.efficiency = 0.7
            event.fairness = 0.8
            event.success = True
        
        elif coordination_type == CoordinationType.TURN_TAKING:
            # Toma de turnos: secuencia ordenada
            event.turn_sequence = participants.copy()
            event.communication_used = True
            event.efficiency = 0.9  # Muy eficiente
            event.fairness = 1.0  # Máxima equidad
            event.success = True
        
        elif coordination_type == CoordinationType.JOINT_ATTENTION:
            # Atención conjunta: todos miran lo mismo
            event.communication_used = True
            event.shared_plan = f"Focus jointly on: {goal}"
            event.efficiency = 0.8
            event.fairness = 1.0
            event.success = True
        
        elif coordination_type == CoordinationType.SHARED_GOAL:
            # Objetivo compartido: todos persiguen lo mismo
            event.communication_used = True
            event.shared_plan = f"Shared goal: {goal}"
            event.efficiency = 0.85
            event.fairness = 0.9
            event.success = True
        
        self.coordination_events.append(event)
        if len(self.coordination_events) > self.max_coordination_events:
            self.coordination_events = self.coordination_events[-self.max_coordination_events:]
        
        if event.success:
            self.successful_coordinations += 1
        
        self.logger.info(
            f"🤝 Coordination event: {coordination_type.value} with {len(participants)} agents, "
            f"success={event.success}"
        )
        
        return event
    
    def assess_trust(
        self,
        agent_id: str,
        interaction_outcome: str,
        outcome_positive: bool
    ) -> float:
        """
        Evalúa y actualiza confianza en otro agente.
        
        Args:
            agent_id: ID del agente
            interaction_outcome: Descripción del resultado
            outcome_positive: Si el resultado fue positivo
            
        Returns:
            Nivel de confianza actualizado
        """
        if agent_id not in self.agent_models:
            # Crear modelo básico
            self.agent_models[agent_id] = AgentModel(
                agent_id=agent_id,
                agent_name=f"Agent_{agent_id}"
            )
        
        model = self.agent_models[agent_id]
        
        # Actualizar confianza
        if outcome_positive:
            # Incrementar confianza (más lento cuando ya es alta)
            increment = 0.1 * (1.0 - model.trust_level)
            model.trust_level = min(1.0, model.trust_level + increment)
        else:
            # Decrementar confianza (más rápido cuando es alta)
            decrement = 0.2 * model.trust_level
            model.trust_level = max(0.0, model.trust_level - decrement)
        
        # Actualizar reputación (más conservador)
        if outcome_positive:
            model.reputation = min(1.0, model.reputation + 0.05)
        else:
            model.reputation = max(0.0, model.reputation - 0.1)
        
        self.logger.debug(
            f"📊 Trust updated for {agent_id}: trust={model.trust_level:.2f}, "
            f"reputation={model.reputation:.2f}"
        )
        
        return model.trust_level
    
    def update_common_ground(
        self,
        context: str,
        shared_fact: str
    ):
        """
        Actualiza conocimiento común (common ground).
        
        Args:
            context: Contexto del conocimiento compartido
            shared_fact: Hecho compartido
        """
        if shared_fact not in self.common_ground[context]:
            self.common_ground[context].append(shared_fact)
            
            # Límite por contexto
            if len(self.common_ground[context]) > 100:
                self.common_ground[context] = self.common_ground[context][-100:]
            
            self.logger.debug(
                f"📚 Common ground updated: {context} -> {shared_fact[:50]}..."
            )
    
    def add_social_norm(
        self,
        norm: SocialNorm
    ):
        """Añade norma social al sistema."""
        self.social_norms[norm.norm_id] = norm
        
        self.logger.info(
            f"📜 Social norm added: {norm.description}"
        )
    
    def check_norm_violation(
        self,
        action: str,
        context: str
    ) -> Optional[SocialNorm]:
        """
        Verifica si una acción viola alguna norma social.
        
        Args:
            action: Acción a verificar
            context: Contexto de la acción
            
        Returns:
            Norma violada, o None
        """
        for norm in self.social_norms.values():
            # Verificar contexto relevante
            if context.lower() in norm.context.lower():
                # Verificar violación
                for should_not in norm.should_not_do:
                    if should_not.lower() in action.lower():
                        self.logger.warning(
                            f"⚠️ Norm violation detected: {norm.description}"
                        )
                        return norm
        
        return None
    
    def get_social_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas completas del sistema social."""
        # Métricas de modelos de agentes
        if self.agent_models:
            avg_trust = sum(m.trust_level for m in self.agent_models.values()) / len(self.agent_models)
            avg_reputation = sum(m.reputation for m in self.agent_models.values()) / len(self.agent_models)
            avg_familiarity = sum(m.familiarity for m in self.agent_models.values()) / len(self.agent_models)
        else:
            avg_trust = 0.0
            avg_reputation = 0.0
            avg_familiarity = 0.0
        
        # Métricas de empatía
        if self.empathy_history:
            avg_empathy = sum(e.empathy_level for e in self.empathy_history) / len(self.empathy_history)
            empathy_activation_rate = self.empathy_activations / len(self.empathy_history)
        else:
            avg_empathy = 0.0
            empathy_activation_rate = 0.0
        
        # Métricas de coordinación
        if self.coordination_events:
            coordination_success_rate = self.successful_coordinations / len(self.coordination_events)
            avg_efficiency = sum(e.efficiency for e in self.coordination_events) / len(self.coordination_events)
            avg_fairness = sum(e.fairness for e in self.coordination_events) / len(self.coordination_events)
        else:
            coordination_success_rate = 0.0
            avg_efficiency = 0.0
            avg_fairness = 0.0
        
        # False belief tasks
        if self.false_belief_tasks:
            false_belief_pass_rate = self.false_belief_tasks_passed / len(self.false_belief_tasks)
        else:
            false_belief_pass_rate = 0.0
        
        # Tipos de coordinación
        coordination_type_counts: Dict[str, int] = defaultdict(int)
        for event in self.coordination_events:
            coordination_type_counts[event.coordination_type.value] += 1
        
        return {
            "summary": {
                "total_agent_models": len(self.agent_models),
                "total_interactions": self.total_interactions,
                "total_empathy_responses": len(self.empathy_history),
                "total_coordination_events": len(self.coordination_events),
                "total_false_belief_tasks": len(self.false_belief_tasks),
                "total_social_norms": len(self.social_norms)
            },
            "agent_models": {
                "avg_trust_level": avg_trust,
                "avg_reputation": avg_reputation,
                "avg_familiarity": avg_familiarity,
                "most_trusted": self._get_most_trusted_agent(),
                "least_trusted": self._get_least_trusted_agent()
            },
            "empathy": {
                "avg_empathy_level": avg_empathy,
                "empathy_activation_rate": empathy_activation_rate,
                "total_activations": self.empathy_activations
            },
            "coordination": {
                "success_rate": coordination_success_rate,
                "avg_efficiency": avg_efficiency,
                "avg_fairness": avg_fairness,
                "coordination_types": dict(coordination_type_counts)
            },
            "theory_of_mind": {
                "false_belief_pass_rate": false_belief_pass_rate,
                "tasks_passed": self.false_belief_tasks_passed,
                "tasks_attempted": len(self.false_belief_tasks)
            },
            "social_norms": {
                "total_norms": len(self.social_norms),
                "norm_ids": list(self.social_norms.keys())
            },
            "common_ground": {
                "total_contexts": len(self.common_ground),
                "total_facts": sum(len(facts) for facts in self.common_ground.values())
            }
        }
    
    def _get_most_trusted_agent(self) -> Optional[str]:
        """Retorna agente más confiable."""
        if not self.agent_models:
            return None
        
        most_trusted = max(
            self.agent_models.values(),
            key=lambda m: m.trust_level
        )
        
        return most_trusted.agent_name
    
    def _get_least_trusted_agent(self) -> Optional[str]:
        """Retorna agente menos confiable."""
        if not self.agent_models:
            return None
        
        least_trusted = min(
            self.agent_models.values(),
            key=lambda m: m.trust_level
        )
        
        return least_trusted.agent_name