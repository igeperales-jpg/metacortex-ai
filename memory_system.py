from __future__ import annotations
#!/usr/bin/env python3
"""
🧠 METACORTEX Memory System v2.0
Sistema avanzado de memoria con búsqueda semántica, knowledge graph y clustering

Evolución v2.0:
    pass  # TODO: Implementar
- SQLite para datos estructurados (compatible con v1)
- ChromaDB para búsqueda vectorial semántica (opcional)
- Knowledge graph con NetworkX (opcional)
- Clustering de conversaciones
- Análisis de patrones
- Export/Import avanzado

Compatible con versión anterior - Los sistemas v2.0 son opcionales
"""

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# 🔥 IMPORTS OBLIGATORIOS - NO LAZY, NO TRY-EXCEPT
# Si falta alguno, el sistema debe fallar inmediatamente con mensaje claro
import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from advanced_cache_system import get_global_cache
from neural_symbiotic_network import get_neural_network
from vector_embedding_system import get_embedding_system

_is_chromadb_available = False
try:
    import chromadb
    from chromadb.config import Settings
    _is_chromadb_available = True
except ImportError:
    chromadb = None
    Settings = None
    # _is_chromadb_available remains False


logger = logging.getLogger(__name__)


# Flags explícitos (siempre True si llegamos aquí)
EMBEDDINGS_V2_AVAILABLE = True
NETWORKX_AVAILABLE = True
CHROMADB_AVAILABLE = _is_chromadb_available


class MemoryType(Enum):
    """Tipos de memoria"""

    CONVERSATION = "conversation"
    PROJECT = "project"
    LEARNING = "learning"
    KNOWLEDGE = "knowledge"
    ERROR = "error"
    SOLUTION = "solution"


@dataclass
class SemanticSearchResult:
    """Resultado de búsqueda semántica"""

    content: str
    similarity: float
    timestamp: datetime
    memory_type: MemoryType
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetacortexMemory:
    """
    Sistema de memoria persistente para METACORTEX v2.0

    Compatible 100% con versión anterior.
    Nuevas características opcionales:
    - Búsqueda semántica con ChromaDB
    - Knowledge graph con NetworkX
    - Clustering de conversaciones
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        enable_semantic_search: bool = True,
        enable_knowledge_graph: bool = True,
    ):
        """
        Inicializar sistema de memoria

        Args:
            db_path: Ruta a la base de datos SQLite
            enable_semantic_search: Habilitar búsqueda semántica (requiere ChromaDB)
            enable_knowledge_graph: Habilitar knowledge graph (requiere NetworkX)
        """
        if db_path is None:
            db_path = str(Path.home() / ".metacortex" / "memory.db")

        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # 🆕 v2.0: Configurar flags ANTES de _init_database
        self.enable_semantic_search = (
            enable_semantic_search and EMBEDDINGS_V2_AVAILABLE and CHROMADB_AVAILABLE
        )
        self.enable_knowledge_graph = enable_knowledge_graph and NETWORKX_AVAILABLE

        # Inicializar base de datos
        self._init_database()

        self.embedding_system: Optional[Any] = None
        self.cache: Optional[Any] = None
        self.chroma_client: Optional["chromadb.Client"] = None
        self.chroma_conversations: Optional["chromadb.Collection"] = None
        self.chroma_learnings: Optional["chromadb.Collection"] = None
        self.chroma_knowledge: Optional["chromadb.Collection"] = None
        self.knowledge_graph: Optional[nx.DiGraph[Any]] = None

        # 🆕 v2.0: Sistema de embeddings (OBLIGATORIO)
        if self.enable_semantic_search:
            self.embedding_system = get_embedding_system()
            self.cache = get_global_cache()
            self._init_chromadb()
            logger.info("✅ Búsqueda semántica habilitada")

        # 🆕 v2.0: Knowledge graph (OBLIGATORIO)
        if self.enable_knowledge_graph:
            self.knowledge_graph = nx.DiGraph()
            self._load_knowledge_graph()
            logger.info("✅ Knowledge graph habilitado")

        # Mensaje informativo sobre modo de operación
        mode_features: List[str] = []
        if self.enable_semantic_search:
            mode_features.append("búsqueda semántica")
        if self.enable_knowledge_graph:
            mode_features.append("knowledge graph")

        if mode_features:
            logger.info(f"🧠 Sistema de memoria inicializado: {db_path}")
            logger.info(f"   Modo v2.0: {', '.join(mode_features)}")
        else:
            logger.info(f"🧠 Sistema de memoria inicializado: {db_path}")
            logger.info(
                "   ℹ️ Sistemas v2.0 no disponibles, funcionando en modo básico (100% compatible)"
            )

        # 🆕 2026: Alias para verify_system compatibility
        # Agregar atributos que verify_system busca
        self.search = (
            self.semantic_search if self.enable_semantic_search else None
        )  # Alias para búsqueda semántica
        self.working_memory: Dict[str, Any] = (
            {}
        )  # Working memory básica (dict para contexto temporal)

        # 🔗 CONEXIÓN OBLIGATORIA CON NEURAL NETWORK
        logger.info("🔗 Conectando con Neural Symbiotic Network...")
        self.neural_network = get_neural_network()
        if not self.neural_network:
            logger.error("❌ Neural Symbiotic Network no disponible - componente crítico")
            raise RuntimeError("❌ Neural Symbiotic Network no disponible - componente crítico")
        
        self.neural_network.register_module("memory_system", self)
        logger.info("   ✅ Conectado con Neural Symbiotic Network")

    def _init_database(self):
        """Crear tablas si no existen"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabla de conversaciones
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_message TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                model_used TEXT,
                tokens_used INTEGER,
                response_time REAL,
                context_hash TEXT,
                metadata TEXT
            )
        """)

        # Tabla de proyectos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL,
                project_path TEXT,
                description TEXT,
                technologies TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',
                metadata TEXT
            )
        """)

        # Tabla de errores y aprendizajes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                error_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                solution TEXT,
                context TEXT,
                project_id INTEGER,
                learned_from TEXT,
                confidence_score REAL DEFAULT 0.5,
                times_encountered INTEGER DEFAULT 1,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        """)

        # Tabla de conocimiento acumulado
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                relevance_score REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                tags TEXT
            )
        """)

        # Tabla de contexto de sesión
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_messages INTEGER DEFAULT 0,
                current_project_id INTEGER,
                context_summary TEXT,
                active INTEGER DEFAULT 1,
                FOREIGN KEY (current_project_id) REFERENCES projects(id)
            )
        """)

        # Índices para búsqueda rápida
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(project_name)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_learnings_error_type ON learnings(error_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge_base(topic)"
        )

        # 🆕 v2.0: Tablas adicionales para búsqueda semántica y knowledge graph
        if self.enable_knowledge_graph:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                target_entity TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_entity_relations_source ON entity_relations(source_entity)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_entity_relations_target ON entity_relations(target_entity)"
            )

        if self.enable_semantic_search:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                cluster_id INTEGER NOT NULL,
                similarity_score REAL DEFAULT 0.0,
                cluster_topic TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_clusters_conversation ON conversation_clusters(conversation_id)"
            )

        conn.commit()
        conn.close()

    def _init_chromadb(self):
        """
        🆕 v2.0: Inicializar ChromaDB para búsqueda semántica

        Crea colecciones para diferentes tipos de memoria:
        - conversations: Historial de conversaciones
        - learnings: Aprendizajes y soluciones
        - knowledge: Base de conocimiento
        """
        if not CHROMADB_AVAILABLE:
            raise RuntimeError("❌ ChromaDB no disponible. Instalar con: pip install chromadb")
        
        assert chromadb is not None, "chromadb is None despite CHROMADB_AVAILABLE being True"

        # FIX: Usar PersistentClient en vez de Client (deprecated)
        chroma_path = str(Path(self.db_path).parent / "chromadb")
        Path(chroma_path).mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        # Crear/obtener colecciones
        self.chroma_conversations = self.chroma_client.get_or_create_collection(
            name="metacortex_conversations",
            metadata={"description": "Historial de conversaciones"},
        )

        self.chroma_learnings = self.chroma_client.get_or_create_collection(
            name="metacortex_learnings",
            metadata={"description": "Aprendizajes y soluciones"},
        )

        self.chroma_knowledge = self.chroma_client.get_or_create_collection(
            name="metacortex_knowledge",
            metadata={"description": "Base de conocimiento"},
        )

        logger.info(f"✅ ChromaDB inicializado: {chroma_path}")

    def _load_knowledge_graph(self) -> None:
        """
        🆕 v2.0: Cargar knowledge graph desde la base de datos

        Carga todas las relaciones entity_relations y construye
        el grafo en memoria usando NetworkX.
        """
        if not NETWORKX_AVAILABLE or not self.enable_knowledge_graph or self.knowledge_graph is None:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
            SELECT source_entity, relation_type, target_entity, weight, metadata
            FROM entity_relations
            """)

            relations = cursor.fetchall()
            conn.close()

            # Construir grafo
            for source, relation, target, weight, metadata in relations:
                self.knowledge_graph.add_edge(
                    source,
                    target,
                    relation=relation,
                    weight=weight,
                    metadata=json.loads(metadata) if metadata else {},
                )

            logger.info(
                f"✅ Knowledge graph cargado: {len(self.knowledge_graph.nodes)} nodos, {len(self.knowledge_graph.edges)} aristas"
            )

        except Exception as e:
            logger.error(f"❌ Error cargando knowledge graph: {e}")
            self.knowledge_graph = nx.DiGraph()  # Grafo vacío en caso de error

    def save_conversation(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        model_used: Optional[str] = None,
        tokens_used: Optional[int] = None,
        response_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Guardar una conversación en la memoria

        Returns:
            ID de la conversación guardada
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Crear hash del contexto para búsquedas rápidas
        context_hash = hashlib.md5(f"{user_message[:100]}".encode()).hexdigest()

        cursor.execute(
            """
            INSERT INTO conversations 
            (session_id, user_message, assistant_response, model_used, 
             tokens_used, response_time, context_hash, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                user_message,
                assistant_response,
                model_used,
                tokens_used,
                response_time,
                context_hash,
                json.dumps(metadata) if metadata else None,
            ),
        )

        conversation_id = cursor.lastrowid

        # 🆕 v2.0: Generar y guardar embedding en ChromaDB
        if self.enable_semantic_search and self.embedding_system and self.chroma_conversations:
            try:
                # Combinar mensaje de usuario y respuesta
                full_text = f"Usuario: {user_message}\nAsistente: {assistant_response}"

                # Generar embedding
                embedding = self.embedding_system.encode(
                    user_message
                )  # Usar encode en lugar de generate_embedding

                # Guardar en ChromaDB
                self.chroma_conversations.add(
                    documents=[full_text],
                    embeddings=[embedding.tolist()],
                    ids=[f"conv_{conversation_id}"],
                    metadatas=[
                        {
                            "session_id": session_id,
                            "conversation_id": conversation_id,
                            "timestamp": datetime.now().isoformat(),
                            "model_used": model_used or "unknown",
                        }
                    ],
                )

                logger.debug(
                    f"✅ Embedding guardado para conversación {conversation_id}"
                )
            except Exception as e:
                logger.warning(f"⚠️ Error guardando embedding: {e}")

        # Actualizar sesión
        cursor.execute(
            """
            UPDATE session_context 
            SET last_activity = CURRENT_TIMESTAMP,
                total_messages = total_messages + 1
            WHERE session_id = ?
        """,
            (session_id,),
        )

        conn.commit()
        conn.close()

        logger.info(f"💾 Conversación guardada: {conversation_id}")
        return conversation_id

    def get_conversation_history(
        self, session_id: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Recuperar historial de conversaciones

        Args:
            session_id: ID de la sesión
            limit: Número máximo de mensajes
            offset: Offset para paginación

        Returns:
            Lista de conversaciones
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """,
            (session_id, limit, offset),
        )

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def search_similar_conversations(
        self, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Buscar conversaciones similares por contenido
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Búsqueda simple por palabras clave
        cursor.execute(
            """
            SELECT * FROM conversations
            WHERE user_message LIKE ? OR assistant_response LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (f"%{query}%", f"%{query}%", limit),
        )

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def save_project(
        self,
        project_name: str,
        project_path: Optional[str] = None,
        description: Optional[str] = None,
        technologies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Guardar información de un proyecto"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO projects 
            (project_name, project_path, description, technologies, metadata)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                project_name,
                project_path,
                description,
                json.dumps(technologies) if technologies else None,
                json.dumps(metadata) if metadata else None,
            ),
        )

        project_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"📁 Proyecto guardado: {project_name} (ID: {project_id})")
        return project_id

    def get_project_by_name(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Recuperar proyecto por nombre"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM projects
            WHERE project_name = ?
            ORDER BY created_at DESC
            LIMIT 1
        """,
            (project_name,),
        )

        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def save_learning(
        self,
        error_type: str,
        error_message: str,
        solution: Optional[str] = None,
        context: Optional[str] = None,
        project_id: Optional[int] = None,
        learned_from: Optional[str] = None,
    ) -> int:
        """
        Guardar un aprendizaje de un error

        Returns:
            ID del aprendizaje guardado
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Verificar si ya existe un error similar
        cursor.execute(
            """
            SELECT id, times_encountered, confidence_score
            FROM learnings
            WHERE error_type = ? AND error_message = ?
        """,
            (error_type, error_message),
        )

        existing = cursor.fetchone()

        if existing:
            # Incrementar contador y mejorar confianza
            learning_id, times, confidence = existing
            new_confidence = min(1.0, confidence + 0.1)

            cursor.execute(
                """
                UPDATE learnings
                SET times_encountered = ?,
                    confidence_score = ?,
                    solution = COALESCE(?, solution),
                    context = COALESCE(?, context)
                WHERE id = ?
            """,
                (times + 1, new_confidence, solution, context, learning_id),
            )

            logger.info(
                f"🔄 Aprendizaje actualizado: {error_type} (visto {times + 1} veces)"
            )
        else:
            # Nuevo aprendizaje
            cursor.execute(
                """
                INSERT INTO learnings
                (error_type, error_message, solution, context, project_id, learned_from)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    error_type,
                    error_message,
                    solution,
                    context,
                    project_id,
                    learned_from,
                ),
            )

            learning_id = cursor.lastrowid
            logger.info(f"📚 Nuevo aprendizaje guardado: {error_type}")

        # 🆕 v2.0: Generar y guardar embedding en ChromaDB
        if self.enable_semantic_search and self.embedding_system and self.chroma_learnings:
            try:
                # Combinar toda la información del aprendizaje
                full_text = f"Error: {error_type}\nMensaje: {error_message}\nSolución: {solution or 'N/A'}\nContexto: {context or 'N/A'}"

                # Generar embedding
                embedding = self.embedding_system.encode(full_text)

                # Guardar en ChromaDB
                learning_id_to_use = learning_id if not existing else existing[0]
                self.chroma_learnings.add(
                    documents=[full_text],
                    embeddings=[
                        embedding.tolist()
                        if hasattr(embedding, "tolist")
                        else embedding
                    ],
                    ids=[f"learning_{learning_id_to_use}"],
                    metadatas=[
                        {
                            "error_type": error_type,
                            "learning_id": learning_id_to_use,
                            "timestamp": datetime.now().isoformat(),
                            "project_id": project_id or 0,
                        }
                    ],
                )

                logger.debug(
                    f"✅ Embedding guardado para aprendizaje {learning_id_to_use}"
                )
            except Exception as e:
                logger.warning(f"⚠️ Error guardando embedding de aprendizaje: {e}")

        conn.commit()
        conn.close()

        return learning_id if not existing else existing[0]

    def get_solution_for_error(
        self, error_type: Optional[str] = None, error_message: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Buscar solución para un error basándose en aprendizajes previos
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if error_message:
            # Búsqueda exacta por mensaje
            cursor.execute(
                """
                SELECT * FROM learnings
                WHERE error_message LIKE ?
                ORDER BY confidence_score DESC, times_encountered DESC
                LIMIT 1
            """,
                (f"%{error_message}%",),
            )
        elif error_type:
            # Búsqueda por tipo
            cursor.execute(
                """
                SELECT * FROM learnings
                WHERE error_type = ?
                ORDER BY confidence_score DESC, times_encountered DESC
                LIMIT 1
            """,
                (error_type,),
            )
        else:
            return None

        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def save_knowledge(
        self,
        topic: str,
        content: str,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        relevance_score: float = 0.5,
    ) -> int:
        """Guardar conocimiento en la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO knowledge_base
            (topic, content, source, relevance_score, tags)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                topic,
                content,
                source,
                relevance_score,
                json.dumps(tags) if tags else None,
            ),
        )

        knowledge_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"💡 Conocimiento guardado: {topic}")
        return knowledge_id

    def search_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Buscar conocimiento relevante"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM knowledge_base
            WHERE topic LIKE ? OR content LIKE ?
            ORDER BY relevance_score DESC, access_count DESC
            LIMIT ?
        """,
            (f"%{query}%", f"%{query}%", limit),
        )

        rows = cursor.fetchall()

        # Actualizar contador de acceso
        if rows:
            ids = [row["id"] for row in rows]
            cursor.execute(
                f"""
                UPDATE knowledge_base
                SET access_count = access_count + 1
                WHERE id IN ({",".join("?" * len(ids))})
            """,
                ids,
            )
            conn.commit()

        conn.close()

        return [dict(row) for row in rows]

    def create_session(self, session_id: str) -> int:
        """Crear nueva sesión"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR IGNORE INTO session_context (session_id)
            VALUES (?)
        """,
            (session_id,),
        )

        session_id_db = cursor.lastrowid
        conn.commit()
        conn.close()

        return session_id_db

    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Obtener contexto de sesión actual"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM session_context
            WHERE session_id = ?
        """,
            (session_id,),
        )

        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def get_context_for_llm(self, session_id: str, max_messages: int = 10) -> str:
        """
        Construir contexto resumido para el LLM basado en la memoria

        Returns:
            String con contexto relevante
        """
        # Obtener conversaciones recientes
        recent_conversations = self.get_conversation_history(
            session_id, limit=max_messages
        )

        # Obtener sesión actual
        session = self.get_session_context(session_id)

        # Construir contexto
        context_parts: List[str] = []

        if session and session.get("current_project_id"):
            context_parts.append("📁 Proyecto actual activo")

        if recent_conversations:
            context_parts.append(
                f"💬 {len(recent_conversations)} mensajes en esta conversación"
            )

        # Buscar aprendizajes relevantes recientes
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT error_type, solution, times_encountered
            FROM learnings
            ORDER BY timestamp DESC
            LIMIT 5
        """)
        learnings = cursor.fetchall()
        conn.close()

        if learnings:
            context_parts.append(
                f"📚 {len(learnings)} aprendizajes recientes disponibles"
            )

        return " | ".join(context_parts) if context_parts else "Nueva conversación"

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la memoria"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Total conversaciones
        cursor.execute("SELECT COUNT(*) FROM conversations")
        stats["total_conversations"] = cursor.fetchone()[0]

        # Total proyectos
        cursor.execute("SELECT COUNT(*) FROM projects")
        stats["total_projects"] = cursor.fetchone()[0]

        # Total aprendizajes
        cursor.execute("SELECT COUNT(*) FROM learnings")
        stats["total_learnings"] = cursor.fetchone()[0]

        # Total conocimiento
        cursor.execute("SELECT COUNT(*) FROM knowledge_base")
        stats["total_knowledge"] = cursor.fetchone()[0]

        # Sesiones activas
        cursor.execute("SELECT COUNT(*) FROM session_context WHERE active = 1")
        stats["active_sessions"] = cursor.fetchone()[0]

        conn.close()

        return stats

    # ============================================================================
    # 🆕 MÉTODOS v2.0: BÚSQUEDA SEMÁNTICA Y KNOWLEDGE GRAPH
    # ============================================================================

    def semantic_search(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        top_k: int = 5,
        min_similarity: float = 0.7,
    ) -> List[SemanticSearchResult]:
        """
        🆕 v2.0: Búsqueda semántica en la memoria usando embeddings

        Args:
            query: Query de búsqueda
            memory_types: Tipos de memoria a buscar (None = todos)
            top_k: Número máximo de resultados
            min_similarity: Similaridad mínima requerida (0-1)

        Returns:
            Lista de resultados ordenados por relevancia
        """
        if not self.enable_semantic_search or not self.embedding_system:
            logger.warning("⚠️ Búsqueda semántica no disponible, usando búsqueda básica")
            return []

        try:
            # Generar embedding del query usando el método correcto
            query_embedding = self.embedding_system.encode(query)

            # Convertir a lista si es necesario
            if hasattr(query_embedding, "tolist"):
                query_embedding = query_embedding.tolist()

            results: List[SemanticSearchResult] = []

            # Buscar en conversaciones
            if (not memory_types or MemoryType.CONVERSATION in memory_types) and self.chroma_conversations:
                conv_results = self.chroma_conversations.query(
                    query_embeddings=[query_embedding], n_results=top_k
                )

                for i, (content, distance) in enumerate(
                    zip(conv_results["documents"][0], conv_results["distances"][0])
                ):
                    similarity = 1.0 - distance  # Convertir distancia a similaridad
                    if similarity >= min_similarity:
                        results.append(
                            SemanticSearchResult(
                                content=content,
                                similarity=similarity,
                                timestamp=datetime.now(),
                                memory_type=MemoryType.CONVERSATION,
                                metadata=conv_results["metadatas"][0][i]
                                if conv_results["metadatas"]
                                else {},
                            )
                        )

            # Buscar en learnings
            if (not memory_types or MemoryType.LEARNING in memory_types) and self.chroma_learnings:
                learning_results = self.chroma_learnings.query(
                    query_embeddings=[query_embedding], n_results=top_k
                )

                for i, (content, distance) in enumerate(
                    zip(
                        learning_results["documents"][0],
                        learning_results["distances"][0],
                    )
                ):
                    similarity = 1.0 - distance
                    if similarity >= min_similarity:
                        results.append(
                            SemanticSearchResult(
                                content=content,
                                similarity=similarity,
                                timestamp=datetime.now(),
                                memory_type=MemoryType.LEARNING,
                                metadata=learning_results["metadatas"][0][i]
                                if learning_results["metadatas"]
                                else {},
                            )
                        )

            # Buscar en knowledge base
            if (not memory_types or MemoryType.KNOWLEDGE in memory_types) and self.chroma_knowledge:
                knowledge_results = self.chroma_knowledge.query(
                    query_embeddings=[query_embedding], n_results=top_k
                )

                for i, (content, distance) in enumerate(
                    zip(
                        knowledge_results["documents"][0],
                        knowledge_results["distances"][0],
                    )
                ):
                    similarity = 1.0 - distance
                    if similarity >= min_similarity:
                        results.append(
                            SemanticSearchResult(
                                content=content,
                                similarity=similarity,
                                timestamp=datetime.now(),
                                memory_type=MemoryType.KNOWLEDGE,
                                metadata=knowledge_results["metadatas"][0][i]
                                if knowledge_results["metadatas"]
                                else {},
                            )
                        )

            # Ordenar por similaridad
            results.sort(key=lambda x: x.similarity, reverse=True)

            return results[:top_k]

        except Exception as e:
            logger.error(f"❌ Error en búsqueda semántica: {e}")
            return []

    def cluster_conversations(
        self, session_id: Optional[str] = None, n_clusters: int = 5, method: str = "kmeans"
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        🆕 v2.0: Agrupar conversaciones similares usando clustering

        Args:
            session_id: ID de sesión (None = todas las conversaciones)
            n_clusters: Número de clusters a crear
            method: Método de clustering ('kmeans' o 'dbscan')

        Returns:
            Diccionario {cluster_id: [conversaciones]}
        """
        if not self.enable_semantic_search or not self.embedding_system:
            logger.warning("⚠️ Clustering no disponible")
            return {}

        
        try:
            # Obtener conversaciones
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if session_id:
                cursor.execute(
                    """
                SELECT id, user_message, assistant_response, timestamp
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp
                """,
                    (session_id,),
                )
            else:
                cursor.execute("""
                SELECT id, user_message, assistant_response, timestamp
                FROM conversations
                ORDER BY timestamp
                """)

            conversations: List[Dict[str, Any]] = [dict(row) for row in cursor.fetchall()]
            conn.close()

            if len(conversations) < n_clusters:
                logger.warning(
                    f"⚠️ Muy pocas conversaciones ({len(conversaciones)}) para {n_clusters} clusters"
                )
                return {0: conversations}

            # Generar embeddings usando el método correcto
            texts = [
                f"{c['user_message']} {c['assistant_response']}" for c in conversations
            ]
            embeddings = [self.embedding_system.encode(text) for text in texts]

            # Convertir embeddings a formato correcto para clustering

            embeddings_array = np.array(
                [emb.tolist() if hasattr(emb, "tolist") else emb for emb in embeddings]
            )

            # Clustering con scikit-learn

            if method == "kmeans":
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            else:  # dbscan
                clusterer = DBSCAN(eps=0.5, min_samples=2)

            clusters = clusterer.fit_predict(embeddings_array)

            # Agrupar por cluster
            result: Dict[int, List[Dict[str, Any]]] = {}
            for conv, cluster_id in zip(conversations, clusters):
                # Asegurarse que cluster_id sea un int estándar de Python
                py_cluster_id = int(cluster_id)
                if py_cluster_id not in result:
                    result[py_cluster_id] = []
                result[py_cluster_id].append(conv)

            # Guardar en base de datos
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for cluster_id, convs in result.items():
                for conv in convs:
                    cursor.execute(
                        """
                    INSERT INTO conversation_clusters
                    (conversation_id, cluster_id, cluster_topic)
                    VALUES (?, ?, ?)
                    """,
                        (conv["id"], cluster_id, f"Cluster {cluster_id}"),
                    )

            conn.commit()
            conn.close()

            logger.info(
                f"✅ Creados {len(result)} clusters con {len(conversations)} conversaciones"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Error en clustering: {e}")
            return {}

    def add_relation(
        self,
        source_entity: str,
        relation_type: str,
        target_entity: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        🆕 v2.0: Agregar relación al knowledge graph

        Args:
            source_entity: Entidad origen
            relation_type: Tipo de relación (ej: "usa", "implementa", "depende_de")
            target_entity: Entidad destino
            weight: Peso de la relación (0-1)
            metadata: Metadatos adicionales

        Returns:
            True si se agregó exitosamente
        """
        if not self.enable_knowledge_graph or self.knowledge_graph is None:
            logger.warning("⚠️ Knowledge graph no disponible")
            return False

        try:
            # Agregar a base de datos
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
            INSERT INTO entity_relations
            (source_entity, relation_type, target_entity, weight, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
                (
                    source_entity,
                    relation_type,
                    target_entity,
                    weight,
                    json.dumps(metadata) if metadata else None,
                ),
            )

            conn.commit()
            conn.close()

            # Agregar al grafo en memoria
            self.knowledge_graph.add_edge(
                source_entity,
                target_entity,
                relation=relation_type,
                weight=weight,
                metadata=metadata or {},
            )

            logger.info(
                f"✅ Relación agregada: {source_entity} --[{relation_type}]--> {target_entity}"
            )

            return True

        except Exception as e:
            logger.error(f"❌ Error agregando relación: {e}")
            return False

    def get_related_entities(
        self, entity: str, max_depth: int = 2, relation_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        🆕 v2.0: Obtener entidades relacionadas en el knowledge graph

        Args:
            entity: Entidad origen
            max_depth: Profundidad máxima de búsqueda
            relation_types: Tipos de relaciones a considerar (None = todos)

        Returns:
            Diccionario con entidades relacionadas y sus relaciones
        """
        if not self.enable_knowledge_graph or self.knowledge_graph is None:
            logger.warning("⚠️ Knowledge graph no disponible")
            return {}

        try:
            if entity not in self.knowledge_graph:
                return {
                    "entity": entity,
                    "relations": [],
                    "message": "Entidad no encontrada",
                }

            # BFS para encontrar entidades relacionadas
            related: Dict[str, Any] = {}
            visited = set()
            queue = [(entity, 0)]  # (nodo, profundidad)

            while queue:
                current, depth = queue.pop(0)

                if current in visited or depth > max_depth:
                    continue

                visited.add(current)

                # Obtener vecinos
                for neighbor in self.knowledge_graph.neighbors(current):
                    edge_data = self.knowledge_graph[current][neighbor]
                    relation = edge_data.get("relation", "unknown")

                    # Filtrar por tipo de relación si se especifica
                    if relation_types and relation not in relation_types:
                        continue

                    if neighbor not in related:
                        related[neighbor] = {"relations": [], "distance": depth + 1}

                    related[neighbor]["relations"].append(
                        {
                            "from": current,
                            "type": relation,
                            "weight": edge_data.get("weight", 1.0),
                            "metadata": edge_data.get("metadata", {}),
                        }
                    )

                    if depth + 1 < max_depth:
                        queue.append((neighbor, depth + 1))

            return {
                "entity": entity,
                "related_entities": related,
                "total_found": len(related),
            }

        except Exception as e:
            logger.error(f"❌ Error obteniendo entidades relacionadas: {e}")
            return {}

    def find_similar_errors(
        self, error_message: str, top_k: int = 5, min_similarity: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        🆕 v2.0: Encontrar errores similares en la base de aprendizajes

        Args:
            error_message: Mensaje de error
            top_k: Número máximo de resultados
            min_similarity: Similaridad mínima

        Returns:
            Lista de errores similares con sus soluciones
        """
        if not self.enable_semantic_search:
            # Fallback a búsqueda básica
            solution = self.get_solution_for_error(error_message=error_message)
            return [solution] if solution else []

        try:
            # Buscar semánticamente
            results = self.semantic_search(
                query=error_message,
                memory_types=[MemoryType.LEARNING],
                top_k=top_k,
                min_similarity=min_similarity,
            )

            # Obtener detalles completos de los learnings
            similar_errors = []

            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            for result in results:
                # Buscar en la base de datos por el contenido
                cursor.execute(
                    """
                SELECT *
                FROM learnings
                WHERE error_message LIKE ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                    (f"%{result.content[:50]}%",),
                )

                row = cursor.fetchone()
                if row:
                    learning = dict(row)
                    learning["similarity_score"] = result.similarity
                    similar_errors.append(learning)

            conn.close()

            logger.info(f"✅ Encontrados {len(similar_errors)} errores similares")

            return similar_errors

        except Exception as e:
            logger.error(f"❌ Error buscando errores similares: {e}")
            return []

    def get_v2_stats(self) -> Dict[str, Any]:
        """
        🆕 v2.0: Estadísticas de características v2.0

        Returns:
            Diccionario con estadísticas de búsqueda semántica y knowledge graph
        """
        stats: Dict[str, Any] = {
            "version": "2.0"
            if (self.enable_semantic_search or self.enable_knowledge_graph)
            else "1.0",
            "semantic_search_enabled": self.enable_semantic_search,
            "knowledge_graph_enabled": self.enable_knowledge_graph,
            "features": {
                "semantic_search": self.enable_semantic_search,
                "knowledge_graph": self.enable_knowledge_graph,
                "chromadb": self.enable_semantic_search and CHROMADB_AVAILABLE,
                "embeddings": self.enable_semantic_search and EMBEDDINGS_V2_AVAILABLE,
                "networkx": self.enable_knowledge_graph and NETWORKX_AVAILABLE,
            },
        }

        if self.enable_semantic_search and self.chroma_conversations and self.chroma_learnings and self.chroma_knowledge:
            try:
                stats["chromadb"] = {
                    "conversations": self.chroma_conversations.count(),
                    "learnings": self.chroma_learnings.count(),
                    "knowledge": self.chroma_knowledge.count(),
                }
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                stats["chromadb"] = {"error": "No se pudo obtener estadísticas"}

        if self.enable_knowledge_graph and self.knowledge_graph is not None:
            stats["knowledge_graph"] = {
                "nodes": len(self.knowledge_graph.nodes),
                "edges": len(self.knowledge_graph.edges),
            }

        return stats

    def store_episode(self, episode_data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Almacenar un episodio completo en memoria

        Args:
            episode_data: Dict con datos del episodio (observations, actions, rewards, etc)
            **kwargs: Cualquier argumento adicional se fusiona con episode_data
        """
        try:
            # Si episode_data es None, usar kwargs como datos
            if episode_data is None:
                episode_data = kwargs
            else:
                # Fusionar kwargs con episode_data
                episode_data = {**episode_data, **kwargs}

            episode_id = episode_data.get(
                "episode_id",
                f"episode_{len(self.episodes) if hasattr(self, 'episodes') else 0}",
            )

            # Crear estructura del episodio
            episode = {
                "id": episode_id,
                "timestamp": episode_data.get("timestamp", time.time()),
                "observations": episode_data.get("observations", []),
                "actions": episode_data.get("actions", []),
                "rewards": episode_data.get("rewards", []),
                "context": episode_data.get("context", {}),
                "metadata": episode_data.get("metadata", {}),
                "content": episode_data.get(
                    "content", ""
                ),  # Soportar 'content' también
                "raw_data": episode_data,  # Guardar datos originales
            }

            # Guardar en memoria episódica
            if not hasattr(self, "episodes"):
                self.episodes: List[Dict[str, Any]] = []

            self.episodes.append(episode)

            # Mantener solo últimos N episodios
            max_episodes = 1000
            if len(self.episodes) > max_episodes:
                self.episodes = self.episodes[-max_episodes:]

            logger.debug(f"📝 Episodio {episode_id} almacenado en memoria")

        except Exception as e:
            logger.error(f"❌ Error almacenando episodio: {e}")


# Singleton global
_global_memory: Optional[MetacortexMemory] = None


def get_memory(db_path: Optional[str] = None) -> MetacortexMemory:
    """Obtener instancia global de memoria"""
    global _global_memory
    if _global_memory is None:
        _global_memory = MetacortexMemory(db_path)
    return _global_memory


def get_memory_system(db_path: Optional[str] = None) -> MetacortexMemory:
    """Alias para get_memory() - compatibilidad con imports"""
    return get_memory(db_path)