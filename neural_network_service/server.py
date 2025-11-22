#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 METACORTEX - Neural Symbiotic Network Service
================================================

Servicio standalone para el Neural Symbiotic Network que:
- Expone capacidades del neural network vía API REST
- Permite monitoreo en tiempo real de módulos y conexiones
- Gestiona comunicación inter-módulos
- Almacena y recupera conocimiento compartido
- Integra con MPS para aceleración GPU

Puerto: 8001 (service standalone)
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Agregar el root al path
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

# Imports de METACORTEX
try:
    from neural_symbiotic_network import get_neural_network, MetacortexNeuralSymbioticNetworkV2
    from unified_logging import get_logger
    from mps_config import get_device, get_system_info
except ImportError as e:
    print(f"⚠️ Error importando módulos: {e}")
    logging.basicConfig(level=logging.INFO)
    
    def get_logger(name: str = "NeuralNetworkService") -> logging.Logger:
        return logging.getLogger(name)
    
    sys.exit(1)

logger = get_logger("NeuralNetworkService")

# ==============================================================================
# MODELOS PYDANTIC
# ==============================================================================

class RegisterModuleRequest(BaseModel):
    """Solicitud para registrar módulo"""
    module_name: str
    capabilities: Optional[List[str]] = []
    metadata: Optional[Dict[str, Any]] = {}


class ShareKnowledgeRequest(BaseModel):
    """Solicitud para compartir conocimiento"""
    source_module: str
    knowledge_type: str
    data: Dict[str, Any]
    target_modules: Optional[List[str]] = None


class QueryKnowledgeRequest(BaseModel):
    """Consulta de conocimiento"""
    query: str
    filters: Optional[Dict[str, Any]] = {}
    limit: int = 10


# ==============================================================================
# APLICACIÓN FASTAPI
# ==============================================================================

app = FastAPI(
    title="METACORTEX Neural Network Service",
    description="API REST para Neural Symbiotic Network",
    version="5.0.0-M4",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# ESTADO GLOBAL
# ==============================================================================

class NeuralServiceState:
    """Estado del servicio neural"""
    
    def __init__(self):
        self.neural_network: Optional[MetacortexNeuralSymbioticNetworkV2] = None
        self.start_time = datetime.now()
        self.request_count = 0
        
        logger.info("🧠 Inicializando Neural Network Service...")
        self._initialize_neural_network()
    
    def _initialize_neural_network(self):
        """Inicializa neural network"""
        try:
            self.neural_network = get_neural_network()
            if self.neural_network:
                self.neural_network.register_module("neural_service", self)
                logger.info(f"✅ Neural Network inicializada")
                logger.info(f"   Módulos registrados: {len(self.neural_network.modules)}")
            else:
                logger.error("❌ No se pudo inicializar Neural Network")
                raise RuntimeError("Neural Network no disponible")
        except Exception as e:
            logger.error(f"❌ Error inicializando neural network: {e}")
            raise


# Instancia global
state = NeuralServiceState()

# ==============================================================================
# ENDPOINTS - STATUS & HEALTH
# ==============================================================================

@app.get("/health", tags=["Status"])
async def health_check():
    """🏥 Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "neural_network": state.neural_network is not None,
        "mps_available": torch.backends.mps.is_available(),
    }


@app.get("/status", tags=["Status"])
async def get_status():
    """📊 Estado del servicio"""
    state.request_count += 1
    
    uptime = (datetime.now() - state.start_time).total_seconds()
    
    # MPS info
    mps_info = {}
    try:
        mps_info = {
            "available": torch.backends.mps.is_available(),
            "device": str(get_device()),
            "system_info": get_system_info(),
        }
    except Exception as e:
        logger.warning(f"⚠️ Error obteniendo MPS info: {e}")
        mps_info = {"available": False, "error": str(e)}
    
    # Neural stats
    neural_stats = {}
    if state.neural_network:
        neural_stats = state.neural_network.get_stats()
    
    return {
        "status": "operational",
        "version": "5.0.0-M4",
        "uptime_seconds": uptime,
        "request_count": state.request_count,
        "mps": mps_info,
        "neural_network": neural_stats,
    }


# ==============================================================================
# ENDPOINTS - MODULES
# ==============================================================================

@app.get("/modules", tags=["Modules"])
async def list_modules():
    """📋 Lista todos los módulos registrados"""
    if not state.neural_network:
        raise HTTPException(status_code=503, detail="Neural Network no disponible")
    
    state.request_count += 1
    
    modules = []
    for name, module in state.neural_network.modules.items():
        modules.append({
            "name": name,
            "connections": len(module.connections),
            "metadata": module.metadata,
        })
    
    return {
        "total": len(modules),
        "modules": modules,
    }


@app.get("/modules/{module_name}", tags=["Modules"])
async def get_module_info(module_name: str):
    """🔍 Información detallada de un módulo"""
    if not state.neural_network:
        raise HTTPException(status_code=503, detail="Neural Network no disponible")
    
    state.request_count += 1
    
    if module_name not in state.neural_network.modules:
        raise HTTPException(status_code=404, detail=f"Módulo '{module_name}' no encontrado")
    
    module = state.neural_network.modules[module_name]
    
    return {
        "name": module_name,
        "connections": [
            {
                "target": conn.target_module,
                "weight": conn.weight,
                "bidirectional": conn.bidirectional,
            }
            for conn in module.connections
        ],
        "metadata": module.metadata,
        "stats": {
            "messages_sent": module.messages_sent,
            "messages_received": module.messages_received,
        }
    }


@app.post("/modules/register", tags=["Modules"])
async def register_module(request: RegisterModuleRequest):
    """➕ Registra un nuevo módulo"""
    if not state.neural_network:
        raise HTTPException(status_code=503, detail="Neural Network no disponible")
    
    state.request_count += 1
    
    try:
        # Registrar módulo (pasando objeto dummy)
        state.neural_network.register_module(
            request.module_name,
            None,  # El módulo puede ser None para registro remoto
            metadata={
                "capabilities": request.capabilities,
                "registered_at": datetime.now().isoformat(),
                **request.metadata
            }
        )
        
        logger.info(f"✅ Módulo '{request.module_name}' registrado")
        
        return {
            "status": "registered",
            "module_name": request.module_name,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"❌ Error registrando módulo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# ENDPOINTS - KNOWLEDGE SHARING
# ==============================================================================

@app.post("/knowledge/share", tags=["Knowledge"])
async def share_knowledge(request: ShareKnowledgeRequest):
    """📤 Comparte conocimiento entre módulos"""
    if not state.neural_network:
        raise HTTPException(status_code=503, detail="Neural Network no disponible")
    
    state.request_count += 1
    
    try:
        state.neural_network.share_knowledge(
            source_module=request.source_module,
            knowledge_type=request.knowledge_type,
            data=request.data,
            target_modules=request.target_modules,
        )
        
        logger.info(f"✅ Conocimiento compartido: {request.source_module} -> {request.knowledge_type}")
        
        return {
            "status": "shared",
            "source": request.source_module,
            "type": request.knowledge_type,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"❌ Error compartiendo conocimiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge/query", tags=["Knowledge"])
async def query_knowledge(request: QueryKnowledgeRequest):
    """🔍 Consulta conocimiento compartido"""
    if not state.neural_network:
        raise HTTPException(status_code=503, detail="Neural Network no disponible")
    
    state.request_count += 1
    
    try:
        results = state.neural_network.query_shared_knowledge(
            query=request.query,
            filters=request.filters,
            limit=request.limit,
        )
        
        return {
            "query": request.query,
            "total": len(results),
            "results": results,
        }
        
    except Exception as e:
        logger.error(f"❌ Error consultando conocimiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge/stats", tags=["Knowledge"])
async def knowledge_stats():
    """📊 Estadísticas de conocimiento compartido"""
    if not state.neural_network:
        raise HTTPException(status_code=503, detail="Neural Network no disponible")
    
    state.request_count += 1
    
    total_entries = len(state.neural_network.shared_knowledge)
    
    # Agrupar por tipo
    by_type: Dict[str, int] = {}
    for entry in state.neural_network.shared_knowledge.values():
        k_type = entry.get("knowledge_type", "unknown")
        by_type[k_type] = by_type.get(k_type, 0) + 1
    
    # Agrupar por módulo fuente
    by_module: Dict[str, int] = {}
    for entry in state.neural_network.shared_knowledge.values():
        module = entry.get("source_module", "unknown")
        by_module[module] = by_module.get(module, 0) + 1
    
    return {
        "total_entries": total_entries,
        "by_type": by_type,
        "by_module": by_module,
    }


# ==============================================================================
# ENDPOINTS - NETWORK GRAPH
# ==============================================================================

@app.get("/graph", tags=["Graph"])
async def get_network_graph():
    """🕸️ Obtiene grafo completo del network"""
    if not state.neural_network:
        raise HTTPException(status_code=503, detail="Neural Network no disponible")
    
    state.request_count += 1
    
    # Construir grafo
    nodes = []
    edges = []
    
    for name, module in state.neural_network.modules.items():
        nodes.append({
            "id": name,
            "metadata": module.metadata,
            "stats": {
                "sent": module.messages_sent,
                "received": module.messages_received,
            }
        })
        
        for conn in module.connections:
            edges.append({
                "source": name,
                "target": conn.target_module,
                "weight": conn.weight,
                "bidirectional": conn.bidirectional,
            })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }
    }


# ==============================================================================
# ENDPOINTS - MESSAGING
# ==============================================================================

@app.post("/messages/send", tags=["Messaging"])
async def send_message(
    source: str,
    target: str,
    message_type: str,
    data: Dict[str, Any],
):
    """📨 Envía mensaje entre módulos"""
    if not state.neural_network:
        raise HTTPException(status_code=503, detail="Neural Network no disponible")
    
    state.request_count += 1
    
    try:
        state.neural_network.send_message(
            source_module=source,
            target_module=target,
            message_type=message_type,
            data=data,
        )
        
        logger.info(f"✅ Mensaje enviado: {source} -> {target} ({message_type})")
        
        return {
            "status": "sent",
            "source": source,
            "target": target,
            "type": message_type,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"❌ Error enviando mensaje: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# STARTUP & SHUTDOWN
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Evento de inicio"""
    logger.info("=" * 80)
    logger.info("🧠 METACORTEX Neural Network Service - STARTING")
    logger.info("=" * 80)
    logger.info(f"🍎 Platform: Apple Silicon M4")
    logger.info(f"🎮 MPS Available: {torch.backends.mps.is_available()}")
    logger.info(f"🔗 Módulos registrados: {len(state.neural_network.modules)}")
    logger.info("=" * 80)
    logger.info("✅ Neural Network Service READY")
    logger.info(f"📍 API: http://localhost:8001")
    logger.info(f"📍 Docs: http://localhost:8001/docs")
    logger.info(f"📍 Graph: http://localhost:8001/graph")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre"""
    logger.info("🛑 Cerrando Neural Network Service...")
    logger.info("✅ Neural Network Service cerrado correctamente")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Punto de entrada principal"""
    try:
        # Configurar puerto
        port = int(os.environ.get("NEURAL_SERVICE_PORT", "8001"))
        host = os.environ.get("NEURAL_SERVICE_HOST", "0.0.0.0")
        
        logger.info(f"🚀 Iniciando servidor en {host}:{port}")
        
        # Iniciar servidor
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
        )
        
    except KeyboardInterrupt:
        logger.info("⚠️ Interrupción por usuario")
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
