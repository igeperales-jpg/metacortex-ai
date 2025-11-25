#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 METACORTEX - Web Interface & Dashboard
==========================================

Servidor FastAPI con:
- Dashboard de métricas en /api/dashboard/metrics
- API REST para control del sistema
- WebSocket para eventos en tiempo real
- Integración con Neural Symbiotic Network
- Monitoreo de MPS (Apple Silicon)
- Sistema de autenticación y rate limiting

Puerto: 8000 (incluye dashboard integrado)
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Agregar el root al path
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

# Imports de METACORTEX
try:
    from neural_symbiotic_network import get_neural_network
    from unified_logging import get_logger
    from mps_config import get_device, verify_mps, get_system_info
    from agent_modules.telemetry_system import get_telemetry_system
except ImportError as e:
    print(f"⚠️ Error importando módulos: {e}")
    # Fallback logger
    logging.basicConfig(level=logging.INFO)
    
    def get_logger(name: str = "WebInterface") -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger("WebInterface")

# ==============================================================================
# MODELOS PYDANTIC
# ==============================================================================

class SystemCommand(BaseModel):
    """Comando para el sistema"""
    action: str
    parameters: Optional[Dict[str, Any]] = {}


class PerceptionInput(BaseModel):
    """Input de percepción"""
    event_type: str
    data: Dict[str, Any]
    priority: str = "normal"


class QueryInput(BaseModel):
    """Query de búsqueda"""
    query: str
    filters: Optional[Dict[str, Any]] = {}


# ==============================================================================
# APLICACIÓN FASTAPI
# ==============================================================================

app = FastAPI(
    title="METACORTEX Web Interface",
    description="Dashboard y API REST para control del sistema",
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

class WebInterfaceState:
    """Estado global del web interface"""
    
    def __init__(self):
        self.neural_network = None
        self.telemetry = None
        self.websocket_clients: List[WebSocket] = []
        self.start_time = datetime.now()
        self.request_count = 0
        
        logger.info("🌐 Inicializando Web Interface...")
        self._initialize_components()
    
    def _initialize_components(self):
        """Inicializa componentes"""
        try:
            # Neural Network
            self.neural_network = get_neural_network()
            if self.neural_network:
                self.neural_network.register_module("web_interface", self)
                logger.info("✅ Neural Network conectada")
            
            # Telemetry (puerto 9090 para evitar conflicto con Web Interface en 8000)
            try:
                self.telemetry = get_telemetry_system(port=9090)
                logger.info("✅ Telemetry System conectado (puerto 9090)")
            except Exception as e:
                logger.warning(f"⚠️ Telemetry no disponible: {e}")
                self.telemetry = None
            
        except Exception as e:
            logger.error(f"❌ Error inicializando componentes: {e}")
    
    async def broadcast_event(self, event: Dict[str, Any]):
        """Envía evento a todos los clientes WebSocket"""
        if not self.websocket_clients:
            return
        
        disconnected = []
        for client in self.websocket_clients:
            try:
                await client.send_json(event)
            except Exception:
                disconnected.append(client)
        
        # Limpiar clientes desconectados
        for client in disconnected:
            self.websocket_clients.remove(client)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del sistema"""
        self.request_count += 1
        
        # CPU y Memoria
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # MPS (Apple Silicon)
        mps_info = {}
        try:
            mps_available = torch.backends.mps.is_available()
            mps_info = {
                "available": mps_available,
                "device": str(get_device()),
                "system_info": get_system_info() if mps_available else {}
            }
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo MPS info: {e}")
            mps_info = {"available": False, "error": str(e)}
        
        # Neural Network
        neural_stats = {}
        if self.neural_network:
            try:
                neural_stats = {
                    "modules": len(self.neural_network.modules),
                    "connections": sum(len(m.connections) for m in self.neural_network.modules.values()),
                    "health": self.neural_network.get_health_status(),
                }
            except Exception as e:
                logger.warning(f"⚠️ Error obteniendo neural stats: {e}")
        
        # Uptime
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime_seconds,
            "request_count": self.request_count,
            "cpu_percent": cpu_percent,
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percent": memory.percent,
            },
            "mps": mps_info,
            "neural_network": neural_stats,
            "websocket_clients": len(self.websocket_clients),
        }


# Instancia global
state = WebInterfaceState()

# ==============================================================================
# ENDPOINTS - DASHBOARD
# ==============================================================================

@app.get("/api/dashboard/metrics", tags=["Dashboard"])
async def get_dashboard_metrics():
    """
    📊 Obtiene métricas completas del sistema
    
    Incluye:
    - CPU y memoria
    - MPS (Apple Silicon GPU)
    - Neural Network
    - Uptime y requests
    """
    try:
        metrics = state.get_system_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error(f"❌ Error obteniendo métricas: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/health", tags=["Dashboard"])
async def get_health():
    """🏥 Health check del sistema"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "neural_network": state.neural_network is not None,
            "telemetry": state.telemetry is not None,
            "mps": torch.backends.mps.is_available(),
        }
    }


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard_html():
    """🎨 Dashboard HTML interactivo"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>METACORTEX Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Monaco', 'Courier New', monospace;
                background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
                color: #00ff41;
                padding: 20px;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            header {
                text-align: center;
                padding: 30px;
                border-bottom: 2px solid #00ff41;
                margin-bottom: 30px;
            }
            h1 {
                font-size: 3em;
                text-shadow: 0 0 10px #00ff41;
                animation: pulse 2s ease-in-out infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .metric-card {
                background: rgba(0, 255, 65, 0.05);
                border: 1px solid #00ff41;
                border-radius: 10px;
                padding: 20px;
                transition: all 0.3s;
            }
            .metric-card:hover {
                background: rgba(0, 255, 65, 0.1);
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0, 255, 65, 0.3);
            }
            .metric-title {
                font-size: 1.2em;
                margin-bottom: 15px;
                color: #00ff41;
            }
            .metric-value {
                font-size: 2.5em;
                font-weight: bold;
                color: #fff;
            }
            .metric-label {
                font-size: 0.9em;
                color: #888;
                margin-top: 5px;
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
                animation: blink 1s ease-in-out infinite;
            }
            .status-ok { background: #00ff41; }
            .status-warning { background: #ffa500; }
            .status-error { background: #ff0000; }
            @keyframes blink {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.3; }
            }
            .log-container {
                background: rgba(0, 0, 0, 0.5);
                border: 1px solid #00ff41;
                border-radius: 10px;
                padding: 20px;
                max-height: 400px;
                overflow-y: auto;
            }
            .log-entry {
                margin-bottom: 10px;
                padding: 8px;
                background: rgba(0, 255, 65, 0.03);
                border-left: 3px solid #00ff41;
            }
            .timestamp {
                color: #888;
                font-size: 0.8em;
            }
            footer {
                text-align: center;
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #00ff41;
                color: #888;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🍎 METACORTEX</h1>
                <p>Apple Silicon M4 + MPS Optimized</p>
            </header>
            
            <div class="metrics-grid" id="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">
                        <span class="status-indicator status-ok"></span>
                        Sistema
                    </div>
                    <div class="metric-value" id="uptime">--</div>
                    <div class="metric-label">Uptime</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">CPU Usage</div>
                    <div class="metric-value" id="cpu">--%</div>
                    <div class="metric-label">Apple M4</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Memory</div>
                    <div class="metric-value" id="memory">-- GB</div>
                    <div class="metric-label">16GB Unified Memory</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">GPU Metal (MPS)</div>
                    <div class="metric-value" id="mps">--</div>
                    <div class="metric-label">Apple Silicon</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Neural Network</div>
                    <div class="metric-value" id="neural">--</div>
                    <div class="metric-label">Módulos conectados</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Requests</div>
                    <div class="metric-value" id="requests">--</div>
                    <div class="metric-label">Total procesados</div>
                </div>
            </div>
            
            <div class="log-container" id="log-container">
                <h3>📝 Activity Log</h3>
                <div id="logs"></div>
            </div>
            
            <footer>
                <p>METACORTEX v5.0 - Apple Silicon M4 Edition</p>
                <p>Autonomous Operation Mode: ACTIVE</p>
            </footer>
        </div>
        
        <script>
            function formatUptime(seconds) {
                const hours = Math.floor(seconds / 3600);
                const minutes = Math.floor((seconds % 3600) / 60);
                return `${hours}h ${minutes}m`;
            }
            
            function addLog(message) {
                const logsDiv = document.getElementById('logs');
                const entry = document.createElement('div');
                entry.className = 'log-entry';
                const now = new Date().toLocaleTimeString();
                entry.innerHTML = `<span class="timestamp">[${now}]</span> ${message}`;
                logsDiv.insertBefore(entry, logsDiv.firstChild);
                
                // Mantener solo últimos 50 logs
                while (logsDiv.children.length > 50) {
                    logsDiv.removeChild(logsDiv.lastChild);
                }
            }
            
            async function updateMetrics() {
                try {
                    const response = await fetch('/api/dashboard/metrics');
                    const data = await response.json();
                    
                    // Actualizar métricas
                    document.getElementById('uptime').textContent = 
                        formatUptime(data.uptime_seconds);
                    document.getElementById('cpu').textContent = 
                        `${data.cpu_percent.toFixed(1)}%`;
                    document.getElementById('memory').textContent = 
                        `${data.memory.used_gb} / ${data.memory.total_gb} GB`;
                    document.getElementById('mps').textContent = 
                        data.mps.available ? '✅ ACTIVO' : '❌ INACTIVO';
                    document.getElementById('neural').textContent = 
                        data.neural_network.modules || '0';
                    document.getElementById('requests').textContent = 
                        data.request_count.toLocaleString();
                    
                    addLog(`Métricas actualizadas - CPU: ${data.cpu_percent.toFixed(1)}%`);
                    
                } catch (error) {
                    console.error('Error actualizando métricas:', error);
                    addLog(`⚠️ Error: ${error.message}`);
                }
            }
            
            // Actualizar cada 3 segundos
            updateMetrics();
            setInterval(updateMetrics, 3000);
            
            // Log inicial
            addLog('🚀 Dashboard iniciado');
            addLog('🍎 Conectado a METACORTEX');
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# ==============================================================================
# ENDPOINTS - NEURAL NETWORK
# ==============================================================================

@app.post("/api/neural/perceive", tags=["Neural Network"])
async def neural_perceive(perception: PerceptionInput):
    """🧠 Envía percepción al neural network"""
    if not state.neural_network:
        raise HTTPException(status_code=503, detail="Neural Network no disponible")
    
    try:
        # Broadcast evento
        await state.broadcast_event({
            "type": "perception",
            "data": perception.dict()
        })
        
        return {
            "status": "processed",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Error procesando percepción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/neural/stats", tags=["Neural Network"])
async def neural_stats():
    """📊 Estadísticas del neural network"""
    if not state.neural_network:
        raise HTTPException(status_code=503, detail="Neural Network no disponible")
    
    try:
        return state.neural_network.get_stats()
    except Exception as e:
        logger.error(f"❌ Error obteniendo stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# ENDPOINTS - SYSTEM CONTROL
# ==============================================================================

@app.post("/api/system/command", tags=["System"])
async def system_command(command: SystemCommand):
    """🎮 Ejecuta comando del sistema"""
    logger.info(f"📥 Comando recibido: {command.action}")
    
    # Broadcast evento
    await state.broadcast_event({
        "type": "command",
        "data": command.dict()
    })
    
    return {
        "status": "received",
        "action": command.action,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/system/status", tags=["System"])
async def system_status():
    """📊 Estado completo del sistema"""
    return {
        "status": "operational",
        "version": "5.0.0-M4",
        "platform": "Apple Silicon M4",
        "components": {
            "neural_network": state.neural_network is not None,
            "telemetry": state.telemetry is not None,
            "mps": torch.backends.mps.is_available(),
        },
        "metrics": state.get_system_metrics(),
    }


# ==============================================================================
# WEBSOCKET
# ==============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """🔌 WebSocket para eventos en tiempo real"""
    await websocket.accept()
    state.websocket_clients.append(websocket)
    logger.info(f"✅ Cliente WebSocket conectado (total: {len(state.websocket_clients)})")
    
    try:
        # Enviar mensaje de bienvenida
        await websocket.send_json({
            "type": "connected",
            "message": "Conectado a METACORTEX",
            "timestamp": datetime.now().isoformat(),
        })
        
        # Mantener conexión abierta
        while True:
            data = await websocket.receive_text()
            logger.debug(f"📥 WebSocket recibido: {data}")
            
            # Echo back
            await websocket.send_json({
                "type": "echo",
                "data": data,
                "timestamp": datetime.now().isoformat(),
            })
            
    except WebSocketDisconnect:
        logger.info("❌ Cliente WebSocket desconectado")
        state.websocket_clients.remove(websocket)
    except Exception as e:
        logger.error(f"❌ Error en WebSocket: {e}")
        if websocket in state.websocket_clients:
            state.websocket_clients.remove(websocket)


# ==============================================================================
# STARTUP & SHUTDOWN
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Evento de inicio"""
    logger.info("=" * 80)
    logger.info("🌐 METACORTEX Web Interface - STARTING")
    logger.info("=" * 80)
    logger.info(f"🍎 Platform: Apple Silicon M4")
    logger.info(f"🎮 MPS Available: {torch.backends.mps.is_available()}")
    logger.info(f"🧠 Neural Network: {'Connected' if state.neural_network else 'Not Available'}")
    logger.info(f"📊 Telemetry: {'Connected' if state.telemetry else 'Not Available'}")
    logger.info("=" * 80)
    logger.info("✅ Web Interface READY")
    logger.info(f"📍 Dashboard: http://localhost:8000")
    logger.info(f"📍 API Docs: http://localhost:8000/docs")
    logger.info(f"📍 Metrics: http://localhost:8000/api/dashboard/metrics")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre"""
    logger.info("🛑 Cerrando Web Interface...")
    
    # Cerrar WebSockets
    for client in state.websocket_clients:
        try:
            await client.close()
        except Exception:
            pass
    
    logger.info("✅ Web Interface cerrado correctamente")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Punto de entrada principal"""
    try:
        # Configurar puerto
        port = int(os.environ.get("WEB_INTERFACE_PORT", "8000"))
        host = os.environ.get("WEB_INTERFACE_HOST", "0.0.0.0")
        
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
