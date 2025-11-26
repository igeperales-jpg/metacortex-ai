#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
METACORTEX ENTERPRISE DASHBOARD - FastAPI
═══════════════════════════════════════════════════════════════════════════
Dashboard web enterprise con FastAPI para monitoreo en tiempo real de:
- 956+ Modelos ML activos
- Sistema Autónomo de Orquestación
- Métricas completas
- WebSocket para actualizaciones en tiempo real
- Telegram Bot integration
- API RESTful completa

Autor: METACORTEX System
Versión: 1.0.0 - Enterprise Grade
═══════════════════════════════════════════════════════════════════════════
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# FastAPI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Singleton Registry para evitar circular imports
try:
    from singleton_registry import get_autonomous_orchestrator
    SINGLETON_AVAILABLE = True
except ImportError:
    SINGLETON_AVAILABLE = False
    print("⚠️  Singleton registry not available - limited functionality")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# FASTAPI APP CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="METACORTEX Enterprise Dashboard",
    description="Dashboard en tiempo real para 956+ modelos ML autónomos",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════
# WEBSOCKET CONNECTION MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    """Gestiona conexiones WebSocket para actualizaciones en tiempo real."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.update_interval = 3  # segundos
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"✅ WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"❌ WebSocket disconnected. Remaining: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Envía mensaje a todos los clientes conectados."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")

manager = ConnectionManager()

# ═══════════════════════════════════════════════════════════════════════════
# AUTONOMOUS ORCHESTRATOR HELPER
# ═══════════════════════════════════════════════════════════════════════════

def get_orchestrator():
    """
    Obtiene instancia del orchestrator via singleton registry.
    Asegura que el orchestrator esté completamente iniciado en modo autónomo.
    """
    if not SINGLETON_AVAILABLE:
        return None
    
    try:
        # Obtener orchestrator con auto_start=True para activar modo autónomo
        orchestrator = get_autonomous_orchestrator(auto_start=True)
        
        # Verificar que esté inicializado
        if orchestrator and not orchestrator.is_running:
            logger.info("🚀 Iniciando Autonomous Orchestrator en modo autónomo...")
            orchestrator._discover_models()
            orchestrator._start_execution_threads()
            logger.info("✅ Orchestrator iniciado completamente")
        
        return orchestrator
    except Exception as e:
        logger.error(f"Error getting orchestrator: {e}")
        return None

def get_system_status() -> Dict[str, Any]:
    """Obtiene status completo del sistema."""
    orchestrator = get_orchestrator()
    
    if orchestrator is None:
        return {
            "status": "unavailable",
            "message": "Autonomous Orchestrator not available",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        status = orchestrator.get_status()
        status["timestamp"] = datetime.now().isoformat()
        status["status"] = "operational"
        return status
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ═══════════════════════════════════════════════════════════════════════════
# HTML DASHBOARD (EMBEDDED)
# ═══════════════════════════════════════════════════════════════════════════

HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>METACORTEX Enterprise Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        
        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 10px;
        }
        
        .status-operational {
            background: #10b981;
        }
        
        .status-error {
            background: #ef4444;
        }
        
        .status-unavailable {
            background: #f59e0b;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card h2 {
            font-size: 1.5em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .card .value {
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .card .label {
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .specializations {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        
        .spec-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .spec-name {
            font-weight: 500;
        }
        
        .spec-count {
            background: rgba(255, 255, 255, 0.2);
            padding: 2px 10px;
            border-radius: 12px;
            font-weight: bold;
        }
        
        .tasks-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
        }
        
        .task-item {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #10b981;
        }
        
        .task-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .task-id {
            font-weight: bold;
            font-family: monospace;
        }
        
        .task-status {
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
        }
        
        .status-active {
            background: #3b82f6;
        }
        
        .status-completed {
            background: #10b981;
        }
        
        .status-failed {
            background: #ef4444;
        }
        
        .connection-status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.7);
            padding: 10px 20px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        .connected {
            background: #10b981;
        }
        
        .disconnected {
            background: #ef4444;
        }
        
        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.5;
            }
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981, #3b82f6);
            border-radius: 4px;
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 METACORTEX Enterprise Dashboard</h1>
            <div class="subtitle">Sistema Autónomo de Orquestación - 956+ Modelos ML</div>
            <div id="systemStatus" class="status-badge status-unavailable">Conectando...</div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>🧠 Modelos Activos</h2>
                <div class="value" id="totalModels">-</div>
                <div class="label">Modelos ML trabajando</div>
            </div>
            
            <div class="card">
                <h2>📝 Cola de Tareas</h2>
                <div class="value" id="queueSize">-</div>
                <div class="label">Tareas pendientes</div>
            </div>
            
            <div class="card">
                <h2>⚡ Tareas Activas</h2>
                <div class="value" id="activeTasks">-</div>
                <div class="label">En ejecución ahora</div>
            </div>
            
            <div class="card">
                <h2>✅ Completadas</h2>
                <div class="value" id="completedTasks">-</div>
                <div class="label">Total ejecutadas</div>
            </div>
            
            <div class="card">
                <h2>❌ Fallidas</h2>
                <div class="value" id="failedTasks">-</div>
                <div class="label">Errores detectados</div>
            </div>
            
            <div class="card">
                <h2>📈 Success Rate</h2>
                <div class="value" id="successRate">-</div>
                <div class="label">Tasa de éxito</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="successProgress" style="width: 0%"></div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>🎯 Modelos por Especialización</h2>
            <div class="specializations" id="specializationsList">
                <div class="spec-item">
                    <span class="spec-name">Cargando...</span>
                    <span class="spec-count">-</span>
                </div>
            </div>
        </div>
        
        <div class="tasks-container">
            <h2 style="margin-bottom: 20px;">⚡ Tareas Activas en Tiempo Real</h2>
            <div id="activeTasks List">
                <div style="text-align: center; opacity: 0.7;">
                    Esperando tareas activas...
                </div>
            </div>
        </div>
    </div>
    
    <div class="connection-status">
        <div class="status-indicator disconnected" id="wsIndicator"></div>
        <span id="wsStatus">Desconectado</span>
    </div>
    
    <script>
        let ws = null;
        let reconnectInterval = null;
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('✅ WebSocket connected');
                document.getElementById('wsIndicator').className = 'status-indicator connected';
                document.getElementById('wsStatus').textContent = 'Conectado';
                
                if (reconnectInterval) {
                    clearInterval(reconnectInterval);
                    reconnectInterval = null;
                }
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            ws.onclose = () => {
                console.log('❌ WebSocket disconnected');
                document.getElementById('wsIndicator').className = 'status-indicator disconnected';
                document.getElementById('wsStatus').textContent = 'Desconectado';
                
                // Reconnect after 5 seconds
                if (!reconnectInterval) {
                    reconnectInterval = setInterval(() => {
                        console.log('🔄 Attempting to reconnect...');
                        connectWebSocket();
                    }, 5000);
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        function updateDashboard(data) {
            // System status
            const statusBadge = document.getElementById('systemStatus');
            statusBadge.className = `status-badge status-${data.status}`;
            statusBadge.textContent = data.status === 'operational' ? '✅ Operacional' : 
                                     data.status === 'error' ? '❌ Error' : '⚠️ No disponible';
            
            // Main metrics
            document.getElementById('totalModels').textContent = data.total_models || '-';
            document.getElementById('queueSize').textContent = data.queue_size || '-';
            document.getElementById('activeTasks').textContent = data.active_tasks || '-';
            document.getElementById('completedTasks').textContent = data.completed_tasks || '-';
            document.getElementById('failedTasks').textContent = data.failed_tasks || '-';
            
            // Success rate
            const successRate = (data.success_rate * 100) || 0;
            document.getElementById('successRate').textContent = `${successRate.toFixed(1)}%`;
            document.getElementById('successProgress').style.width = `${successRate}%`;
            
            // Specializations
            if (data.models_by_specialization) {
                const specList = document.getElementById('specializationsList');
                specList.innerHTML = '';
                
                for (const [spec, count] of Object.entries(data.models_by_specialization)) {
                    const item = document.createElement('div');
                    item.className = 'spec-item';
                    item.innerHTML = `
                        <span class="spec-name">${spec}</span>
                        <span class="spec-count">${count}</span>
                    `;
                    specList.appendChild(item);
                }
            }
            
            // Active tasks
            if (data.active_tasks_details && data.active_tasks_details.length > 0) {
                const tasksList = document.getElementById('activeTasksList');
                tasksList.innerHTML = '';
                
                data.active_tasks_details.forEach(task => {
                    const item = document.createElement('div');
                    item.className = 'task-item';
                    item.innerHTML = `
                        <div class="task-header">
                            <span class="task-id">${task.task_id}</span>
                            <span class="task-status status-${task.status}">${task.status}</span>
                        </div>
                        <div>${task.description}</div>
                        <div style="margin-top: 8px; opacity: 0.8; font-size: 0.9em;">
                            Especialización: ${task.specialization} | Prioridad: ${task.priority}
                        </div>
                    `;
                    tasksList.appendChild(item);
                });
            }
        }
        
        // Initialize WebSocket connection
        connectWebSocket();
        
        // Fetch initial data via REST API
        fetch('/api/status')
            .then(response => response.json())
            .then(data => updateDashboard(data))
            .catch(error => console.error('Error fetching initial data:', error));
    </script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════════════════
# REST API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Dashboard principal HTML."""
    return HTML_DASHBOARD

@app.get("/api/status")
async def get_status() -> JSONResponse:
    """Obtiene status completo del sistema."""
    status = get_system_status()
    return JSONResponse(content=status)

@app.get("/api/models")
async def get_models() -> JSONResponse:
    """Lista todos los modelos disponibles."""
    orchestrator = get_orchestrator()
    
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    
    try:
        models = []
        for model_id, profile in orchestrator.model_profiles.items():
            models.append({
                "model_id": model_id,
                "specializations": [s.value for s in profile.specializations],
                "accuracy": profile.accuracy,
                "success_count": profile.success_count,
                "failure_count": profile.failure_count,
                "success_rate": profile.success_rate
            })
        
        return JSONResponse(content={"models": models, "total": len(models)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks")
async def get_tasks() -> JSONResponse:
    """Lista todas las tareas (activas, pendientes, completadas)."""
    orchestrator = get_orchestrator()
    
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    
    try:
        tasks = {
            "active": [
                {
                    "task_id": task.task_id,
                    "description": task.description,
                    "specialization": task.specialization.value,
                    "priority": task.priority.value,
                    "status": task.status.value,
                    "started_at": task.started_at.isoformat() if task.started_at else None
                }
                for task in orchestrator.active_tasks.values()
            ],
            "queue": [
                {
                    "task_id": task.task_id,
                    "description": task.description,
                    "specialization": task.specialization.value,
                    "priority": task.priority.value
                }
                for task in list(orchestrator.task_queue)[:10]  # Solo primeras 10
            ],
            "completed": [
                {
                    "task_id": task.task_id,
                    "description": task.description,
                    "status": task.status.value,
                    "execution_time": task.execution_time
                }
                for task in list(orchestrator.completed_tasks)[:20]  # Últimas 20
            ]
        }
        
        return JSONResponse(content=tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/task")
async def create_task(task_data: Dict[str, Any]) -> JSONResponse:
    """Crea una nueva tarea."""
    orchestrator = get_orchestrator()
    
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    
    try:
        from autonomous_model_orchestrator import Task, TaskPriority, ModelSpecialization
        
        task = Task(
            description=task_data.get("description", "Manual task"),
            specialization=ModelSpecialization(task_data.get("specialization", "ANALYSIS")),
            priority=TaskPriority(task_data.get("priority", "MEDIUM")),
            data=task_data.get("data", {})
        )
        
        orchestrator.add_task(task)
        
        return JSONResponse(content={
            "success": True,
            "task_id": task.task_id,
            "message": "Task created successfully"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(content={
        "status": "healthy",
        "service": "METACORTEX Enterprise Dashboard",
        "timestamp": datetime.now().isoformat()
    })

# ═══════════════════════════════════════════════════════════════════════════
# WEBSOCKET ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket para actualizaciones en tiempo real."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Enviar status cada 3 segundos
            status = get_system_status()
            await websocket.send_json(status)
            await asyncio.sleep(manager.update_interval)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ═══════════════════════════════════════════════════════════════════════════
# BACKGROUND TASK - BROADCAST STATUS
# ═══════════════════════════════════════════════════════════════════════════

async def broadcast_status_loop():
    """Broadcast system status to all connected clients."""
    while True:
        try:
            if len(manager.active_connections) > 0:
                status = get_system_status()
                await manager.broadcast(status)
        except Exception as e:
            logger.error(f"Error in broadcast loop: {e}")
        
        await asyncio.sleep(manager.update_interval)

@app.on_event("startup")
async def startup_event():
    """Ejecuta al iniciar la app."""
    logger.info("🚀 METACORTEX Enterprise Dashboard starting...")
    logger.info("📊 Dashboard available at: http://localhost:8300")
    logger.info("📡 WebSocket endpoint: ws://localhost:8300/ws")
    logger.info("📚 API docs: http://localhost:8300/api/docs")
    
    # Start broadcast loop
    asyncio.create_task(broadcast_status_loop())

@app.on_event("shutdown")
async def shutdown_event():
    """Ejecuta al cerrar la app."""
    logger.info("🛑 METACORTEX Enterprise Dashboard shutting down...")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8300,
        log_level="info",
        access_log=True
    )
