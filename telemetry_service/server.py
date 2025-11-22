#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 METACORTEX - Telemetry System Service
========================================

Servicio de telemetría con Prometheus que:
- Expone métricas del sistema en formato Prometheus
- Monitorea health de componentes
- Rastrea performance y uso de recursos
- Integra con MPS para métricas GPU
- Dashboard de métricas en tiempo real

Puerto: 9090 (Prometheus metrics)
"""

import logging
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

# Agregar el root al path
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

# Imports de METACORTEX
try:
    from agent_modules.telemetry_system import get_telemetry_system, TelemetrySystem
    from unified_logging import get_logger
    from mps_config import get_device, get_system_info
except ImportError as e:
    print(f"⚠️ Error importando módulos: {e}")
    logging.basicConfig(level=logging.INFO)
    
    def get_logger(name: str = "TelemetryService") -> logging.Logger:
        return logging.getLogger(name)
    
    sys.exit(1)

logger = get_logger("TelemetryService")

# ==============================================================================
# MODELOS PYDANTIC
# ==============================================================================

class MetricUpdate(BaseModel):
    """Actualización de métrica"""
    metric_name: str
    value: float
    labels: Optional[Dict[str, str]] = {}


class AlertRule(BaseModel):
    """Regla de alerta"""
    name: str
    metric: str
    condition: str  # "gt", "lt", "eq"
    threshold: float
    severity: str  # "info", "warning", "critical"


# ==============================================================================
# APLICACIÓN FASTAPI
# ==============================================================================

app = FastAPI(
    title="METACORTEX Telemetry Service",
    description="Servicio de telemetría y métricas Prometheus",
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

class TelemetryServiceState:
    """Estado del servicio de telemetría"""
    
    def __init__(self):
        self.telemetry: Optional[TelemetrySystem] = None
        self.start_time = datetime.now()
        self.request_count = 0
        self.alert_rules: List[AlertRule] = []
        self.metrics_cache: Dict[str, Any] = {}
        self.update_thread: Optional[threading.Thread] = None
        self.running = False
        
        logger.info("📊 Inicializando Telemetry Service...")
        self._initialize_telemetry()
        self._start_metrics_updater()
    
    def _initialize_telemetry(self):
        """Inicializa sistema de telemetría"""
        try:
            self.telemetry = get_telemetry_system(port=9090)
            if self.telemetry:
                logger.info("✅ Telemetry System inicializado")
                # Iniciar servidor de métricas Prometheus
                self.telemetry.start_server()
            else:
                logger.error("❌ No se pudo inicializar Telemetry System")
        except Exception as e:
            logger.error(f"❌ Error inicializando telemetry: {e}")
    
    def _start_metrics_updater(self):
        """Inicia thread de actualización de métricas"""
        self.running = True
        self.update_thread = threading.Thread(
            target=self._metrics_update_loop,
            daemon=True
        )
        self.update_thread.start()
        logger.info("✅ Metrics updater thread iniciado")
    
    def _metrics_update_loop(self):
        """Loop de actualización de métricas"""
        while self.running:
            try:
                self._update_system_metrics()
                time.sleep(5)  # Actualizar cada 5 segundos
            except Exception as e:
                logger.error(f"❌ Error en metrics update loop: {e}")
                time.sleep(5)
    
    def _update_system_metrics(self):
        """Actualiza métricas del sistema"""
        if not self.telemetry:
            return
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.metrics_cache["cpu_percent"] = cpu_percent
            
            # Memoria
            memory = psutil.virtual_memory()
            self.metrics_cache["memory_percent"] = memory.percent
            self.metrics_cache["memory_used_gb"] = memory.used / (1024**3)
            
            # Disco
            disk = psutil.disk_usage('/')
            self.metrics_cache["disk_percent"] = disk.percent
            
            # MPS (si está disponible)
            if torch.backends.mps.is_available():
                self.metrics_cache["mps_available"] = 1.0
            else:
                self.metrics_cache["mps_available"] = 0.0
            
        except Exception as e:
            logger.warning(f"⚠️ Error actualizando métricas: {e}")
    
    def stop(self):
        """Detiene el servicio"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2)


# Instancia global
state = TelemetryServiceState()

# ==============================================================================
# ENDPOINTS - PROMETHEUS METRICS
# ==============================================================================

@app.get("/metrics", response_class=PlainTextResponse, tags=["Prometheus"])
async def prometheus_metrics():
    """
    📊 Endpoint de métricas en formato Prometheus
    
    Este es el endpoint que Prometheus scrapeará para recolectar métricas.
    """
    state.request_count += 1
    
    try:
        # Generar métricas en formato Prometheus
        metrics_output = []
        
        # HELP y TYPE headers
        metrics_output.append("# HELP metacortex_cpu_percent CPU usage percentage")
        metrics_output.append("# TYPE metacortex_cpu_percent gauge")
        metrics_output.append(f"metacortex_cpu_percent {state.metrics_cache.get('cpu_percent', 0.0)}")
        
        metrics_output.append("# HELP metacortex_memory_percent Memory usage percentage")
        metrics_output.append("# TYPE metacortex_memory_percent gauge")
        metrics_output.append(f"metacortex_memory_percent {state.metrics_cache.get('memory_percent', 0.0)}")
        
        metrics_output.append("# HELP metacortex_memory_used_gb Memory used in GB")
        metrics_output.append("# TYPE metacortex_memory_used_gb gauge")
        metrics_output.append(f"metacortex_memory_used_gb {state.metrics_cache.get('memory_used_gb', 0.0)}")
        
        metrics_output.append("# HELP metacortex_disk_percent Disk usage percentage")
        metrics_output.append("# TYPE metacortex_disk_percent gauge")
        metrics_output.append(f"metacortex_disk_percent {state.metrics_cache.get('disk_percent', 0.0)}")
        
        metrics_output.append("# HELP metacortex_mps_available MPS (Apple Silicon GPU) availability")
        metrics_output.append("# TYPE metacortex_mps_available gauge")
        metrics_output.append(f"metacortex_mps_available {state.metrics_cache.get('mps_available', 0.0)}")
        
        metrics_output.append("# HELP metacortex_uptime_seconds Service uptime in seconds")
        metrics_output.append("# TYPE metacortex_uptime_seconds counter")
        uptime = (datetime.now() - state.start_time).total_seconds()
        metrics_output.append(f"metacortex_uptime_seconds {uptime}")
        
        metrics_output.append("# HELP metacortex_requests_total Total requests processed")
        metrics_output.append("# TYPE metacortex_requests_total counter")
        metrics_output.append(f"metacortex_requests_total {state.request_count}")
        
        return "\n".join(metrics_output) + "\n"
        
    except Exception as e:
        logger.error(f"❌ Error generando métricas Prometheus: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# ENDPOINTS - STATUS & HEALTH
# ==============================================================================

@app.get("/health", tags=["Status"])
async def health_check():
    """🏥 Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "telemetry": state.telemetry is not None,
        "metrics_updater": state.running,
    }


@app.get("/status", tags=["Status"])
async def get_status():
    """📊 Estado del servicio"""
    uptime = (datetime.now() - state.start_time).total_seconds()
    
    return {
        "status": "operational",
        "version": "5.0.0-M4",
        "uptime_seconds": uptime,
        "request_count": state.request_count,
        "metrics_cache": state.metrics_cache,
        "alert_rules": len(state.alert_rules),
        "telemetry_available": state.telemetry is not None,
    }


# ==============================================================================
# ENDPOINTS - METRICS API
# ==============================================================================

@app.get("/api/metrics/current", tags=["Metrics"])
async def get_current_metrics():
    """📊 Métricas actuales en formato JSON"""
    state.request_count += 1
    
    return {
        "timestamp": datetime.now().isoformat(),
        "metrics": state.metrics_cache,
        "system": {
            "uptime_seconds": (datetime.now() - state.start_time).total_seconds(),
            "mps_available": torch.backends.mps.is_available(),
        }
    }


@app.post("/api/metrics/update", tags=["Metrics"])
async def update_metric(update: MetricUpdate):
    """📝 Actualiza una métrica manualmente"""
    state.request_count += 1
    
    try:
        # Guardar en cache
        cache_key = f"{update.metric_name}_{update.labels}" if update.labels else update.metric_name
        state.metrics_cache[cache_key] = update.value
        
        logger.info(f"✅ Métrica actualizada: {update.metric_name} = {update.value}")
        
        return {
            "status": "updated",
            "metric": update.metric_name,
            "value": update.value,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"❌ Error actualizando métrica: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# ENDPOINTS - ALERTS
# ==============================================================================

@app.get("/api/alerts/rules", tags=["Alerts"])
async def list_alert_rules():
    """📋 Lista reglas de alerta configuradas"""
    return {
        "total": len(state.alert_rules),
        "rules": [rule.dict() for rule in state.alert_rules],
    }


@app.post("/api/alerts/rules", tags=["Alerts"])
async def create_alert_rule(rule: AlertRule):
    """➕ Crea una nueva regla de alerta"""
    state.alert_rules.append(rule)
    logger.info(f"✅ Regla de alerta creada: {rule.name}")
    
    return {
        "status": "created",
        "rule": rule.dict(),
        "timestamp": datetime.now().isoformat(),
    }


@app.delete("/api/alerts/rules/{rule_name}", tags=["Alerts"])
async def delete_alert_rule(rule_name: str):
    """🗑️ Elimina una regla de alerta"""
    state.alert_rules = [r for r in state.alert_rules if r.name != rule_name]
    logger.info(f"✅ Regla de alerta eliminada: {rule_name}")
    
    return {
        "status": "deleted",
        "rule_name": rule_name,
        "timestamp": datetime.now().isoformat(),
    }


# ==============================================================================
# ENDPOINTS - DASHBOARD
# ==============================================================================

@app.get("/dashboard", tags=["Dashboard"])
async def dashboard():
    """📊 Dashboard de telemetría"""
    return JSONResponse({
        "status": "operational",
        "metrics": state.metrics_cache,
        "uptime": (datetime.now() - state.start_time).total_seconds(),
        "requests": state.request_count,
        "alerts": len(state.alert_rules),
    })


# ==============================================================================
# STARTUP & SHUTDOWN
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Evento de inicio"""
    logger.info("=" * 80)
    logger.info("📊 METACORTEX Telemetry Service - STARTING")
    logger.info("=" * 80)
    logger.info(f"🍎 Platform: Apple Silicon M4")
    logger.info(f"🎮 MPS Available: {torch.backends.mps.is_available()}")
    logger.info(f"📊 Telemetry System: {'Connected' if state.telemetry else 'Not Available'}")
    logger.info("=" * 80)
    logger.info("✅ Telemetry Service READY")
    logger.info(f"📍 Prometheus Metrics: http://localhost:9090/metrics")
    logger.info(f"📍 API Docs: http://localhost:9090/docs")
    logger.info(f"📍 Dashboard: http://localhost:9090/dashboard")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre"""
    logger.info("🛑 Cerrando Telemetry Service...")
    state.stop()
    logger.info("✅ Telemetry Service cerrado correctamente")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Punto de entrada principal"""
    try:
        # Configurar puerto
        port = int(os.environ.get("TELEMETRY_SERVICE_PORT", "9090"))
        host = os.environ.get("TELEMETRY_SERVICE_HOST", "0.0.0.0")
        
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
