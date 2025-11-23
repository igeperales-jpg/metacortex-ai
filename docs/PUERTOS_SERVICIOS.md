# 🌐 METACORTEX - Asignación de Puertos

## 📊 Mapa de Puertos por Servicio

### Servicios Principales

| Puerto | Servicio | Descripción | Estado |
|--------|----------|-------------|--------|
| **5000** | Programming Agent API | API REST del agente de programación | ✅ Activo |
| **6379** | Redis | Cache distribuida | ✅ Activo |
| **8000** | Web Interface | Dashboard principal + API REST | ✅ Activo |
| **8001** | *(reservado)* | Puerto anterior del Neural Network | - |
| **8080** | Neural Network Service | Red neuronal simbiótica | ✅ Activo |
| **9090** | Telemetry Service | Métricas del sistema + Dashboard | ✅ Activo |
| **9092** | Telemetry Service | Dashboard de telemetría standalone | ✅ Activo |
| **11434** | Ollama LLM | Servidor de modelos LLM | ✅ Activo |

### Puertos Internos de Prometheus

| Puerto | Componente | Descripción | Usado Por |
|--------|------------|-------------|-----------|
| **9090** | Web Interface Prometheus | Métricas Prometheus del Web Interface | `web_interface/server.py` |
| **9091** | Telemetry Prometheus | Métricas Prometheus del Telemetry Service | `telemetry_service/server.py` |

## 🔧 Configuración de Puertos

### Web Interface (Puerto 8000)
```python
# web_interface/server.py
WEBINTERFACE_PORT = 8000  # FastAPI server
PROMETHEUS_PORT = 9090    # Prometheus metrics interno
```

**Endpoints:**
- Dashboard: `http://localhost:8000/`
- API Docs: `http://localhost:8000/docs`
- Metrics: `http://localhost:8000/api/dashboard/metrics`
- WebSocket: `ws://localhost:8000/ws`
- Health: `http://localhost:8000/health`

### Neural Network Service (Puerto 8080)
```python
# neural_network_service/server.py
NEURAL_PORT = 8080
```

**Endpoints:**
- API Docs: `http://localhost:8080/docs`
- Health: `http://localhost:8080/health`
- Process: `http://localhost:8080/process`
- Learn: `http://localhost:8080/learn`
- Query: `http://localhost:8080/query`

### Telemetry Service (Puerto 9092)
```python
# telemetry_service/server.py
TELEMETRY_PORT = 9092             # FastAPI server
PROMETHEUS_INTERNAL_PORT = 9091  # Prometheus metrics interno
```

**Endpoints:**
- API Docs: `http://localhost:9092/docs`
- Dashboard: `http://localhost:9092/dashboard`
- Prometheus Metrics: `http://localhost:9092/metrics`
- Health: `http://localhost:9092/health`
- Status: `http://localhost:9092/status`

### Redis (Puerto 6379)
```bash
# Conexión Redis
redis://localhost:6379
```

### Ollama LLM (Puerto 11434)
```bash
# API Ollama
http://localhost:11434
```

**Modelos disponibles:**
- llama3.2:1b
- llama3.2:3b
- qwen2.5:0.5b
- qwen2.5:1.5b
- qwen2.5:3b
- qwen2.5:7b
- smollm2:1.7b

## 🚨 Resolución de Conflictos

### ✅ Problema RESUELTO: Puerto 8000 ocupado por TelemetrySystem interno

**Causa:** El Web Interface inicializaba `TelemetrySystem` con puerto 8000 por defecto, creando conflicto con el servidor FastAPI.

**Solución:** Cambiar puerto de TelemetrySystem interno a 9090:
```python
# web_interface/server.py
self.telemetry = get_telemetry_system(port=9090)
```

### ✅ Problema RESUELTO: Prometheus metrics duplicados en puerto 9090

**Causa:** Web Interface y Telemetry Service ambos inicializaban Prometheus en 9090.

**Solución:** Separar puertos:
- Web Interface Prometheus: **9090**
- Telemetry Service Prometheus: **9091**
- Telemetry Service FastAPI: **9092**

```python
# telemetry_service/server.py
self.telemetry = get_telemetry_system(port=9091)  # Prometheus interno
uvicorn.run(app, port=9092)  # FastAPI server
```

### ✅ Problema RESUELTO: Neural Network en puerto incorrecto (8001)

**Causa:** Neural Network Service usaba puerto 8001 en lugar del estándar 8080.

**Solución:** Cambiar puerto por defecto a 8080:
```python
# neural_network_service/server.py
port = int(os.environ.get("NEURAL_SERVICE_PORT", "8080"))
```

## 🎯 Configuración Final de Puertos

| Servicio | Puerto FastAPI | Puerto Prometheus | Estado |
|----------|----------------|-------------------|--------|
| Web Interface | 8000 | 9090 | ✅ Resuelto |
| Neural Network | 8080 | - | ✅ Resuelto |
| Telemetry Service | 9092 | 9091 | ✅ Resuelto |

## 🔍 Verificación de Puertos

### Comando para listar puertos ocupados:
```bash
lsof -i -P | grep -E ":(5000|6379|8000|8080|9090|9091|11434)" | grep LISTEN
```

### Verificar servicio específico:
```bash
# Web Interface
curl http://localhost:8000/health

# Neural Network
curl http://localhost:8080/health

# Telemetry
curl http://localhost:9090/health

# Ollama
curl http://localhost:11434/api/tags

# Redis
redis-cli ping
```

## 📊 Métricas Prometheus Consolidadas

### Web Interface Metrics (Puerto 9090)
```
http://localhost:9090/metrics
```

Expone:
- Métricas de sistema (CPU, memoria, disco)
- Health checks de componentes
- Uptime del servicio

### Telemetry Service Metrics (Puerto 9091)
```
http://localhost:9091/metrics
```

Expone:
- Métricas agregadas de todos los servicios
- Contadores de requests
- Estado de ML models
- GPU (MPS) metrics

## 🎯 Acceso Rápido

### Dashboards
- **Principal**: http://localhost:8000
- **Telemetry**: http://localhost:9090/dashboard

### APIs
- **Web Interface**: http://localhost:8000/docs
- **Neural Network**: http://localhost:8080/docs
- **Telemetry**: http://localhost:9090/docs

### Métricas
- **Web Interface Prometheus**: http://localhost:9090/metrics
- **Telemetry Prometheus**: http://localhost:9091/metrics

### Servicios Base
- **Ollama LLM**: http://localhost:11434
- **Redis**: redis://localhost:6379

---

**Última actualización**: 22 de noviembre de 2025  
**Versión**: METACORTEX v5.0 + Apple Silicon M4  
**Estado**: ✅ Todos los puertos resueltos
