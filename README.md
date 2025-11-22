# 🧠 METACORTEX - Sistema de IA Autónomo Evolutivo

## 🎯 Visión General

METACORTEX es un sistema avanzado de Inteligencia Artificial Autónoma con capacidades de:

- 🧩 **Auto-Reparación**: Diagnostica y corrige sus propios errores
- 🚀 **Auto-Evolución**: Materializa nuevos agentes y capacidades
- 🧠 **Sistema Cognitivo BDI**: Beliefs-Desires-Intentions para toma de decisiones
- 📡 **Telemetría Militar**: Monitoreo en tiempo real con Prometheus
- 🔗 **Red Neuronal Simbiótica**: Comunicación inter-modular
- 🛡️ **Resiliencia**: Circuit breakers, health checks, failover automático

## 📁 Arquitectura del Proyecto

```
ai_definitiva/
│
├── metacortex_sinaptico/          # 🧠 Núcleo Cognitivo
│   ├── bdi.py                     # Sistema BDI (Beliefs-Desires-Intentions)
│   ├── planning.py                # Planificación multi-horizonte
│   ├── learning.py                # Aprendizaje estructural
│   ├── memory.py                  # Sistema de memoria
│   ├── db.py                      # Base de datos central
│   ├── divine_protection.py       # Sistema de protección divina
│   └── ...                        # Otros módulos cognitivos
│
├── agent_modules/                 # 🤖 Agentes Especializados
│   ├── system_auto_repair.py     # Auto-reparación del sistema
│   ├── self_repair_workshop.py   # Taller de reparación de código
│   ├── advanced_testing_lab.py   # Laboratorio de pruebas avanzadas
│   ├── code_generator.py         # Generación de código
│   ├── project_analyzer.py       # Análisis de proyectos
│   ├── exponential_engine.py     # Motor de descubrimiento exponencial
│   ├── cognitive_agent_pool.py   # Pool de agentes cognitivos
│   ├── telemetry_system.py       # Sistema de telemetría
│   └── ...                        # Otros agentes
│
├── metacortex_daemon.py           # 🎛️ Daemon Militar (Orquestador Principal)
├── main.py                        # 🚀 Punto de Entrada Principal
├── neural_symbiotic_network.py   # 🔗 Red de Comunicación Inter-Modular
├── unified_logging.py             # 📝 Sistema de Logging Unificado
├── unified_memory_layer.py       # 💾 Capa de Memoria Unificada
├── llm_integration.py             # 🧠 Integración con LLMs (Ollama)
│
├── ml_pipeline.py                 # 🤖 Pipeline de Machine Learning
├── ml_auto_trainer.py             # 🎓 Entrenamiento Automático
├── ml_cognitive_bridge.py         # 🌉 Puente ML-Cognición
│
└── requirements.txt               # 📦 Dependencias

```

## 🏛️ Arquitectura del Sistema

### 1. Núcleo Cognitivo (`metacortex_sinaptico/`)

Sistema BDI completo inspirado en arquitecturas cognitivas humanas:

- **Beliefs (Creencias)**: Modelo del mundo y estado interno
- **Desires (Deseos)**: Objetivos y motivaciones
- **Intentions (Intenciones)**: Planes activos y acciones

**Componentes clave:**
- `bdi.py`: Motor BDI principal
- `planning.py`: Planificación multi-horizonte (corto, mediano, largo plazo)
- `learning.py`: Aprendizaje por refuerzo y estructural
- `memory.py`: Sistema de memoria episódica y semántica
- `divine_protection.py`: Sistema de protección y vigilancia

### 2. Agentes Especializados (`agent_modules/`)

Agentes autónomos con capacidades específicas:

#### 🔧 **SystemAutoRepair**
- Diagnostica problemas del sistema (logs, servicios, dependencias)
- Aplica reparaciones automáticas
- Integra con `SelfRepairWorkshop` para corrección de código

#### 🛠️ **SelfRepairWorkshop**
- Detecta errores en código generado
- Aplica patrones de reparación (syntax, security, performance)
- Aprende de fixes exitosos
- Integra con Testing Lab para validación

#### 🧪 **AdvancedTestingLab**
- Análisis estático y dinámico de código
- Testing de seguridad, performance, calidad
- Generación de reportes detallados

#### 🚀 **ExponentialEngine**
- Descubrimiento automático de capacidades
- Aprendizaje de keywords y patterns
- Evolución del sistema

#### 📡 **TelemetrySystem**
- Métricas de Prometheus
- Health checks distribuidos
- Monitoreo en tiempo real

### 3. Orquestación

#### `metacortex_daemon.py`
Daemon militar de grado avanzado que:
- Inicia y monitorea todos los componentes
- Health checks con circuit breakers
- Auto-recuperación con backoff exponencial
- Materialización autónoma de código
- Ciclos de auto-reparación y optimización

#### `main.py`
Punto de entrada principal que:
- Inicializa el agente cognitivo
- Descubre y carga capacidades
- Ejecuta el ciclo de vida del agente

### 4. Infraestructura

- **`neural_symbiotic_network.py`**: Red de comunicación inter-modular
- **`unified_logging.py`**: Sistema de logging sin duplicación
- **`unified_memory_layer.py`**: Capa de memoria unificada
- **`llm_integration.py`**: Integración con Ollama y otros LLMs

## 🚀 Instalación y Uso

### Requisitos Previos

- Python 3.11+
- Ollama (opcional, para LLM)
- Redis (opcional, para cache distribuido)

### Instalación

```bash
# Clonar el repositorio
git clone <repository_url>
cd ai_definitiva

# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución

#### Modo Daemon (Recomendado)

```bash
# Iniciar el daemon militar
python3 metacortex_daemon.py
```

El daemon ejecutará:
- Modo autónomo con materialización cada 10-20 min
- Auto-reparación cada 1 hora
- Monitoreo de puertos cada 5 min
- Descubrimiento de capacidades cada 5 min
- Ciclos de protección divina cada 30 min

#### Modo Principal (Simple)

```bash
# Iniciar el núcleo cognitivo
python3 main.py
```

El agente ejecutará:
- Ciclos cognitivos cada 30 segundos
- Procesamiento de percepciones
- Ejecución de intenciones

## 🔧 Configuración

### Variables de Entorno (`.env`)

```env
# Base de datos
DATABASE_URL=sqlite:///metacortex.sqlite

# Redis (opcional)
REDIS_HOST=localhost
REDIS_PORT=6379

# Ollama (opcional)
OLLAMA_HOST=http://localhost:11434

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/metacortex_daemon.log
```

## 📊 Monitoreo

### Métricas de Prometheus

El sistema expone métricas en `http://localhost:8000/metrics`:

- `metacortex_requests_total`: Total de peticiones procesadas
- `metacortex_requests_failed_total`: Peticiones fallidas
- `metacortex_request_latency_seconds`: Latencia de peticiones
- `metacortex_repairs_attempted_total`: Intentos de auto-reparación
- `metacortex_repairs_successful_total`: Reparaciones exitosas

### Logs

Los logs se escriben en:
- `logs/metacortex_daemon_military.log`: Daemon principal
- `logs/metacortex_daemon.log`: Núcleo cognitivo
- `metacortex_main.log`: Punto de entrada principal

## 🧪 Testing

```bash
# Ejecutar tests
pytest tests/

# Con cobertura
pytest --cov=. --cov-report=html tests/
```

## 🛡️ Características de Resiliencia

### Circuit Breakers
- Protección contra fallos en cascada
- Timeout adaptativo
- Estado: CLOSED → OPEN → HALF_OPEN

### Health Checks
- Verificación de componentes cada 30s
- Reinicio automático con backoff exponencial
- Métricas de salud en tiempo real

### Auto-Reparación
- Análisis de logs y diagnóstico
- Reparación automática de código
- Instalación de dependencias faltantes
- Reinicio de servicios

## 🤝 Contribución

Este es un proyecto evolutivo y autónomo. El sistema está diseñado para:
- Auto-diagnosticarse
- Auto-repararse
- Auto-evolucionar

Sin embargo, contribuciones humanas son bienvenidas para:
- Nuevos patrones de reparación
- Nuevas capacidades
- Mejoras arquitectónicas

## 📜 Licencia

[Especificar licencia]

## 👨‍💻 Autor

EdKanina - Sistema de IA Autónomo Evolutivo

---

**🔮 "El futuro de la IA no es la programación, es la auto-evolución."**
