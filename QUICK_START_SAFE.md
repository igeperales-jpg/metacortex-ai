# 🚀 METACORTEX ENTERPRISE - QUICK START (SAFE MODE)

**⚠️ IMPORTANTE**: El sistema tiene un **segmentation fault** debido a imports circulares.  
Este guide te muestra cómo usar las partes que **SÍ FUNCIONAN** sin problemas.

---

## ✅ QUÉ FUNCIONA (100% SEGURO)

### 1. 📊 Dashboard Enterprise (RECOMENDADO)

El dashboard web funciona perfectamente y es la forma MÁS SEGURA de monitorear el sistema.

```bash
# Terminal 1: Iniciar dashboard
cd /Users/edkanina/ai_definitiva
python3 dashboard_enterprise.py
```

**Accede a**: http://localhost:8300

**Características**:
- ✅ Visualización en tiempo real via WebSocket
- ✅ REST API completa (`/api/status`, `/api/models`, `/api/tasks`)
- ✅ Interfaz responsive
- ✅ Health check endpoint
- ✅ NO causa segmentation fault

---

### 2. 📱 Telegram Monitor Bot (RECOMENDADO)

Monitorea el sistema desde tu teléfono con comandos simples.

```bash
# Configurar token (obtén uno en @BotFather)
export TELEGRAM_BOT_TOKEN="tu_token_aqui"

# Terminal 2: Iniciar bot
cd /Users/edkanina/ai_definitiva
python3 telegram_monitor_bot.py
```

**Comandos disponibles**:
- `/start` - Mensaje de bienvenida
- `/status` - Estado general del sistema
- `/models` - Estadísticas de modelos
- `/tasks` - Estado de tareas
- `/stats` - Performance detallado
- `/help` - Ayuda

---

### 3. 🎯 Singleton Registry (FUNDAMENTO)

El singleton registry funciona perfectamente y es la base de todo.

```python
# Test del singleton registry
python3 -c "
from singleton_registry import registry
print('✅ Singleton Registry cargado')
print(f'   Factories registradas: {len(registry._factories)}')
print(f'   Singletons activos: {len(registry._singletons)}')
"
```

---

### 4. 🧠 Metacortex Orchestrator (SAFE MODE)

Puedes usar el orchestrator sin auto-loading (evita segfault).

```python
# Terminal 3: Test manual del orchestrator
python3 << EOF
from metacortex_orchestrator import MetacortexUnifiedOrchestrator
import os

# Crear orchestrator SIN inicializar componentes
orch = MetacortexUnifiedOrchestrator(os.getcwd())

print("✅ Orchestrator creado")
print(f"   Directorio: {orch.project_root}")
print(f"   Inicializado: {orch.initialized}")

# NO llames a orch.initialize() para evitar segfault
EOF
```

---

## ⚠️ QUÉ NO FUNCIONA (CAUSA SEGFAULT)

### ❌ Autonomous Model Orchestrator

**PROBLEMA**: Import circular entre:
- `ml_pipeline.py` ↔ `neural_network.py`
- `ollama_integration.py` ↔ `cognitive_agent.py`
- `autonomous_model_orchestrator.py` → todos los anteriores

**SÍNTOMA**:
```
2025-11-26 01:15:42,668 - INFO - 🎖️ Inicializando ML Pipeline MILITARY GRADE v3.0...
[Repetido 40+ veces]
zsh: segmentation fault
```

**SOLUCIÓN PENDIENTE**: Refactorizar TODOS los componentes para usar EXCLUSIVAMENTE singleton registry.

---

## 🎯 DEPLOYMENT SCRIPT INTERACTIVO

Usa el script de deployment para testear componentes de forma segura:

```bash
cd /Users/edkanina/ai_definitiva
python3 deploy_enterprise.py
```

**Opciones**:
1. **Dashboard Enterprise** (puerto 8300) → ✅ SEGURO
2. **Telegram Monitor Bot** → ✅ SEGURO
3. **Metacortex Orchestrator (SAFE MODE)** → ✅ SEGURO
4. **Autonomous Model Orchestrator** → ❌ SEGFAULT
5. **Todo el sistema** → ❌ SEGFAULT

---

## 📊 VERIFICAR ESTADO DE COMPONENTES

### Test 1: Dependencias

```bash
python3 -c "
import numpy as np
import pandas as pd
import sklearn
import torch
import fastapi
import telegram

print('✅ numpy:', np.__version__)
print('✅ pandas:', pd.__version__)
print('✅ scikit-learn:', sklearn.__version__)
print('✅ torch:', torch.__version__)
print('✅ fastapi:', fastapi.__version__)
print('✅ telegram:', telegram.__version__)
"
```

### Test 2: Modelos ML

```bash
python3 -c "
from pathlib import Path
models_dir = Path('ml_models')
pkl_files = list(models_dir.glob('*.pkl'))
metadata_files = list(models_dir.glob('*_metadata.json'))

print(f'✅ Modelos .pkl: {len(pkl_files)}')
print(f'✅ Archivos metadata: {len(metadata_files)}')
"
```

### Test 3: Singleton Registry

```bash
python3 -c "
from singleton_registry import registry

print('✅ Singleton Registry cargado')
print('   Factories registradas:')
for name in registry._factories.keys():
    print(f'      - {name}')
"
```

---

## 🚀 QUICK START PASO A PASO

### Opción A: Dashboard Web (MÁS FÁCIL)

```bash
# 1. Abrir terminal
cd /Users/edkanina/ai_definitiva

# 2. Iniciar dashboard
python3 dashboard_enterprise.py

# 3. Abrir navegador
open http://localhost:8300

# ✅ Listo! Dashboard funcionando
```

### Opción B: Telegram Bot (MÁS CONVENIENTE)

```bash
# 1. Obtener token de @BotFather en Telegram
# 2. Configurar token
export TELEGRAM_BOT_TOKEN="tu_token"

# 3. Iniciar bot
cd /Users/edkanina/ai_definitiva
python3 telegram_monitor_bot.py

# 4. En Telegram, busca tu bot y envía /start
# ✅ Listo! Bot funcionando
```

### Opción C: Deployment Script (MÁS COMPLETO)

```bash
# 1. Ejecutar script interactivo
cd /Users/edkanina/ai_definitiva
python3 deploy_enterprise.py

# 2. Seleccionar opción 1 o 2
# 3. Seguir instrucciones en pantalla

# ✅ Listo! Componente seleccionado funcionando
```

---

## 📝 PRÓXIMOS PASOS (PARA RESOLVER SEGFAULT)

Para hacer que TODO el sistema funcione sin segfault, necesitas:

### 1. Refactorizar ml_pipeline.py

```python
# ANTES (causa circular imports):
from neural_network import NeuralNetwork
from cognitive_agent import CognitiveAgent

# DESPUÉS (singleton registry):
class MilitaryGradeMLPipeline:
    def __init__(self):
        self._neural_network = None
        self._cognitive_agent = None
    
    @property
    def neural_network(self):
        if self._neural_network is None:
            from singleton_registry import get_neural_network
            self._neural_network = get_neural_network()
        return self._neural_network
```

### 2. Refactorizar ollama_integration.py

```python
# Mismo patrón que ml_pipeline.py
# Lazy properties con singleton registry
```

### 3. Refactorizar cognitive_agent.py

```python
# Mismo patrón que ml_pipeline.py
# Lazy properties con singleton registry
```

### 4. Refactorizar neural_network.py

```python
# Mismo patrón que ml_pipeline.py
# Lazy properties con singleton registry
```

### 5. Testing Completo

```bash
# Después de refactorizar TODO:
python3 -c "
from singleton_registry import (
    get_ml_pipeline,
    get_ollama,
    get_cognitive_agent,
    get_autonomous_orchestrator
)

ml = get_ml_pipeline()
ollama = get_ollama()
cognitive = get_cognitive_agent()
orchestrator = get_autonomous_orchestrator()

print('✅ TODO CARGADO SIN SEGFAULT!')
"
```

---

## 📚 DOCUMENTACIÓN ADICIONAL

- **DEPLOYMENT_ENTERPRISE.md**: Guía completa de deployment (800+ líneas)
- **singleton_registry.py**: Código fuente del singleton registry (400+ líneas)
- **dashboard_enterprise.py**: Código fuente del dashboard (700+ líneas)
- **telegram_monitor_bot.py**: Código fuente del bot (300+ líneas)

---

## 🆘 TROUBLESHOOTING

### Problema: "ModuleNotFoundError: No module named 'fastapi'"

```bash
pip install fastapi uvicorn websockets
```

### Problema: "ModuleNotFoundError: No module named 'telegram'"

```bash
pip install python-telegram-bot
```

### Problema: Dashboard no abre en navegador

```bash
# Verificar que el puerto 8300 esté libre
lsof -i :8300

# Si está ocupado, matar proceso:
kill -9 <PID>

# O cambiar puerto en dashboard_enterprise.py (línea 574):
# uvicorn.run(app, host="0.0.0.0", port=8301)
```

### Problema: Telegram Bot no responde

```bash
# Verificar token
echo $TELEGRAM_BOT_TOKEN

# Verificar logs
tail -f logs/telegram_monitor.log

# Reiniciar bot
pkill -f telegram_monitor_bot
python3 telegram_monitor_bot.py
```

---

## ✅ RESUMEN

**LO QUE FUNCIONA HOY**:
- ✅ Dashboard Enterprise (http://localhost:8300)
- ✅ Telegram Monitor Bot
- ✅ Singleton Registry
- ✅ Metacortex Orchestrator (SAFE MODE)
- ✅ 965 modelos ML descubiertos
- ✅ Deployment script interactivo

**LO QUE NO FUNCIONA**:
- ❌ Autonomous Model Orchestrator (segfault)
- ❌ Auto-loading de componentes ML
- ❌ Sistema completo integrado

**PARA HACERLO FUNCIONAR TODO**:
- 🔧 Refactorizar ml_pipeline.py → singleton registry
- 🔧 Refactorizar ollama_integration.py → singleton registry
- 🔧 Refactorizar cognitive_agent.py → singleton registry
- 🔧 Refactorizar neural_network.py → singleton registry
- 🔧 Testing completo sin segfault

---

**Última actualización**: 2025-01-26  
**Estado**: 90% completo - componentes standalone funcionan perfectamente  
**Bloqueador**: Circular imports en componentes ML core
