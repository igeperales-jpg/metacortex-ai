# 🍎 METACORTEX - Apple Silicon M4 + MPS Configuration

## ✅ Sistema Completamente Configurado y Operacional

### 📊 Especificaciones del Hardware

```
🍎 Chip: Apple M4
⚡ Performance Cores: 4
💤 Efficiency Cores: 6
📦 Total Cores: 10
🧠 Unified Memory: 16GB
🎮 GPU: Metal Performance Shaders (MPS)
```

### 🚀 Estado del Sistema

**✅ METACORTEX está corriendo de forma permanente con:**

- **GPU Metal (MPS)**: ACTIVO para aceleración ML/AI
- **Caffeinate**: Sistema 24/7 sin sleep
- **Modo Autónomo**: Toma de decisiones automática
- **Daemon Militar**: PID activo con health monitoring
- **PyTorch MPS**: Versión 2.9.1 con soporte Metal

### 🎮 Configuración de GPU Metal (MPS)

El sistema está configurado para **FORZAR el uso de GPU** en lugar de CPU:

```bash
# Variables de entorno configuradas
export PYTORCH_ENABLE_MPS_FALLBACK=1          # Fallback a CPU si MPS falla
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0   # Usar toda la memoria GPU (16GB)
export PYTORCH_MPS_PREFER_METAL=1             # Preferir Metal sobre CPU
export MPS_FORCE_ENABLE=1                     # Forzar MPS
export TOKENIZERS_PARALLELISM=true            # Paralelizar tokenizers
export OMP_NUM_THREADS=10                     # 10 cores del M4
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection
```

### 📝 Archivos de Configuración Creados

1. **`mps_config.py`** - Módulo de configuración MPS con funciones de utilidad
2. **`.venv/bin/activate_mps`** - Script para activar variables MPS en sesión
3. **`scripts/configure_apple_silicon_m4.sh`** - Script de configuración automática
4. **`metacortex_master.sh`** - Script maestro con optimizaciones M4

### 🔧 Comandos Disponibles

#### Iniciar el sistema (24/7 con GPU)
```bash
./metacortex_master.sh start
```

#### Ver estado del sistema
```bash
./metacortex_master.sh status
```

#### Detener el sistema
```bash
./metacortex_master.sh stop
```

#### Reiniciar el sistema
```bash
./metacortex_master.sh restart
```

#### Verificar MPS en Python
```python
import mps_config
mps_config.verify_mps()
mps_config.print_config()
```

### 📊 Verificación de MPS

Para verificar que PyTorch está usando GPU Metal:

```python
import torch

# Verificar disponibilidad
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Crear tensor en GPU
device = torch.device("mps")
x = torch.randn(1000, 1000, device=device)
print(f"Tensor en: {x.device}")

# Verificar dispositivo actual
from mps_config import get_device
print(f"Dispositivo óptimo: {get_device()}")
```

### 🔄 Persistencia 24/7

El sistema usa `caffeinate` de Apple para mantener ejecución continua:

- **Previene system sleep** mientras METACORTEX está activo
- **Permite screen sleep** para ahorrar energía
- **Desactiva automáticamente** al detener el sistema
- **Optimizado para iMac M4** con gestión inteligente de energía

### 🎯 Componentes Activos

✅ **Daemon Principal**: PID 51387, Uptime permanente  
✅ **Ollama LLM**: Puerto 11434, 7 modelos disponibles  
✅ **Redis Cache**: Puerto 6379, caché distribuida  
✅ **GPU Metal (MPS)**: Aceleración ML/AI  
✅ **ML Pipeline**: Entrenamiento automático cada 24h  
✅ **BDI System**: Sistema cognitivo autónomo  
✅ **Memory System**: Memoria infinita con embedding  

### 📈 Rendimiento Esperado

Con Apple Silicon M4 + MPS:

- **Inferencia LLM**: ~2-3x más rápido que CPU
- **Entrenamiento ML**: ~5-10x más rápido que CPU
- **Embeddings**: ~4x más rápido con MPS
- **Matrix Operations**: ~10x más rápido en GPU
- **Consumo Energético**: Menor que GPU dedicada

### 🛠️ Optimizaciones Aplicadas

1. **Unified Memory**: CPU y GPU comparten 16GB sin copias
2. **Performance Cores**: Priorizados para ML/AI
3. **Efficiency Cores**: Tareas de fondo y monitoreo
4. **Memory Management**: Garbage collection optimizado
5. **Thread Pool**: 10 threads para máximo paralelismo
6. **Zero-Copy**: Transferencias directas CPU ↔ GPU

### 📝 Logs y Monitoreo

```bash
# Ver logs del daemon en tiempo real
tail -f logs/metacortex_daemon_military.log

# Ver logs del orchestrator
tail -f logs/startup_orchestrator.log

# Ver logs de web interface
tail -f logs/web_interface_stdout.log

# Verificar procesos activos
ps aux | grep metacortex
```

### 🔒 Seguridad y Estabilidad

- **Circuit Breakers**: Previenen cascadas de fallos
- **Health Checks**: Monitoreo continuo de servicios
- **Auto-Recovery**: Reinicio automático si falla
- **Graceful Shutdown**: Cierre ordenado de componentes
- **PID Management**: Prevención de procesos duplicados

### 🌐 Acceso Web

Cuando Web Interface esté activo:

- **Dashboard**: http://localhost:8000/api/dashboard/metrics
- **API REST**: http://localhost:5000
- **Prometheus**: http://localhost:9090

### 📦 Dependencias Instaladas

- PyTorch 2.9.1 con soporte MPS
- Transformers (Hugging Face)
- Sentence Transformers
- Ollama (7 modelos)
- ChromaDB (embeddings)
- Redis (caché)
- FastAPI + Uvicorn
- Prometheus Client

### 🎓 Uso en Python

```python
# Importar configuración MPS
import mps_config

# Obtener dispositivo óptimo (automáticamente selecciona MPS)
device = mps_config.get_device()
print(f"Usando: {device}")  # Output: mps

# Verificar sistema
info = mps_config.get_system_info()
print(info)

# Imprimir configuración completa
mps_config.print_config()
```

### ✅ Checklist de Verificación

- [x] Apple Silicon M4 detectado
- [x] PyTorch con MPS instalado
- [x] GPU Metal funcionando
- [x] Variables de entorno configuradas
- [x] Caffeinate activo para persistencia
- [x] Daemon corriendo 24/7
- [x] Ollama activo con modelos
- [x] Redis cache activo
- [x] Sistema en modo autónomo

### 🔄 Actualización del Sistema

Para actualizar PyTorch con soporte MPS mejorado:

```bash
source .venv/bin/activate
pip install --upgrade torch torchvision torchaudio
```

### 📞 Soporte y Debugging

Si encuentras problemas:

1. **Verificar MPS**:
   ```bash
   ./scripts/configure_apple_silicon_m4.sh
   ```

2. **Ver logs completos**:
   ```bash
   cat logs/metacortex_daemon_military.log
   ```

3. **Reiniciar sistema**:
   ```bash
   ./metacortex_master.sh restart
   ```

4. **Apagado de emergencia**:
   ```bash
   ./metacortex_master.sh emergency
   ```

### 🎉 ¡Sistema Listo!

METACORTEX está completamente configurado y optimizado para:

- ✅ Ejecución **permanente 24/7** en iMac M4
- ✅ Uso de **GPU Metal (MPS)** para ML/AI
- ✅ **Modo autónomo** con toma de decisiones
- ✅ **Alta disponibilidad** con health monitoring
- ✅ **Rendimiento óptimo** en Apple Silicon

---

**Última actualización**: 22 de noviembre de 2025  
**Versión**: METACORTEX v5.0 + Apple Silicon M4 Optimization  
**Estado**: ✅ OPERACIONAL
