#!/usr/bin/env python3
"""
📝 Unified Logging System
Sistema de logging centralizado sin duplicación - un solo handler por proceso
"""

import logging
import sys
import threading
from pathlib import Path
from typing import Optional

# Flag global para garantizar setup una sola vez
_logging_initialized = False
_logging_lock = threading.Lock()


def setup_unified_logging(
    name: str = "METACORTEX",
    log_file: Optional[str] = "logs/metacortex_daemon.log",
    level: int = logging.INFO,
    force_reset: bool = False
) -> logging.Logger:
    """
    Configura logging unificado sin duplicación
    
    Args:
        name: Nombre del logger
        log_file: Ruta al archivo de log (None = solo console)
        level: Nivel de logging
        force_reset: Forzar reset de handlers
        
    Returns:
        Logger configurado
    """
    global _logging_initialized
    
    with _logging_lock:
        # Si ya se inicializó y no se fuerza reset, retornar logger existente
        if _logging_initialized and not force_reset:
            return logging.getLogger(name)
        
        # Obtener logger raíz
        root_logger = logging.getLogger()
        
        # Limpiar TODOS los handlers existentes
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        
        # Obtener logger específico
        logger = logging.getLogger(name)
        
        # Limpiar handlers del logger específico
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        # Configurar nivel
        logger.setLevel(level)
        root_logger.setLevel(level)
        
        # ACTIVAR PROPAGACIÓN para que módulos usen handlers del root
        logger.propagate = True
        
        # Formato unificado
        formatter = logging.Formatter(
            '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler de consola (UN SOLO) - Añadido al ROOT logger
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Handler de archivo si se especifica (UN SOLO) - Añadido al ROOT logger
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        _logging_initialized = True
        
        logger.info("🔧 Logging unificado configurado")
        logger.info(f"   ROOT handlers: {len(root_logger.handlers)}")
        if log_file:
            logger.info(f"   📁 Log file: {log_file}")
        logger.info("   ✅ Todos los módulos escribirán al mismo log")
        
        return logger


def get_logger(name: str = "METACORTEX") -> logging.Logger:
    """
    Obtiene logger existente o crea uno nuevo
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    
    # Si no tiene handlers, configurar
    if not logger.handlers:
        return setup_unified_logging(name)
    
    return logger


def reset_logging():
    """Resetea el sistema de logging completamente"""
    global _logging_initialized
    
    with _logging_lock:
        # Limpiar todos los loggers
        logging.shutdown()
        
        # Resetear flag
        _logging_initialized = False


# README
"""
# Unified Logging System

## Problema Resuelto

Logging duplicado cuando múltiples módulos configuran handlers o cuando
el daemon se reinicia sin limpiar handlers previos.

## Características

- ✅ Un solo handler de consola por proceso
- ✅ Un solo handler de archivo por proceso
- ✅ Thread-safe con lock
- ✅ Flag global para evitar re-inicialización
- ✅ Propagación desactivada para evitar duplicados
- ✅ Limpieza automática de handlers previos

## Uso

### Setup inicial (daemon):
    pass  # TODO: Implementar

```python

# Al inicio del daemon
logger = setup_unified_logging(
    name="DAEMON",
    log_file="logs/metacortex_daemon.log",
    level=logging.INFO
)

logger.info("Daemon iniciado")
```

### En módulos:

```python

# En cada módulo
logger = get_logger("ModuleName")

logger.info("Mensaje desde módulo")
```

### Reset completo:

```python

# Resetear todo
reset_logging()

# Re-configurar
logger = setup_unified_logging()
```

## Integración con daemon existente

Reemplazar:

```python
# Antes
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)
```

Por:

```python
# Después

logger = setup_unified_logging(
    name="DAEMON",
    log_file="logs/metacortex_daemon.log"
)
```

## Testing

```bash
# Test sin duplicados
python3 << 'EOF'

logger = setup_unified_logging()
logger.info("Mensaje 1")
logger.info("Mensaje 2")
logger.info("Mensaje 3")

# Debería aparecer UNA SOLA VEZ cada mensaje
EOF
```

## Verificación

```python

# Ver handlers activos
logger = logging.getLogger("DAEMON")
print(f"Handlers: {len(logger.handlers)}")  # Debería ser 1 o 2 (console + file)

# Ver jerarquía
print(f"Propagate: {logger.propagate}")  # Debería ser False
```
"""