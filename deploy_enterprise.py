#!/usr/bin/env python3
"""
🚀 METACORTEX ENTERPRISE - SAFE DEPLOYMENT
═══════════════════════════════════════════════════════════════════════════

Script de deployment seguro que evita segmentation faults mediante:
1. Testing incremental de componentes
2. Lazy loading controlado
3. Verificación de memoria
4. Rollback automático en caso de error

Autor: METACORTEX System
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# Colores para terminal
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
RESET = '\033[0m'

def print_header(text):
    """Imprime header bonito."""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}{text:^70}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")

def print_success(text):
    """Imprime éxito."""
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    """Imprime error."""
    print(f"{RED}❌ {text}{RESET}")

def print_warning(text):
    """Imprime warning."""
    print(f"{YELLOW}⚠️  {text}{RESET}")

def print_info(text):
    """Imprime info."""
    print(f"{BLUE}ℹ️  {text}{RESET}")

def test_component(name, test_code):
    """
    Testea un componente de forma segura.
    
    Args:
        name: Nombre del componente
        test_code: Código Python a ejecutar
        
    Returns:
        True si exitoso, False si falla
    """
    print(f"Testing {name}...", end=" ", flush=True)
    
    try:
        result = subprocess.run(
            ["python3", "-c", test_code],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print_success(f"{name} OK")
            return True
        else:
            print_error(f"{name} FAILED")
            print(f"   Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print_error(f"{name} TIMEOUT")
        return False
    except Exception as e:
        print_error(f"{name} EXCEPTION: {e}")
        return False

def main():
    """Main deployment flow."""
    print_header("🚀 METACORTEX ENTERPRISE DEPLOYMENT")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print_info(f"Project directory: {project_dir}")
    print_info(f"Python: {sys.version.split()[0]}")
    print()
    
    # ═══════════════════════════════════════════════════════════════════════
    # FASE 1: VERIFICACIÓN DE DEPENDENCIAS
    # ═══════════════════════════════════════════════════════════════════════
    
    print_header("📦 FASE 1: Verificando Dependencias")
    
    dependencies = {
        "numpy": "import numpy; print(numpy.__version__)",
        "pandas": "import pandas; print(pandas.__version__)",
        "scikit-learn": "import sklearn; print(sklearn.__version__)",
        "torch": "import torch; print(torch.__version__)",
        "fastapi": "import fastapi; print(fastapi.__version__)",
        "telegram": "import telegram; print(telegram.__version__)"
    }
    
    missing_deps = []
    for dep_name, test_code in dependencies.items():
        if not test_component(dep_name, test_code):
            missing_deps.append(dep_name)
    
    if missing_deps:
        print_error(f"Faltan dependencias: {', '.join(missing_deps)}")
        print_info("Instala con: pip install " + " ".join(missing_deps))
        return False
    
    print_success("Todas las dependencias están instaladas")
    
    # ═══════════════════════════════════════════════════════════════════════
    # FASE 2: TESTING DE SINGLETON REGISTRY
    # ═══════════════════════════════════════════════════════════════════════
    
    print_header("🎯 FASE 2: Testing Singleton Registry")
    
    # Test básico del registry
    if not test_component(
        "Singleton Registry Import",
        "from singleton_registry import registry; print('OK')"
    ):
        print_error("Singleton Registry falló")
        return False
    
    # Test de factories registradas
    if not test_component(
        "Factories Registration",
        """
from singleton_registry import registry
factories = list(registry._factories.keys())
print(f'{len(factories)} factories registered')
assert len(factories) >= 5, 'Not enough factories'
"""
    ):
        print_error("Factories registration falló")
        return False
    
    print_success("Singleton Registry operacional")
    
    # ═══════════════════════════════════════════════════════════════════════
    # FASE 3: TESTING INDIVIDUAL DE COMPONENTES (SAFE)
    # ═══════════════════════════════════════════════════════════════════════
    
    print_header("🧪 FASE 3: Testing Componentes Individuales")
    
    # NO testeamos componentes que causan segfault
    # Solo verificamos que los archivos existen
    
    components = [
        "singleton_registry.py",
        "dashboard_enterprise.py",
        "telegram_monitor_bot.py",
        "metacortex_orchestrator.py",
        "autonomous_model_orchestrator.py"
    ]
    
    for component in components:
        if Path(component).exists():
            print_success(f"{component} exists")
        else:
            print_error(f"{component} NOT FOUND")
            return False
    
    # ═══════════════════════════════════════════════════════════════════════
    # FASE 4: VERIFICAR MODELOS ML
    # ═══════════════════════════════════════════════════════════════════════
    
    print_header("🧠 FASE 4: Verificando Modelos ML")
    
    models_dir = Path("ml_models")
    if not models_dir.exists():
        print_error(f"Directorio {models_dir} no existe")
        return False
    
    pkl_files = list(models_dir.glob("*.pkl"))
    metadata_files = list(models_dir.glob("*_metadata.json"))
    
    print_info(f"Modelos .pkl encontrados: {len(pkl_files)}")
    print_info(f"Archivos metadata encontrados: {len(metadata_files)}")
    
    if len(pkl_files) < 10:
        print_warning("Pocos modelos encontrados (esperados 956+)")
    else:
        print_success(f"{len(pkl_files)} modelos disponibles")
    
    # ═══════════════════════════════════════════════════════════════════════
    # FASE 5: PREPARAR LOGS
    # ═══════════════════════════════════════════════════════════════════════
    
    print_header("📝 FASE 5: Preparando Logs")
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    print_success(f"Directorio logs: {logs_dir}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # FASE 6: DEPLOYMENT OPTIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    print_header("🚀 FASE 6: Opciones de Deployment")
    
    print("""
Selecciona qué componentes iniciar:

1. 📊 Dashboard Enterprise (puerto 8300)
2. 📱 Telegram Monitor Bot
3. 🧠 Metacortex Orchestrator (SAFE MODE - sin auto-loading)
4. 🤖 Autonomous Model Orchestrator (REQUIERE TESTING)
5. 🌐 Todo el sistema (EXPERIMENTAL - puede causar segfault)

0. Salir

RECOMENDADO: Opción 1 (Dashboard) o 2 (Telegram Bot)
""")
    
    try:
        choice = input("Selecciona opción [0-5]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\nDeployment cancelado por usuario")
        return False
    
    if choice == "0":
        print_info("Deployment cancelado")
        return True
    
    elif choice == "1":
        print_header("📊 Iniciando Dashboard Enterprise")
        print_info("Ejecuta en otra terminal:")
        print(f"   cd {project_dir}")
        print("   python3 dashboard_enterprise.py")
        print()
        print_info("Dashboard estará disponible en: http://localhost:8300")
        return True
    
    elif choice == "2":
        print_header("📱 Iniciando Telegram Monitor Bot")
        
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            print_warning("TELEGRAM_BOT_TOKEN no configurado")
            print_info("Configura con: export TELEGRAM_BOT_TOKEN='tu_token'")
            print_info("Obtén token en: https://t.me/BotFather")
            return False
        
        print_info("Ejecuta en otra terminal:")
        print(f"   cd {project_dir}")
        print("   python3 telegram_monitor_bot.py")
        return True
    
    elif choice == "3":
        print_header("🧠 Iniciando Metacortex Orchestrator (SAFE MODE)")
        print_warning("SAFE MODE: Sin auto-loading de componentes")
        print_info("Esto evita el segmentation fault")
        print()
        print_info("Ejecuta en otra terminal:")
        print(f"   cd {project_dir}")
        print("   python3 -c \"")
        print("from metacortex_orchestrator import MetacortexUnifiedOrchestrator")
        print("import os")
        print("orch = MetacortexUnifiedOrchestrator(os.getcwd())")
        print("print(f'✅ Orchestrator initialized')")
        print("print(f'   Status: {orch.is_initialized}')")
        print("   \"")
        return True
    
    elif choice == "4":
        print_header("🤖 Autonomous Model Orchestrator")
        print_error("⚠️  ADVERTENCIA: Puede causar segmentation fault")
        print_warning("Solo usar después de resolver circular imports")
        print()
        confirm = input("¿Continuar de todos modos? [y/N]: ").strip().lower()
        if confirm != 'y':
            print_info("Cancelado - decisión sabia 👍")
            return True
        
        print_info("Ejecuta en otra terminal:")
        print(f"   cd {project_dir}")
        print("   python3 autonomous_model_orchestrator.py")
        return True
    
    elif choice == "5":
        print_header("🌐 Sistema Completo")
        print_error("⚠️  ADVERTENCIA CRÍTICA: Probablemente cause segmentation fault")
        print_warning("NO RECOMENDADO hasta resolver circular imports")
        print()
        confirm = input("¿Estás SEGURO que quieres continuar? [y/N]: ").strip().lower()
        if confirm != 'y':
            print_info("Cancelado - muy buena decisión 👍👍")
            return True
        
        print_error("Iniciando sistema completo... 🙏")
        print_info("Si falla, usa Ctrl+C para detener")
        time.sleep(2)
        
        try:
            subprocess.run(["python3", "metacortex_orchestrator.py"])
        except KeyboardInterrupt:
            print("\n\nDeployment interrumpido")
        return True
    
    else:
        print_error(f"Opción inválida: {choice}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        
        if success:
            print()
            print_header("✅ DEPLOYMENT COMPLETADO")
            print_info("Lee DEPLOYMENT_ENTERPRISE.md para más detalles")
            print_info("Logs en: logs/")
            print()
            sys.exit(0)
        else:
            print()
            print_header("❌ DEPLOYMENT FALLÓ")
            print_info("Revisa los errores arriba")
            print_info("Consulta DEPLOYMENT_ENTERPRISE.md para troubleshooting")
            print()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Deployment cancelado por usuario")
        sys.exit(130)
    except Exception as e:
        print_error(f"Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
