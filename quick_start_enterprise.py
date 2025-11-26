#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
METACORTEX SYSTEM - QUICK START (Sin Circular Imports)
═══════════════════════════════════════════════════════════════════════════
Script de inicio rápido que evita el segmentation fault causado por
circular imports en el sistema.

Inicia SOLO los componentes enterprise nuevos:
- Dashboard Web (FastAPI)
- Telegram Bot Monitor (opcional)

NO inicia el full system orchestrator (para evitar circular imports).

Autor: METACORTEX System
Versión: 1.0.0
═══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path

# Colores para terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header():
    """Imprime header del sistema."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 70}")
    print("🤖 METACORTEX ENTERPRISE SYSTEM - QUICK START")
    print(f"{'═' * 70}{Colors.END}\n")

def print_success(msg):
    """Imprime mensaje de éxito."""
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_error(msg):
    """Imprime mensaje de error."""
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_warning(msg):
    """Imprime mensaje de advertencia."""
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def print_info(msg):
    """Imprime mensaje informativo."""
    print(f"{Colors.CYAN}ℹ️  {msg}{Colors.END}")

def check_dependencies():
    """Verifica dependencias necesarias."""
    print_info("Verificando dependencias...")
    
    required = {
        'fastapi': 'pip install fastapi',
        'uvicorn': 'pip install uvicorn',
        'websockets': 'pip install websockets'
    }
    
    missing = []
    for package, install_cmd in required.items():
        try:
            __import__(package)
            print_success(f"{package} instalado")
        except ImportError:
            missing.append((package, install_cmd))
            print_error(f"{package} NO instalado")
    
    if missing:
        print_warning("\nDependencias faltantes:")
        for package, cmd in missing:
            print(f"   {cmd}")
        return False
    
    return True

def check_telegram_bot():
    """Verifica si el bot de Telegram está configurado."""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if token:
        print_success("Telegram Bot Token configurado")
        return True
    else:
        print_warning("Telegram Bot Token NO configurado (opcional)")
        print(f"   {Colors.CYAN}Para configurar: export TELEGRAM_BOT_TOKEN='tu_token'{Colors.END}")
        return False

def start_dashboard(project_root):
    """Inicia el dashboard enterprise."""
    print_info("Iniciando Dashboard Enterprise...")
    
    dashboard_script = project_root / "dashboard_enterprise.py"
    
    if not dashboard_script.exists():
        print_error(f"No se encontró: {dashboard_script}")
        return None
    
    try:
        # Iniciar dashboard en background
        process = subprocess.Popen(
            [sys.executable, str(dashboard_script)],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Esperar un poco para verificar que inició
        time.sleep(2)
        
        if process.poll() is None:
            print_success("Dashboard iniciado correctamente")
            print_info(f"   URL: {Colors.BOLD}http://localhost:8300{Colors.END}")
            print_info(f"   API Docs: {Colors.BOLD}http://localhost:8300/api/docs{Colors.END}")
            return process
        else:
            print_error("Dashboard falló al iniciar")
            return None
            
    except Exception as e:
        print_error(f"Error iniciando dashboard: {e}")
        return None

def start_telegram_bot(project_root):
    """Inicia el bot de Telegram."""
    if not check_telegram_bot():
        return None
    
    print_info("Iniciando Telegram Bot Monitor...")
    
    bot_script = project_root / "telegram_monitor_bot.py"
    
    if not bot_script.exists():
        print_error(f"No se encontró: {bot_script}")
        return None
    
    try:
        # Verificar dependencia
        try:
            __import__('telegram')
        except ImportError:
            print_error("python-telegram-bot NO instalado")
            print_info("   Instalar con: pip install python-telegram-bot")
            return None
        
        # Iniciar bot en background
        process = subprocess.Popen(
            [sys.executable, str(bot_script)],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Esperar un poco para verificar que inició
        time.sleep(2)
        
        if process.poll() is None:
            print_success("Telegram Bot iniciado correctamente")
            print_info("   Busca tu bot en Telegram y ejecuta /start")
            return process
        else:
            print_error("Telegram Bot falló al iniciar")
            return None
            
    except Exception as e:
        print_error(f"Error iniciando Telegram Bot: {e}")
        return None

def show_status():
    """Muestra el status del sistema."""
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'─' * 70}")
    print("📊 SISTEMA OPERACIONAL")
    print(f"{'─' * 70}{Colors.END}\n")
    
    print(f"{Colors.CYAN}Servicios Activos:{Colors.END}")
    print(f"   🌐 Dashboard Enterprise: http://localhost:8300")
    print(f"   📱 Telegram Bot: Disponible (si configurado)")
    
    print(f"\n{Colors.CYAN}Funcionalidades:{Colors.END}")
    print(f"   ✅ Monitoreo en tiempo real (WebSocket)")
    print(f"   ✅ REST API completa")
    print(f"   ✅ Control remoto vía Telegram")
    print(f"   ✅ Dashboard responsive")
    
    print(f"\n{Colors.YELLOW}Nota:{Colors.END}")
    print(f"   El sistema completo con 956+ modelos ML requiere resolver")
    print(f"   circular imports para evitar segmentation fault.")
    print(f"   Por ahora, estos servicios enterprise están operacionales.\n")

def cleanup(processes):
    """Limpia procesos al salir."""
    print(f"\n{Colors.YELLOW}Deteniendo servicios...{Colors.END}")
    
    for name, process in processes.items():
        if process and process.poll() is None:
            print_info(f"Deteniendo {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
                print_success(f"{name} detenido")
            except subprocess.TimeoutExpired:
                print_warning(f"Forzando cierre de {name}...")
                process.kill()
    
    print_success("Servicios detenidos correctamente\n")

def main():
    """Función principal."""
    print_header()
    
    # Directorio del proyecto
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print_info(f"Directorio de trabajo: {project_root}\n")
    
    # Verificar dependencias
    if not check_dependencies():
        print_error("\nInstala las dependencias faltantes y vuelve a ejecutar.")
        sys.exit(1)
    
    print()
    
    # Procesos a monitorear
    processes = {}
    
    try:
        # Iniciar Dashboard
        dashboard_proc = start_dashboard(project_root)
        if dashboard_proc:
            processes['Dashboard'] = dashboard_proc
        
        print()
        
        # Iniciar Telegram Bot (opcional)
        bot_proc = start_telegram_bot(project_root)
        if bot_proc:
            processes['Telegram Bot'] = bot_proc
        
        if not processes:
            print_error("\nNo se pudo iniciar ningún servicio.")
            sys.exit(1)
        
        # Mostrar status
        show_status()
        
        # Mantener activo
        print(f"{Colors.BOLD}Presiona Ctrl+C para detener todos los servicios{Colors.END}\n")
        
        while True:
            time.sleep(1)
            
            # Verificar que los procesos sigan activos
            for name, process in list(processes.items()):
                if process.poll() is not None:
                    print_error(f"{name} se detuvo inesperadamente")
                    del processes[name]
            
            if not processes:
                print_error("Todos los servicios se detuvieron")
                break
    
    except KeyboardInterrupt:
        print(f"\n\n{Colors.CYAN}Ctrl+C detectado{Colors.END}")
    
    finally:
        cleanup(processes)
        print(f"{Colors.GREEN}✨ Hasta luego!{Colors.END}\n")

if __name__ == "__main__":
    main()
