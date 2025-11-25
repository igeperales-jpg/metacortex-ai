#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀🤖⚡ START AUTONOMOUS SYSTEM - Activa todos los 956+ modelos ML para trabajar 24/7
═══════════════════════════════════════════════════════════════════════════════

MISIÓN: Activar el sistema autónomo completo con todos los modelos trabajando.

CARACTERÍSTICAS:
✅ Activa Autonomous Model Orchestrator
✅ Conecta con ML Pipeline, Ollama, Internet Search, World Model
✅ Genera tareas automáticas del mundo real
✅ Asigna modelos a tareas especializadas
✅ Dashboard en tiempo real
✅ Auto-mejora continua

USO:
    python3 start_autonomous_system.py
    
    # O para background:
    nohup python3 start_autonomous_system.py > autonomous_system.log 2>&1 &

AUTOR: METACORTEX AUTONOMOUS SYSTEM
FECHA: 2025-11-26
"""

import json
import logging
import time
import signal
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import orchestrator
try:
    from autonomous_model_orchestrator import (
        get_autonomous_orchestrator,
        Task,
        ModelSpecialization,
        TaskPriority
    )
except ImportError as e:
    logger.error(f"Failed to import orchestrator: {e}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# 🎨 BEAUTIFUL DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def clear_screen():
    """Limpia pantalla."""
    print("\033[2J\033[H", end="")


def print_banner():
    """Banner del sistema."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   🤖⚡🧠 METACORTEX AUTONOMOUS MODEL ORCHESTRATOR - RUNNING ⚡🧠🤖           ║
║                                                                           ║
║   Sistema Autónomo con 956+ Modelos ML trabajando 24/7                   ║
║   Integrated with Ollama Mistral + Internet Search + World Model         ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_status_dashboard(status: Dict[str, Any]):
    """Dashboard de estado en tiempo real."""
    
    print("\n" + "=" * 79)
    print("📊 SYSTEM STATUS - Live Dashboard")
    print("=" * 79)
    
    # Estado general
    print(f"\n🟢 System Running: {'YES' if status['is_running'] else 'NO'}")
    print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Modelos
    print(f"\n🧠 Total Models Active: {status['total_models']}")
    print("\n   Models by Specialization:")
    for spec, count in status['models_by_specialization'].items():
        print(f"      • {spec:20s}: {count:4d} models")
    
    # Tareas
    print(f"\n📝 Task Queue: {status['queue_size']} pending")
    print(f"⚡ Active Tasks: {status['active_tasks']} executing")
    print(f"✅ Completed: {status['completed_tasks']}")
    print(f"❌ Failed: {status['failed_tasks']}")
    print(f"📈 Success Rate: {status['success_rate']:.1%}")
    print(f"🎲 Auto-Generated Tasks: {status['total_tasks_generated']}")
    
    # Recursos
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        print(f"\n💻 CPU Usage: {cpu:.1f}%")
        print(f"🧮 Memory: {memory.percent:.1f}% ({memory.used / 1e9:.2f}GB / {memory.total / 1e9:.2f}GB)")
    except ImportError:
        pass
    
    print("\n" + "=" * 79)
    print("Press Ctrl+C to stop")
    print("=" * 79 + "\n")


def print_task_details(orchestrator):
    """Muestra detalles de tareas activas."""
    active = orchestrator.active_tasks
    
    if active:
        print("\n🔥 ACTIVE TASKS:")
        for task_id, task in list(active.items())[:5]:  # Top 5
            elapsed = (datetime.now() - task.started_at).total_seconds()
            print(f"   • {task_id}: {task.task_type.value} ({elapsed:.1f}s)")
    
    # Últimas completadas
    recent = list(orchestrator.completed_tasks)[-5:]  # Last 5
    if recent:
        print("\n✅ RECENT COMPLETED:")
        for task in recent:
            print(f"   • {task.task_id}: {task.task_type.value} - {task.execution_time:.2f}s")


# ═══════════════════════════════════════════════════════════════════════════
# 🎯 MAIN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class AutonomousSystemRunner:
    """Runner principal del sistema autónomo."""
    
    def __init__(self):
        self.orchestrator = None
        self.is_running = False
        
        # Signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
    
    def handle_shutdown(self, signum, frame):
        """Maneja señales de shutdown."""
        logger.info("\n⚠️  Shutdown signal received")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Inicia el sistema autónomo completo."""
        logger.info("🚀 Starting Autonomous Model Orchestrator System...")
        
        try:
            # Inicializar orchestrator
            logger.info("   Initializing orchestrator...")
            self.orchestrator = get_autonomous_orchestrator(
                max_parallel_tasks=50,
                enable_auto_task_generation=True
            )
            
            self.is_running = True
            logger.info("✅ System started successfully!")
            
            # Banner inicial
            clear_screen()
            print_banner()
            
            # Main loop con dashboard
            self.run_dashboard_loop()
            
        except Exception as e:
            logger.error(f"❌ Failed to start system: {e}", exc_info=True)
            sys.exit(1)
    
    def run_dashboard_loop(self):
        """Loop principal con dashboard en tiempo real."""
        dashboard_refresh_interval = 3.0  # segundos
        
        while self.is_running:
            try:
                # Obtener estado
                status = self.orchestrator.get_status()
                
                # Actualizar dashboard
                clear_screen()
                print_banner()
                print_status_dashboard(status)
                print_task_details(self.orchestrator)
                
                # Sleep
                time.sleep(dashboard_refresh_interval)
                
            except KeyboardInterrupt:
                logger.info("\n⚠️  Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in dashboard loop: {e}")
                time.sleep(5.0)
    
    def stop(self):
        """Detiene el sistema de forma limpia."""
        logger.info("🛑 Stopping system...")
        self.is_running = False
        
        if self.orchestrator:
            self.orchestrator.shutdown()
        
        logger.info("✅ System stopped")


# ═══════════════════════════════════════════════════════════════════════════
# 🏃 ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Entry point principal."""
    
    print("\n" + "=" * 79)
    print("🤖⚡ METACORTEX AUTONOMOUS SYSTEM - STARTUP")
    print("=" * 79)
    print("\nInitializing all 956+ ML models for autonomous operation...")
    print("This will activate:")
    print("  • Autonomous Model Orchestrator")
    print("  • ML Pipeline (Military Grade v3.0)")
    print("  • Ollama Integration (7 LLM models)")
    print("  • Internet Search Engine")
    print("  • World Model (real-world interaction)")
    print("  • Auto Task Generator")
    print("\n" + "=" * 79 + "\n")
    
    # Dar tiempo para cancelar si es necesario
    print("Starting in 3 seconds... (Press Ctrl+C to cancel)")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user\n")
        sys.exit(0)
    
    # Iniciar sistema
    runner = AutonomousSystemRunner()
    runner.start()


if __name__ == "__main__":
    main()
