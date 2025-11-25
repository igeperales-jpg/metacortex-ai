#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metacortex Unificado - Punto de Entrada Principal
=================================================

Orquestador minimalista que inicializa el Agente Cognitivo y
le proporciona sus capacidades para actuar en el mundo.

Arquitectura:
1.  **main.py (Este archivo):** Carga el agente cognitivo y el gestor de capacidades.
2.  **CognitiveAgent:** El cerebro. Procesa percepciones y toma decisiones.
3.  **CapabilityManager:** Descubre y carga las "herramientas" del agente (Programación, Búsqueda, etc.).
4.  **Capabilities (Herramientas):** Módulos especializados que el agente usa para actuar.
"""
import time
import logging
import asyncio
from pathlib import Path
from metacortex_sinaptico.core import get_cognitive_agent
from capability_manager import CapabilityManager

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("metacortex_main.log"),
    ],
)
logger = logging.getLogger(__name__)


async def main():
    """Función principal asíncrona."""
    logger.info("🚀 Iniciando Metacortex Unificado...")

    try:
        # 1. Definir rutas y inicializar el Gestor de Capacidades
        logger.info("🔧 Descubriendo y cargando capacidades del agente...")
        
        # Rutas donde se buscarán los módulos de capacidades.
        project_root = Path(__file__).parent
        capability_paths = [
            project_root,
            project_root / "agent_modules",
        ]
        
        capability_manager = CapabilityManager(capability_paths=capability_paths)
        capabilities = capability_manager.discover_and_initialize_capabilities()
        logger.info(f"✅ {len(capabilities)} capacidades cargadas: {list(capabilities.keys())}")

        # 2. Obtener la instancia del Agente Cognitivo (Singleton)
        logger.info("🧠 Inicializando el núcleo cognitivo...")
        cognitive_agent = get_cognitive_agent()

        # 3. Proporcionar las capacidades al agente
        cognitive_agent.set_capabilities(capabilities)
        logger.info("✅ Capacidades inyectadas en el núcleo cognitivo.")

        # 4. Iniciar el ciclo de vida del agente
        logger.info("✨ El agente cognitivo ha tomado el control. Iniciando ciclo de vida...")
        
        # Bucle principal asíncrono
        tick_interval = 30  # Intervalo entre ciclos en segundos
        while True:
            logger.info("-------------------")
            logger.info(f"⏳ Iniciando nuevo ciclo cognitivo (Tick #{cognitive_agent.tick_count + 1})")
            
            # Ejecutar un ciclo del agente
            cognitive_agent.tick()

            # Simular percepción del entorno (esto sería reemplazado por percepciones reales)
            cognitive_agent.perceive("system_heartbeat", {"timestamp": time.time()})

            logger.info(f"💡 Intención actual: {cognitive_agent.get_current_state().get('current_intention')}")
            logger.info(f"😊 Bienestar: {cognitive_agent.get_current_state().get('wellbeing'):.2f}")
            
            await asyncio.sleep(tick_interval)

    except Exception as e:
        logger.critical(f"❌ Error fatal en el bucle principal: {e}", exc_info=True)
    finally:
        logger.info("🛑 Metacortex Unificado ha sido detenido.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Detención solicitada por el usuario.")

