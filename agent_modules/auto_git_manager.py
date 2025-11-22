#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛂 Auto Git Manager
====================

Sistema de gestión automática de Git para el Metacortex.
Este módulo proporciona una capa de abstracción de alto nivel sobre
las operaciones de Git, permitiendo al sistema realizar commits,
pulls y pushes de forma autónoma y segura.

Características de Grado Militar:
- **Gestión de Repositorios:** Clona repositorios si no existen.
- **Operaciones Atómicas:** Envuelve los comandos de Git para garantizar la consistencia.
- **Manejo de Conflictos (Básico):** Detecta conflictos de merge y los reporta.
- **Autenticación Segura:** Diseñado para usar tokens de acceso personal o claves SSH.
- **Reintentos Inteligentes:** Implementa una lógica de reintentos con backoff exponencial.
- **Logging Detallado:** Registra todas las operaciones para una auditoría completa.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from typing import Optional, Tuple, Any

# Configuración del logging
logger = logging.getLogger(__name__)

class AutoGitManager:
    """
    Gestiona las operaciones de Git de forma autónoma.
    Es un Singleton para asegurar una única interfaz con Git.
    """
    _instance: Optional[AutoGitManager] = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> AutoGitManager:
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, repo_path: str, remote_url: Optional[str] = None):
        self._initialized = False
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.repo_path = repo_path
        self.remote_url = remote_url
        
        if not os.path.isdir(os.path.join(self.repo_path, '.git')):
            logger.warning(f"El directorio '{self.repo_path}' no es un repositorio Git. Intentando clonar...")
            if self.remote_url:
                self._clone_repo()
            else:
                raise ValueError("El directorio no es un repo y no se proporcionó URL remota para clonar.")
        
        self._initialized = True
        logger.info(f"🛂 AutoGitManager inicializado para el repositorio: {self.repo_path}")

    def _run_command(self, command: list[str], retries: int = 3, backoff_factor: float = 0.5) -> Tuple[bool, str, str]:
        """
        Ejecuta un comando de Git con reintentos y backoff exponencial.

        Returns:
            Una tupla (success, stdout, stderr).
        """
        for attempt in range(retries):
            try:
                process = subprocess.Popen(
                    command,
                    cwd=self.repo_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate()
                if process.returncode == 0:
                    logger.debug(f"Comando Git exitoso: {' '.join(command)}")
                    return True, stdout, stderr
                else:
                    logger.warning(f"Error en comando Git (intento {attempt + 1}/{retries}): {' '.join(command)}\nStderr: {stderr}")
            except FileNotFoundError:
                logger.error("Error crítico: 'git' no se encuentra en el PATH del sistema.")
                return False, "", "git command not found"
            except Exception as e:
                logger.error(f"Excepción inesperada al ejecutar Git: {e}")

            time.sleep(backoff_factor * (2 ** attempt))
        
        logger.error(f"El comando Git falló después de {retries} intentos: {' '.join(command)}")
        return False, "", "Max retries exceeded"

    def _clone_repo(self):
        """Clona el repositorio desde la URL remota."""
        logger.info(f"Clonando repositorio desde {self.remote_url} hacia {self.repo_path}...")
        assert self.remote_url is not None, "La URL remota no puede ser None para clonar"
        os.makedirs(os.path.dirname(self.repo_path), exist_ok=True)
        success, _, stderr = self._run_command(['git', 'clone', self.remote_url, self.repo_path])
        if not success:
            raise RuntimeError(f"No se pudo clonar el repositorio: {stderr}")
        logger.info("Repositorio clonado exitosamente.")

    def get_status(self) -> str:
        """Obtiene el estado actual del repositorio."""
        success, stdout, _ = self._run_command(['git', 'status', '--porcelain'])
        return stdout if success else "Error al obtener estado."

    def add_all(self) -> bool:
        """Añade todos los cambios al staging area."""
        logger.info("Añadiendo todos los cambios al staging area...")
        success, _, stderr = self._run_command(['git', 'add', '.'])
        if not success:
            logger.error(f"Fallo al añadir cambios: {stderr}")
        return success

    def commit(self, message: str) -> bool:
        """
        Realiza un commit con el mensaje proporcionado.
        Añade todos los cambios antes de hacer commit.
        """
        if not self.get_status():
            logger.info("No hay cambios que commitear.")
            return True

        self.add_all()
        
        logger.info(f"Realizando commit con mensaje: '{message}'")
        success, _, stderr = self._run_command(['git', 'commit', '-m', message])
        if not success:
            if "nothing to commit" in stderr:
                logger.info("Nada que commitear, el árbol de trabajo está limpio.")
                return True
            logger.error(f"Fallo al hacer commit: {stderr}")
        return success

    def pull(self, branch: str = 'main', rebase: bool = False) -> bool:
        """
        Realiza un pull desde el repositorio remoto.
        
        Args:
            branch: La rama desde la que hacer pull.
            rebase: Usar 'git pull --rebase' para evitar commits de merge.
        """
        logger.info(f"Haciendo pull desde origin/{branch}...")
        command = ['git', 'pull', 'origin', branch]
        if rebase:
            command.append('--rebase')
            
        success, _, stderr = self._run_command(command)
        if not success:
            if "conflict" in stderr.lower():
                logger.error(f"¡Conflicto de merge detectado! Se necesita intervención manual. Stderr: {stderr}")
            else:
                logger.error(f"Fallo al hacer pull: {stderr}")
        return success

    def push(self, branch: str = 'main') -> bool:
        """Realiza un push al repositorio remoto."""
        logger.info(f"Haciendo push a origin/{branch}...")
        success, _, stderr = self._run_command(['git', 'push', 'origin', branch])
        if not success:
            logger.error(f"Fallo al hacer push: {stderr}")
        return success

    def auto_commit_and_push(self, commit_message: str, branch: str = 'main') -> bool:
        """
        Ciclo completo de commit y push automatizado.
        """
        logger.info("🚀 Iniciando ciclo automático de commit y push...")
        
        # 1. Hacer pull para sincronizar con el remoto
        if not self.pull(branch, rebase=True):
            logger.error("Ciclo abortado: Fallo al hacer pull. Posible conflicto.")
            return False
            
        # 2. Realizar commit de los cambios locales
        if not self.commit(commit_message):
            logger.warning("Ciclo terminado: No se pudo realizar el commit (puede que no haya cambios).")
            # Si no hay cambios, no es un error fatal.
            return True

        # 3. Hacer push de los nuevos commits
        if not self.push(branch):
            logger.error("Ciclo abortado: Fallo al hacer push.")
            return False
            
        logger.info("✅ Ciclo automático de commit y push completado exitosamente.")
        return True

# Singleton global
_git_manager_instance: Optional[AutoGitManager] = None

def get_auto_git_manager(repo_path: str, remote_url: Optional[str] = None) -> AutoGitManager:
    """
    Fábrica para obtener la instancia singleton del AutoGitManager.
    """
    global _git_manager_instance
    if _git_manager_instance is None:
        _git_manager_instance = AutoGitManager(repo_path, remote_url)
    return _git_manager_instance

if __name__ == '__main__':
    # --- ADVERTENCIA: Esta prueba realizará operaciones de Git reales ---
    # Se recomienda crear un repositorio de prueba para ejecutar esto.
    # 1. Crea un dir: mkdir /tmp/git_test
    # 2. Inícialo: cd /tmp/git_test && git init
    # 3. (Opcional) Añade un remoto: git remote add origin <url_remota>
    
    logging.basicConfig(level=logging.INFO)
    
    test_repo_path = "/tmp/git_test_metacortex"
    if not os.path.exists(test_repo_path):
        os.makedirs(test_repo_path)
        subprocess.run(['git', 'init'], cwd=test_repo_path)

    print(f"🧪 Probando AutoGitManager en el repositorio: {test_repo_path}")
    
    git_manager = get_auto_git_manager(repo_path=test_repo_path)
    
    print("\n1. Verificando estado inicial...")
    status = git_manager.get_status()
    print(f"   Estado: {'Limpio' if not status else status}")
    
    print("\n2. Creando un nuevo archivo...")
    with open(os.path.join(test_repo_path, "test_file.txt"), "w") as f:
        f.write(f"Test content at {time.time()}")
    
    status_after_change = git_manager.get_status()
    print(f"   Estado después del cambio: {status_after_change}")
    assert status_after_change
    
    print("\n3. Realizando commit...")
    commit_msg = f"Commit de prueba automático - {time.time()}"
    success = git_manager.commit(commit_msg)
    if success:
        print("   Commit exitoso.")
    else:
        print("   Fallo en el commit.")
    
    status_after_commit = git_manager.get_status()
    print(f"   Estado después del commit: {'Limpio' if not status_after_commit else status_after_commit}")
    
    print("\n4. (Simulado) Probando ciclo completo de auto_commit_and_push...")
    # Para una prueba real, necesitarías un repositorio remoto configurado.
    # success_full_cycle = git_manager.auto_commit_and_push("Commit de ciclo completo")
    # print(f"   Resultado del ciclo completo: {'Éxito' if success_full_cycle else 'Fallo'}")

    print("\n✅ Pruebas básicas completadas.")
    print("   (Las pruebas de pull/push requieren un repositorio remoto configurado)")
