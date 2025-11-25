#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Disk Space Manager Module (Military-Grade)
============================================

Monitorea, analiza y gestiona proactivamente el espacio en disco para
garantizar la operatividad continua del sistema.
"""

import shutil
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
import zipfile
from datetime import datetime, timedelta

try:
    # Asumiendo que unified_logging está en la ruta del proyecto
    from unified_logging import get_logger
except ImportError:
    # Fallback si la importación directa falla
    def get_logger(name: str = "DefaultLogger") -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

class DiskSpaceManager:
    """
    💾 Gestor de Espacio en Disco (Grado Militar)

    Implementa un sistema robusto para el monitoreo y la gestión del espacio
    de almacenamiento, con estrategias de limpieza configurables.
    """

    def __init__(
        self,
        paths_to_monitor: List[str],
        warning_threshold: float = 80.0,
        critical_threshold: float = 95.0,
        check_interval_seconds: int = 3600,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Inicializa el gestor de espacio en disco.

        Args:
            paths_to_monitor: Lista de rutas de directorios a monitorear.
            warning_threshold: Porcentaje de uso para lanzar una advertencia.
            critical_threshold: Porcentaje de uso para iniciar la limpieza.
            check_interval_seconds: Intervalo en segundos para las verificaciones en segundo plano.
            logger: Instancia del logger unificado.
        """
        self.paths_to_monitor = [Path(p).resolve() for p in paths_to_monitor]
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval_seconds
        self.logger = logger or get_logger(__name__)
        
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        self.logger.info(f"💾 DiskSpaceManager (Grado Militar) inicializado. Umbrales: {warning_threshold}% (warn), {critical_threshold}% (crit).")

    def get_disk_usage(self, path: Path) -> Optional[Dict[str, Any]]:
        """
        Obtiene el uso del disco para la partición donde reside una ruta.

        Args:
            path: La ruta para la cual verificar el uso del disco.

        Returns:
            Un diccionario con el total, usado y espacio libre en GB, y el porcentaje de uso.
        """
        try:
            total, used, free = shutil.disk_usage(path)
            usage_gb = {
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
                "percent_used": round((used / total) * 100, 2),
            }
            return usage_gb
        except FileNotFoundError:
            self.logger.error(f"La ruta '{path}' no existe. No se puede verificar el uso del disco.")
            return None
        except Exception as e:
            self.logger.error(f"Error al obtener el uso del disco para '{path}': {e}", exc_info=True)
            return None

    def run_check(self):
        """
        Ejecuta una verificación única del espacio en disco y aplica acciones si es necesario.
        """
        self.logger.info("Running periodic disk space check...")
        unique_partitions = {p.anchor for p in self.paths_to_monitor}

        for partition_path in unique_partitions:
            usage = self.get_disk_usage(Path(partition_path))
            if not usage:
                continue

            percent_used = usage["percent_used"]
            self.logger.info(f"Partición '{partition_path}': {percent_used}% usado. ({usage['free_gb']} GB libres)")

            if percent_used >= self.critical_threshold:
                self.logger.critical(
                    f"¡NIVEL CRÍTICO! Uso del disco en {percent_used}%, superando el umbral de {self.critical_threshold}%. "
                    "Iniciando estrategias de limpieza."
                )
                self.run_cleanup_strategies()
            elif percent_used >= self.warning_threshold:
                self.logger.warning(
                    f"ADVERTENCIA: Uso del disco en {percent_used}%, superando el umbral de {self.warning_threshold}%."
                )

    def run_cleanup_strategies(self):
        """
        Ejecuta una serie de estrategias de limpieza para liberar espacio.
        """
        self.logger.info("Ejecutando estrategias de limpieza de disco...")
        
        # Estrategia 1: Limpiar logs antiguos (más de 7 días)
        self._cleanup_old_files(days_old=7, patterns=["*.log"], subfolder="logs")

        # Estrategia 2: Limpiar cachés de Python
        self._cleanup_pycache()

        # Estrategia 3: Comprimir artefactos grandes (archivos > 100MB, más de 30 días)
        self._compress_large_files(days_old=30, min_size_mb=100)

        self.logger.info("Estrategias de limpieza finalizadas.")

    def _cleanup_old_files(self, days_old: int, patterns: List[str], subfolder: str):
        """Elimina archivos antiguos que coinciden con un patrón en un subdirectorio."""
        self.logger.info(f"Buscando archivos con patrones {patterns} más antiguos de {days_old} días en subdirectorios '{subfolder}'...")
        cutoff_date = datetime.now() - timedelta(days=days_old)
        files_deleted_count = 0
        space_freed_mb = 0.0

        for base_path in self.paths_to_monitor:
            target_dir = base_path / subfolder
            if not target_dir.is_dir():
                continue
            
            for pattern in patterns:
                for file_path in target_dir.rglob(pattern):
                    try:
                        file_mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_mod_time < cutoff_date:
                            file_size = file_path.stat().st_size
                            self.logger.info(f"Eliminando archivo antiguo: {file_path}")
                            file_path.unlink()
                            files_deleted_count += 1
                            space_freed_mb += file_size / (1024 * 1024)
                    except Exception as e:
                        self.logger.error(f"No se pudo eliminar el archivo {file_path}: {e}")
        
        if files_deleted_count > 0:
            self.logger.info(f"Se eliminaron {files_deleted_count} archivos, liberando {space_freed_mb:.2f} MB.")

    def _cleanup_pycache(self):
        """Elimina todos los directorios __pycache__."""
        self.logger.info("Limpiando directorios __pycache__...")
        dirs_deleted_count = 0
        for base_path in self.paths_to_monitor:
            for path in base_path.rglob("__pycache__"):
                if path.is_dir():
                    try:
                        self.logger.info(f"Eliminando directorio de caché: {path}")
                        shutil.rmtree(path)
                        dirs_deleted_count += 1
                    except Exception as e:
                        self.logger.error(f"No se pudo eliminar el directorio {path}: {e}")
        if dirs_deleted_count > 0:
            self.logger.info(f"Se eliminaron {dirs_deleted_count} directorios __pycache__.")

    def _compress_large_files(self, days_old: int, min_size_mb: int):
        """Comprime archivos grandes y antiguos."""
        self.logger.info(f"Buscando archivos de más de {min_size_mb}MB y más antiguos de {days_old} días para comprimir...")
        cutoff_date = datetime.now() - timedelta(days=days_old)
        min_size_bytes = min_size_mb * 1024 * 1024
        files_compressed_count = 0

        for base_path in self.paths_to_monitor:
            for file_path in base_path.rglob("*"):
                if not file_path.is_file() or file_path.suffix == '.zip':
                    continue
                try:
                    file_stat = file_path.stat()
                    file_mod_time = datetime.fromtimestamp(file_stat.st_mtime)
                    if file_mod_time < cutoff_date and file_stat.st_size > min_size_bytes:
                        self.logger.info(f"Comprimiendo archivo grande: {file_path}")
                        zip_path = file_path.with_suffix(file_path.suffix + '.zip')
                        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                            zf.write(file_path, file_path.name)
                        
                        # Si la compresión fue exitosa, eliminar el original
                        file_path.unlink()
                        files_compressed_count += 1
                except Exception as e:
                    self.logger.error(f"No se pudo comprimir el archivo {file_path}: {e}")
        
        if files_compressed_count > 0:
            self.logger.info(f"Se comprimieron {files_compressed_count} archivos grandes.")

    def start_monitoring(self):
        """Inicia el monitoreo en segundo plano."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self.logger.warning("El monitoreo en segundo plano ya está en ejecución.")
            return

        self.logger.info(f"Iniciando monitoreo de disco en segundo plano (intervalo: {self.check_interval}s).")
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Detiene el monitoreo en segundo plano."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self.logger.info("Deteniendo el monitoreo de disco en segundo plano.")
            self._stop_event.set()
            self._monitor_thread.join(timeout=5)
            if self._monitor_thread.is_alive():
                self.logger.error("El hilo de monitoreo no se detuvo correctamente.")
        else:
            self.logger.info("El monitoreo en segundo plano no estaba en ejecución.")

    def _monitor_loop(self):
        """El bucle principal para el monitoreo en segundo plano."""
        while not self._stop_event.is_set():
            self.run_check()
            self._stop_event.wait(self.check_interval)

# --- Singleton Factory ---
_disk_space_manager_instance: Optional[DiskSpaceManager] = None

def get_disk_space_manager(
    paths_to_monitor: List[str],
    force_new: bool = False,
    **kwargs: Any
) -> DiskSpaceManager:
    """
    Factory para obtener una instancia del DiskSpaceManager.

    Args:
        paths_to_monitor: Lista de rutas a monitorear (requerido).
        force_new: Si es True, crea una nueva instancia.
        **kwargs: Argumentos adicionales para el constructor de DiskSpaceManager.

    Returns:
        Una instancia de DiskSpaceManager.
    """
    global _disk_space_manager_instance
    if _disk_space_manager_instance is None or force_new:
        _disk_space_manager_instance = DiskSpaceManager(
            paths_to_monitor=paths_to_monitor,
            **kwargs
        )
    return _disk_space_manager_instance

if __name__ == '__main__':
    print("Ejecutando DiskSpaceManager en modo de prueba...")
    
    # Obtener la ruta del proyecto actual
    project_root = Path(__file__).resolve().parent.parent
    
    # Rutas a monitorear para la prueba
    test_paths = [str(project_root), str(project_root / "logs"), str(project_root / "chromadb")]
    
    print(f"Monitoreando rutas: {test_paths}")
    
    # Crear una instancia del gestor con un intervalo corto para la prueba
    manager = get_disk_space_manager(
        paths_to_monitor=test_paths,
        warning_threshold=1.0,  # Umbral bajo para forzar la activación
        critical_threshold=2.0, # Umbral bajo para forzar la activación
        check_interval_seconds=10,
        force_new=True
    )
    
    # Ejecutar una verificación manual
    manager.run_check()
    
    print("\nPrueba finalizada. El gestor de espacio en disco está diseñado para ser importado como un módulo.")
