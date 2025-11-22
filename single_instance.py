import sys
import os
import fcntl
import tempfile
import logging

logger = logging.getLogger(__name__)

class SingleInstanceException(BaseException):
    """Excepción personalizada para errores de instancia única."""
    pass

class SingleInstance:
    """
    Asegura que solo una instancia de la aplicación se esté ejecutando a la vez
    utilizando un archivo de bloqueo.

    Este mecanismo es robusto y funciona en sistemas operativos tipo Unix (como macOS y Linux).
    Crea un archivo de bloqueo en el directorio temporal del sistema. Si el archivo ya está
    bloqueado por otra instancia, se lanza una `SingleInstanceException`.

    Uso como gestor de contexto (recomendado):
    
    try:
        with SingleInstance():
            # Tu código de aplicación aquí
            ...
    except SingleInstanceException:
        logger.error("Ya hay otra instancia de la aplicación en ejecución.")
        sys.exit(1)

    """

    def __init__(self, flavor_id="default"):
        # Generar un nombre de archivo de bloqueo único basado en la ruta del script
        script_path = os.path.abspath(sys.argv[0])
        # Reemplazar caracteres no válidos para nombres de archivo
        safe_path = script_path.replace('/', '_').replace('\\', '_').replace(':', '')
        lock_filename = f"metacortex_{safe_path}_{flavor_id}.lock"
        
        self.lockfile = os.path.join(tempfile.gettempdir(), lock_filename)
        self.fp = None
        self._lock_acquired = False

        logger.debug(f"Intentando adquirir bloqueo en: {self.lockfile}")
        self.fp = open(self.lockfile, 'w')
        try:
            # Intentar adquirir un bloqueo exclusivo sin bloquear el proceso
            fcntl.lockf(self.fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_acquired = True
            logger.info("Bloqueo de instancia única adquirido con éxito.")
        except IOError:
            self.fp.close()
            self.fp = None
            logger.warning(f"No se pudo adquirir el bloqueo. Otra instancia podría estar en ejecución.")
            raise SingleInstanceException("Otra instancia ya está en ejecución.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def release(self):
        """Libera el archivo de bloqueo."""
        if not self._lock_acquired or not self.fp:
            return
        
        try:
            # Liberar el bloqueo
            fcntl.lockf(self.fp, fcntl.LOCK_UN)
            self.fp.close()
            self.fp = None
            # Eliminar el archivo de bloqueo
            os.remove(self.lockfile)
            self._lock_acquired = False
            logger.info("Bloqueo de instancia única liberado.")
        except Exception as e:
            logger.error(f"Error al liberar el bloqueo de instancia única: {e}", exc_info=True)

def ensure_single_instance(flavor_id="default"):
    """
    Función de fábrica para crear y devolver un objeto SingleInstance.
    Esta es la función que será importada por otros módulos.
    
    Lanza SingleInstanceException si no se puede adquirir el bloqueo.
    """
    return SingleInstance(flavor_id=flavor_id)
