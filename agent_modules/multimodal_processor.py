#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Modal Processor Module (Military-Grade)
=============================================

Módulo avanzado para el procesamiento unificado de datos multimodales
(texto, imágenes, audio) utilizando modelos de lenguaje y visión.
"""

import base64
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

# --- Configuración de Ruta para Importaciones ---
import sys
try:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except IndexError:
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
# --- Fin Configuración de Ruta ---

# --- Unified Robust Imports ---
try:
    from PIL import Image
    import librosa
    import soundfile as sf
    from unified_logging import get_logger
    from agent_modules.llm_integration import get_llm_integration, LLMIntegration
    from agent_modules.telemetry_system import get_telemetry_system, TelemetrySystem
    from agent_modules.advanced_cache_system import get_cache_system, AdvancedCacheSystem

    imports_were_successful = True
except ImportError:
    imports_were_successful = False
    print("ADVERTENCIA: Módulos críticos no encontrados. MultiModalProcessor funcionará en modo degradado.")

    # --- Fallbacks ---
    def get_logger(name: str = "DefaultLogger") -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    class _DummyMetric:
        def labels(self, *args: Any, **kwargs: Any) -> '_DummyMetric': return self
        def observe(self, *args: Any, **kwargs: Any) -> None: pass
        def inc(self, *args: Any, **kwargs: Any) -> None: pass

    class _DummyTelemetry:
        multimodal_processing_latency = _DummyMetric()
        multimodal_requests_total = _DummyMetric()
        multimodal_failures_total = _DummyMetric()

    TelemetrySystem = _DummyTelemetry # type: ignore

    def get_telemetry_system(force_new: bool = False, **kwargs: Any) -> Any:
        return _DummyTelemetry()

    class _DummyLLM:
        def query_vision_model(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {"success": False, "error": "LLM not available"}

    LLMIntegration = _DummyLLM # type: ignore
    
    def get_llm_integration(**kwargs: Any) -> Any:
        return _DummyLLM()

    class _DummyCache:
        def get(self, key: str) -> Optional[Any]: return None
        def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None: pass

    AdvancedCacheSystem = _DummyCache # type: ignore

    def get_cache_system(**kwargs: Any) -> Any:
        return _DummyCache()

    # Fallbacks for optional libraries
    Image = None
    librosa = None
    sf = None

class MultiModalProcessor:
    """
    🧠 Procesador Multi-Modal (Grado Militar)

    Orquesta el análisis y la extracción de información de diversas
    fuentes de datos.
    """
    llm: "LLMIntegration"
    telemetry: "TelemetrySystem"
    cache: "AdvancedCacheSystem"

    def __init__(self, cache_enabled: bool = True, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.llm = get_llm_integration()
        self.telemetry = get_telemetry_system()
        self.cache_enabled = cache_enabled
        if self.cache_enabled:
            self.cache = get_cache_system(namespace="multimodal")
        
        self.logger.info("🧠 MultiModalProcessor (Grado Militar) inicializado.")
        if not imports_were_successful:
            self.logger.warning("Dependencias opcionales no encontradas. La funcionalidad estará limitada.")

    def _generate_cache_key(self, data: bytes) -> str:
        """Genera una clave de caché SHA256 para un bloque de datos."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _encode_image_to_base64(image_path: Path) -> str:
        """Codifica una imagen a base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def process_image(self, image_path: Path, prompt: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Analiza una imagen utilizando un modelo de visión.
        """
        start_time = time.time()
        self.telemetry.multimodal_requests_total.labels(type='image').inc()

        if not Image:
            self.logger.error("Pillow no está instalado. No se puede procesar la imagen.")
            self.telemetry.multimodal_failures_total.labels(type='image', reason='dependency_missing').inc()
            return {"success": False, "error": "Pillow library not found."}

        if not image_path.exists():
            self.logger.error(f"El archivo de imagen no existe: {image_path}")
            self.telemetry.multimodal_failures_total.labels(type='image', reason='file_not_found').inc()
            return {"success": False, "error": "Image file not found."}

        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            cache_key = self._generate_cache_key(image_data + prompt.encode('utf-8'))
            
            if self.cache_enabled and not force_reprocess:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.logger.info(f"Resultado para imagen '{image_path.name}' y prompt obtenido de caché.")
                    return cached_result

            # Codificar la imagen y enviarla al LLM
            encoded_image = self._encode_image_to_base64(image_path)
            result = self.llm.query_vision_model(
                prompt=prompt,
                images_base64=[encoded_image]
            )

            if self.cache_enabled and result.get("success"):
                self.cache.set(cache_key, result, ttl=3600) # Cache por 1 hora

            return result

        except Exception as e:
            self.logger.error(f"Error procesando imagen '{image_path}': {e}", exc_info=True)
            self.telemetry.multimodal_failures_total.labels(type='image', reason='processing_error').inc()
            return {"success": False, "error": str(e)}
        finally:
            latency = time.time() - start_time
            self.telemetry.multimodal_processing_latency.labels(type='image').observe(latency)

    def process_audio(self, audio_path: Path, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Analiza un archivo de audio, extrayendo características básicas.
        """
        start_time = time.time()
        self.telemetry.multimodal_requests_total.labels(type='audio').inc()

        if not librosa or not sf:
            self.logger.error("Librosa o SoundFile no están instalados. No se puede procesar audio.")
            self.telemetry.multimodal_failures_total.labels(type='audio', reason='dependency_missing').inc()
            return {"success": False, "error": "Audio processing libraries not found."}

        if not audio_path.exists():
            self.logger.error(f"El archivo de audio no existe: {audio_path}")
            self.telemetry.multimodal_failures_total.labels(type='audio', reason='file_not_found').inc()
            return {"success": False, "error": "Audio file not found."}

        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            cache_key = self._generate_cache_key(audio_data)

            if self.cache_enabled and not force_reprocess:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.logger.info(f"Resultado para audio '{audio_path.name}' obtenido de caché.")
                    return cached_result

            y, sr = librosa.load(audio_path)
            
            result = {
                "success": True,
                "duration_seconds": librosa.get_duration(y=y, sr=sr),
                "sample_rate": sr,
                "channels": y.ndim,
                "tempo_bpm": librosa.beat.tempo(y=y, sr=sr)[0],
            }

            if self.cache_enabled:
                self.cache.set(cache_key, result, ttl=3600)

            return result

        except Exception as e:
            self.logger.error(f"Error procesando audio '{audio_path}': {e}", exc_info=True)
            self.telemetry.multimodal_failures_total.labels(type='audio', reason='processing_error').inc()
            return {"success": False, "error": str(e)}
        finally:
            latency = time.time() - start_time
            self.telemetry.multimodal_processing_latency.labels(type='audio').observe(latency)

# --- Singleton Factory ---
_multimodal_processor_instance: Optional[MultiModalProcessor] = None

def get_multimodal_processor(force_new: bool = False, **kwargs: Any) -> MultiModalProcessor:
    """
    Factory para obtener una instancia de MultiModalProcessor.
    """
    global _multimodal_processor_instance
    if _multimodal_processor_instance is None or force_new:
        _multimodal_processor_instance = MultiModalProcessor(**kwargs)
    return _multimodal_processor_instance

if __name__ == '__main__':
    print("Ejecutando MultiModalProcessor en modo de prueba...")
    
    # Crear archivos de prueba si no existen
    test_dir = Path("./test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Crear imagen de prueba
    img_path: Optional[Path] = None
    try:
        from PIL import Image
        img_path = test_dir / "test_image.png"
        if not img_path.exists():
            Image.new('RGB', (100, 100), color = 'red').save(img_path)
            print(f"Imagen de prueba creada en: {img_path}")
    except ImportError:
        print("Pillow no instalado, no se puede crear o probar imagen.")

    # Crear audio de prueba
    audio_path: Optional[Path] = None
    try:
        import numpy as np
        import soundfile as sf
        audio_path = test_dir / "test_audio.wav"
        if not audio_path.exists():
            sr = 22050
            duration = 1
            t = np.linspace(0., duration, int(sr * duration))
            amplitude = np.iinfo(np.int16).max * 0.5
            data = amplitude * np.sin(2. * np.pi * 440. * t)
            sf.write(audio_path, data.astype(np.int16), sr)
            print(f"Audio de prueba creado en: {audio_path}")
    except (ImportError, NameError):
        print("Numpy o Soundfile no instalados, no se puede crear o probar audio.")

    processor = get_multimodal_processor(force_new=True)

    if not imports_were_successful:
        print("\nADVERTENCIA: Ejecutando en modo degradado. Las pruebas serán limitadas.")

    # --- Prueba de Imagen ---
    if img_path:
        print("\n--- Prueba 1: Procesamiento de Imagen ---")
        prompt = "Describe esta imagen."
        result = processor.process_image(img_path, prompt)
        print(f"Resultado del procesamiento de imagen: {result}")
        if not result.get("success"):
            print("NOTA: El fallo puede ser esperado si el LLM de visión no está configurado.")

    # --- Prueba de Audio ---
    if audio_path:
        print("\n--- Prueba 2: Procesamiento de Audio ---")
        result = processor.process_audio(audio_path)
        print(f"Resultado del procesamiento de audio: {result}")

    print("\nPrueba finalizada.")
