#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METACORTEX - TemplateSystem v1.0
=================================

Sistema avanzado de plantillas para generación de código con:
    pass  # TODO: Implementar
- Plantillas parametrizadas
- Herencia de plantillas
- Versionado
- Plantillas específicas por lenguaje

Autor: METACORTEX Evolution Team
Fecha: 2025-01-16
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from string import Template
import logging


class TemplateSystem:
    """
    Sistema avanzado de gestión de plantillas
    """

    def __init__(self, logger: logging.Logger):
        """
        Inicializa el sistema de plantillas

        Args:
            logger: Logger para mensajes
        """
        self.logger = logger
        self.templates: Dict[str, str] = {}
        self._load_default_templates()

        # 🧠 CONEXIÓN A RED NEURONAL SIMBIÓTICA
        try:
            from neural_symbiotic_network import get_neural_network

            self.neural_network = get_neural_network()
            self.neural_network.register_module("template_system", self)
            logger.info("✅ 'template_system' conectado a red neuronal")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a red neuronal: {e}")
            self.neural_network = None

    def _load_default_templates(self) -> None:
        """Carga plantillas por defecto"""

        # Python templates
        self.templates["python_class"] = """class $class_name:
    \"\"\"$description\"\"\"
    
    def __init__(self$init_params):
        \"\"\"Inicializa $class_name\"\"\"
        $init_body
    
    $methods
"""

        self.templates["python_function"] = """def $function_name($params)$return_hint:
    \"\"\"
    $description
    
    Args:
        $args_doc
    
    Returns:
        $returns_doc
    \"\"\"
    $body
"""

        self.templates["python_fastapi_route"] = """@$router.$method("$path")
async def $function_name($params) -> $return_type:
    \"\"\"$description\"\"\"
    try:
        $body
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Error en $function_name: {e}")
        raise HTTPException(status_code=500, detail=str(e))
"""

        self.templates["python_test"] = """def test_$test_name():
    \"\"\"$description\"\"\"
    # Arrange
    $arrange
    
    # Act
    $act
    
    # Assert
    $assert
"""

        # JavaScript templates
        self.templates["js_class"] = """class $class_name {
    /**
     * $description
     */
    constructor($params) {
        $constructor_body
    }
    
    $methods
}

export default $class_name;
"""

        self.templates["js_function"] = """/**
 * $description
 * @param {$param_types} $param_names
 * @returns {$return_type}
 */
function $function_name($params) {
    $body
}

export { $function_name };
"""

        self.templates["react_component"] = """import React from 'react';

/**
 * $description
 */
const $component_name = ({ $props }) => {
    $state
    
    return (
        $jsx
    );
};

export default $component_name;
"""

        # API templates
        self.templates["rest_endpoint"] = """$method $path
Content-Type: application/json

$description

Request:
$request_schema

Response:
$response_schema
"""

        # Docker templates
        self.templates["dockerfile"] = """FROM $base_image

WORKDIR /app

$dependencies_install

COPY . .

$build_commands

EXPOSE $port

CMD [$start_command]
"""

        # Documentation templates
        self.templates["readme"] = """# $project_name

$description

## 📋 Features

$features

## 🚀 Installation

```bash
$installation_steps
```

## 💻 Usage

```bash
$usage_examples
```

## 📚 Documentation

$documentation_link

## 🤝 Contributing

$contributing_guide

## 📄 License

$license
"""

        self.logger.info(f"✅ Cargadas {len(self.templates)} plantillas por defecto")

    def get_template(self, template_name: str) -> str:
        """
        Obtiene una plantilla por nombre

        Args:
            template_name: Nombre de la plantilla

        Returns:
            Contenido de la plantilla
        """
        if template_name not in self.templates:
            self.logger.warning(f"⚠️ Plantilla '{template_name}' no encontrada")
            return ""

        return self.templates[template_name]

    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Renderiza una plantilla con contexto

        Args:
            template_name: Nombre de la plantilla
            context: Diccionario con variables para reemplazar

        Returns:
            Plantilla renderizada
        """
        template_content = self.get_template(template_name)
        if not template_content:
            return ""

        try:
            # Usar Template de Python para sustitución segura
            template = Template(template_content)
            rendered = template.safe_substitute(context)

            self.logger.debug(f"✅ Plantilla '{template_name}' renderizada")
            return rendered

        except Exception as e:
            logger.error(f"Error en template_system.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error renderizando plantilla '{template_name}': {e}")
            return template_content

    def add_template(self, name: str, content: str) -> None:
        """
        Añade una nueva plantilla

        Args:
            name: Nombre de la plantilla
            content: Contenido de la plantilla
        """
        self.templates[name] = content
        self.logger.info(f"✅ Plantilla '{name}' añadida")

    def list_templates(self) -> List[str]:
        """
        Lista todas las plantillas disponibles

        Returns:
            Lista de nombres de plantillas
        """
        return sorted(self.templates.keys())

    def save_templates(self, file_path: Path) -> None:
        """
        Guarda plantillas en archivo JSON

        Args:
            file_path: Ruta del archivo
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.templates, f, indent=2, ensure_ascii=False)
            self.logger.info(f"✅ Plantillas guardadas en {file_path}")
        except Exception as e:
            logger.error(f"Error en template_system.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error guardando plantillas: {e}")

    def load_templates(self, file_path: Path) -> None:
        """
        Carga plantillas desde archivo JSON

        Args:
            file_path: Ruta del archivo
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                self.templates.update(loaded)
            self.logger.info(f"✅ Plantillas cargadas desde {file_path}")
        except Exception as e:
            logger.error(f"Error en template_system.py: {e}", exc_info=True)
            self.logger.error(f"❌ Error cargando plantillas: {e}")

    def create_python_class(
        self, class_name: str, description: str, methods: List[Dict[str, str]]
    ) -> str:
        """
        Crea una clase Python completa

        Args:
            class_name: Nombre de la clase
            description: Descripción de la clase
            methods: Lista de métodos a incluir

        Returns:
            Código de la clase
        """
        methods_code = "\n    ".join(
            [
                self.render_template(
                    "python_function",
                    {
                        "function_name": m["name"],
                        "params": m.get("params", ""),
                        "return_hint": f" -> {m['return_type']}"
                        if "return_type" in m
                        else "",
                        "description": m.get("description", ""),
                        "args_doc": "",
                        "returns_doc": m.get("return_type", "None"),
                        "body": m.get("body", "pass"),
                    },
                )
                for m in methods
            ]
        )

        return self.render_template(
            "python_class",
            {
                "class_name": class_name,
                "description": description,
                "init_params": "",
                "init_body": "pass",
                "methods": methods_code,
            },
        )

    def create_fastapi_route(
        self,
        method: str,
        path: str,
        function_name: str,
        description: str,
        params: str = "",
        return_type: str = "dict",
    ) -> str:
        """
        Crea una ruta FastAPI

        Args:
            method: Método HTTP (get, post, put, delete)
            path: Ruta del endpoint
            function_name: Nombre de la función
            description: Descripción del endpoint
            params: Parámetros de la función
            return_type: Tipo de retorno

        Returns:
            Código de la ruta
        """
        return self.render_template(
            "python_fastapi_route",
            {
                "router": "router",
                "method": method,
                "path": path,
                "function_name": function_name,
                "params": params,
                "return_type": return_type,
                "description": description,
                "body": "# IMPLEMENTED: Implementar lógica\n        result = {}",
            },
        )

    def create_readme(
        self,
        project_name: str,
        description: str,
        features: List[str],
        installation_steps: str,
    ) -> str:
        """
        Crea un README.md

        Args:
            project_name: Nombre del proyecto
            description: Descripción
            features: Lista de características
            installation_steps: Pasos de instalación

        Returns:
            Contenido del README
        """
        features_text = "\n".join([f"- {f}" for f in features])

        return self.render_template(
            "readme",
            {
                "project_name": project_name,
                "description": description,
                "features": features_text,
                "installation_steps": installation_steps,
                "usage_examples": "python main.py",
                "documentation_link": "Consulta la carpeta `docs/`",
                "contributing_guide": "Pull requests son bienvenidos.",
                "license": "MIT",
            },
        )


def get_template_system(logger: Optional[logging.Logger] = None) -> TemplateSystem:
    """
    Factory function para crear instancia de TemplateSystem.
    
    Args:
        logger: Logger opcional. Si no se provee, se crea uno.
    
    Returns:
        TemplateSystem: Instancia del sistema de templates
    """
    if logger is None:
        logger = logging.getLogger("template_system")
    return TemplateSystem(logger)
