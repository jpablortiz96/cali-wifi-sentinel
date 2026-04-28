from __future__ import annotations

import os

from dotenv import load_dotenv

from src.config import PROJECT_ROOT

try:
    from google import genai
except ImportError:  # pragma: no cover - proteccion simple para entorno incompleto
    genai = None


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
NOT_CONFIGURED_MESSAGE = (
    "Gemini API no esta configurada. Crea un archivo .env con GEMINI_API_KEY."
)


def load_gemini_config() -> dict[str, str]:
    """Carga configuracion de Gemini desde .env y variables de entorno."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL

    return {
        "api_key": api_key,
        "model": model,
    }


def is_gemini_configured() -> bool:
    """Indica si existe una API key valida en el entorno local."""
    return bool(load_gemini_config()["api_key"])


def generate_gemini_text(prompt: str) -> str:
    """Genera texto con Gemini usando solo un prompt textual."""
    config = load_gemini_config()
    api_key = config["api_key"]
    model = config["model"]

    if not api_key:
        return NOT_CONFIGURED_MESSAGE

    if genai is None:
        return (
            "La libreria google-genai no esta instalada. Ejecuta "
            "`pip install -r requirements.txt`."
        )

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        generated_text = getattr(response, "text", "") or ""
        generated_text = generated_text.strip()

        if generated_text:
            return generated_text

        return "Gemini no devolvio contenido en texto para este analisis."
    except Exception as error:  # noqa: BLE001
        return (
            "No fue posible generar el analisis con Gemini. Revisa la API key, "
            f"el modelo configurado o tu conexion. Detalle: {error}"
        )
    finally:
        close_method = getattr(client, "close", None)
        if callable(close_method):
            close_method()
