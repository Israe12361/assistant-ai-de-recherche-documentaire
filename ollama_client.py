from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class OllamaClient:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.2:3b"
    timeout_seconds: float = 150.0
    num_predict: int = 450
    temperature: float = 0.1

    def _tags_response(self) -> requests.Response:
        try:
            response = requests.get(
                f"{self.base_url.rstrip('/')}/api/tags",
                timeout=5,
            )
            response.raise_for_status()
            return response
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Ollama n'est pas démarré ou n'est pas accessible à "
                f"{self.base_url}. Lancez Ollama, puis vérifiez avec :\n"
                "  ollama list"
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Ollama ne répond pas dans le délai prévu à {self.base_url}."
            ) from exc
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"L'adresse Ollama {self.base_url} a renvoyé HTTP {exc.response.status_code}."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Impossible de contacter Ollama à {self.base_url}.") from exc

    def diagnose(self) -> tuple[bool, list[str], str]:
        try:
            response = self._tags_response()
            models = response.json().get("models", [])
            names = [str(item.get("name")) for item in models if item.get("name")]
            if self.model not in names:
                return True, names, (
                    f"Le modèle '{self.model}' n'est pas installé. "
                    "Installez-le avec : ollama pull " + self.model
                )
            return True, names, f"Ollama est disponible avec le modèle '{self.model}'."
        except (RuntimeError, ValueError, TypeError) as exc:
            return False, [], str(exc)

    def list_models(self) -> list[str]:
        available, names, _message = self.diagnose()
        return names if available else []

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        available, names, message = self.diagnose()
        if not available:
            raise RuntimeError(message)
        if self.model not in names:
            raise RuntimeError(
                f"Le modèle Ollama '{self.model}' n'est pas installé. "
                f"Modèles disponibles : {', '.join(names) or 'aucun'}.\n"
                f"Commande : ollama pull {self.model}"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "num_ctx": 4096,
                "keep_alive": "10m",
                "repeat_penalty": 1.05,
            },
            "stop": ["\n\n\n"],
        }
        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"La connexion à Ollama a été interrompue pendant la génération à {self.base_url}."
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError(
                f"La génération Ollama a dépassé le délai de {self.timeout_seconds:.0f} secondes."
            ) from exc
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Ollama a refusé la génération (HTTP {exc.response.status_code}). "
                "Vérifiez le nom du modèle et son installation."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError("La requête vers Ollama a échoué.") from exc
        try:
            data = response.json()
            content = data.get("message", {}).get("content", "")
        except (ValueError, TypeError, AttributeError) as exc:
            raise RuntimeError("Ollama a renvoyé une réponse JSON invalide.") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama a renvoyé une réponse vide ou invalide.")
        return content.strip()
