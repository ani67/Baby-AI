import httpx
import time
from dataclasses import dataclass
from .prompts import build_prompt


class TeacherError(Exception):
    """Base class for all teacher bridge errors."""


class TeacherUnavailableError(TeacherError):
    """Ollama is not running or not reachable at the configured host."""


class TeacherTimeoutError(TeacherError):
    """The model took longer than TEACHER_TIMEOUT seconds to respond."""


class TeacherModelNotFoundError(TeacherError):
    """The configured model is not pulled in Ollama."""


@dataclass
class TeacherResponse:
    answer: str
    model: str
    duration_ms: int
    tokens_used: int
    truncated: bool


class TeacherBridge:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llava",
        max_tokens: int = 200,
        temperature: float = 0.3,
        timeout: float = 30.0,
    ):
        self.host = host
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def ask(
        self,
        question: str,
        stage: int,
        context: str | None = None,
        image_bytes: bytes | None = None,
    ) -> TeacherResponse | None:
        prompt = build_prompt(question, context, stage)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
                "stop": ["\n\n", "###"],
            },
        }
        if image_bytes is not None:
            import base64
            payload["images"] = [base64.b64encode(image_bytes).decode("ascii")]
        t0 = time.monotonic()
        try:
            response = await self._client.post(
                f"{self.host}/api/generate",
                json=payload,
            )
            response.raise_for_status()
        except httpx.ConnectError:
            raise TeacherUnavailableError(
                f"Cannot reach Ollama at {self.host}"
            )
        except httpx.TimeoutException:
            raise TeacherTimeoutError(
                f"Model '{self.model}' timed out after {self.timeout}s"
            )

        duration_ms = int((time.monotonic() - t0) * 1000)
        data = response.json()

        answer = data["response"].strip()

        # Repetition check — skip if any word appears more than 5 times
        words = answer.lower().split()
        if words:
            from collections import Counter
            counts = Counter(words)
            most_common_word, most_common_count = counts.most_common(1)[0]
            if most_common_count > 5:
                print(f"[teacher] repetition detected (\"{most_common_word}\" x{most_common_count}), skipping step", flush=True)
                return None

            # Sequence repetition check — any 10-word chunk appearing more than once
            if len(words) >= 20:
                chunks = Counter()
                for i in range(len(words) - 9):
                    chunk = " ".join(words[i:i + 10])
                    chunks[chunk] += 1
                    if chunks[chunk] > 1:
                        print(f"[teacher] sequence repetition detected, skipping step", flush=True)
                        return None

        return TeacherResponse(
            answer=answer,
            model=data.get("model", self.model),
            duration_ms=duration_ms,
            tokens_used=data.get("eval_count", 0)
            + data.get("prompt_eval_count", 0),
            truncated=data.get("done_reason") == "length",
        )

    async def health_check(self) -> bool:
        try:
            response = await self._client.get(
                f"{self.host}/api/tags",
                timeout=5.0,
            )
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False

    async def list_available_models(self) -> list[str]:
        try:
            response = await self._client.get(
                f"{self.host}/api/tags",
                timeout=5.0,
            )
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except Exception:
            return []

    async def close(self):
        """Call on backend shutdown to release the HTTP client."""
        await self._client.aclose()
