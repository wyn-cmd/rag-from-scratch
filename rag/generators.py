"""Produce an answer from a prompt.

Same shape as the embedders: one dependency free implementation and one that
loads a real model, both behind `generate(prompt, context="")`.

The context argument exists so the extractive fallback can answer without a
model. Real generators only need the prompt, since the context is already
formatted into it by the pipeline.
"""

from typing import List, Optional


class Generator:
    """Base class documenting the interface."""

    def generate(self, prompt: str, context: str = "") -> str:
        raise NotImplementedError


class ExtractiveGenerator(Generator):
    """Answer by quoting the best passage. No model, no download, no invention.

    This is a development fallback, not a question answerer. It exists so the
    pipeline can be exercised end to end and asserted on in tests without a
    multi gigabyte install, and so `ask()` never returns an empty string just
    because torch is missing.
    """

    def __init__(self, max_chars: int = 600) -> None:
        self.max_chars = max_chars

    def generate(self, prompt: str, context: str = "") -> str:
        passage = (context or "").strip()
        if not passage:
            return "No context was retrieved, so there is nothing to quote."
        if len(passage) > self.max_chars:
            passage = passage[:self.max_chars].rsplit(" ", 1)[0] + " ..."
        return (
            "(extractive fallback, no language model loaded)\n\n"
            f"Most relevant passage found:\n{passage}"
        )


class TransformersGenerator(Generator):
    """Text generation through transformers with a causal language model.

    Kept deliberately plain: a pipeline, a token budget and a prompt. Batching,
    quantisation, GPU placement and streaming all belong to whatever you build
    on top, and none of them change what the retrieval half of this project does.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
                 max_new_tokens: int = 256, temperature: float = 0.2,
                 token: Optional[str] = None, device: Optional[str] = None,
                 dtype: str = "auto") -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.token = token
        self.device = device
        self.dtype = dtype
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            try:
                import torch
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "transformers and torch are not installed. "
                    "Install the optional stack with: pip install -r requirements.txt"
                ) from exc

            kwargs = {"token": self.token} if self.token else {}
            kwargs["torch_dtype"] = self.dtype if self.dtype != "auto" else "auto"
            if self.device:
                kwargs["device"] = self.device
            self._pipeline = pipeline("text-generation", model=self.model_name, **kwargs)
        return self._pipeline

    def generate(self, prompt: str, context: str = "") -> str:
        pipe = self._load()
        do_sample = self.temperature > 0
        outputs = pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            return_full_text=False,
        )
        text = outputs[0].get("generated_text", "") if outputs else ""
        return text.strip()


class RecordingGenerator(Generator):
    """Records prompts and returns a fixed reply. Handy in tests."""

    def __init__(self, reply: str = "recorded reply") -> None:
        self.reply = reply
        self.prompts: List[str] = []

    def generate(self, prompt: str, context: str = "") -> str:
        self.prompts.append(prompt)
        return self.reply
