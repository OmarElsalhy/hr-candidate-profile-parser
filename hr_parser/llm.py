"""Model loading and text generation.

``torch`` and ``transformers`` are imported lazily so the rest of the package
(and the test-suite) works without the heavy ML dependencies installed.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Advanced: set HR_PARSER_MODEL to use another instruct model (e.g. a larger one on a big GPU).
MODEL_ID = os.environ.get("HR_PARSER_MODEL", DEFAULT_MODEL)

# The model stops by itself when the JSON is complete; this is only a safety cap.
MAX_NEW_TOKENS = 1024


def pick_device():
    """Return ``(device, dtype)`` for the best available hardware."""
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    # bfloat16 halves RAM use on CPU compared to float32.
    return "cpu", torch.bfloat16


def load_model(model_id: str | None = None):
    """Download (first run only) and load the tokenizer and model, then warm them up."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = model_id or MODEL_ID
    device, dtype = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()

    # The very first generation pays one-off start-up costs (GPU kernels, caches).
    # Doing a tiny one here means the user's first real parse is as fast as the rest.
    try:
        warmup = tokenizer("Hello", return_tensors="pt").to(model.device)
        with torch.inference_mode():
            model.generate(
                **warmup,
                max_new_tokens=2,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
            )
    except Exception:  # noqa: BLE001 - warm-up is best effort
        pass
    return tokenizer, model


def generate(tokenizer, model, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Run the prompt through the model's chat template and return only the reply.

    Greedy decoding (no sampling) is the most reliable choice for structured extraction.
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)