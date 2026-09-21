"""
Qwen3-Reranker text cross-encoder for LookBench
Optional PEFT LoRA adapter on top of the base reranker
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image

from utils.logging import get_logger
from .registry import register_model
from .reranker_base import BaseReranker

logger = get_logger(__name__)


@register_model("qwen3-reranker", metadata={"task": "reranking", "modality": "text"})
class Qwen3Reranker(BaseReranker):
    """Official Qwen3-Reranker scoring, with an optional LoRA adapter.

    The model is a causal LM prompted to answer "yes" or "no" to whether a
    document satisfies a query. The score is the softmax probability of "yes"
    against "no" at the final position -- not a generated string -- so scoring
    costs one forward pass per (query, candidate) and is deterministic.

    The prompt template, the left padding, the truncation that trims only the
    body (never the template), and the two-way softmax are all load-bearing: a
    reranker scored under a different template is a different system, and its
    numbers are not comparable to published ones.
    """

    supports_multimodal = False

    DEFAULT_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    PREFIX = (
        '<|im_start|>system\n'
        'Judge whether the Document meets the requirements based on the Query and the '
        'Instruct provided. Note that the answer can only be "yes" or "no".'
        '<|im_end|>\n'
        '<|im_start|>user\n'
    )
    SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        instruction: str = DEFAULT_INSTRUCTION,
        max_length: int = 4096,
        batch_size: int = 16,
    ):
        """
        Initialize the scorer

        Args:
            model: Loaded causal LM, optionally already wrapped with a LoRA adapter
            tokenizer: Matching tokenizer, left-padded
            instruction: Task instruction placed in the prompt
            max_length: Maximum prompt length in tokens, template included
            batch_size: Candidates scored per forward pass
        """
        self.model = model
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.max_length = max_length
        self.batch_size = batch_size

        self.yes_token_id = self._resolve_token_id("yes")
        self.no_token_id = self._resolve_token_id("no")
        self.prefix_token_ids = tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_token_ids = tokenizer.encode(self.SUFFIX, add_special_tokens=False)

    @classmethod
    def load_model(
        cls,
        model_name: str,
        model_path: Optional[str] = None,
        instruction: str = DEFAULT_INSTRUCTION,
        max_length: int = 4096,
        batch_size: int = 16,
        **kwargs: Any,
    ) -> Tuple[Any, "Qwen3Reranker"]:
        """
        Load a Qwen3-Reranker, optionally applying a LoRA adapter.

        Args:
            model_name: Base model repo id or local path
                (e.g., ``Qwen/Qwen3-Reranker-4B``)
            model_path: Optional PEFT adapter repo id or path applied on top
                (e.g., ``srpone/zoowork-shopranker-4b``)
            instruction: Task instruction placed in the prompt
            max_length: Maximum prompt length in tokens
            batch_size: Candidates scored per forward pass
            **kwargs: Passed through to ``from_pretrained``

        Returns:
            Tuple of (underlying model, scorer instance)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Qwen3-Reranker base: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if device == "cuda":
            load_kwargs["dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["dtype"] = torch.float32
        load_kwargs.update(kwargs)

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        model.config.pad_token_id = tokenizer.pad_token_id

        if model_path:
            from peft import PeftModel

            logger.info("Loading LoRA adapter: %s", model_path)
            model = PeftModel.from_pretrained(model, model_path)

        model.eval()
        return model, cls(model, tokenizer, instruction=instruction,
                          max_length=max_length, batch_size=batch_size)

    def _resolve_token_id(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            vocab = self.tokenizer.get_vocab()
            if token not in vocab:
                raise ValueError(f"Could not resolve token id for {token!r}")
            token_id = vocab[token]
        return int(token_id)

    def _input_device(self) -> torch.device:
        if hasattr(self.model, "device"):
            return self.model.device
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            return torch.device(next(iter(device_map.values())))
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _last_token_indices(self, attention_mask: torch.Tensor) -> torch.Tensor:
        # Correct for both left- and right-padded batches.
        return attention_mask.size(1) - 1 - attention_mask.long().flip(dims=[1]).argmax(dim=1)

    def _format(self, query: str, document: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def _tokenize(self, query: str, documents: Sequence[str]) -> Dict[str, torch.Tensor]:
        usable = self.max_length - len(self.prefix_token_ids) - len(self.suffix_token_ids)
        if usable <= 0:
            raise ValueError(
                f"max_length={self.max_length} is too small for the Qwen3 reranker template"
            )

        prompts = []
        for document in documents:
            body = self.tokenizer.encode(
                self._format(query, document),
                add_special_tokens=False,
                truncation=True,
                max_length=usable,
            )
            prompts.append(self.prefix_token_ids + body + self.suffix_token_ids)

        width = max(len(ids) for ids in prompts)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention = [], []
        for ids in prompts:
            pad = width - len(ids)
            input_ids.append([pad_id] * pad + ids)
            attention.append([0] * pad + [1] * len(ids))

        device = self._input_device()
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
        }

    def score(
        self,
        query: str,
        candidates: Sequence[str],
        query_image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
    ) -> List[float]:
        """
        Score candidates as P("yes") against P("no") at the final position.

        Args:
            query: Query text
            candidates: Candidate documents
            query_image: Ignored; this reranker is text-only

        Returns:
            One probability in [0, 1] per candidate, in the order given
        """
        if query_image is not None:
            logger.debug("Qwen3Reranker is text-only; ignoring query_image")

        candidates = list(candidates)
        scores: List[float] = []
        for start in range(0, len(candidates), self.batch_size):
            encoded = self._tokenize(query, candidates[start:start + self.batch_size])
            with torch.no_grad():
                logits = self.model(**encoded).logits
                last = self._last_token_indices(encoded["attention_mask"])
                rows = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
                yn = logits[rows, last, :][:, [self.no_token_id, self.yes_token_id]]
                probs = torch.softmax(yn.float(), dim=-1)[:, 1]
            scores.extend(probs.detach().cpu().tolist())
        return scores
