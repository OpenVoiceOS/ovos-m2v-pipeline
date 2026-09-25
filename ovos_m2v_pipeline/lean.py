"""A numpy forward path for the shipped model2vec checkpoints.

Step 2 of the lean-runtime plan (`knowledge/wiki/plans/m2v-lean-inference.md`,
Miro's ruling of 2026-09-23: keep `tokenizers`, steps 1 to 3, ONNX optional).

A published checkpoint holds an embedding matrix, a Unigram tokenizer and a
two-layer MLP head. The forward pass is: tokenize, gather the embedding rows,
mean-pool, L2-normalise, `Linear`, ReLU, `Linear`, softmax. Every step is
numpy, and the tokenizer is the Rust `tokenizers` wheel, so this module reads
a checkpoint and answers a prediction without `model2vec` on the path.

The module reproduces `model2vec`, it does not improve on it. Each step here
is written to match the reference operation for operation and dtype for
dtype, because the parity bar the plan names is per-locale label agreement at
100 percent, and a different accumulation order moves labels near a decision
boundary. Where the reference does something that looks wrong (the `1e-32`
added to the norm, the float64 zero vector for an empty token list), this
module does the same thing.

What is deliberately NOT here: training, distillation, quantization, the
sentence-transformers folder layouts and the legacy `pipeline.skops` head.
A checkpoint this module cannot read raises `LeanUnsupportedCheckpoint`, and
the caller falls back to `model2vec`.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: the files a checkpoint must ship for the lean path to read it
_EMBEDDINGS_FILE = "model.safetensors"
_HEAD_FILE = "head.safetensors"
_TOKENIZER_FILE = "tokenizer.json"
_CONFIG_FILE = "config.json"


class LeanUnsupportedCheckpoint(ValueError):
    """The checkpoint is not in the layout the lean loader reads."""


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis.

    Same shift-by-max as `model2vec.inference.mlp`, so the two agree to
    floating-point noise rather than to an algebraic identity.
    """
    shifted = x - x.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid, branch for branch as the reference."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class LeanHead:
    """The MLP head: a list of linear layers, ReLU between them.

    `model2vec`'s own `MLPHead` is already plain numpy; this is the same
    arithmetic with no import behind it.
    """

    def __init__(self, weights: List[np.ndarray], biases: List[np.ndarray],
                 activation: str, classes: Optional[np.ndarray]) -> None:
        self.weights = weights
        self.biases = biases
        self.activation = activation
        self.classes_ = classes

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Forward through every layer, ReLU on all but the last."""
        out = x
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            out = out @ weight.T + bias
            if index != last:
                out = np.maximum(out, 0.0)
        return out

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """The output activation applied to the logits."""
        logits = self.logits(x)
        if self.activation == "softmax":
            return _softmax(logits)
        if self.activation == "sigmoid":
            return _sigmoid(logits)
        return logits

    def predict_index(self, x: np.ndarray) -> np.ndarray:
        """The index of the largest logit, per row."""
        return self.logits(x).argmax(axis=1)


class LeanStaticModel:
    """The embedding half: tokenize, gather, mean-pool, normalise.

    Stands in for `model2vec.StaticModel` on the surface this plugin uses:
    `encode`, `dim`, `tokenize`, `tokenizer` and `embedding`. The prototype
    store calls `encode(..., use_multiprocessing=False)`, so `encode` accepts
    and ignores the reference's batching and multiprocessing keywords; the
    lean path is single-process by construction.
    """

    def __init__(self, embedding: np.ndarray, tokenizer: Any,
                 config: Optional[Dict[str, Any]] = None,
                 weights: Optional[np.ndarray] = None,
                 token_mapping: Optional[np.ndarray] = None) -> None:
        self.embedding = embedding
        self.tokenizer = tokenizer
        self.config = config or {}
        self.weights = weights
        self.token_mapping = token_mapping
        self.normalize = bool(self.config.get("normalize", False))

        tokens, _ = zip(*sorted(tokenizer.get_vocab().items(),
                                key=lambda item: item[1]))
        self.tokens = tokens
        # the reference truncates by characters before it truncates by
        # tokens, using the median token length as the characters-per-token
        # estimate; the same estimate is needed or long inputs diverge
        self.median_token_length = int(np.median([len(t) for t in tokens]))

        model = getattr(tokenizer, "model", None)
        unk = getattr(model, "unk_token", None)
        self.unk_token_id = tokenizer.get_vocab()[unk] if unk is not None else None
        self._can_encode_fast = hasattr(tokenizer, "encode_batch_fast")

    @property
    def dim(self) -> int:
        """The embedding width."""
        return int(self.embedding.shape[1])

    def tokenize(self, sentences: Sequence[str],
                 max_length: Optional[int] = None) -> List[List[int]]:
        """Token ids per sentence, unknown tokens dropped."""
        if max_length is not None:
            limit = max_length * self.median_token_length
            sentences = [sentence[:limit] for sentence in sentences]

        if self._can_encode_fast:
            encodings = self.tokenizer.encode_batch_fast(
                list(sentences), add_special_tokens=False)
        else:
            encodings = self.tokenizer.encode_batch(
                list(sentences), add_special_tokens=False)

        ids = [encoding.ids for encoding in encodings]
        if self.unk_token_id is not None:
            ids = [[i for i in row if i != self.unk_token_id] for row in ids]
        if max_length is not None:
            ids = [row[:max_length] for row in ids]
        return ids

    def encode(self, sentences: Sequence[str], max_length: Optional[int] = 512,
               **_ignored: Any) -> np.ndarray:
        """Mean-pooled, optionally L2-normalised sentence vectors.

        `_ignored` swallows `show_progress_bar`, `batch_size`,
        `use_multiprocessing` and `multiprocessing_threshold`: the callers in
        this plugin pass them, and the lean path has no use for any of them.
        """
        was_single = isinstance(sentences, str)
        if was_single:
            sentences = [sentences]

        ids = self.tokenize(sentences, max_length=max_length)
        rows: List[np.ndarray] = []
        for row_ids in ids:
            if row_ids:
                remapped = (row_ids if self.token_mapping is None
                            else self.token_mapping[row_ids])
                vectors = self.embedding[remapped]
                if self.weights is not None:
                    vectors = vectors * self.weights[row_ids][:, None]
                rows.append(vectors.mean(axis=0))
            else:
                # float64, as the reference: an all-unknown utterance answers
                # a zero vector and the dtype of that row is not float16
                rows.append(np.zeros(self.dim))

        out = np.stack(rows)
        if self.normalize:
            # the 1e-32 is the reference's guard against a zero row; it is
            # kept so a zero row answers zeros in both paths
            out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-32)
        if was_single:
            return out[0]
        return out


class LeanStaticModelPipeline:
    """Embedding plus head, on the surface `StaticModelPipeline` offers.

    The plugin reads `.model`, `.classes_` and `.predict_proba`; the shared
    model cache also re-points `.model` at an embedding another instance
    already loaded. All four work here.
    """

    def __init__(self, model: LeanStaticModel, head: LeanHead) -> None:
        self.model = model
        self.head = head
        self.classes_ = head.classes_

    def predict_proba(self, sentences: Sequence[str],
                      max_length: Optional[int] = 512,
                      **_ignored: Any) -> np.ndarray:
        """Class probabilities, one row per sentence."""
        encoded = self.model.encode(sentences, max_length=max_length)
        if np.ndim(encoded) == 1:
            encoded = encoded[None, :]
        return self.head.predict_proba(encoded)

    def predict(self, sentences: Sequence[str],
                max_length: Optional[int] = 512,
                threshold: float = 0.5, **_ignored: Any) -> np.ndarray:
        """The predicted label per sentence."""
        encoded = self.model.encode(sentences, max_length=max_length)
        if np.ndim(encoded) == 1:
            encoded = encoded[None, :]
        if self.classes_ is None:
            return self.head.logits(encoded)
        if self.head.activation == "sigmoid":
            proba = self.head.predict_proba(encoded)
            return np.asarray([self.classes_[row > threshold] for row in proba],
                              dtype=object)
        return self.classes_[self.head.predict_index(encoded)]

    @classmethod
    def from_pretrained(cls, path: str,
                        token: Optional[str] = None) -> "LeanStaticModelPipeline":
        """Read a checkpoint from a local folder or from the hub cache."""
        folder = _resolve_folder(path, token)
        model = _load_embedding(folder)
        head = _load_head(folder, model.config)
        return cls(model, head)


def load_lean_embedding(path: str, token: Optional[str] = None) -> LeanStaticModel:
    """The embedding half alone, for the prototype mode."""
    return _load_embedding(_resolve_folder(path, token))


def _resolve_folder(path: str, token: Optional[str]) -> Path:
    """A local folder, or the hub snapshot for a repo id.

    `huggingface_hub` does the download and the caching; this module does not
    reimplement either. The token is never put on a command line.
    """
    local = Path(path)
    if local.exists():
        return local
    import huggingface_hub
    return Path(huggingface_hub.snapshot_download(path, repo_type="model",
                                                  token=token))


def _load_embedding(folder: Path) -> LeanStaticModel:
    """`model.safetensors`, `tokenizer.json` and `config.json` from a folder."""
    import safetensors
    from tokenizers import Tokenizer

    for name in (_EMBEDDINGS_FILE, _TOKENIZER_FILE, _CONFIG_FILE):
        if not (folder / name).exists():
            raise LeanUnsupportedCheckpoint(
                f"{folder} has no `{name}`: the lean runtime reads the "
                f"model2vec folder layout only. Set the pipeline's "
                f"`runtime` back to `model2vec` for this checkpoint."
            )

    opened = safetensors.safe_open(str(folder / _EMBEDDINGS_FILE),
                                   framework="numpy")
    embedding = opened.get_tensor("embeddings")
    weights = _maybe_tensor(opened, "weights")
    mapping = _maybe_tensor(opened, "mapping")
    # the reference does this on every load: a NaN row would otherwise poison
    # the mean of every utterance that touches it
    embedding = np.nan_to_num(embedding)

    tokenizer = Tokenizer.from_file(str(folder / _TOKENIZER_FILE))
    with open(folder / _CONFIG_FILE) as handle:
        config = json.load(handle)
    return LeanStaticModel(embedding=embedding, tokenizer=tokenizer,
                           config=config, weights=weights,
                           token_mapping=mapping)


def _maybe_tensor(opened: Any, name: str) -> Optional[np.ndarray]:
    """A named tensor, or None when the file does not carry it."""
    try:
        return opened.get_tensor(name)
    except Exception:
        # safetensors does not export its own error type; the reference
        # catches everything here too
        return None


def _load_head(folder: Path, config: Dict[str, Any]) -> LeanHead:
    """`head.safetensors` plus the `head_config` block of `config.json`."""
    from safetensors.numpy import load_file

    head_path = folder / _HEAD_FILE
    if not head_path.exists():
        raise LeanUnsupportedCheckpoint(
            f"{folder} ships no `{_HEAD_FILE}`: it is a legacy "
            f"`pipeline.skops` checkpoint, which the lean runtime does not "
            f"read. Set the pipeline's `runtime` back to `model2vec`, or "
            f"convert the checkpoint with model2vec's "
            f"`convert_legacy_pipeline`."
        )

    head_config = config.get("head_config", {})
    n_layers = head_config.get("n_layers", 0)
    classes = head_config.get("classes")
    tensors = load_file(str(head_path))
    weights = [tensors[f"head.{index}.weight"] for index in range(n_layers)]
    biases = [tensors[f"head.{index}.bias"] for index in range(n_layers)]
    return LeanHead(weights=weights, biases=biases,
                    activation=head_config.get("activation", "identity"),
                    classes=np.asarray(classes) if classes is not None else None)
