"""ONNX Runtime session wrapper with thread tuning and async offload.

onnxruntime's InferenceSession.run() is thread-safe on CPU/CUDA providers,
so we offload the synchronous call onto a worker thread via
`asyncio.to_thread`, keeping the event loop responsive under concurrency.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class RuntimeConfig:
    intra_op_num_threads: int = 0
    inter_op_num_threads: int = 0


def _build_session_options(cfg: RuntimeConfig) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    if cfg.intra_op_num_threads > 0:
        opts.intra_op_num_threads = cfg.intra_op_num_threads
    if cfg.inter_op_num_threads > 0:
        opts.inter_op_num_threads = cfg.inter_op_num_threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts


def _labels_from_hf_config(config_path: Path) -> list[str]:
    """Read the HuggingFace `config.json` and return labels ordered by class id."""
    cfg = json.loads(config_path.read_text())
    id2label: dict[str, str] = cfg["id2label"]
    # JSON object keys are strings; sort numerically to recover label order.
    return [id2label[str(i)] for i in range(len(id2label))]


class ModelRuntime:
    """Owns an ONNX Runtime session and exposes sync + async inference."""

    def __init__(
        self,
        onnx_path: Path,
        labels: list[str],
        runtime_cfg: RuntimeConfig | None = None,
    ) -> None:
        self._onnx_path = onnx_path
        self._labels = labels
        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=_build_session_options(runtime_cfg or RuntimeConfig()),
            providers=["CPUExecutionProvider"],
        )

    @classmethod
    def from_artifacts(
        cls,
        onnx_path: Path,
        tokenizer_dir: Path,
        runtime_cfg: RuntimeConfig | None = None,
    ) -> ModelRuntime:
        labels = _labels_from_hf_config(tokenizer_dir / "config.json")
        return cls(onnx_path=onnx_path, labels=labels, runtime_cfg=runtime_cfg)

    @property
    def labels(self) -> list[str]:
        return self._labels

    @property
    def onnx_path(self) -> Path:
        return self._onnx_path

    def run(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Synchronous inference. Returns raw logits of shape (batch, num_labels)."""
        return self._session.run(
            ["logits"],
            {
                "input_ids": input_ids.astype(np.int64, copy=False),
                "attention_mask": attention_mask.astype(np.int64, copy=False),
            },
        )[0]

    async def run_async(
        self, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        return await asyncio.to_thread(self.run, input_ids, attention_mask)
