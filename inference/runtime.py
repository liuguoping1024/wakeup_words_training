"""
microWakeWord 推理运行时
兼容 ai-edge-litert / tflite-runtime / tensorflow 三种后端，自动选择可用的。
"""
from __future__ import annotations

import collections
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ─── TFLite 后端自动选择 ────────────────────────────────────────────────────

def _load_interpreter(model_path: str):
    """
    按优先级尝试加载 TFLite 解释器，全部使用 CPU（无 GPU delegate）。
    优先级：ai-edge-litert > tflite-runtime > tensorflow
    """
    errors = []

    import os
    num_threads = int(os.environ.get("TFLITE_NUM_THREADS", "4"))

    try:
        from ai_edge_litert.interpreter import Interpreter
        interp = Interpreter(model_path=model_path, num_threads=num_threads)
        interp.allocate_tensors()
        return interp
    except Exception as e:
        errors.append(f"ai-edge-litert: {e}")

    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
        interp.allocate_tensors()
        return interp
    except Exception as e:
        errors.append(f"tflite-runtime: {e}")

    try:
        import tensorflow as tf
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        interp = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        interp.allocate_tensors()
        return interp
    except Exception as e:
        errors.append(f"tensorflow: {e}")

    raise ImportError(
        "没有可用的 TFLite 后端，请安装以下任意一个：\n"
        "  pip install ai-edge-litert       （推荐，最轻量）\n"
        "  pip install tflite-runtime\n"
        "  pip install tensorflow-cpu\n"
        "详细错误：\n" + "\n".join(errors)
    )


# ─── 流式检测器 ─────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
STRIDE_SAMPLES = 160   # 10ms @ 16kHz，每次喂给模型的步长


class WakeWordDetector:
    """
    流式唤醒词检测器。

    用法：
        detector = WakeWordDetector("help_me.tflite", cutoff=0.10)
        triggered = detector.feed(pcm_int16)   # 返回 True 表示检测到唤醒词
    """

    def __init__(
        self,
        model_path: str,
        cutoff: float = 0.10,
        window_count: int = 5,
    ):
        self.cutoff = cutoff
        self.window_count = window_count

        self._interp = _load_interpreter(model_path)
        self._fe = self._load_frontend()

        in_details = self._interp.get_input_details()
        out_details = self._interp.get_output_details()

        # 第一个输入/输出是特征/概率，其余是 streaming 状态
        self._in_idx = in_details[0]["index"]
        self._out_idx = out_details[0]["index"]
        self._in_scale, self._in_zero = in_details[0]["quantization"]
        self._out_scale, self._out_zero = out_details[0]["quantization"]
        self._state_ins  = [d for d in in_details  if d["index"] != self._in_idx]
        self._state_outs = [d for d in out_details if d["index"] != self._out_idx]
        self._reset_states()

        self._audio_buf = np.zeros(0, dtype=np.int16)
        self._score_buf: collections.deque[float] = collections.deque(maxlen=window_count)

        # 模型输入形状 (1, N_FRAMES, 40)，需要攒够 N_FRAMES 帧才推理一次
        in_shape = in_details[0]["shape"]   # e.g. (1, 3, 40)
        self._n_frames = int(in_shape[1]) if len(in_shape) >= 3 else 1
        self._n_feats  = int(in_shape[-1])
        # 帧缓冲，每收到一帧特征就 append，满 n_frames 后推理并清空
        self._frame_buf: list[np.ndarray] = []

    # ── 公开方法 ──────────────────────────────────────────────────────────

    def feed(self, pcm_int16: np.ndarray) -> bool:
        """
        喂入 16kHz 单声道 int16 PCM（任意长度）。
        检测到唤醒词时返回 True，否则返回 False。
        """
        self._audio_buf = np.concatenate([self._audio_buf, pcm_int16])
        triggered = False
        while len(self._audio_buf) >= STRIDE_SAMPLES:
            chunk = self._audio_buf[:STRIDE_SAMPLES]
            self._audio_buf = self._audio_buf[STRIDE_SAMPLES:]
            prob = self._infer(chunk)
            self._score_buf.append(prob)
            if (
                len(self._score_buf) == self.window_count
                and all(s >= self.cutoff for s in self._score_buf)
            ):
                triggered = True
                self._score_buf.clear()
        return triggered

    def feed_and_score(self, pcm_int16: np.ndarray) -> list[float]:
        """
        与 feed() 相同，但返回每 10ms 帧的概率列表，方便调试/可视化。
        """
        self._audio_buf = np.concatenate([self._audio_buf, pcm_int16])
        scores = []
        while len(self._audio_buf) >= STRIDE_SAMPLES:
            chunk = self._audio_buf[:STRIDE_SAMPLES]
            self._audio_buf = self._audio_buf[STRIDE_SAMPLES:]
            scores.append(self._infer(chunk))
        return scores

    def reset(self):
        """重置 streaming 状态（换段音频时调用）"""
        self._reset_states()
        self._audio_buf = np.zeros(0, dtype=np.int16)
        self._score_buf.clear()
        self._frame_buf.clear()

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _load_frontend(self):
        try:
            from pymicro_features import MicroFrontend
            return MicroFrontend()
        except ImportError:
            raise ImportError(
                "pymicro-features 未安装：pip install pymicro-features"
            )

    def _infer(self, chunk: np.ndarray) -> float:
        # process_samples 接收 bytes，返回 MicroFrontendOutput
        # out.features 是长度 40 的一维序列（一帧频谱特征）
        raw = chunk.astype(np.int16).tobytes()
        out = self._fe.process_samples(raw)

        if not out.features:
            return 0.0

        # 一次得到一帧 (40,)，加入帧缓冲
        frame = np.array(out.features, dtype=np.float32)   # (40,)
        self._frame_buf.append(frame)

        # 未攒够 n_frames 帧，暂不推理
        if len(self._frame_buf) < self._n_frames:
            return 0.0

        # 取最新的 n_frames 帧拼成 (1, n_frames, 40)
        frames = np.stack(self._frame_buf[-self._n_frames:], axis=0)  # (n_frames, 40)
        feat_q = (frames / self._in_scale + self._in_zero).astype(np.int8)
        feat_q = feat_q.reshape(1, self._n_frames, self._n_feats)

        self._interp.set_tensor(self._in_idx, feat_q)
        self._interp.invoke()

        out_q = float(self._interp.get_tensor(self._out_idx).flat[0])
        self._update_states()
        return float(np.clip((out_q - self._out_zero) * self._out_scale, 0.0, 1.0))

    def _reset_states(self):
        for d in self._state_ins:
            self._interp.set_tensor(d["index"], np.zeros(d["shape"], dtype=d["dtype"]))

    def _update_states(self):
        for si, so in zip(self._state_ins, self._state_outs):
            self._interp.set_tensor(si["index"], self._interp.get_tensor(so["index"]))
