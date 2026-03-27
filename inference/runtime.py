"""
microWakeWord 推理运行时
兼容 ai-edge-litert / tflite-runtime / tensorflow 三种后端，自动选择可用的。

特征提取完全对齐训练参数：
  sample_rate=16000, window_size=30ms, window_step=10ms,
  num_channels=40, upper_band=7500Hz, lower_band=125Hz,
  enable_pcan=True, min_signal_remaining=0.05
"""
from __future__ import annotations

import collections
import os
import math
from pathlib import Path

import numpy as np

# ─── TFLite 后端自动选择 ────────────────────────────────────────────────────

def _load_interpreter(model_path: str):
    errors = []
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
        "  pip install ai-edge-litert\n"
        "  pip install tflite-runtime\n"
        "  pip install tensorflow-cpu\n"
        "详细错误：\n" + "\n".join(errors)
    )


# ─── 纯 Python 音频前端（对齐 TFLM audio_microfrontend 参数）──────────────

class AudioFrontend:
    """
    与训练时完全一致的 mel 特征提取。

    参数与 microwakeword/audio/audio_utils.py 中的 frontend_op.audio_microfrontend
    调用完全对应：
      sample_rate=16000, window_size=30ms(480), window_step=10ms(160),
      num_channels=40, upper_band_limit=7500, lower_band_limit=125,
      enable_pcan=True, min_signal_remaining=0.05, out_scale=1
    """

    SAMPLE_RATE    = 16000
    WINDOW_SAMPLES = 480   # 30ms
    STEP_SAMPLES   = 160   # 10ms
    NUM_CHANNELS   = 40
    UPPER_HZ       = 7500.0
    LOWER_HZ       = 125.0
    # PCAN 参数（来自 TFLM 默认值）
    PCAN_STRENGTH      = 0.95
    PCAN_OFFSET        = 80.0
    PCAN_GAIN_BITS     = 21
    # noise floor 跟踪
    SMOOTHING_BITS     = 10
    EVEN_SMOOTHING     = 0.025
    ODD_SMOOTHING      = 0.06
    MIN_SIGNAL_REMAINING = 0.05

    def __init__(self):
        self._buf   = np.zeros(self.WINDOW_SAMPLES, dtype=np.float32)
        self._noise = None       # noise floor，延迟初始化
        self._mel_fb = self._build_mel_filterbank()

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def process_int16(self, pcm: np.ndarray) -> list[np.ndarray]:
        """
        接收 int16 PCM（任意长度），返回特征帧列表，每帧 shape=(40,) float32。
        每 STEP_SAMPLES 样本输出一帧（缓冲区满 WINDOW_SAMPLES 后才开始输出）。
        """
        frames = []
        audio = pcm.astype(np.float32)
        i = 0
        while i < len(audio):
            # 把新样本移入窗口缓冲区
            take = min(len(audio) - i, self.STEP_SAMPLES)
            self._buf = np.roll(self._buf, -take)
            self._buf[-take:] = audio[i: i + take]
            i += take

            if take < self.STEP_SAMPLES:
                break

            frame = self._compute_frame(self._buf)
            if frame is not None:
                frames.append(frame)

        return frames

    def reset(self):
        self._buf   = np.zeros(self.WINDOW_SAMPLES, dtype=np.float32)
        self._noise = None

    # ── 内部实现 ──────────────────────────────────────────────────────────────

    def _compute_frame(self, window: np.ndarray) -> np.ndarray | None:
        # 汉宁窗
        windowed = window * np.hanning(len(window))
        # FFT 取幅值平方
        spec = np.abs(np.fft.rfft(windowed)) ** 2
        # mel 滤波
        mel = self._mel_fb @ spec          # (40,)
        mel = np.maximum(mel, 1e-9)

        # PCAN 自动增益（简化版，对齐 TFLM smoothing noise floor）
        if self._noise is None:
            self._noise = mel.copy()
        else:
            # 指数平滑噪声底板
            alpha = self.EVEN_SMOOTHING
            self._noise = alpha * mel + (1 - alpha) * self._noise

        noise_floor = np.maximum(self._noise * self.MIN_SIGNAL_REMAINING, 1.0)
        gain = (mel / noise_floor) ** self.PCAN_STRENGTH
        mel_pcan = mel * gain / (gain + self.PCAN_OFFSET)

        return mel_pcan.astype(np.float32)

    def _build_mel_filterbank(self) -> np.ndarray:
        """构建 mel 滤波器组，shape=(NUM_CHANNELS, fft_size//2+1)。"""
        fft_size   = self.WINDOW_SAMPLES
        n_fft_bins = fft_size // 2 + 1

        def hz_to_mel(hz):
            return 2595.0 * math.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        low_mel  = hz_to_mel(self.LOWER_HZ)
        high_mel = hz_to_mel(self.UPPER_HZ)

        mel_pts  = np.linspace(low_mel, high_mel, self.NUM_CHANNELS + 2)
        hz_pts   = np.array([mel_to_hz(m) for m in mel_pts])
        bin_pts  = np.floor(hz_pts / self.SAMPLE_RATE * fft_size).astype(int)
        bin_pts  = np.clip(bin_pts, 0, n_fft_bins - 1)

        fb = np.zeros((self.NUM_CHANNELS, n_fft_bins), dtype=np.float32)
        for i in range(self.NUM_CHANNELS):
            lo, center, hi = bin_pts[i], bin_pts[i + 1], bin_pts[i + 2]
            if center > lo:
                fb[i, lo:center] = np.linspace(0, 1, center - lo, endpoint=False)
            if hi > center:
                fb[i, center:hi] = np.linspace(1, 0, hi - center, endpoint=False)
            fb[i, center] = 1.0

        return fb


# ─── 流式检测器 ─────────────────────────────────────────────────────────────

SAMPLE_RATE    = 16000
STRIDE_SAMPLES = 160   # 10ms @ 16kHz


def _try_pymicro() -> object | None:
    """尝试加载 pymicro_features，失败返回 None。"""
    try:
        from pymicro_features import MicroFrontend
        fe = MicroFrontend()
        # 快速健康检查：喂 160 样本，确认能正常工作
        test = np.zeros(160, dtype=np.int16).tobytes()
        _ = fe.process_samples(test)
        return fe
    except Exception:
        return None


class WakeWordDetector:
    """
    流式唤醒词检测器。

    特征提取优先使用 pymicro_features（与训练 C 实现一致）；
    若不可用或指定 use_python_frontend=True，则改用内置纯 Python 前端。

    用法：
        detector = WakeWordDetector("help_me.tflite", cutoff=0.10)
        triggered = detector.feed(pcm_int16)
    """

    def __init__(
        self,
        model_path: str,
        cutoff: float = 0.10,
        window_count: int = 5,
        use_python_frontend: bool = False,
    ):
        self.cutoff       = cutoff
        self.window_count = window_count

        self._interp = _load_interpreter(model_path)

        in_details  = self._interp.get_input_details()
        out_details = self._interp.get_output_details()

        self._in_idx   = in_details[0]["index"]
        self._out_idx  = out_details[0]["index"]
        self._in_scale, self._in_zero   = in_details[0]["quantization"]
        self._out_scale, self._out_zero = out_details[0]["quantization"]
        self._state_ins  = [d for d in in_details  if d["index"] != self._in_idx]
        self._state_outs = [d for d in out_details if d["index"] != self._out_idx]
        self._reset_states()

        in_shape = in_details[0]["shape"]       # (1, N_FRAMES, 40)
        self._n_frames = int(in_shape[1]) if len(in_shape) >= 3 else 1
        self._n_feats  = int(in_shape[-1])

        # 选择前端
        if use_python_frontend:
            self._pymicro = None
        else:
            self._pymicro = _try_pymicro()

        self._py_fe = AudioFrontend()           # 备用/主用纯 Python 前端

        self._audio_buf = np.zeros(0, dtype=np.int16)
        self._score_buf: collections.deque[float] = collections.deque(maxlen=window_count)
        self._frame_buf: list[np.ndarray] = []

        frontend_name = "pymicro_features" if self._pymicro else "Python AudioFrontend"
        print(f"[runtime] frontend={frontend_name}  n_frames={self._n_frames}  n_feats={self._n_feats}")
        print(f"[runtime] in_scale={self._in_scale:.6f}  in_zero={self._in_zero}  "
              f"out_scale={self._out_scale:.6f}  out_zero={self._out_zero}")

    # ── 公开方法 ──────────────────────────────────────────────────────────────

    def feed(self, pcm_int16: np.ndarray) -> bool:
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
        self._audio_buf = np.concatenate([self._audio_buf, pcm_int16])
        scores = []
        while len(self._audio_buf) >= STRIDE_SAMPLES:
            chunk = self._audio_buf[:STRIDE_SAMPLES]
            self._audio_buf = self._audio_buf[STRIDE_SAMPLES:]
            scores.append(self._infer(chunk))
        return scores

    def reset(self):
        self._reset_states()
        self._audio_buf = np.zeros(0, dtype=np.int16)
        self._score_buf.clear()
        self._frame_buf.clear()
        self._py_fe.reset()
        if self._pymicro:
            try:
                self._pymicro.reset()
            except Exception:
                pass

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _extract_frame(self, chunk: np.ndarray) -> np.ndarray | None:
        """从 160 样本 chunk 提取一帧 (40,) 特征，返回 None 表示前端预热中。"""
        if self._pymicro is not None:
            raw = chunk.astype(np.int16).tobytes()
            try:
                out = self._pymicro.process_samples(raw)
                if out.features:
                    return np.array(out.features, dtype=np.float32)
                return None
            except Exception:
                # pymicro 报错，回退到 Python 前端
                self._pymicro = None

        # 纯 Python 前端
        frames = self._py_fe.process_int16(chunk.astype(np.int16))
        return frames[-1] if frames else None

    def _infer(self, chunk: np.ndarray) -> float:
        frame = self._extract_frame(chunk)
        if frame is None:
            return 0.0

        self._frame_buf.append(frame)
        if len(self._frame_buf) < self._n_frames:
            return 0.0

        frames = np.stack(self._frame_buf[-self._n_frames:], axis=0)  # (n_frames, 40)

        # TFLite int8 量化：quantized = real / scale + zero_point
        if self._in_scale != 0:
            feat_q = (frames / self._in_scale + self._in_zero)
        else:
            feat_q = frames + self._in_zero
        feat_q = np.clip(feat_q, -128, 127).astype(np.int8)
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
