#!/usr/bin/env python3
"""
验证 jiuming_v4.tflite 模型：
1. 用 edge-tts 生成测试音频（正样本 + 负样本 + 边界样本）
2. 用 MWW 推理检测每个样本的概率
3. 报告哪些被误判

在宿主机运行第一步（生成音频），在 Docker 内运行第二步（推理）。
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


# ── 测试用例 ──
# 正样本：应该触发（prob > cutoff）
POSITIVE_TESTS = [
    "救命",
    "救命啊",
    "快救命",
    "救命救命",
    "来人救命",
    "救命呀",
]

# 方言变体：也应该触发
DIALECT_TESTS = [
    "救民",
    "久名",
    "揪命",
    "纠命",
    "九命",
]

# 负样本：不应该触发（prob < cutoff）
NEGATIVE_TESTS = [
    "救火",
    "救护车",
    "救人",
    "救援",
    "生命",
    "革命",
    "拼命",
    "要命",
    "玩命",
    "光明",
    "聪明",
    "文明",
    "说明",
    "你好",
    "谢谢",
    "再见",
    "打开灯",
    "播放音乐",
    "你好树实",
    "小爱同学",
    "天猫精灵",
    "你好小度",
    "你好百度",
    "你好谷歌",
    "着火了",
    "危险",
    "帮帮我",
]

# 中文声音列表（选几个代表性的）
VOICES = [
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-YunxiNeural",
    "zh-CN-XiaoyiNeural",
]


async def generate_test_audio(output_dir: str):
    """用 edge-tts 生成测试音频"""
    import edge_tts

    os.makedirs(output_dir, exist_ok=True)

    all_tests = []
    for text in POSITIVE_TESTS:
        all_tests.append(("positive", text))
    for text in DIALECT_TESTS:
        all_tests.append(("dialect", text))
    for text in NEGATIVE_TESTS:
        all_tests.append(("negative", text))

    manifest = []
    for category, text in all_tests:
        for voice in VOICES:
            safe_text = text.replace(" ", "_")
            safe_voice = voice.split("-")[-1].replace("Neural", "")
            fname = f"{category}_{safe_text}_{safe_voice}.wav"
            fpath = os.path.join(output_dir, fname)

            if os.path.exists(fpath) and os.path.getsize(fpath) > 1000:
                manifest.append({
                    "file": fname,
                    "text": text,
                    "category": category,
                    "voice": voice,
                })
                continue

            try:
                comm = edge_tts.Communicate(text, voice)
                # edge-tts 输出 mp3，需要转 wav
                mp3_path = fpath.replace(".wav", ".mp3")
                await comm.save(mp3_path)

                # 转换为 16kHz mono wav
                import subprocess
                subprocess.run([
                    "ffmpeg", "-y", "-i", mp3_path,
                    "-ar", "16000", "-ac", "1", "-f", "wav", fpath
                ], capture_output=True, timeout=10)
                os.remove(mp3_path)

                if os.path.exists(fpath) and os.path.getsize(fpath) > 1000:
                    manifest.append({
                        "file": fname,
                        "text": text,
                        "category": category,
                        "voice": voice,
                    })
                    print(f"  ✓ {fname}")
                else:
                    print(f"  ✗ {fname} (转换失败)")
            except Exception as e:
                print(f"  ✗ {fname}: {e}")

    # 保存 manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n生成 {len(manifest)} 个测试音频 → {output_dir}")
    print(f"Manifest: {manifest_path}")
    return manifest


def run_inference(test_dir: str, model_path: str, cutoff: float = 0.5, window: int = 3):
    """在 Docker 内运行：用 MWW 模型推理所有测试音频"""
    import numpy as np

    try:
        import tensorflow as tf
        interpreter_class = tf.lite.Interpreter
    except ImportError:
        from tflite_runtime.interpreter import Interpreter as interpreter_class

    # 加载 manifest
    manifest_path = os.path.join(test_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # 加载模型
    interpreter = interpreter_class(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_shape = input_details[0]["shape"]  # [1, 3, 40]
    in_dtype = input_details[0]["dtype"]
    in_scale = input_details[0]["quantization"][0]
    in_zero = input_details[0]["quantization"][1]
    out_scale = output_details[0]["quantization"][0]
    out_zero = output_details[0]["quantization"][1]

    print(f"模型: {model_path}")
    print(f"输入: {in_shape}, dtype={in_dtype}, scale={in_scale}, zero={in_zero}")
    print(f"输出: scale={out_scale}, zero={out_zero}")
    print(f"cutoff={cutoff}, window={window}")
    print()

    # 需要 micro_features 前端
    try:
        from microwakeword.audio.audio_utils import AudioFeatureGenerator
        has_mww = True
    except ImportError:
        has_mww = False

    if not has_mww:
        # 手动实现 micro features
        try:
            from micro_features import MicroFrontend
            has_micro = True
        except ImportError:
            has_micro = False

    results = []
    for item in manifest:
        wav_path = os.path.join(test_dir, item["file"])
        if not os.path.exists(wav_path):
            continue

        # 读取音频
        try:
            import wave
            with wave.open(wav_path, "rb") as wf:
                assert wf.getsampwidth() == 2
                assert wf.getnchannels() == 1
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            # 如果不是 16kHz，重采样
            if sr != 16000:
                from scipy.signal import resample
                audio = resample(audio, int(len(audio) * 16000 / sr))
        except Exception as e:
            print(f"  跳过 {item['file']}: {e}")
            continue

        # 生成 mel 特征（使用 MWW 的前端）
        try:
            from microwakeword.audio import preprocessor
            features = preprocessor.generate_features_for_clip(
                audio, sample_rate=16000
            )
        except Exception:
            # fallback: 使用 pymicro_features
            try:
                sys.path.insert(0, "/workspace/work/micro-wake-word")
                from microwakeword.audio import preprocessor
                features = preprocessor.generate_features_for_clip(
                    audio, sample_rate=16000
                )
            except Exception as e2:
                print(f"  跳过 {item['file']}: 无法生成特征 {e2}")
                continue

        # 滑窗推理
        n_frames = features.shape[0]
        stride = in_shape[1]  # 3
        max_prob = 0.0
        probs = []

        for start in range(0, n_frames - stride + 1, stride):
            chunk = features[start:start + stride]  # [3, 40]
            if chunk.shape[0] < stride:
                break

            # 量化输入
            if in_dtype == np.int8:
                chunk_q = np.clip(
                    np.round(chunk / in_scale + in_zero), -128, 127
                ).astype(np.int8)
            else:
                chunk_q = chunk.astype(np.float32)

            chunk_q = chunk_q.reshape(in_shape)
            interpreter.set_tensor(input_details[0]["index"], chunk_q)
            interpreter.invoke()

            raw_out = interpreter.get_tensor(output_details[0]["index"])
            if out_scale > 0:
                prob = float((raw_out.flatten()[0] - out_zero) * out_scale)
            else:
                prob = float(raw_out.flatten()[0])

            prob = max(0.0, min(1.0, prob))
            probs.append(prob)
            if prob > max_prob:
                max_prob = prob

        # 滑窗检测
        detected = False
        if len(probs) >= window:
            for i in range(len(probs) - window + 1):
                if all(p >= cutoff for p in probs[i:i + window]):
                    detected = True
                    break

        item["max_prob"] = round(max_prob, 4)
        item["detected"] = detected
        item["n_frames"] = len(probs)
        results.append(item)

    # 打印结果
    print("=" * 80)
    print(f"{'类别':<10} {'文本':<12} {'声音':<12} {'最大概率':>8} {'检测':>6} {'判定':>8}")
    print("=" * 80)

    errors = []
    for r in sorted(results, key=lambda x: (x["category"], x["text"])):
        cat = r["category"]
        detected = r["detected"]
        max_p = r["max_prob"]

        # 判定是否正确
        if cat in ("positive", "dialect"):
            correct = detected  # 应该触发
            mark = "✓" if correct else "✗ 漏报"
        else:
            correct = not detected  # 不应该触发
            mark = "✓" if correct else "✗ 误触发"

        if not correct:
            errors.append(r)

        voice_short = r["voice"].split("-")[-1].replace("Neural", "")
        print(f"{cat:<10} {r['text']:<12} {voice_short:<12} {max_p:>8.4f} {'是' if detected else '否':>6} {mark:>8}")

    print("=" * 80)
    print(f"总计: {len(results)} 个样本, {len(errors)} 个错误")

    if errors:
        print("\n⚠️  错误详情:")
        for e in errors:
            cat = e["category"]
            if cat in ("positive", "dialect"):
                print(f"  漏报: {e['text']} ({e['voice']}) max_prob={e['max_prob']}")
            else:
                print(f"  误触发: {e['text']} ({e['voice']}) max_prob={e['max_prob']}")

    # 保存结果
    results_path = os.path.join(test_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果保存: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "infer", "both"], default="both")
    parser.add_argument("--test-dir", default="outputs/verify_jiuming_v4")
    parser.add_argument("--model", default="outputs/jiuming_v4.tflite")
    parser.add_argument("--cutoff", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args()

    if args.mode in ("generate", "both"):
        print("=== 第一步：生成测试音频 ===")
        asyncio.run(generate_test_audio(args.test_dir))

    if args.mode in ("infer", "both"):
        print("\n=== 第二步：模型推理 ===")
        run_inference(args.test_dir, args.model, args.cutoff, args.window)
