#!/usr/bin/env python3
"""
Docker 内运行：对 verify 测试音频跑 MWW TFLite 推理。
用法: python3 infer_verify.py --test-dir /workspace/outputs/verify_jiuming_v4 \
                               --model /workspace/outputs/jiuming_v4.tflite
"""
import argparse, json, os, sys, wave
import numpy as np

def load_wav_16k(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16)
    if ch > 1:
        audio = audio[::ch]
    if sr != 16000:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * 16000 / sr)).astype(np.int16)
    return audio

def generate_micro_features(audio_int16):
    """生成 MWW 的 40-ch mel filterbank 特征"""
    try:
        # 尝试用 MWW 内置的特征生成
        sys.path.insert(0, "/workspace/work/micro-wake-word/")
        from microwakeword.feature_generation import generate_features_for_clip
        return generate_features_for_clip(audio_int16)
    except Exception:
        pass

    try:
        from microwakeword.audio.preprocessor import generate_features_for_clip
        return generate_features_for_clip(audio_int16)
    except Exception:
        pass

    # 手动实现 micro_features (30ms window, 10ms step, 40 mel bins)
    # 使用 tflite micro_features 前端
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow as tf
        tflite = tf.lite

    # 最后手段：用 python-speech-features
    try:
        from python_speech_features import logfbank
        audio_f = audio_int16.astype(np.float32) / 32768.0
        feats = logfbank(audio_f, samplerate=16000, winlen=0.030,
                         winstep=0.010, nfilt=40, nfft=512)
        return feats.astype(np.float32)
    except ImportError:
        pass

    raise RuntimeError("无法生成特征，请安装 microwakeword 或 python_speech_features")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cutoff", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=1)
    args = parser.parse_args()

    # 加载 manifest
    with open(os.path.join(args.test_dir, "manifest.json")) as f:
        manifest = json.load(f)

    # 加载模型
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=args.model)
    except ImportError:
        from tflite_runtime.interpreter import Interpreter
        interpreter = Interpreter(model_path=args.model)

    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    in_shape = inp["shape"]       # [1, 3, 40]
    stride = in_shape[1]          # 3
    n_feats = in_shape[2]         # 40
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]

    print(f"模型: {args.model}")
    print(f"输入: shape={list(in_shape)}, scale={in_scale}, zero={in_zero}")
    print(f"输出: scale={out_scale}, zero={out_zero}")
    print(f"cutoff={args.cutoff}, window={args.window}")
    print()

    results = []
    for item in manifest:
        wav_path = os.path.join(args.test_dir, item["file"])
        if not os.path.exists(wav_path):
            continue

        try:
            audio = load_wav_16k(wav_path)
            features = generate_micro_features(audio)
        except Exception as e:
            print(f"  跳过 {item['file']}: {e}")
            continue

        # 滑窗推理
        n_frames = features.shape[0]
        probs = []
        for start in range(0, n_frames - stride + 1, stride):
            chunk = features[start:start + stride]
            if chunk.shape != (stride, n_feats):
                break

            # 量化
            if inp["dtype"] == np.int8:
                chunk_q = np.clip(
                    np.round(chunk / in_scale + in_zero), -128, 127
                ).astype(np.int8)
            else:
                chunk_q = chunk.astype(np.float32)

            interpreter.set_tensor(inp["index"], chunk_q.reshape(in_shape))
            interpreter.invoke()
            raw = interpreter.get_tensor(out["index"]).flatten()[0]

            if out_scale > 0:
                prob = float((raw - out_zero) * out_scale)
            else:
                prob = float(raw)
            prob = max(0.0, min(1.0, prob))
            probs.append(prob)

        max_prob = max(probs) if probs else 0.0

        # 滑窗检测
        detected = False
        w = args.window
        if len(probs) >= w:
            for i in range(len(probs) - w + 1):
                if all(p >= args.cutoff for p in probs[i:i + w]):
                    detected = True
                    break

        item["max_prob"] = round(max_prob, 4)
        item["detected"] = detected
        results.append(item)

    # 打印结果表
    print("=" * 85)
    print(f"{'类别':<10} {'文本':<14} {'声音':<12} {'最大概率':>8} {'检测':>6} {'判定':>10}")
    print("=" * 85)

    errors = []
    for r in sorted(results, key=lambda x: (x["category"], x["text"], x["voice"])):
        cat = r["category"]
        det = r["detected"]
        mp = r["max_prob"]
        voice_short = r["voice"].split("-")[-1].replace("Neural", "")

        if cat in ("positive", "dialect"):
            mark = "✓ 正确" if det else "✗ 漏报"
            correct = det
        else:
            mark = "✓ 正确" if not det else "✗ 误触发"
            correct = not det

        if not correct:
            errors.append(r)

        print(f"{cat:<10} {r['text']:<14} {voice_short:<12} {mp:>8.4f} {'是' if det else '否':>6} {mark:>10}")

    print("=" * 85)
    print(f"\n总计: {len(results)} 样本, 错误: {len(errors)}")

    # 按类别统计
    for cat in ["positive", "dialect", "negative"]:
        cat_items = [r for r in results if r["category"] == cat]
        if not cat_items:
            continue
        if cat in ("positive", "dialect"):
            hits = sum(1 for r in cat_items if r["detected"])
            print(f"  {cat}: {hits}/{len(cat_items)} 触发 (recall={hits/len(cat_items)*100:.1f}%)")
        else:
            fps = sum(1 for r in cat_items if r["detected"])
            print(f"  {cat}: {fps}/{len(cat_items)} 误触发 (FPR={fps/len(cat_items)*100:.1f}%)")

    if errors:
        print("\n⚠️  错误列表:")
        for e in errors:
            cat = e["category"]
            tp = "漏报" if cat in ("positive", "dialect") else "误触发"
            print(f"  [{tp}] {e['text']} ({e['voice'].split('-')[-1]}) max_prob={e['max_prob']}")

    # 保存
    with open(os.path.join(args.test_dir, "results.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
