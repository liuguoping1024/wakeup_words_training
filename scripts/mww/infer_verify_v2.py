#!/usr/bin/env python3
"""
用 pymicro_features（MWW 官方前端）对测试音频跑 TFLite 推理。
必须在 wakeword-mww Docker 内运行。
"""
import argparse, json, os, sys, wave, struct
import numpy as np
from pymicro_features import MicroFrontend


def load_wav_16k_int16(path):
    """读取 WAV 并转为 16kHz int16"""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sw == 2:
        audio = np.frombuffer(raw, dtype=np.int16)
    elif sw == 4:
        audio = (np.frombuffer(raw, dtype=np.int32) >> 16).astype(np.int16)
    else:
        raise ValueError(f"Unsupported sample width: {sw}")

    if ch > 1:
        audio = audio[::ch]

    if sr != 16000:
        from scipy.signal import resample
        audio_f = audio.astype(np.float32)
        audio = resample(audio_f, int(len(audio_f) * 16000 / sr)).astype(np.int16)

    return audio


def generate_features(audio_int16):
    """用 pymicro_features 生成 40-ch mel filterbank 特征（和 MWW 训练一致）"""
    frontend = MicroFrontend()

    # process_samples 每次喂 160 个 int16 样本（10ms step）
    # 内部积累够 30ms 窗口后输出 40 维特征
    step_samples = 160
    features = []
    offset = 0
    while offset + step_samples <= len(audio_int16):
        chunk = audio_int16[offset:offset + step_samples]
        result = frontend.process_samples(chunk)
        if result.features:
            features.append(result.features)
        offset += result.samples_read

    if not features:
        return np.zeros((0, 40), dtype=np.float32)

    return np.array(features, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cutoff", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=1)
    args = parser.parse_args()

    import tensorflow as tf

    # 加载 manifest
    with open(os.path.join(args.test_dir, "manifest.json")) as f:
        manifest = json.load(f)

    # 加载模型
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    in_shape = inp["shape"]       # [1, 3, 40]
    stride = in_shape[1]          # 3
    n_feats = in_shape[2]         # 40
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]

    print(f"模型: {args.model}")
    print(f"输入: shape={list(in_shape)}, dtype={inp['dtype']}, scale={in_scale}, zero={in_zero}")
    print(f"输出: scale={out_scale}, zero={out_zero}")
    print(f"cutoff={args.cutoff}, window={args.window}")
    print()

    results = []
    for item in manifest:
        wav_path = os.path.join(args.test_dir, item["file"])
        if not os.path.exists(wav_path):
            continue

        try:
            audio = load_wav_16k_int16(wav_path)
            features = generate_features(audio)
        except Exception as e:
            print(f"  跳过 {item['file']}: {e}")
            continue

        if features.shape[0] < stride:
            print(f"  跳过 {item['file']}: 太短 ({features.shape[0]} frames)")
            continue

        # 滑窗推理
        probs = []
        for start in range(0, features.shape[0] - stride + 1, stride):
            chunk = features[start:start + stride]  # [3, 40]

            # 量化输入
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
        item["all_probs"] = [round(p, 4) for p in probs]
        results.append(item)

    # 打印结果
    print("=" * 90)
    print(f"{'类别':<10} {'文本':<14} {'声音':<12} {'最大概率':>8} {'检测':>6} {'判定':>10} {'概率序列'}")
    print("=" * 90)

    errors = []
    for r in sorted(results, key=lambda x: (x["category"], x["text"], x["voice"])):
        cat = r["category"]
        det = r["detected"]
        mp = r["max_prob"]
        voice_short = r["voice"].split("-")[-1].replace("Neural", "")
        probs_str = ",".join(f"{p:.2f}" for p in r.get("all_probs", []))

        if cat in ("positive", "dialect"):
            mark = "✓" if det else "✗漏报"
            correct = det
        else:
            mark = "✓" if not det else "✗误触"
            correct = not det

        if not correct:
            errors.append(r)

        print(f"{cat:<10} {r['text']:<14} {voice_short:<12} {mp:>8.4f} {'是' if det else '否':>6} {mark:>10}  [{probs_str}]")

    print("=" * 90)
    print(f"\n总计: {len(results)} 样本, 错误: {len(errors)}")

    for cat in ["positive", "dialect", "negative"]:
        items = [r for r in results if r["category"] == cat]
        if not items:
            continue
        if cat in ("positive", "dialect"):
            hits = sum(1 for r in items if r["detected"])
            print(f"  {cat}: {hits}/{len(items)} 触发 (recall={hits/len(items)*100:.1f}%)")
        else:
            fps = sum(1 for r in items if r["detected"])
            print(f"  {cat}: {fps}/{len(items)} 误触发 (FPR={fps/len(items)*100:.1f}%)")

    if errors:
        print("\n⚠️  错误:")
        for e in errors:
            cat = e["category"]
            tp = "漏报" if cat in ("positive", "dialect") else "误触发"
            print(f"  [{tp}] {e['text']} ({e['voice'].split('-')[-1]}) max_prob={e['max_prob']}")

    with open(os.path.join(args.test_dir, "results_v2.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
