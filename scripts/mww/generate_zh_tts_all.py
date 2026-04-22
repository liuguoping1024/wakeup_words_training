#!/usr/bin/env python3
"""
用多个中文 TTS 引擎生成正样本和对抗性负样本。
在宿主机上运行（edge-tts 需要网络，Piper 离线）。

TTS 引擎：
  1. edge-tts: 微软在线 API，19 种中文声音
  2. Piper: 离线，zh_CN-huayan-medium（1 种声音）

CosyVoice2 单独在 Docker 内运行，见 generate_cosyvoice_jiuming.py

用法：
  python3 scripts/mww/generate_zh_tts_all.py \
    --keyword "救命" --keyword-id jiuming \
    --pos-dir data/tts_positive/jiuming \
    --neg-dir data/tts_negative/jiuming \
    --n-pos-edge 1500 --n-pos-piper 200 \
    --n-neg-edge 2000 --n-neg-piper 300
"""
import argparse
import asyncio
import logging
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)

# ── edge-tts 中文声音 ──
EDGE_VOICES = [
    "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-CN-YunjianNeural",
    "zh-CN-XiaoyiNeural", "zh-CN-YunyangNeural", "zh-CN-XiaochenNeural",
    "zh-CN-XiaohanNeural", "zh-CN-XiaomengNeural", "zh-CN-XiaomoNeural",
    "zh-CN-XiaoqiuNeural", "zh-CN-XiaoruiNeural", "zh-CN-XiaoshuangNeural",
    "zh-CN-XiaoxuanNeural", "zh-CN-XiaoyanNeural", "zh-CN-XiaozhenNeural",
    "zh-CN-YunfengNeural", "zh-CN-YunhaoNeural", "zh-CN-YunxiaNeural",
    "zh-CN-YunzeNeural",
]
EDGE_RATES = ["-25%", "-15%", "-10%", "-5%", "+0%", "+5%", "+10%", "+15%", "+25%"]

# ── 对抗性负样本短语 ──
ADVERSARIAL = {
    "jiuming": [
        "旧命", "就命", "酒名", "九命", "久明", "纠明",
        "救民", "救星", "救人", "救火", "救护", "救急",
        "救", "命", "救了", "命令", "生命", "性命", "拼命",
        "揪命", "究命", "灸命",
        "小爱同学", "天猫精灵", "你好小度",
        "打开灯", "关闭窗帘", "今天天气", "播放音乐",
        "谢谢", "再见", "对不起", "没关系", "早上好",
        "你好", "帮帮我", "快来", "怎么办", "着火了",
        "报警", "快跑", "小心", "危险", "停下",
    ],
    "nihao_shushi": [
        "你好叔叔", "你好书式", "你好树枝", "你好属实",
        "你好舒适", "你好数十", "你好书市", "你好熟食",
        "你好", "树实", "你好啊", "你好吗",
        "你好世界", "你好师傅", "你好同学", "你好朋友",
        "泥好树实", "拟好树实", "你浩树实", "你号树实",
        "小爱同学", "天猫精灵", "你好小度",
        "打开灯", "关闭窗帘", "今天天气", "播放音乐",
    ],
}


# ═══════════════════════════════════════════════════════════════
# edge-tts 生成
# ═══════════════════════════════════════════════════════════════

async def edge_tts_batch(output_dir, texts, n_target, prefix, is_single=False):
    """用 edge-tts 批量生成 WAV。"""
    import edge_tts
    os.makedirs(output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".wav") and f.startswith(prefix)])
    if existing >= n_target * 0.95:
        log.info(f"  edge-tts {prefix}: 已有 {existing}/{n_target}，跳过")
        return existing

    log.info(f"  edge-tts {prefix}: 生成 {n_target - existing} 条到 {output_dir}")
    count = existing
    for i in range(existing, n_target):
        text = texts if is_single else random.choice(texts)
        voice = random.choice(EDGE_VOICES)
        rate = random.choice(EDGE_RATES)
        mp3 = os.path.join(output_dir, f"{prefix}_{i:05d}.mp3")
        wav = os.path.join(output_dir, f"{prefix}_{i:05d}.wav")
        if os.path.exists(wav):
            count += 1
            continue
        for attempt in range(3):
            try:
                c = edge_tts.Communicate(text, voice, rate=rate)
                await c.save(mp3)
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3,
                     "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav],
                    check=True)
                if os.path.exists(mp3):
                    os.remove(mp3)
                count += 1
                break
            except Exception:
                await asyncio.sleep(1 + attempt * 2)
        if i % 10 == 0:
            await asyncio.sleep(0.3)
        if count % 500 == 0 and count > existing:
            log.info(f"    进度: {count}/{n_target}")
    log.info(f"  edge-tts {prefix}: 完成 {count} 条")
    return count


# ═══════════════════════════════════════════════════════════════
# Piper 生成
# ═══════════════════════════════════════════════════════════════

def piper_batch(output_dir, texts, n_target, prefix, piper_model, piper_config,
                is_single=False):
    """用 Piper TTS 批量生成 WAV。"""
    os.makedirs(output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".wav") and f.startswith(prefix)])
    if existing >= n_target * 0.95:
        log.info(f"  piper {prefix}: 已有 {existing}/{n_target}，跳过")
        return existing

    # 检查 piper 命令
    piper_cmd = shutil.which("piper")
    if piper_cmd is None:
        log.warning("  piper 命令不可用，跳过 Piper 生成")
        return existing

    if not os.path.exists(piper_model):
        log.warning(f"  Piper 模型不存在: {piper_model}，跳过")
        return existing

    log.info(f"  piper {prefix}: 生成 {n_target - existing} 条到 {output_dir}")
    noise_scales = [0.4, 0.5, 0.667, 0.8, 1.0]
    length_scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    noise_w_scales = [0.5, 0.8, 1.0]
    count = existing

    for i in range(existing, n_target):
        text = texts if is_single else random.choice(texts)
        wav = os.path.join(output_dir, f"{prefix}_{i:05d}.wav")
        if os.path.exists(wav):
            count += 1
            continue
        ns = random.choice(noise_scales)
        ls = random.choice(length_scales)
        nw = random.choice(noise_w_scales)
        try:
            raw_wav = wav + ".raw.wav"
            proc = subprocess.run(
                [piper_cmd, "--model", piper_model,
                 "--noise-scale", str(ns), "--length-scale", str(ls),
                 "--noise-w-scale", str(nw),
                 "--output-file", raw_wav],
                input=text.encode("utf-8"), capture_output=True, timeout=30)
            if proc.returncode == 0 and os.path.exists(raw_wav):
                # Piper 输出 22050Hz，重采样到 16kHz
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", raw_wav,
                     "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav],
                    check=True)
                os.remove(raw_wav)
                count += 1
            else:
                if proc.stderr:
                    log.warning(f"    piper 错误: {proc.stderr.decode()[:100]}")
        except Exception as e:
            log.warning(f"    piper 生成失败 {i}: {e}")
        if count % 100 == 0 and count > existing:
            log.info(f"    进度: {count}/{n_target}")
    log.info(f"  piper {prefix}: 完成 {count} 条")
    return count


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="多 TTS 引擎生成中文正/负样本")
    parser.add_argument("--keyword", required=True, help="唤醒词文本，如 '救命'")
    parser.add_argument("--keyword-id", required=True, help="唤醒词 ID，如 jiuming")
    parser.add_argument("--pos-dir", required=True, help="正样本输出目录")
    parser.add_argument("--neg-dir", required=True, help="负样本输出目录")
    # edge-tts 数量
    parser.add_argument("--n-pos-edge", type=int, default=1500)
    parser.add_argument("--n-neg-edge", type=int, default=2000)
    # Piper 数量
    parser.add_argument("--n-pos-piper", type=int, default=200)
    parser.add_argument("--n-neg-piper", type=int, default=300)
    # Piper 模型路径
    parser.add_argument("--piper-model",
                        default="work/piper-sample-generator-oww/models/piper_onnx/zh_CN-huayan-medium.onnx")
    parser.add_argument("--piper-config",
                        default="work/piper-sample-generator-oww/models/piper_onnx/zh_CN-huayan-medium.onnx.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    neg_phrases = ADVERSARIAL.get(args.keyword_id, ADVERSARIAL.get("jiuming"))

    # ── 1. edge-tts 正样本 ──
    log.info(f"=== edge-tts 正样本 ({args.keyword}) ===")
    asyncio.run(edge_tts_batch(
        args.pos_dir, args.keyword, args.n_pos_edge, "edge_pos", is_single=True))

    # ── 2. edge-tts 负样本 ──
    log.info(f"=== edge-tts 负样本 ({len(neg_phrases)} 短语) ===")
    asyncio.run(edge_tts_batch(
        args.neg_dir, neg_phrases, args.n_neg_edge, "edge_neg"))

    # ── 3. Piper 正样本 ──
    log.info(f"=== Piper 正样本 ({args.keyword}) ===")
    piper_batch(args.pos_dir, args.keyword, args.n_pos_piper, "piper_pos",
                args.piper_model, args.piper_config, is_single=True)

    # ── 4. Piper 负样本 ──
    log.info(f"=== Piper 负样本 ===")
    piper_batch(args.neg_dir, neg_phrases, args.n_neg_piper, "piper_neg",
                args.piper_model, args.piper_config)

    # ── 统计 ──
    pos_count = len([f for f in os.listdir(args.pos_dir) if f.endswith(".wav")])
    neg_count = len([f for f in os.listdir(args.neg_dir) if f.endswith(".wav")]) if os.path.isdir(args.neg_dir) else 0
    log.info(f"=== 完成: 正样本 {pos_count} 条, 负样本 {neg_count} 条 ===")


if __name__ == "__main__":
    main()
