#!/usr/bin/env python3
"""从已有的测试音频文件名重建 manifest.json"""
import json, os, sys

test_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/verify_jiuming_v4"
manifest = []
for f in sorted(os.listdir(test_dir)):
    if not f.endswith(".wav"):
        continue
    parts = f.replace(".wav", "").split("_", 2)
    if len(parts) < 3:
        continue
    category = parts[0]
    # text is between first and last underscore-separated voice
    # format: category_text_voice.wav
    voice_map = {"Xiaoxiao": "zh-CN-XiaoxiaoNeural", "Yunxi": "zh-CN-YunxiNeural", "Xiaoyi": "zh-CN-XiaoyiNeural"}
    voice_short = None
    for v in voice_map:
        if f.endswith(f"_{v}.wav"):
            voice_short = v
            break
    if not voice_short:
        continue
    text = f.replace(".wav", "").replace(f"{category}_", "", 1).replace(f"_{voice_short}", "", 1)
    manifest.append({
        "file": f,
        "text": text,
        "category": category,
        "voice": voice_map[voice_short],
    })

out = os.path.join(test_dir, "manifest.json")
with open(out, "w", encoding="utf-8") as fp:
    json.dump(manifest, fp, ensure_ascii=False, indent=2)
print(f"写入 {len(manifest)} 条 → {out}")
