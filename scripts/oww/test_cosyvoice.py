#!/usr/bin/env python3
"""Quick test for CosyVoice2 in Docker."""
import sys
sys.path.insert(0, '/workspace/CosyVoice')
sys.path.insert(0, '/workspace/CosyVoice/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

print("Loading CosyVoice2...")
cosyvoice = AutoModel(model_dir='/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B')
print("Model loaded! Sample rate:", cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_zero_shot(
    '你好树实',
    '希望你以后能够做的比我还好呦。',
    '/workspace/CosyVoice/asset/zero_shot_prompt.wav',
    stream=False
)):
    torchaudio.save('/workspace/outputs/test_cosyvoice.wav', j['tts_speech'], cosyvoice.sample_rate)
    print("Generated!", j['tts_speech'].shape)
    break

print("Done!")
