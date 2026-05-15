# CLAUDE.md — Wake Word Training Project

This document summarizes all work performed on this wake word training project.

## Project Overview

Custom wake word model training using two frameworks:
- **MWW** (micro-wake-word) — TensorFlow → quantized TFLite for ESP32/edge devices
- **OWW** (openWakeWord) — PyTorch → ONNX for Linux/Android (experimental)

Hardware: 4× GTX 1080 Ti (sm_61, 11GB), CUDA 12.3, Ubuntu 20.04.
All training and TTS generation runs in Docker containers.

## Wake Words Trained

### 1. 你好树实 (nihao_shushi) — Chinese
- **Status**: Done (v2 deployed)
- **Real recordings**: 99,454 samples from 40 speakers
- **MWW v1**: 99.6% recall on real voices
- **MWW v2**: 97.6% recall, better generalization (CosyVoice + edge-tts augmented)
- **OWW v6**: 91.6% recall, 6.6% FPR (best OWW result before abandoning OWW)
- **OWW abandoned**: Consistently underperformed MWW for this task

### 2. 救命 (jiuming) — Chinese, single utterance
- **Status**: Done (v6 best version)
- **Real recordings**: 100 samples (single speaker)
- **Final model**: `outputs/jiuming_v6.tflite`
- **Recall on 100 real recordings**: 90% (cutoff=0.88, window=3)
- **Iterations**:
  - v1: 100% recall but only on training speaker
  - v3: 66.7% on 12 ARM test recordings
  - v4: 91.7% ARM, 57% on 100 real with proper streaming inference
  - v5: 25% (over-aggressive adversarial weight 20/3.0)
  - **v6**: **90%** real, 83.3% ARM, 9.9% FPR (real recordings × 10 + adversarial weight 8/1.5)

### 3. 救命救命 (jiuming2) — Chinese, double utterance
- **Status**: Done (v7 with cleaned data)
- **No real recordings** — pure TTS
- **Final model**: `outputs/jiuming2_v7.tflite`
- **Training metric**: AUC 0.988, FRR@0.57 = 3.1%
- **Iterations**:
  - v1-v3: 5K-12K positive samples, recall on TTS test ~35-60%
  - v4-v5: Data merging bugs (Piper subdirs not flattened, mmap empty)
  - **v6**: 13,810 positives with strict reduplicated phrases (救命救命/救民救民/etc)
  - **v7**: User manually cleaned ~2,260 bad samples → AUC jumped from 0.13 to 0.99
- **Key insight**: Bad TTS samples (mispronunciation) catastrophically hurt training; cleaning is essential

### 4. help help — English (in progress)
- **Status**: In progress (v6 latest)
- **No real recordings** — pure TTS
- **Final model**: `outputs/help_help_v6.tflite`
- **TTS positive recall**: 100%, FPR ~29% on similar phonemes
- **Iterations**:
  - v1: 95% recall on 100 real recordings (×5 replication)
  - v2-v3: Increased data + cleaning, but real recall dropped (over-aggressive negatives)
  - v4: Strict 3 positive phrases (help help / help help! / help! help!) but Piper too fast → "hope hope" sound, CosyVoice English from Chinese refs failed
  - v5: Failed (path bug, mmap empty)
  - **v6**: Slow Piper (length_scales 2.0/2.5/3.0) + edge-tts at -15% rate, no CosyVoice English

### 5. help me — English (DEPRECATED)
- **Status**: Removed (only 3 syllables, deemed insufficient by user)
- All artifacts cleaned up

## Key Technical Discoveries

### 1. Streaming Inference Requires State Maintenance
The MWW model has **internal state tensors** that must be carried across frames during inference. Initial verification scripts ignored these states, producing falsely high recall numbers. The correct inference pipeline is in `inference/runtime.py` (`WakeWordDetector` class).

**Impact**: All early "high recall" numbers were unreliable. After fixing inference (script: `scripts/mww/infer_verify_v3.py`), v4 recall on 100 jiuming samples dropped from claimed 99% to actual 57%.

### 2. Real Recordings × 10 Replication
For wake words with limited real recordings (100 samples), **replicating real recordings 10 times** in the training set dramatically improves recall:
- jiuming v4 (real ×0): 57% recall
- jiuming v6 (real ×10): **90% recall**

### 3. TTS Quality Matters More Than Quantity
After training a model, scan all positive samples through it. Files where `max_prob < 0.3` are likely bad TTS outputs (mispronunciations, truncations, wrong content):
- jiuming救命救命: 2,260 of 13,810 bad samples (16%) → cleaning improved AUC from 0.13 to 0.99
- help help v2: 699 of 4,932 bad samples (14%) → cleaning helped FPR but hurt recall

Tool: `scripts/mww/scan_suspicious_positives.py`

### 4. Cross-Lingual TTS Fails
CosyVoice2 with `inference_cross_lingual` using Chinese speaker refs to generate English produces garbage. The Chinese voice cloning fundamentally cannot synthesize quality English. **Don't mix languages between speaker refs and target text.**

### 5. Piper Speech Speed Control
Piper's `--length-scales` controls speed (default 0.75/1.0/1.25). For two-syllable English wake words like "help help", the default produces 0.6-1s audio that sounds like "hope hope". Use `2.0/2.5/3.0` for clearer articulation.

### 6. edge-tts Rate Format
edge-tts requires explicit sign: `"+0%"` not `"0%"`. `"-15%"` works for slower speech. Wrong format silently fails generation.

### 7. Adversarial Negative Sample Weights
Sweet spot for adversarial negative weights:
- Too low (5/1.0): Model confuses similar phonemes
- **Right (10/1.5)**: Best balance of recall and FPR
- Too high (15/2.0 or 20/3.0): Recall collapses, model rejects everything

### 8. Strict Positive Phrase Definition
For "save lives" wake words like 救命救命 / help help, **only count exact reduplications** as positives:
- ✅ "help help", "help help!", "help! help!"
- ❌ "help help please", "somebody help help" (treat as noise, not positives)
- ❌ Mixed reduplications like "救民救命" (unnatural in dialect)

## TTS Engines Used

| Engine | Type | Voice Count | Speed | Best For |
|--------|------|-------------|-------|----------|
| CosyVoice2 (0.5B) | Offline GPU | 40 (voice cloning) | ~18/min/GPU | Chinese multi-speaker positives |
| Piper huayan (zh_CN) | Offline GPU | 1 | ~75/min/GPU | Bulk Chinese positives |
| Piper libritts (en_US) | Offline GPU | Multi-speaker | ~75/min/GPU | English positives (use length_scale 2.0+) |
| edge-tts (Microsoft) | Online | 14 zh, 22 en | ~6/min | High-quality multi-voice |

4 GPU × parallel CosyVoice processes for ~12,000 samples in 3 hours.

## Adversarial Negative Sample Strategy

For Chinese wake words, the most effective negatives target the **rhyme** of the wake word:
- 救命 model errors: 说明/革命/聪明 (anything ending in -ming)
- Solution: Generate adversarial negatives heavy in -ming endings, weight=10/1.5
- AISHELL-1 (~30K random Chinese) provides general background coverage

For English help help, target **adjacent phonemes**:
- -elp: yelp, kelp, whelp, helping
- -ell: shell, smell, swell, fell, tell
- -elf: self, shelf, myself, yourself
- -elm: helm, helmet, elm, realm

## Pipeline Architecture

```
Step 1: TTS generation (parallel)
  - edge-tts (host, Python 3.12)
  - Piper (Docker oww image, GPU)
  - CosyVoice (Docker cosyvoice image, 4 GPUs in parallel)

Step 2: Merge audio with prefix tagging
  - data/positive_augmented/<keyword>_v<N>/
  - outputs/<keyword>_adversarial_v<N>/

Step 3: Generate mmap features (Docker mww image)
  - data/generated_augmented_features/  (positives)
  - data/negative_datasets/<adv>_v<N>/  (adversarial)

Step 4: Train (mixednet 64,64,64,64, 25k-30k steps, GPU 0)

Step 5: Export quantized TFLite to outputs/<keyword>_v<N>.tflite

Step 6: Verify with inference/runtime.py (NOT raw tflite inference)
```

## Important File Locations

- Models: `outputs/<keyword>_v<N>.tflite` + `.json`
- Positive samples: `data/positive_augmented/<keyword>_v<N>/`
- Adversarial negatives: `outputs/<keyword>_adversarial_v<N>/`
- Common negatives (mmap): `data/negative_datasets/`
- Inference runtime: `inference/runtime.py` (state-aware)
- Verification script: `scripts/mww/infer_verify_v3.py`
- Sample scanner (find bad TTS): `scripts/mww/scan_suspicious_positives.py`

## Final Models Summary

| Wake Word | Model | Real Recordings | Recall (real) | Recall (TTS) | Notes |
|-----------|-------|-----------------|---------------|--------------|-------|
| 你好树实 | `nihao_shushi_v2.tflite` | 99K | 97.6% | — | Production |
| 救命 | `jiuming_v6.tflite` | 100 (×10) | 90% | — | Production |
| 救命救命 | `jiuming2_v7.tflite` | 0 | — | (AUC 0.988) | Pure TTS, needs ARM verification |
| help help | `help_help_v6.tflite` | 0 | — | 100% / FPR 29% | English, needs improvement |

## Lessons Learned

1. **Verify with the deployment inference pipeline**, not the training framework's evaluation. Streaming model state tensors matter.
2. **Listen to TTS samples before training**. A self-scan can identify bad samples but human ears are still required for ambiguous cases (wrong text, wrong speech).
3. **Real recordings beat any amount of TTS** for personalization. ×10 replication is a cheap and effective trick.
4. **Different TTS engines have orthogonal failure modes**. CosyVoice may produce mispronunciation, Piper may speak too fast, edge-tts has rate format quirks. Mix all sources for diversity.
5. **Don't trust early metrics**. Pre-streaming-inference results were systematically wrong; always re-verify after pipeline changes.
6. **Strict positive phrase definition prevents drift**. Loose phrases like "somebody help help" dilute the signal.
