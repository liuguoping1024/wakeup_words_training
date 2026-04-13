import asyncio, edge_tts, os, random, subprocess

phrases = ['你好叔叔','你好书式','你好树枝','你好属实','你好舒适','你好数十',
           '你好','树实','你好啊','你好吗','你好世界','你好师傅',
           '小爱同学','天猫精灵','你好小度','打开灯','关闭窗帘',
           '你好吃吗','你好看吗','你好厉害','树上有鸟','实在太好']
voices = ['zh-CN-XiaoxiaoNeural','zh-CN-YunxiNeural','zh-CN-YunjianNeural',
          'zh-CN-XiaoyiNeural','zh-CN-YunyangNeural']
rates = ['-20%','-10%','+0%','+10%','+20%']

async def gen(out_dir, n, prefix):
    os.makedirs(out_dir, exist_ok=True)
    existing = len([f for f in os.listdir(out_dir) if f.endswith('.wav')])
    if existing >= n:
        print(f'{out_dir}: already have {existing}')
        return
    count = existing
    for i in range(existing, n):
        text = random.choice(phrases)
        voice = random.choice(voices)
        rate = random.choice(rates)
        mp3 = f'{out_dir}/{prefix}_{i:05d}.mp3'
        wav = f'{out_dir}/{prefix}_{i:05d}.wav'
        if os.path.exists(wav):
            count += 1
            continue
        for attempt in range(3):
            try:
                c = edge_tts.Communicate(text, voice, rate=rate)
                await c.save(mp3)
                subprocess.run(['ffmpeg','-y','-loglevel','error','-i',mp3,
                    '-ac','1','-ar','16000','-sample_fmt','s16',wav], check=True)
                os.remove(mp3)
                count += 1
                break
            except:
                await asyncio.sleep(1 + attempt * 2)
        if count % 100 == 0 and count > 0:
            print(f'  {count}/{n}', flush=True)
    print(f'{out_dir}: done {count}', flush=True)

async def main():
    random.seed(42)
    await gen('outputs/oww/nihao_shushi/negative_train', 1000, 'neg')
    await gen('outputs/oww/nihao_shushi/negative_test', 200, 'neg')
    print('ALL DONE')

asyncio.run(main())
