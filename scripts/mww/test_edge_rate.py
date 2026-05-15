import asyncio, edge_tts, os

async def test(rate, name):
    try:
        comm = edge_tts.Communicate('hello', 'en-US-AriaNeural', rate=rate)
        await comm.save(f'/tmp/test_{name}.mp3')
        ok = os.path.exists(f'/tmp/test_{name}.mp3') and os.path.getsize(f'/tmp/test_{name}.mp3') > 500
        print(f'  rate={rate!r}: {"ok" if ok else "FAIL"}')
    except Exception as e:
        print(f'  rate={rate!r}: EXCEPTION {type(e).__name__}: {e}')

async def main():
    await test('-15%', 'neg15')
    await test('0%', 'zero')
    await test('+0%', 'pluszero')
    await test('-10%', 'neg10')

asyncio.run(main())
