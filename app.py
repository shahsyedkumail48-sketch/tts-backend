import asyncio
import io

import edge_tts
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Edge TTS backend is running"}


async def generate_audio(text: str, voice: str, rate: str, pitch: str, max_retries: int = 3) -> bytes:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            audio_buffer = io.BytesIO()
            got_audio = False
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
                    got_audio = True

            if got_audio and audio_buffer.tell() > 0:
                return audio_buffer.getvalue()

            # No audio received this attempt — treat as failure and retry
            raise edge_tts.exceptions.NoAudioReceived(
                "No audio was received. Please verify that your parameters are correct."
            )
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                # small backoff before retrying
                await asyncio.sleep(0.7 * attempt)
                continue
            break

    raise last_error


@app.get("/test-voices")
async def test_voices(voices: str = Query(..., description="Comma-separated list of voice short-names")):
    """
    Test a batch of voices with a short sample text and report which ones
    actually return audio vs which ones fail. Use this once to find out
    which voices are currently broken on Microsoft's side.
    Example: /test-voices?voices=en-US-AdamNeural,en-US-GuyNeural
    """
    voice_list = [v.strip() for v in voices.split(",") if v.strip()]
    results = {}

    async def check_one(v: str):
        try:
            await generate_audio("testing one two three", v, "+0%", "+0Hz", max_retries=1)
            results[v] = "working"
        except Exception:
            results[v] = "broken"

    # run in small batches so we don't hammer Microsoft's server / Railway's memory
    batch_size = 5
    for i in range(0, len(voice_list), batch_size):
        batch = voice_list[i : i + batch_size]
        await asyncio.gather(*(check_one(v) for v in batch))

    working = [v for v, s in results.items() if s == "working"]
    broken = [v for v, s in results.items() if s == "broken"]
    return {"total": len(voice_list), "working": working, "broken": broken}


@app.get("/speak")
async def speak(
    text: str = Query(...),
    voice: str = Query("en-US-AriaNeural"),
    rate: str = Query("+0%"),
    pitch: str = Query("+0Hz"),
):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        audio_bytes = await generate_audio(text, voice, rate, pitch)
    except edge_tts.exceptions.NoAudioReceived:
        raise HTTPException(
            status_code=502,
            detail="Microsoft TTS server se audio nahi mila, dobara try karein (ho sakta hai sirf is waqt server busy ho).",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=speech.mp3"},
    )
