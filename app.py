import asyncio
import io

import edge_tts
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from gtts import gTTS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "TTS backend is running"}


# ---------- ENGINE: Microsoft Edge Neural TTS ----------
async def generate_edge_audio(text: str, voice: str, rate: str, pitch: str, max_retries: int = 3) -> bytes:
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

            raise edge_tts.exceptions.NoAudioReceived(
                "No audio was received. Please verify that your parameters are correct."
            )
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                await asyncio.sleep(0.7 * attempt)
                continue
            break

    raise last_error


# ---------- ENGINE: Google TTS (backup, more stable, fewer voices) ----------
def generate_google_audio(text: str, lang: str) -> bytes:
    buf = io.BytesIO()
    tts = gTTS(text=text, lang=lang)
    tts.write_to_fp(buf)
    return buf.getvalue()


async def generate_audio(text: str, voice: str, rate: str, pitch: str, engine: str, max_retries: int = 3) -> bytes:
    if engine == "google":
        lang = voice.split("-")[0] if "-" in voice else voice
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, generate_google_audio, text, lang)
    return await generate_edge_audio(text, voice, rate, pitch, max_retries=max_retries)


@app.get("/test-voices")
async def test_voices(voices: str = Query(...), engine: str = Query("edge")):
    """
    Test a batch of voices with a short sample text and report which ones
    actually return audio vs which ones fail.
    Example: /test-voices?voices=en-US-AdamNeural,en-US-GuyNeural&engine=edge
    """
    voice_list = [v.strip() for v in voices.split(",") if v.strip()]
    results = {}

    async def check_one(v: str):
        try:
            await generate_audio("testing one two three", v, "+0%", "+0Hz", engine, max_retries=1)
            results[v] = "working"
        except Exception:
            results[v] = "broken"

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
    engine: str = Query("edge", description="edge or google"),
):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        audio_bytes = await generate_audio(text, voice, rate, pitch, engine)
    except edge_tts.exceptions.NoAudioReceived:
        raise HTTPException(
            status_code=502,
            detail="Microsoft TTS server se audio nahi mila, dobara try karein (ya 'Google' engine try karein).",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=speech.mp3"},
    )


@app.get("/preview")
async def preview(
    voice: str = Query(...),
    engine: str = Query("edge"),
):
    """Short fixed sample so users can hear what a voice sounds like before using it."""
    sample_text = "This is a quick preview of this voice."
    try:
        audio_bytes = await generate_audio(sample_text, voice, "+0%", "+0Hz", engine, max_retries=2)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Preview failed: {str(e)}")

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=preview.mp3"},
    )
