import asyncio
import io
import os
import struct

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

# ---------- EXPRESSION / MOOD PRESETS ----------
# rate/pitch overrides apply to Edge TTS; gemini_style is a natural-language
# instruction prepended to the text for Gemini TTS (which understands tone/emotion directly).
EXPRESSION_PRESETS = {
    "normal":  {"rate": 0,   "pitch": 0,   "gemini_style": None},
    "chill":   {"rate": -15, "pitch": -5,  "gemini_style": "Say in a calm, relaxed, laid-back tone:"},
    "fast":    {"rate": 35,  "pitch": 0,   "gemini_style": "Say quickly, in a fast-paced and energetic tone:"},
    "slow":    {"rate": -30, "pitch": 0,   "gemini_style": "Say slowly and deliberately, in a calm tone:"},
    "excited": {"rate": 20,  "pitch": 15,  "gemini_style": "Say in an excited, enthusiastic, upbeat tone:"},
    "sad":     {"rate": -10, "pitch": -12, "gemini_style": "Say in a sad, somber, melancholic tone:"},
    "whisper": {"rate": -10, "pitch": -5,  "gemini_style": "Whisper softly and gently:"},
    "angry":   {"rate": 10,  "pitch": 12,  "gemini_style": "Say in an angry, intense, forceful tone:"},
    "dramatic":{"rate": -5,  "pitch": 8,   "gemini_style": "Say in a dramatic, theatrical, suspenseful tone:"},
}


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


# ---------- ENGINE: Google Gemini TTS (real AI character voices) ----------
def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Gemini TTS returns raw PCM audio — wrap it in a WAV header so browsers can play it."""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    )
    return header + pcm_bytes


def generate_gemini_audio(text: str, voice: str, style: str = "normal") -> bytes:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set on the server.")

    from google import genai
    from google.genai import types

    preset = EXPRESSION_PRESETS.get(style, EXPRESSION_PRESETS["normal"])
    gemini_style = preset.get("gemini_style")
    final_text = f"{gemini_style} {text}" if gemini_style else text

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=final_text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        ),
    )
    pcm_data = response.candidates[0].content.parts[0].inline_data.data
    return _pcm_to_wav(pcm_data)


def engine_mime_type(engine: str) -> str:
    return "audio/wav" if engine == "gemini" else "audio/mpeg"


async def generate_audio(text: str, voice: str, rate: str, pitch: str, engine: str, style: str = "normal", max_retries: int = 3) -> bytes:
    preset = EXPRESSION_PRESETS.get(style, EXPRESSION_PRESETS["normal"])

    if engine == "google":
        lang = voice.split("-")[0] if "-" in voice else voice
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, generate_google_audio, text, lang)
    if engine == "gemini":
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, generate_gemini_audio, text, voice, style)

    # Edge TTS: if an expression preset is chosen (not "normal"), it overrides the manual rate/pitch sliders
    if style != "normal":
        rate = f"{'+' if preset['rate'] >= 0 else ''}{preset['rate']}%"
        pitch = f"{'+' if preset['pitch'] >= 0 else ''}{preset['pitch']}Hz"
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

    if engine == "gemini":
        # Gemini's free tier has strict per-minute rate limits — go one at a time
        # with a small gap so the check doesn't trigger a false "all broken" result.
        for v in voice_list:
            await check_one(v)
            await asyncio.sleep(2.0)
    else:
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
    engine: str = Query("edge", description="edge, google, or gemini"),
    style: str = Query("normal", description="normal, chill, fast, slow, excited, sad, whisper, angry, dramatic"),
):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        audio_bytes = await generate_audio(text, voice, rate, pitch, engine, style=style)
    except edge_tts.exceptions.NoAudioReceived:
        raise HTTPException(
            status_code=502,
            detail="Microsoft TTS server se audio nahi mila, dobara try karein (ya 'Google'/'Gemini' engine try karein).",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

    mime = engine_mime_type(engine)
    ext = "wav" if engine == "gemini" else "mp3"
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type=mime,
        headers={"Content-Disposition": f"inline; filename=speech.{ext}"},
    )


@app.get("/preview")
async def preview(
    voice: str = Query(...),
    engine: str = Query("edge"),
    style: str = Query("normal"),
):
    """Short fixed sample so users can hear what a voice sounds like before using it."""
    sample_text = "This is a quick preview of this voice."
    try:
        audio_bytes = await generate_audio(sample_text, voice, "+0%", "+0Hz", engine, style=style, max_retries=2)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Preview failed: {str(e)}")

    mime = engine_mime_type(engine)
    ext = "wav" if engine == "gemini" else "mp3"
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type=mime,
        headers={"Content-Disposition": f"inline; filename=preview.{ext}"},
    )
