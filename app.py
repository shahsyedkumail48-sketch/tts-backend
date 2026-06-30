from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import edge_tts
import io

app = FastAPI()

# Allow requests from any website (so your HTML page hosted anywhere can call this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"status": "ok", "message": "Edge TTS backend is running"}


@app.get("/voices")
async def list_voices():
    voices = await edge_tts.list_voices()
    return [
        {"name": v["ShortName"], "gender": v["Gender"], "locale": v["Locale"]}
        for v in voices
    ]


@app.get("/speak")
async def speak(
    text: str = Query(..., description="Text to convert to speech"),
    voice: str = Query("en-US-JennyNeural", description="Voice name"),
    rate: str = Query("+0%", description="Speech rate, e.g. +10%, -20%"),
    pitch: str = Query("+0Hz", description="Pitch, e.g. +5Hz, -10Hz"),
):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)

    audio_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])

    audio_buffer.seek(0)
    return StreamingResponse(audio_buffer, media_type="audio/mpeg")
