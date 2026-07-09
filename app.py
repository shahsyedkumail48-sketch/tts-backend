from fastapi import FastAPI, Query, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import edge_tts
import io
import asyncio
import os
from supabase import create_client, Client

app = FastAPI()

# Supabase setup
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://bkjggmanvcjcsjevqglo.supabase.co')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

if not SUPABASE_SERVICE_KEY:
    print("WARNING: SUPABASE_SERVICE_ROLE_KEY not set!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Allow requests from any website
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


async def synthesize_with_retry(text, voice, rate, pitch, max_retries=3):
    last_error = None
    for attempt in range(max_retries):
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            audio_buffer = io.BytesIO()
            got_audio = False
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
                    got_audio = True
            if got_audio:
                audio_buffer.seek(0)
                return audio_buffer
            raise Exception("No audio data received")
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(1)  # wait a bit before retrying
            continue
    raise last_error


@app.get("/speak")
async def speak(
    text: str = Query(..., description="Text to convert to speech"),
    voice: str = Query("en-US-JennyNeural", description="Voice name"),
    rate: str = Query("+0%", description="Speech rate, e.g. +10%, -20%"),
    pitch: str = Query("+0Hz", description="Pitch, e.g. +5Hz, -10Hz"),
    user_id: str = Query(None, description="User ID for credit deduction"),
):
    try:
        # Generate audio
        audio_buffer = await synthesize_with_retry(text, voice, rate, pitch)
        
        # Deduct credits if user_id provided
        if user_id and SUPABASE_SERVICE_KEY:
            try:
                # Calculate credits to deduct (1 credit per character, minimum 1)
                char_count = len(text.strip())
                credits_to_deduct = max(1, char_count)
                
                # Get user's current credits
                user_data = supabase.table("profiles").select("credits").eq("id", user_id).single().execute()
                current_credits = user_data.data.get("credits", 0) if user_data.data else 0
                
                # Check if user has enough credits
                if current_credits < credits_to_deduct:
                    return JSONResponse(
                        status_code=402,
                        content={"error": "Insufficient credits", "required": credits_to_deduct, "available": current_credits},
                    )
                
                # Deduct credits
                new_credits = current_credits - credits_to_deduct
                supabase.table("profiles").update({"credits": new_credits}).eq("id", user_id).execute()
                
            except Exception as credit_error:
                print(f"Credit deduction error: {credit_error}")
                # Don't fail the request if credit deduction fails
                pass
        
        return StreamingResponse(audio_buffer, media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Speech generation failed", "detail": str(e)},
        )
