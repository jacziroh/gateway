import base64
import io
import math
import struct
import time
import wave
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
 
app = FastAPI(title="KAI Mock Upstream")
 
 
def generate_test_image(prompt: str, width: int = 512, height: int = 512) -> str:
    try:
        from PIL import Image, ImageDraw, ImageFont
        h = hash(prompt) % 360
        c, x, m = 0.6, 0.6 * (1 - abs((h / 60) % 2 - 1)), 0.2
        if h < 60:    r, g, b = c, x, 0
        elif h < 120: r, g, b = x, c, 0
        elif h < 180: r, g, b = 0, c, x
        elif h < 240: r, g, b = 0, x, c
        elif h < 300: r, g, b = x, 0, c
        else:         r, g, b = c, 0, x
        color = (int((r+m)*255), int((g+m)*255), int((b+m)*255))
 
        img = Image.new("RGB", (width, height), color=color)
        draw = ImageDraw.Draw(img)
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except Exception:
            font_large = ImageFont.load_default()
            font_small = font_large
 
        draw.text((20, 20), "KAI Image API", fill="white", font=font_large)
        draw.text((20, 60), "Demo Output", fill=(200, 200, 200), font=font_small)
 
        words = prompt.split()
        lines, line = [], ""
        for word in words:
            if len(line + " " + word) > 40:
                lines.append(line)
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            lines.append(line)
 
        y = height // 2 - len(lines) * 12
        for lt in lines[:6]:
            draw.text((30, y), f'"{lt}"', fill="white", font=font_small)
            y += 24
 
        draw.text((20, height - 40), f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", fill=(200, 200, 200), font=font_small)
 
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        pixel = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x02"
            b"\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return base64.b64encode(pixel).decode()
 
 
def generate_test_audio(duration_sec: float = 2.0) -> bytes:
    sample_rate = 16000
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(int(sample_rate * duration_sec)):
            t = i / sample_rate
            envelope = min(t * 10, 1.0) * min((duration_sec - t) * 10, 1.0)
            sample = int(16000 * envelope * math.sin(2 * math.pi * 440 * t))
            wf.writeframes(struct.pack("<h", max(-32768, min(32767, sample))))
    return buf.getvalue()
 
 
# ── Image ──
 
@app.post("/images/generations")
async def images_generate(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "no prompt")
    n = body.get("n", 1)
    model = body.get("model", "unknown")
    print(f"[mock] /images/generations | model={model} | prompt=\"{prompt[:80]}\" | n={n}")
 
    data = [{"b64_json": generate_test_image(f"{prompt} #{i+1}")} for i in range(min(n, 4))]
    return JSONResponse({
        "created": int(time.time()),
        "data": data,
        "model": model,
        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": 0, "total_tokens": len(prompt.split())},
    })
 
 
@app.post("/images/edits")
async def images_edit(req: Request):
    form = await req.form()
    model = form.get("model", "unknown")
    prompt = form.get("prompt", "no prompt")
    image = form.get("image")
    image_size = len(await image.read()) if image and hasattr(image, "read") else 0
    print(f"[mock] /images/edits | model={model} | prompt=\"{prompt[:80]}\" | image={image_size} bytes")
 
    return JSONResponse({
        "created": int(time.time()),
        "data": [{"b64_json": generate_test_image(f"EDITED: {prompt}")}],
        "model": model,
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    })
 
 
# ── Audio ──
 
@app.post("/audio/transcriptions")
async def audio_transcribe(req: Request):
    form = await req.form()
    model = form.get("model", "unknown")
    audio = form.get("file")
    audio_size = len(await audio.read()) if audio and hasattr(audio, "read") else 0
    language = form.get("language", "en")
    print(f"[mock] /audio/transcriptions | model={model} | audio={audio_size} bytes | lang={language}")
 
    return JSONResponse({
        "text": "Welcome to the quarterly review meeting. Today we will discuss the progress on Project Alpha and review the budget for Q3.",
        "model": model,
        "language": language,
        "duration": 12.5,
        "usage": {"prompt_tokens": audio_size // 100, "total_tokens": audio_size // 100},
    })
 
 
@app.post("/audio/translations")
async def audio_translate(req: Request):
    form = await req.form()
    model = form.get("model", "unknown")
    audio = form.get("file")
    audio_size = len(await audio.read()) if audio and hasattr(audio, "read") else 0
    print(f"[mock] /audio/translations | model={model} | audio={audio_size} bytes")
 
    return JSONResponse({
        "text": "Hello, my name is Wolfgang and I come from Germany. Today I want to present our new product to the international team.",
        "model": model,
        "language": "en",
        "duration": 8.3,
        "usage": {"prompt_tokens": audio_size // 100, "total_tokens": audio_size // 100},
    })
 
 
@app.post("/audio/speech")
async def audio_speech(req: Request):
    body = await req.json()
    model = body.get("model", "unknown")
    text = body.get("input", "")
    voice = body.get("voice", "nova")
    print(f"[mock] /audio/speech | model={model} | voice={voice} | text=\"{text[:80]}\"")
 
    return Response(content=generate_test_audio(2.0), media_type="audio/wav")
 
 
# ── Embeddings ──
 
@app.post("/embeddings")
async def embeddings(req: Request):
    body = await req.json()
    model = body.get("model", "unknown")
    input_text = body.get("input", "")
    texts = [input_text] if isinstance(input_text, str) else input_text if isinstance(input_text, list) else [str(input_text)]
    print(f"[mock] /embeddings | model={model} | inputs={len(texts)}")
 
    data = []
    for i, t in enumerate(texts):
        h = hash(t)
        data.append({"object": "embedding", "index": i, "embedding": [math.sin(h + j * 0.1) * 0.01 for j in range(1536)]})
 
    return JSONResponse({
        "object": "list",
        "data": data,
        "model": model,
        "usage": {"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": sum(len(t.split()) for t in texts)},
    })
 
 
# ── Health & Models ──
 
@app.get("/health")
async def health():
    return {"status": "ok", "server": "kai-mock-upstream"}
 
@app.get("/models")
async def models_list():
    return {
        "object": "list",
        "data": [
            {"id": "test-vit-model", "object": "model", "owned_by": "kai"},
            {"id": "test-whisper", "object": "model", "owned_by": "kai"},
            {"id": "test-tts", "object": "model", "owned_by": "kai"},
            {"id": "test-embed", "object": "model", "owned_by": "kai"},
        ]
    }
 
 
if __name__ == "__main__":
    import uvicorn
    print()
    print("=" * 55)
    print("  KAI Mock Upstream — All 6 APIs Ready")
    print("=" * 55)
    print("  Images:     /images/generations, /images/edits")
    print("  Audio:      /audio/transcriptions, /audio/translations, /audio/speech")
    print("  Embeddings: /embeddings")
    print()
    print("  Running on: http://localhost:8080")
    print("=" * 55)
    print()
    uvicorn.run(app, host="0.0.0.0", port=8080)