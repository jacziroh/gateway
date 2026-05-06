import tempfile
from pathlib import Path
from typing import List, Union

import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration,
    pipeline,
    BertForMaskedLM,
)
from PIL import Image


app = FastAPI(title="Temporary Kompact AI Test Upstream")


# ----------------------------
# Model caches
# ----------------------------
_embedding_models = {}
_sentiment_models = {}
_text_generation_models = {}
_asr_pipelines = {}
_image_pipelines = {}
_fill_mask_models = {}


# ----------------------------
# Request schemas
# ----------------------------
class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]


class SentimentRequest(BaseModel):
    model: str
    input: Union[str, List[str]]


class TextGenerationRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    max_new_tokens: int = 128

class FillMaskRequest(BaseModel):
    model: str
    input: str
    top_k: int = 5

# ----------------------------
# Helpers
# ----------------------------
def as_list(value):
    if isinstance(value, str):
        return [value]
    return value


# ----------------------------
# Embeddings
# ----------------------------
@app.post("/embeddings")
def embeddings(req: EmbeddingRequest):
    try:
        texts = as_list(req.input)

        if req.model not in _embedding_models:
            print(f"Loading embedding model: {req.model}")
            _embedding_models[req.model] = SentenceTransformer(req.model)

        model = _embedding_models[req.model]
        vectors = model.encode(texts)

        return {
            "object": "list",
            "model": req.model,
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": vector.tolist(),
                }
                for i, vector in enumerate(vectors)
            ],
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error"}},
        )


# ----------------------------
# Sentiment
# ----------------------------
@app.post("/sentiment")
def sentiment(req: SentimentRequest):
    try:
        texts = as_list(req.input)

        if req.model not in _sentiment_models:
            print(f"Loading sentiment model: {req.model}")
            tokenizer = BertTokenizer.from_pretrained(req.model)
            model = BertForSequenceClassification.from_pretrained(req.model)
            model.eval()
            _sentiment_models[req.model] = (tokenizer, model)

        tokenizer, model = _sentiment_models[req.model]
        labels = {0: "Positive", 1: "Negative"}

        results = []

        for index, text in enumerate(texts):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=70,
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                predicted = torch.argmax(probs, dim=1).item()
                score = probs[0][predicted].item()

            results.append(
                {
                    "object": "sentiment",
                    "index": index,
                    "label": labels.get(predicted, str(predicted)),
                    "score": score,
                }
            )

        return {
            "object": "list",
            "model": req.model,
            "data": results,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error"}},
        )


# ----------------------------
# Text generation / Flan-T5
# ----------------------------
@app.post("/text/translations")
def text_generations(req: TextGenerationRequest):
    try:
        texts = as_list(req.input)

        if req.model not in _text_generation_models:
            print(f"Loading text generation model: {req.model}")
            tokenizer = T5Tokenizer.from_pretrained(req.model)
            model = T5ForConditionalGeneration.from_pretrained(req.model)
            model.eval()
            _text_generation_models[req.model] = (tokenizer, model)

        tokenizer, model = _text_generation_models[req.model]

        results = []

        for index, text in enumerate(texts):
            input_ids = tokenizer(text, return_tensors="pt").input_ids

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=req.max_new_tokens,
                )

            generated_text = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

            results.append(
                {
                    "object": "text.generation",
                    "index": index,
                    "output": generated_text,
                }
            )

        return {
            "object": "list",
            "model": req.model,
            "data": results,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error"}},
        )


# ----------------------------
# Audio transcription / Whisper
# ----------------------------
@app.post("/audio/transcriptions")
async def audio_transcriptions(
    model: str = Form(...),
    file: UploadFile = File(...),
):
    try:
        file_bytes = await file.read()

        if not file_bytes:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Uploaded audio file is empty"}},
            )

        if model not in _asr_pipelines:
            print(f"Loading ASR model: {model}")
            _asr_pipelines[model] = pipeline(
                task="automatic-speech-recognition",
                model=model,
            )

        asr = _asr_pipelines[model]

        suffix = Path(file.filename or "audio.wav").suffix or ".wav"

        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            result = asr(tmp.name, return_timestamps=True)

        text = result.get("text", "") if isinstance(result, dict) else str(result)

        return {
            "text": text,
            "model": model,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error"}},
        )


# ----------------------------
# Image classification / ViT
# ----------------------------
@app.post("/image/classify")
async def image_classify(
    model: str = Form(...),
    image: UploadFile = File(...),
):
    try:
        image_bytes = await image.read()

        if not image_bytes:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Uploaded image file is empty"}},
            )

        if model not in _image_pipelines:
            print(f"Loading image classification model: {model}")
            _image_pipelines[model] = pipeline(
                task="image-classification",
                model=model,
            )

        classifier = _image_pipelines[model]

        suffix = Path(image.filename or "image.jpg").suffix or ".jpg"

        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            tmp.write(image_bytes)
            tmp.flush()

            img = Image.open(tmp.name).convert("RGB")
            result = classifier(img)

        return {
            "object": "list",
            "model": model,
            "data": result,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error"}},
        )


@app.post("/fill-mask")
def fill_mask(req: FillMaskRequest):
    try:
        if "[MASK]" not in req.input:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Input must contain [MASK] token",
                        "type": "invalid_request_error",
                    }
                },
            )

        if req.top_k <= 0:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "top_k must be greater than 0",
                        "type": "invalid_request_error",
                    }
                },
            )

        if req.model not in _fill_mask_models:
            print(f"Loading fill-mask model: {req.model}")
            tokenizer = BertTokenizer.from_pretrained(req.model)
            model = BertForMaskedLM.from_pretrained(req.model)
            model.eval()
            _fill_mask_models[req.model] = (tokenizer, model)

        tokenizer, model = _fill_mask_models[req.model]

        inputs = tokenizer(req.input, return_tensors="pt")

        mask_positions = torch.where(
            inputs["input_ids"] == tokenizer.mask_token_id
        )[1]

        if len(mask_positions) == 0:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "No [MASK] token found after tokenization",
                        "type": "invalid_request_error",
                    }
                },
            )

        # For now, support first [MASK] position
        mask_token_index = mask_positions[0]

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        mask_token_logits = logits[0, mask_token_index, :]

        top_tokens = torch.topk(
            mask_token_logits,
            req.top_k,
            dim=0,
        ).indices.tolist()

        results = []

        for index, token_id in enumerate(top_tokens):
            token = tokenizer.decode([token_id]).strip()
            sequence = req.input.replace(tokenizer.mask_token, token, 1)
            score = torch.softmax(mask_token_logits, dim=0)[token_id].item()

            results.append(
                {
                    "index": index,
                    "token": token,
                    "token_id": token_id,
                    "sequence": sequence,
                    "score": score,
                }
            )

        return {
            "object": "list",
            "model": req.model,
            "data": results,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                }
            },
        )

@app.get("/health")
def health():
    return {"status": "ok"}