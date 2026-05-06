GATEWAY="http://localhost:9000"
OUTPUT_DIR="./demo_output"
 
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'
 
pass_msg() { echo -e "  ${GREEN}✓${NC} $1"; }
fail_msg() { echo -e "  ${RED}✗ FAIL${NC} — $1"; FAILED=1; }
header()   { echo -e "\n${CYAN}━━━ $1 ━━━${NC}"; }
 
FAILED=0
mkdir -p "$OUTPUT_DIR"
 
echo ""
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "${BOLD}        KAI Gateway — API Demo${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
 
# ── JWT ──
header "Step 1: Generating Access Token"
 
JWT_TOKEN=$(python3 -c "
import os, jwt, time
secret = os.environ.get('KAI_JWT_SECRET', '')
if not secret: print('ERROR'); exit(1)
token = jwt.encode({
    'sub': 'demo-user',
    'models': ['test-vit-model', 'test-whisper', 'test-tts', 'test-embed'],
    'iat': int(time.time()),
    'exp': int(time.time()) + 3600
}, secret, algorithm='HS256')
print(token)
" 2>/dev/null)
 
if [[ -z "$JWT_TOKEN" || "$JWT_TOKEN" == "ERROR" ]]; then
    fail_msg "Set KAI_JWT_SECRET env var first."
    exit 1
fi
pass_msg "JWT token generated for demo-user"
 
# ── Gateway check ──
header "Step 2: Checking Services"
 
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "$GATEWAY/docs" 2>/dev/null)
if [[ "$HTTP_CODE" == "200" ]]; then
    pass_msg "Gateway is running at $GATEWAY"
else
    fail_msg "Gateway not running at $GATEWAY"
    exit 1
fi
 
# ══════════════════════════════════════════════════════════════
# API 1: Image Generation
# ══════════════════════════════════════════════════════════════
header "API 1: POST /images/generations"
echo -e "  ${YELLOW}→${NC} model=test-vit-model | prompt=\"A futuristic city at sunset\""
 
HTTP_CODE=$(curl -s -o "$OUTPUT_DIR/1_image_gen.json" -w "%{http_code}" \
  -X POST "$GATEWAY/images/generations" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-vit-model",
    "prompt": "A futuristic city skyline at sunset with flying cars and neon lights",
    "size": "1024x1024",
    "quality": "high",
    "n": 1
  }')
 
if [[ "$HTTP_CODE" == "200" ]]; then
    pass_msg "Status: 200 OK"
    python3 -c "
import json, base64
d = json.load(open('$OUTPUT_DIR/1_image_gen.json'))
img = base64.b64decode(d['data'][0]['b64_json'])
open('$OUTPUT_DIR/1_generated_image.png', 'wb').write(img)
print(f'    Saved: $OUTPUT_DIR/1_generated_image.png ({len(img):,} bytes)')
" 2>/dev/null || fail_msg "Could not decode image"
else
    fail_msg "Expected 200, got $HTTP_CODE"
    echo -e "  ${YELLOW}Response:${NC}"
    cat "$OUTPUT_DIR/1_image_gen.json" 2>/dev/null
    echo ""
fi
 
# ══════════════════════════════════════════════════════════════
# API 2: Image Edit
# ══════════════════════════════════════════════════════════════
header "API 2: POST /images/edits"
 
if [[ -f "$OUTPUT_DIR/1_generated_image.png" ]]; then
    echo -e "  ${YELLOW}→${NC} image from API 1 + prompt=\"Add rain and dark clouds\""
 
    HTTP_CODE=$(curl -s -o "$OUTPUT_DIR/2_image_edit.json" -w "%{http_code}" \
      -X POST "$GATEWAY/images/edits" \
      -H "Authorization: Bearer $JWT_TOKEN" \
      -F "model=test-vit-model" \
      -F "prompt=Add rain and dark storm clouds to the scene" \
      -F "image=@$OUTPUT_DIR/1_generated_image.png" \
      -F "size=1024x1024")
 
    if [[ "$HTTP_CODE" == "200" ]]; then
        pass_msg "Status: 200 OK"
        python3 -c "
import json, base64
d = json.load(open('$OUTPUT_DIR/2_image_edit.json'))
img = base64.b64decode(d['data'][0]['b64_json'])
open('$OUTPUT_DIR/2_edited_image.png', 'wb').write(img)
print(f'    Saved: $OUTPUT_DIR/2_edited_image.png ({len(img):,} bytes)')
" 2>/dev/null || fail_msg "Could not decode edited image"
    else
        fail_msg "Expected 200, got $HTTP_CODE"
        echo -e "  ${YELLOW}Response:${NC}"
        cat "$OUTPUT_DIR/2_image_edit.json" 2>/dev/null
        echo ""
    fi
else
    fail_msg "No image from API 1 to edit"
fi
 
# ══════════════════════════════════════════════════════════════
# API 3: Audio Transcription
# ══════════════════════════════════════════════════════════════
header "API 3: POST /audio/transcriptions"
 
python3 -c "
import wave, struct, math
f = wave.open('$OUTPUT_DIR/test_audio.wav', 'w')
f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
for i in range(32000):
    s = int(8000 * math.sin(2 * 3.14159 * 440 * i / 16000))
    f.writeframes(struct.pack('<h', s))
f.close()
" 2>/dev/null
 
echo -e "  ${YELLOW}→${NC} model=test-whisper | file=test_audio.wav | language=en"
 
HTTP_CODE=$(curl -s -o "$OUTPUT_DIR/3_transcription.json" -w "%{http_code}" \
  -X POST "$GATEWAY/audio/transcriptions" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -F "model=test-whisper" \
  -F "file=@$OUTPUT_DIR/test_audio.wav" \
  -F "language=en")
 
if [[ "$HTTP_CODE" == "200" ]]; then
    pass_msg "Status: 200 OK"
    TRANSCRIPT=$(python3 -c "import json; d=json.load(open('$OUTPUT_DIR/3_transcription.json')); print(d.get('text','')[:100])" 2>/dev/null)
    echo -e "    Transcript: \"$TRANSCRIPT\""
else
    fail_msg "Expected 200, got $HTTP_CODE"
    echo -e "  ${YELLOW}Response:${NC}"
    cat "$OUTPUT_DIR/3_transcription.json" 2>/dev/null
    echo ""
fi
 
# ══════════════════════════════════════════════════════════════
# API 4: Audio Translation
# ══════════════════════════════════════════════════════════════
header "API 4: POST /audio/translations"
echo -e "  ${YELLOW}→${NC} model=test-whisper | file=test_audio.wav"
 
HTTP_CODE=$(curl -s -o "$OUTPUT_DIR/4_translation.json" -w "%{http_code}" \
  -X POST "$GATEWAY/audio/translations" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -F "model=test-whisper" \
  -F "file=@$OUTPUT_DIR/test_audio.wav")
 
if [[ "$HTTP_CODE" == "200" ]]; then
    pass_msg "Status: 200 OK"
    TRANSLATION=$(python3 -c "import json; d=json.load(open('$OUTPUT_DIR/4_translation.json')); print(d.get('text','')[:100])" 2>/dev/null)
    echo -e "    Translation: \"$TRANSLATION\""
else
    fail_msg "Expected 200, got $HTTP_CODE"
    echo -e "  ${YELLOW}Response:${NC}"
    cat "$OUTPUT_DIR/4_translation.json" 2>/dev/null
    echo ""
fi
 
# ══════════════════════════════════════════════════════════════
# API 5: Text-to-Speech
# ══════════════════════════════════════════════════════════════
header "API 5: POST /audio/speech"
echo -e "  ${YELLOW}→${NC} model=test-tts | voice=nova | text=\"Welcome to Kompact AI\""
 
HTTP_CODE=$(curl -s -o "$OUTPUT_DIR/5_speech.wav" -w "%{http_code}" \
  -X POST "$GATEWAY/audio/speech" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-tts",
    "input": "Welcome to Kompact AI. Our platform enables sovereign AI deployment on any Intel hardware.",
    "voice": "nova"
  }')
 
if [[ "$HTTP_CODE" == "200" ]]; then
    AUDIO_SIZE=$(wc -c < "$OUTPUT_DIR/5_speech.wav" 2>/dev/null || echo 0)
    pass_msg "Status: 200 OK"
    echo "    Saved: $OUTPUT_DIR/5_speech.wav ($AUDIO_SIZE bytes)"
else
    fail_msg "Expected 200, got $HTTP_CODE"
    echo -e "  ${YELLOW}Response:${NC}"
    cat "$OUTPUT_DIR/5_speech.wav" 2>/dev/null
    echo ""
fi
 
# ══════════════════════════════════════════════════════════════
# API 6: Embeddings
# ══════════════════════════════════════════════════════════════
header "API 6: POST /embeddings"
echo -e "  ${YELLOW}→${NC} model=test-embed | input=[2 sentences]"
 
HTTP_CODE=$(curl -s -o "$OUTPUT_DIR/6_embeddings.json" -w "%{http_code}" \
  -X POST "$GATEWAY/embeddings" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-embed",
    "input": ["How do I reset my password?", "I forgot my login credentials"]
  }')
 
if [[ "$HTTP_CODE" == "200" ]]; then
    pass_msg "Status: 200 OK"
    python3 -c "
import json
d = json.load(open('$OUTPUT_DIR/6_embeddings.json'))
for item in d.get('data', []):
    vec = item.get('embedding', [])
    print(f'    Vector {item[\"index\"]}: {len(vec)} dimensions')
" 2>/dev/null || fail_msg "Could not parse embeddings"
else
    fail_msg "Expected 200, got $HTTP_CODE"
    echo -e "  ${YELLOW}Response:${NC}"
    cat "$OUTPUT_DIR/6_embeddings.json" 2>/dev/null
    echo ""
fi
 
# ══════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
if [[ "$FAILED" == "0" ]]; then
    echo -e "${GREEN}${BOLD}  ALL 6 APIs WORKING ✓${NC}"
else
    echo -e "${RED}${BOLD}  SOME TESTS FAILED ✗${NC}"
fi
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/ 2>/dev/null | grep -v "^total" | awk '{print "  " $NF " (" $5 ")"}'
echo ""