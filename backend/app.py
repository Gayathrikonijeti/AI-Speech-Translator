from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import whisper
import os
import uuid
from deep_translator import GoogleTranslator, MyMemoryTranslator
from gtts import gTTS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Whisper tiny model
model = whisper.load_model("tiny")


def translate_text(text, target_lang):
    try:
        translated = GoogleTranslator(
            source="auto",
            target=target_lang
        ).translate(text)

        if translated is None or "Error" in translated:
            raise Exception("Google failed")

        return translated

    except Exception:
        return MyMemoryTranslator(
            source="auto",
            target=target_lang
        ).translate(text)


@app.route("/translate", methods=["POST"])
def translate_audio():
    audio_file = request.files.get("audio")
    target_lang = request.form.get("language")

    if not audio_file or not target_lang:
        return jsonify({"error": "Audio or language missing"}), 400

    # File paths
    input_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}")
    wav_path = input_path + ".wav"
    mp3_path = input_path + ".mp3"

    # Save uploaded file (mic / audio / video)
    audio_file.save(input_path)

    # 🔥 Convert ANY input to WAV using FFmpeg
    os.system(
        f'ffmpeg -y -i "{input_path}" -ac 1 -ar 16000 "{wav_path}"'
    )

    # Whisper transcription
    result = model.transcribe(
        wav_path,
        fp16=False,
        condition_on_previous_text=False
    )

    original_text = result["text"].strip()
    print("📝 Original:", original_text)

    if not original_text:
        return jsonify({
            "original_text": "",
            "translated_text": "",
            "audio_url": ""
        })

    # Translation
    translated_text = translate_text(original_text, target_lang)
    print("🌍 Translated:", translated_text)

    # Text → Speech
    gTTS(translated_text, lang=target_lang).save(mp3_path)

    return jsonify({
        "original_text": original_text,
        "translated_text": translated_text,
        "audio_url": f"/audio/{os.path.basename(mp3_path)}"
    })


@app.route("/audio/<filename>")
def get_audio(filename):
    return send_file(
        os.path.join(UPLOAD_FOLDER, filename),
        mimetype="audio/mpeg"
    )


if __name__ == "__main__":
    app.run(debug=True)
