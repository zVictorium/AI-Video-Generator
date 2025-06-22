import os
import json
import base64
from pathlib import Path
import re
from time import sleep

import requests
from moviepy.editor import AudioFileClip, concatenate_audioclips, CompositeAudioClip

API_KEY = os.getenv("ELEVENLABS_API_KEY")
# VOICE_ID = "CoAqFXxZEa3kpJmE7rDr"
VOICE_ID = "hKUnzqLzU3P9IVhYHREu"
OUTPUT_DIR = "audio"
MAX_CHUNK_LENGTH = 5000
MODEL_ID = "eleven_multilingual_v2"  # "eleven_turbo_v2_5"
VOICE_SETTINGS = {
    "speed": 1.0,
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style_exaggeration": 0.3,
    "speaker_boost": True,
}
# VOICE_SETTINGS = {
#     "stability": 0.3,
#     "similarity_boost": 0.6,
#     # "style_exaggeration": 0.3,
#     # "speaker_boost": True,
# }


def split_text(text: str) -> list[str]:
    """Split the input text into chunks of maximum length."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= MAX_CHUNK_LENGTH:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def generate_audio(text: str) -> tuple[bytes | None, list[dict] | None]:
    """Generate audio from text using ElevenLabs API."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/with-timestamps"
    headers = {"Content-Type": "application/json", "xi-api-key": API_KEY}
    data = {"text": text, "model_id": MODEL_ID, "voice_settings": VOICE_SETTINGS}

    response = requests.post(url, json=data, headers=headers)
    while response.status_code == 429:
        print("Rate limit exceeded. Retrying in 30 seconds...")
        sleep(30)
        response = requests.post(url, json=data, headers=headers)

    if response.status_code != 200:
        error_message = (
            response.json().get("detail", {}).get("message", "Unknown error")
        )
        print(f"API Error: {response.status_code} - {error_message}")
        return None, None

    response_data = response.json()
    audio = base64.b64decode(response_data["audio_base64"])
    alignment = response_data["alignment"]["characters"]
    char_times = response_data["alignment"]

    timestamps = []
    word, start_time = "", None
    for i, char in enumerate(alignment):
        if not word:
            start_time = char_times["character_start_times_seconds"][i]
        word += char
        if char in " .,!?;:—-" or i == len(alignment) - 1:
            end_time = char_times["character_end_times_seconds"][i]
            timestamps.append(
                {
                    "word": word.strip(),
                    "start": start_time,
                    "end": end_time,
                }
            )
            word, start_time = "", None

    return audio, timestamps


def process_text(text_sections: list[str]) -> list[float]:
    """Process the input text, generate audio, and save the results. Returns start times of each section."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_timestamps, audio_clips = [], []
    current_time = 0.0
    section_start_times = []

    # Manejamos archivos temporales de forma clara.
    temp_files = []

    try:
        for section_index, section in enumerate(text_sections):
            chunks = split_text(section)
            section_start_times.append(current_time)

            for chunk_index, chunk in enumerate(chunks):
                audio, alignment = generate_audio(chunk)
                if not audio:
                    return []

                # Guardar cada audio temporalmente
                temp_filename = f"section{section_index}_chunk{chunk_index}.mp3"
                temp_path = output_dir / temp_filename
                temp_path.write_bytes(audio)
                temp_files.append(temp_path)

                # Crear clip de audio y calcular duraciones
                clip = AudioFileClip(str(temp_path))
                audio_clips.append(clip)

                # Ajustar timestamps
                for info in alignment:
                    info["start"] += current_time
                    info["end"] += current_time
                    total_timestamps.append(info)

                current_time += clip.duration

        # Concatenar todos los clips de audio en orden
        if audio_clips:
            combined_audio = concatenate_audioclips(audio_clips)

            # Cargar música de fondo y ajustar su volumen
            background_music = (
                AudioFileClip("resources/music.mp3").volumex(0.025).set_start(0)
            )

            # Reproducir música de fondo en bucle si es necesario
            if combined_audio.duration > background_music.duration:
                loop_count = (
                    int(combined_audio.duration // background_music.duration) + 1
                )
                background_music = concatenate_audioclips(
                    [background_music] * loop_count
                ).set_duration(combined_audio.duration)
            else:
                background_music = background_music.set_duration(
                    combined_audio.duration
                )

            # Crear un clip de audio compuesto
            final_audio = CompositeAudioClip([combined_audio, background_music])
            final_audio.fps = combined_audio.fps

            # Guardar el audio final
            final_audio.write_audiofile(str(output_dir / "output.mp3"))

            # Guardar timestamps en archivo JSON
            with open(output_dir / "timestamps.json", "w", encoding="utf-8") as f:
                json.dump(total_timestamps, f, indent=4, ensure_ascii=False)

            # Cerrar clips individuales
            for clip in audio_clips:
                clip.close()

        return section_start_times

    finally:
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except PermissionError:
                pass
                temp_file.unlink()


def generate_audio_from_script(script_sections: list) -> tuple[float, list[float]]:
    """Generate audio from script and return the time length in seconds and section start times."""
    section_start_times = process_text(script_sections)
    output_audio_path = Path(OUTPUT_DIR) / "output.mp3"
    if not output_audio_path.exists():
        return 0.0, []

    audio_clip = AudioFileClip(str(output_audio_path))
    duration = audio_clip.duration
    audio_clip.close()
    return duration, section_start_times
