import os
import json
from pathlib import Path
import re
import time

import requests
from moviepy.editor import AudioFileClip, concatenate_audioclips, CompositeAudioClip
from typing import List, Dict, Any, Tuple, Optional
import replicate
from replicate import Client

# Configuration
OUTPUT_DIR = "audio"
MAX_CHUNK_LENGTH = 1000  # Maximum length of text chunk for TTS processing

# Initialize Replicate client
REPLICATE_API_TOKEN: str = os.getenv("REPLICATE_API_TOKEN")
client = Client(api_token=REPLICATE_API_TOKEN)

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

def generate_audio(text: str, output_path: str = "", voice: str = "richard", emotion: str = "") -> Tuple[bytes, List[Dict[str, Any]]]:
    """
    Generate audio from text using Replicate API.
    
    Args:
        text: The text to convert to speech
        output_path: Optional path to save the audio file
        voice: Voice model to use
        emotion: Emotion for the voice (if supported)
    
    Returns:
        Tuple of (audio_bytes, word_alignments)
    """
    # Dictionary of available voice samples
    voice_samples = {
        "richard": "https://replicate.delivery/pbxt/MUEtXI54W68rj2eUER8rrkaRNUPjtqZdVXN5hQnhmRVMBqwC/richard_sample.wav",
        # Add more voice samples as needed
    }
    
    # Default to richard if the specified voice is not available
    audio_sample = voice_samples.get(voice.lower(), voice_samples["richard"])
    
    # Maximum retries
    max_retries = 5
    retries = 0
    
    while retries < max_retries:
        try:
            output = client.run(
                "jaaari/zonos:79caaf88e47605d71197442eb35361be922488dfb2d55de8ae757cc73d6d2a15",
                input={
                    "seed": 1,
                    "text": text,
                    "audio": audio_sample,
                    "emotion": emotion,
                    "language": "en-us",
                    "model_type": "transformer",
                    "speaking_rate": 15
                }
            )
            
            if not output:
                raise ValueError("No output returned from Replicate API")
            
            # Download the audio file
            response = requests.get(output, timeout=30)
            if response.status_code != 200:
                raise ValueError(f"Failed to download audio file, status code: {response.status_code}")
            
            # Get audio content as bytes
            audio_bytes = response.content
            
            # Save the audio file if output_path is provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(audio_bytes)
                print(f"Audio saved to {output_path}")
            
            # Create more accurate word-level timestamps
            # First, if we have the output file, get the actual duration
            actual_duration = None
            if output_path and os.path.exists(output_path):
                try:
                    temp_clip = AudioFileClip(output_path)
                    actual_duration = temp_clip.duration
                    temp_clip.close()
                except Exception as e:
                    print(f"Warning: Could not get audio duration: {e}")
            
            # Parse text into words
            words = text.split()
            word_count = len(words)
            
            if word_count == 0:
                return audio_bytes, []
                
            # If we have actual duration, use it; otherwise estimate
            if actual_duration is not None:
                total_duration = actual_duration
            else:
                # Estimate: ~3 words per second on average
                total_duration = word_count / 3.0
            
            # Calculate character count to estimate relative word durations
            total_chars = sum(len(word) for word in words)
            
            # Generate timestamps for each word based on relative length
            alignment = []
            current_time = 0.0
            
            for word in words:
                # Word duration is proportional to its length relative to total text
                # Add a small fixed component to account for spaces between words
                word_duration = (len(word) / total_chars * total_duration * 0.85) + (total_duration * 0.15 / word_count)
                
                word_info = {
                    "word": word,
                    "start": current_time,
                    "end": current_time + word_duration
                }
                alignment.append(word_info)
                current_time += word_duration
            
            # Normalize to ensure the last word ends exactly at total_duration
            if alignment:
                scale_factor = total_duration / alignment[-1]["end"]
                for info in alignment:
                    info["start"] *= scale_factor
                    info["end"] *= scale_factor
            
            return audio_bytes, alignment
            
        except Exception as e:
            retries += 1
            print(f"Error generating audio (attempt {retries}/{max_retries}): {e}")
            time.sleep(2)  # Wait before retrying
    
    raise Exception(f"Failed to generate audio after {max_retries} attempts")

def generate_audio_for_script(script: List[dict], output_dir: str = OUTPUT_DIR) -> List[str]:
    """
    Generate audio files for each segment in the script.
    
    Args:
        script: List of dictionaries with 'text' and optional 'voice' keys
        output_dir: Directory to save audio files
    
    Returns:
        List of paths to the generated audio files
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_files = []
    
    for i, segment in enumerate(script):
        text = segment['text']
        voice = segment.get('voice', 'richard')
        emotion = segment.get('emotion', '')
        
        output_path = os.path.join(output_dir, f"segment_{i:03d}.mp3")
        try:
            audio_bytes, _ = generate_audio(text, output_path, voice, emotion)
            audio_files.append(output_path)
        except Exception as e:
            print(f"Failed to generate audio for segment {i}: {e}")
    
    return audio_files

def process_text(text_sections: list[str]) -> list[float]:
    """Process the input text, generate audio, and save the results. Returns start times of each section."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_timestamps, audio_clips = [], []
    current_time = 0.0
    section_start_times = []

    # Manage temp files clearly
    temp_files = []

    try:
        for section_index, section in enumerate(text_sections):
            chunks = split_text(section)
            section_start_times.append(current_time)

            for chunk_index, chunk in enumerate(chunks):
                # Generate audio for this chunk
                temp_filename = f"section{section_index}_chunk{chunk_index}.mp3"
                temp_path = output_dir / temp_filename
                audio_bytes, alignment = generate_audio(chunk, str(temp_path))
                temp_files.append(temp_path)

                # Create audio clip and calculate durations
                clip = AudioFileClip(str(temp_path))
                audio_clips.append(clip)
                
                # The actual clip duration might differ from our estimation
                # Adjust timestamps to match the actual audio duration
                clip_duration = clip.duration
                if alignment and alignment[-1]["end"] > 0:
                    duration_ratio = clip_duration / alignment[-1]["end"]
                    for info in alignment:
                        info["start"] *= duration_ratio
                        info["end"] *= duration_ratio
                        
                        # Add the current accumulated time to make timestamps absolute
                        info["start"] += current_time
                        info["end"] += current_time
                        total_timestamps.append(info)

                current_time += clip_duration

        # Concatenate all audio clips in order
        if audio_clips:
            combined_audio = concatenate_audioclips(audio_clips)

            # Add background music if file exists
            background_music_path = "resources/music.mp3"
            if os.path.exists(background_music_path):
                # Load background music and adjust volume
                background_music = (
                    AudioFileClip(background_music_path).volumex(0.025).set_start(0)
                )

                # Loop background music if needed
                if combined_audio.duration > background_music.duration:
                    loop_count = (
                        int(combined_audio.duration // background_music.duration) + 1
                    )
                    background_music = concatenate_audioclips(
                        [background_music] * loop_count
                    ).set_duration(combined_audio.duration)
                else:
                    background_music = background_music.set_duration(combined_audio.duration)

                # Create composite audio clip
                final_audio = CompositeAudioClip([combined_audio, background_music])
                final_audio.fps = combined_audio.fps
            else:
                final_audio = combined_audio

            # Save final audio
            final_audio.write_audiofile(str(output_dir / "output.mp3"))

            # Save timestamps to JSON file
            with open(output_dir / "timestamps.json", "w", encoding="utf-8") as f:
                json.dump(total_timestamps, f, indent=4, ensure_ascii=False)

            # Close individual clips
            for clip in audio_clips:
                clip.close()

        return section_start_times

    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                print(f"Failed to delete temporary file {temp_file}: {e}")

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

if __name__ == "__main__":
    # Test the audio generation with generate_audio_from_script
    try:
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Test script sections
        test_script_sections = [
            "This is the first section of our test script using the Replicate API.",
            "And here is the second section with additional content for testing purposes."
        ]
        
        # Generate audio using the main function that will be called from main.py
        duration, section_start_times = generate_audio_from_script(test_script_sections)
        
        print(f"Audio generation completed successfully!")
        print(f"Total audio duration: {duration:.2f} seconds")
        print(f"Section start times: {section_start_times}")
        print(f"Output file location: {os.path.join(OUTPUT_DIR, 'output.mp3')}")
        
        # Verify timestamps file was created
        timestamps_file = os.path.join(OUTPUT_DIR, "timestamps.json")
        if os.path.exists(timestamps_file):
            print(f"Timestamps file created at: {timestamps_file}")
        else:
            print("Warning: Timestamps file was not created")
            
    except Exception as e:
        print(f"Test audio generation failed: {e}")
