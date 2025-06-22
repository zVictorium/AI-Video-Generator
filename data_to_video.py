import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Union, Tuple, Any, Generator
from PIL import ImageFont, ImageDraw, Image
from dataclasses import dataclass
import time
import numpy.typing as npt
from moviepy.editor import VideoFileClip, AudioFileClip
import re


ImageArray = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float64]


@dataclass
class VideoConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 24
    font_size: int = 45
    words_per_clip: int = 10
    max_chars_per_line: int = 35
    vignette_gamma: float = 0.7
    contrast_alpha: float = 1.5
    contrast_beta: int = 0


def resize_with_aspect_ratio(
    image: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    max_dim = 4000
    if new_w > max_dim or new_h > max_dim:
        scale = min(max_dim / new_w, max_dim / new_h)
        new_w, new_h = int(new_w * scale), int(new_h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    result[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_image
    return result


def generate_zoom_frames(
    image: np.ndarray,
    start_size: float,
    end_size: float,
    num_frames: int,
    video_width: int,
    video_height: int,
) -> Generator[np.ndarray, None, None]:
    image = resize_with_aspect_ratio(image, video_width, video_height)
    original_height, original_width, _ = image.shape
    t_values = np.linspace(0, 1, num_frames)
    scales = start_size + (end_size - start_size) * (
        0.5 - 0.5 * np.cos(np.pi * t_values)
    )

    for scale in scales:
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        frame_resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
        )

        x_center, y_center = new_width // 2, new_height // 2
        x1 = max(0, x_center - video_width // 2)
        y1 = max(0, y_center - video_height // 2)
        x2 = min(x1 + video_width, new_width)
        y2 = min(y1 + video_height, new_height)

        frame_cropped = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        crop_height = y2 - y1
        crop_width = x2 - x1
        frame_cropped[:crop_height, :crop_width] = frame_resized[y1:y2, x1:x2]
        yield frame_cropped


class SubtitleFrame:
    def __init__(self, config: VideoConfig, font: ImageFont.FreeTypeFont):
        self.config = config
        self.font = font

    def create_text_overlay(
        self, text: str, text_position: Tuple[int, int]
    ) -> np.ndarray:
        overlay = np.zeros((self.config.height, self.config.width, 4), dtype=np.uint8)
        pil_img = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil_img)

        border_width = 3
        border_color = (0, 0, 0, 255)
        draw.multiline_text(
            text_position,
            text,
            font=self.font,
            fill=(255, 255, 255, 255),
            align="center",
            stroke_width=border_width,
            stroke_fill=border_color,
        )

        # Convert to numpy array and apply Gaussian blur to the alpha channel
        overlay_np = np.array(pil_img)
        alpha_channel = overlay_np[:, :, 3]
        blurred_alpha = cv2.GaussianBlur(alpha_channel, (5, 5), 0)
        overlay_np[:, :, 3] = blurred_alpha

        return overlay_np


class CombinedVideoGenerator:
    def __init__(
        self,
        json_path: Union[str, Path],
        images_folder: Union[str, Path],
        font_path: Union[str, Path],
        vignette_path: Union[str, Path],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        config: VideoConfig = VideoConfig(),
    ):
        self.config = config
        self.json_path = Path(json_path)
        self.images_folder = Path(images_folder)
        self.font_path = Path(font_path)
        self.vignette_path = Path(vignette_path)
        self.audio_path = Path(audio_path)
        self.output_path = Path(output_path)

        self.font = ImageFont.truetype(str(self.font_path), config.font_size)
        self.frame_handler = SubtitleFrame(config, self.font)
        self.subtitle_data = self._load_subtitle_data()

        # Calculate video duration and total frames
        self.audio_clip = AudioFileClip(audio_path)
        self.total_frames = self.calculate_total_frames()

        # Initialize vignette video capture
        if not self.vignette_path.exists():
            raise FileNotFoundError(f"Vignette video not found: {self.vignette_path}")
        self.vignette_cap = cv2.VideoCapture(str(self.vignette_path))
        if not self.vignette_cap.isOpened():
            raise RuntimeError("Failed to open vignette video")
        self.vignette_frame_count = int(self.vignette_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _load_subtitle_data(self) -> List[Dict[str, Any]]:
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Loaded {len(data)} words from {self.json_path}")
            return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading subtitle data: {e}")
            raise

    def calculate_total_frames(self) -> int:
        total_frames = int(self.audio_clip.duration * self.config.fps)
        return total_frames

    def _get_vignette_frame(self, frame_idx: int) -> ImageArray:
        vignette_pos = frame_idx % self.vignette_frame_count
        self.vignette_cap.set(cv2.CAP_PROP_POS_FRAMES, vignette_pos)
        ret, frame = self.vignette_cap.read()
        if not ret:
            raise RuntimeError("Failed to read vignette frame")
        return cv2.resize(frame, (self.config.width, self.config.height))

    def _apply_vignette_effect(
        self, frame: ImageArray, vignette_frame: ImageArray
    ) -> ImageArray:
        gray_image: ImageArray = cv2.cvtColor(vignette_frame, cv2.COLOR_BGR2GRAY)
        adjusted_image: ImageArray = cv2.convertScaleAbs(
            np.power(gray_image.astype(np.float32) / 255.0, 3) * 255, alpha=1.5
        )
        brightness_normalized: FloatArray = 1 - (
            adjusted_image.astype(np.float32) / 255.0
        )
        final_mask: FloatArray = np.power(brightness_normalized, 2.0)
        mask = cv2.GaussianBlur(final_mask, (5, 5), 0).astype(np.float32)
        mask = np.stack([mask] * 3, axis=-1)
        black_tint = np.zeros_like(vignette_frame)
        return (black_tint * mask + frame * (1 - mask)).astype(np.uint8)

    def _get_background_frame_generator(self) -> Generator[np.ndarray, None, None]:
        images = [
            img
            for img in os.listdir(self.images_folder)
            if img.endswith((".png", ".jpg", ".jpeg")) and img != "thumbnail.png"
        ]
        images.sort()
        if not images:
            raise ValueError("No images found in the specified folder.")

        frames_per_image = int(self.total_frames / len(images))

        for image_name in images:
            image_path = os.path.join(self.images_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            start_zoom, end_zoom = np.random.uniform(1.0, 1.5, 2)
            while abs(start_zoom - end_zoom) <= 0.15:
                start_zoom, end_zoom = np.random.uniform(1.0, 1.5, 2)

            yield from generate_zoom_frames(
                image,
                start_zoom,
                end_zoom,
                frames_per_image,
                self.config.width,
                self.config.height,
            )

    def _calculate_text_position(self, text: str) -> Tuple[int, int]:
        bbox = ImageDraw.Draw(
            Image.new("RGB", (self.config.width, self.config.height))
        ).textbbox((0, 0), text, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        return (
            (self.config.width - text_width) // 2,
            (self.config.height - text_height) // 2,
        )

    def _prepare_text_block(self, words_group: List[Dict[str, Any]]) -> str:
        lines = []
        current_line = ""
        for word_data in words_group:
            word = word_data["word"]
            if len(current_line) + len(word) <= self.config.max_chars_per_line:
                if current_line and word:
                    current_line += " "
                current_line += " " * len(word)
            else:
                lines.append(current_line)
                current_line = " " * len(word)
        if current_line:
            lines.append(current_line)
        return "\n".join(lines)

    def _group_by_phrases(self, subtitle_data: list):
        grouped_phrases = []
        current_phrase = []
        i = 0
        while i < len(subtitle_data):
            end_phrase = re.search(r"[.,;:—!?]", subtitle_data[i]["word"])
            subtitle_data[i]["word"] = re.sub(r"[.,;:—]", "", subtitle_data[i]["word"])
            current_phrase.append(subtitle_data[i])
            if end_phrase:
                if i + 1 < len(subtitle_data) and subtitle_data[i + 1]["word"] == "":
                    current_phrase.append(subtitle_data[i + 1])
                    subtitle_data.pop(i + 1)
                else:
                    current_phrase.append(
                        {
                            "word": "",
                            "start": subtitle_data[i]["end"],
                            "end": subtitle_data[i]["end"],
                        }
                    )
                grouped_phrases.append(current_phrase)
                current_phrase = []
            i += 1
        if current_phrase:
            grouped_phrases.append(current_phrase)

        # Split phrases longer than 12 words into two halves
        split_phrases = []
        for phrase in grouped_phrases:
            if len(phrase) > 12:
                mid_index = len(phrase) // 2
                first_half = phrase[:mid_index]
                second_half = phrase[mid_index:]
                # Add pause at end of the first half
                first_half.append(
                    {
                        "word": "",
                        "start": first_half[-1]["end"],
                        "end": first_half[-1]["end"],
                    }
                )
                split_phrases.append(first_half)
                split_phrases.append(second_half)
            else:
                split_phrases.append(phrase)

        grouped_phrases = split_phrases

        # Adjust timing within each phrase
        for phrase in grouped_phrases:
            if len(phrase) > 1:
                phrase_start = phrase[0]["start"]
                phrase_end = phrase[-1]["end"]
                total_duration = phrase_end - phrase_start

                # Calculate exponential weights with more dramatic timing difference
                word_count = len(phrase)
                weights = np.exp(
                    np.linspace(0.1, 2.5, word_count)
                )  # Increased to 2.5 from 1.5
                weights = weights / np.sum(weights)

                # Apply additional power to increase contrast
                weights = np.power(weights, 1.5)  # Added power function
                weights = weights / np.sum(weights)  # Renormalize

                # Distribute time according to exponential weights
                current_time = phrase_start
                for i, word in enumerate(phrase):
                    word["start"] = current_time
                    duration = total_duration * weights[i]
                    current_time += duration
                    word["end"] = current_time

                # Ensure exact end time for last word
                phrase[-1]["end"] = phrase_end

        return grouped_phrases

    def create_short_version(self, input_path: str) -> None:
        """Creates a 30-second vertical (9:16) version of the input video with letterboxing and zoom."""
        print("Creating short vertical version of the video...")
        output_short_path = str(Path(input_path).parent / "short.mp4")

        video = VideoFileClip(input_path)
        short_video = video.subclip(0, 59)  # Get first 30 seconds

        # Calculate dimensions for 9:16 aspect ratio
        target_width = 1080  # Standard vertical video width
        target_height = 1920  # 9:16 ratio height

        # Apply 80% zoom (1/0.8 = 1.25 zoom factor)
        zoom_factor = 1.0/0.5

        # Calculate resize dimensions maintaining aspect ratio with zoom
        video_aspect = short_video.w / short_video.h
        target_aspect = target_width / target_height

        if video_aspect > target_aspect:  # Video is wider
            new_width = int(target_width * zoom_factor)
            new_height = int((target_width * zoom_factor) / video_aspect)
        else:  # Video is taller
            new_height = int(target_height * zoom_factor)
            new_width = int((target_height * zoom_factor) * video_aspect)

        resized_video = short_video.resize((new_width, new_height))

        # Create black background
        final_video = resized_video.on_color(
            size=(target_width, target_height), color=(0, 0, 0), pos="center"
        )

        final_video.write_videofile(
            output_short_path, codec="libx264", audio_codec="aac"
        )
        video.close()
        final_video.close()
        print("Short vertical version created successfully")

    def generate_video(self) -> None:
        start_time = time.time()
        print(f"Total frames to process: {self.total_frames}")
        print("Starting video generation with combined effects")
        frame_count = 0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            "videos/temp.mp4",
            fourcc,
            self.config.fps,
            (self.config.width, self.config.height),
        )
        try:
            background_generator = self._get_background_frame_generator()
            phrases = self._group_by_phrases(self.subtitle_data)

            for words_group in phrases:
                current_text = self._prepare_text_block(words_group)
                text_position = self._calculate_text_position(current_text)
                current_index = 0

                for word_data in words_group:
                    word = word_data["word"]
                    word_start = word_data["start"]
                    word_end = word_data["end"]

                    # Cálculo del tiempo para sincronización precisa
                    word_duration = word_end - word_start
                    frame_time = frame_count / self.config.fps

                    # Si es un silencio
                    if not word:
                        text_overlay = self.frame_handler.create_text_overlay(
                            current_text, text_position
                        )
                        while frame_time < word_end:
                            try:
                                background_frame = next(background_generator)
                            except StopIteration:
                                pass

                            vignette_frame = self._get_vignette_frame(frame_count)
                            alpha = text_overlay[:, :, 3] / 255.0
                            for c in range(3):
                                background_frame[:, :, c] = (
                                    background_frame[:, :, c] * (1 - alpha)
                                    + text_overlay[:, :, c] * alpha
                                )

                            frame = self._apply_vignette_effect(
                                background_frame, vignette_frame
                            )
                            video_writer.write(frame)
                            frame_count += 1
                            frame_time = frame_count / self.config.fps
                            if frame_count % 100 == 0:
                                print(f"Processed {frame_count} frames (silence)")

                    else:
                        for i, letter in enumerate(word):
                            # Calcular el tiempo de inicio y fin para cada letra
                            letter_end_time = (
                                word_start + ((i + 1) / len(word)) * word_duration
                            )
                            text_overlay = self.frame_handler.create_text_overlay(
                                current_text, text_position
                            )

                            # Añadir letra al texto actual
                            current_text = (
                                current_text[:current_index]
                                + letter
                                + current_text[current_index + 1 :]
                            )
                            current_index += 1

                            # Renderizar frames hasta el tiempo de finalización de la letra
                            while frame_time < letter_end_time:
                                try:
                                    background_frame = next(background_generator)
                                except StopIteration:
                                    pass

                                vignette_frame = self._get_vignette_frame(frame_count)
                                alpha = text_overlay[:, :, 3] / 255.0
                                for c in range(3):
                                    background_frame[:, :, c] = (
                                        background_frame[:, :, c] * (1 - alpha)
                                        + text_overlay[:, :, c] * alpha
                                    )

                                frame = self._apply_vignette_effect(
                                    background_frame, vignette_frame
                                )
                                video_writer.write(frame)
                                frame_count += 1
                                frame_time = frame_count / self.config.fps
                                if frame_count % 100 == 0:
                                    print(f"Processed {frame_count} frames (sync)")
                        current_index += 1

            while frame_count <= self.total_frames:
                try:
                    background_frame = next(background_generator)
                except StopIteration:
                    pass

                vignette_frame = self._get_vignette_frame(frame_count)
                text_overlay = self.frame_handler.create_text_overlay(
                    current_text, text_position
                )
                alpha = text_overlay[:, :, 3] / 255.0
                for c in range(3):
                    background_frame[:, :, c] = (
                        background_frame[:, :, c] * (1 - alpha)
                        + text_overlay[:, :, c] * alpha
                    )
                frame = self._apply_vignette_effect(background_frame, vignette_frame)
                video_writer.write(frame)
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames (final delay filling)")

        except Exception as e:
            print(f"Error during video generation: {str(e)}")
            raise
        finally:
            self.vignette_cap.release()
            video_writer.release()
            duration = time.time() - start_time
            print(
                f"Video generation completed in {duration:.2f} seconds\n"
                f"Total frames processed: {frame_count}\n"
                "Starting merging audio..."
            )
            video_clip = VideoFileClip("videos/temp.mp4")
            video_clip = video_clip.set_duration(self.audio_clip.duration)
            video_clip = video_clip.set_audio(self.audio_clip)
            video_clip.write_videofile(
                str(self.output_path), codec="libx264", audio_codec="aac"
            )
            video_clip.close()
            os.remove("videos/temp.mp4")

            # Create short version after main video is generated
            self.create_short_version(str(self.output_path))