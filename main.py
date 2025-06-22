from text_to_text import (
    generate_script,
    generate_image_prompts,
    generate_thumbnail_prompt,
    generate_random_topic,
)
from text_to_audio import generate_audio_from_script
from text_to_image import generate_thumbnail, generate_images
from data_to_video import VideoConfig, CombinedVideoGenerator
from video_to_youtube import upload_video, save_video_details
from datetime import datetime, timedelta
import time
import os
import glob


def main() -> None:
    while True:
        # Borrar archivos .png, .txt y .mp3
        for file in glob.glob("images/*.png"):
            os.remove(file)
        for file in glob.glob("text/*.txt"):
            os.remove(file)
        for file in glob.glob("audio/*.mp3"):
            os.remove(file)
        if os.path.exists("audio/timestamps.json"):
            os.remove("audio/timestamps.json")

        # Keep only the 5 most recent videos
        video_files = glob.glob("videos/*.mp4")
        if len(video_files) > 5:
            video_files.sort(key=os.path.getctime)
            for file in video_files[:-5]:
                os.remove(file)

        now = datetime.now()

        # Generar el texto usando IA
        topic = generate_random_topic()
        (
            title,
            description,
            thumbnail_title,
            thumbnail_description,
            script,
            script_sections,
            comment,
            hashtags,
        ) = generate_script(topic, 8)

        # Generar el audio
        duration, section_start_times = generate_audio_from_script(
            [ss["script"] for ss in script_sections]
        )
        print(f"Duration: {str(timedelta(seconds=int(duration)))}")

        durations_str = ""
        for i in range(len(section_start_times)):
            minutes, seconds = divmod(section_start_times[i], 60)
            durations_str += (
                f"\n{script_sections[i]['title']} {int(minutes):02}:{int(seconds):02}"
            )
        durations_str = durations_str.strip()
        # description = f"{description}\n\n{topic}\n\nChapters:\n{durations_str}"
        try:
            if "video" in topic.split(",")[0].lower():
                topic = ",".join(topic.split(",")[1:])
                topic = topic[0].upper() + topic[1:]
        except:
            pass
        description = f"📜 {description}\n\n📚 Chapters:\n{durations_str}\n\n🔥 If you enjoyed the video, don't forget to support it and share your thoughts.\n\n🎯 Topics:\n{' '.join(hashtags)}"

        # Generar los prompts de imágenes
        thumbnail_prompt = generate_thumbnail_prompt(thumbnail_description)
        image_prompts = generate_image_prompts(
            script, title, description, duration // 45
        )

        # Generar imagenes
        generate_thumbnail(thumbnail_prompt, thumbnail_title)
        generate_images(image_prompts)

        # Generar el video
        generator = CombinedVideoGenerator(
            json_path="audio/timestamps.json",
            images_folder="images",
            font_path="resources/font.otf",
            vignette_path="resources/vignette.mp4",
            audio_path="audio/output.mp3",
            output_path="videos/output.mp4",
            config=VideoConfig(),
        )
        generator.generate_video()
        save_video_details(now, title, description, comment)
        upload_video("videos/output.mp4", title, description, comment, hashtags)

        # Renombrar el archivo de salida a la fecha de creación en formato dd-mm-yy.mp4
        creation_date = now.strftime("%d-%m-%Y_%H-%M-%S")
        os.rename("videos/output.mp4", f"videos/video-{creation_date}.mp4")
        os.rename("videos/short.mp4", f"videos/short-{creation_date}.mp4")

        # Esperar hasta las 0am del día siguiente
        DAY_DELAY = 2
        next_run = (now + timedelta(days=DAY_DELAY)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        wait_time = (next_run - datetime.now()).total_seconds()
        if wait_time > 0:
            print(f"Waiting until {next_run.strftime('%d/%m/%Y %H:%M:%S')} to generate next video...")
            time.sleep(wait_time)
        else:
            print("Target time has already passed, continuing immediately...")


if __name__ == "__main__":
    main()
