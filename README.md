# 📹 YouTube AI Content Generator

An automated YouTube content generation system that creates videos using AI. This tool handles the entire content creation pipeline: from topic brainstorming to script writing, audio narration, image generation, video composition, and YouTube uploading.

## 🔍 Overview

This project creates educational and philosophical content videos with minimal human intervention. It leverages several AI APIs to generate engaging content for YouTube.

## 📁 Project Structure

```
.
├── audio/                   # Generated audio files
├── images/                  # Generated images for videos
├── resources/               # Assets for video creation
│   ├── prompts/             # Text prompts for AI generation
│   ├── font files           # Typography for videos
│   └── additional assets    # Other resources
├── text/                    # Generated text content
├── videos/                  # Output video files
├── .env                     # Environment variables
├── browser.py               # Browser automation utilities
├── client_secrets.json      # YouTube API credentials
├── data_to_video.py         # Video composition module
├── history.json             # Record of generated content
├── image.py                 # Image processing utilities
├── main.py                  # Main application entry point
├── port_utils.py            # Networking utilities
├── requirements.txt         # Python dependencies
├── setup scripts            # Configuration scripts
├── suggestions.json         # Stored content suggestions
├── text_to_audio*.py        # Text-to-speech modules
├── text_to_image*.py        # Text-to-image generation modules
├── text_to_text*.py         # Text content generation modules
├── token.json               # API authentication tokens
└── video_to_youtube.py      # YouTube upload functionality
```

## ⚙️ Workflow

1. **Topic Generation**: Creates a random topic using the OpenAI or DeepSeek APIs
2. **Script Creation**: Generates a detailed script based on the topic
3. **Audio Narration**: Converts the script to spoken audio using ElevenLabs or Replicate
4. **Image Prompt Generation**: Creates descriptive prompts for visuals
5. **Image Creation**: Generates images using OpenAI or Replicate APIs
6. **Thumbnail Creation**: Designs an eye-catching thumbnail
7. **Video Assembly**: Combines images, audio, and text into a complete video
8. **YouTube Upload**: Uploads the finished video to YouTube

## 🚀 Setup

1. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

2. **Set Up Environment Variables**:
    Create a `.env` file with your API keys:
    ```
    ELEVENLABS_API_KEY=<your-elevenlabs-api-key>
    REPLICATE_API_TOKEN=<your-replicate-api-token>
    OPENAI_API_KEY=<your-openai-api-key>
    ```

3. **Run the Setup Script**:
    - For Windows:
        ```sh
        setup.bat
        ```
    - For Unix-based systems:
        ```sh
        setup.sh
        ```

## ▶️ Running the Project

Start the content generation process:
    - For Windows:
        ```sh
        start.bat
        ```
    - For Unix-based systems:
        ```sh
        start.sh
        ```

The script will continuously generate and upload videos based on the configured workflow.

## 📝 Note

This code isn't perfect and may require adjustments based on your specific needs and API changes. However, it successfully achieves the core job of automating content creation from concept to published video.

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See the [LICENSE](LICENSE) file for details.

