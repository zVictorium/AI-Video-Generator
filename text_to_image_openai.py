import os
from typing import List
import requests
from openai import OpenAI
import random
import base64
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from io import BytesIO

# Constants
API_KEY: str = os.getenv("OPENAI_API_KEY")
OUTPUT_DIR: str = "images"
USE_DIRECT_PROMPT_TITLE: bool = True  # Toggle between direct prompt title integration and post-processing

client = OpenAI(api_key=API_KEY)


def generate_image(prompt: str, output_dir: str, index: int) -> None:
    """Generate, save, and enhance an image for the given prompt."""
    retries = 5
    while retries > 0:
        try:
            response = client.images.generate(
                model="gpt-image-1",
                prompt=f"{prompt}",
                size="1536x1024",
                quality="high",
                n=1,
            )
            
            # Get image base64 data from OpenAI response
            image_base64 = response.data[0].b64_json
            if not image_base64:
                raise ValueError("No image data returned from the client.")
            break
        except Exception as e:
            print(f"Error generating image: {e}. Retrying... ({5 - retries}/5)")
            retries -= 1
            if retries == 0:
                raise ValueError("Failed to generate image after multiple attempts")

    os.makedirs(output_dir, exist_ok=True)
    
    # Decode the base64 image data
    image_bytes = base64.b64decode(image_base64)
    file_path = os.path.join(output_dir, f"output{index}.png")
    with open(file_path, "wb") as file:
        file.write(image_bytes)
    print(f"Image saved as {file_path}")


def generate_images(prompts: List[str]) -> None:
    """Generate and save images for the given prompts."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, prompt in enumerate(prompts, 1):
        try:
            generate_image(prompt, OUTPUT_DIR, i)
        except Exception as e:
            print(f"An error occurred for prompt {i}: {e}")


def generate_thumbnail_image(prompt: str, title: str) -> None:
    """Generate an image for the given prompt, add a title using a custom TTF font."""
    retries = 5

    if USE_DIRECT_PROMPT_TITLE:
        # Integrate title text and border directly in the prompt
        enhanced_prompt = (
            f"{prompt} oil painting, professional, high detail, 4k. "
            f"The image should prominently display the text '{title}' in a visible, elegent, bold font. "
            f"The text should be centered and clearly readable. "
            f"The image should also have a thin white rectangular border near the edges of the frame."
        )
        
        # Step 1: Generate the Image with title and border included in prompt
        while retries > 0:
            try:
                response = client.images.generate(
                    model="gpt-image-1",
                    prompt=enhanced_prompt,
                    size="1536x1024",
                    quality="high",
                    n=1,
                )
                
                # Get base64 image data from OpenAI response
                image_base64 = response.data[0].b64_json
                if not image_base64:
                    raise ValueError("No image data returned from the client.")
                break
            except Exception as e:
                print(f"Error generating image: {e}. Retrying... ({5 - retries}/5)")
                retries -= 1
                if retries == 0:
                    raise ValueError("Failed to generate image after multiple attempts")

        # Decode the base64 image data
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes))
        
        # Resize to YouTube thumbnail dimensions
        image = image.resize((1280, 720), Image.Resampling.LANCZOS)
        
        # Save the image
        main_path = os.getcwd()
        file_path = os.path.join(main_path, "images/thumbnail.png")
        image.save(file_path)
        print(f"Thumbnail image saved as {file_path}")
        
    else:
        # Original method: Generate image first, then add text with PIL
        # Step 1: Generate the Image
        while retries > 0:
            try:
                response = client.images.generate(
                    model="gpt-image-1",
                    prompt=f"{prompt} oil painting, professional, high detail, 4k",
                    size="1536x1024",
                    quality="high",
                    n=1,
                )
                
                # Get base64 image data from OpenAI response
                image_base64 = response.data[0].b64_json
                if not image_base64:
                    raise ValueError("No image data returned from the client.")
                break
            except Exception as e:
                print(f"Error generating image: {e}. Retrying... ({5 - retries}/5)")
                retries -= 1
                if retries == 0:
                    raise ValueError("Failed to generate image after multiple attempts")

        # Decode the base64 image data
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes))
        draw = ImageDraw.Draw(image)

        # Load the custom TTF font
        main_path = os.getcwd()
        ttf_path = os.path.join(main_path, "resources/thumbnail.ttf")
        if not os.path.isfile(ttf_path):
            raise FileNotFoundError(f"TTF font file not found at {ttf_path}")

        # Calculate initial font size based on image width
        width_ratio = 0.8  # Use 80% of image width
        target_width = image.width * width_ratio
        font_size = random.randint(300, 400)  # Random font size between 200 and 400
        font = ImageFont.truetype(ttf_path, font_size)

        # Function to wrap text
        def wrap_text(text, font, max_width):
            words = text.split()
            lines = []
            current_line = []

            for word in words:
                current_line.append(word)
                bbox = draw.textbbox((0, 0), " ".join(current_line), font=font)
                if bbox[2] - bbox[0] > max_width and len(current_line) > 1:
                    lines.append(" ".join(current_line[:-1]))
                    current_line = [word]

            if current_line:
                lines.append(" ".join(current_line))
            return lines

        # Adjust font size until text fits
        max_height = image.height * 0.3  # Maximum 30% of image height for text
        wrapped_lines = wrap_text(title, font, target_width)
        bbox = draw.textbbox((0, 0), "\n".join(wrapped_lines), font=font)
        
        # Keep reducing font size until text fits both width and height constraints
        while (bbox[2] - bbox[0] > target_width or bbox[3] - bbox[1] > max_height):
            font_size = int(font_size * 0.9)
            font = ImageFont.truetype(ttf_path, font_size)
            wrapped_lines = wrap_text(title, font, target_width)
            bbox = draw.textbbox((0, 0), "\n".join(wrapped_lines), font=font)

        # Calculate text position with padding
        text = "\n".join(wrapped_lines)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (image.width - text_width) // 2
        
        # Add padding at the bottom (5% of image height)
        bottom_padding = int(image.height * 0.15)
        if random.random() < 0.5:
            y = max(0, random.randint(2 * image.height // 3 - text_height - bottom_padding, image.height - text_height - bottom_padding))
        else:
            y = max(0, random.randint(0, image.height // 3 - text_height - bottom_padding))
        
        if y < bottom_padding: y = bottom_padding
        if y > image.height - bottom_padding: y = image.height - bottom_padding

        text_position = (x, y)

        # Make border width larger - now 1/3 of font size
        border_width = max(8, font_size // 3)

        # Adjust border position for multiline text
        border_position = (text_position[0], text_position[1])

        border_color = (0, 0, 0, 217)  # 85% opaque (15% transparent)

        # Create a temporary image for the border effect
        border_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
        border_draw = ImageDraw.Draw(border_image)
        
        # Draw the text border multiple times with increasing blur for a softer effect
        # Further increased iterations for even thicker border
        for offset in range(border_width * 2):
            for i, line in enumerate(wrapped_lines):
                line_bbox = draw.textbbox((0, 0), line, font=font)
                line_height = line_bbox[3] - line_bbox[1]
                line_width = line_bbox[2] - line_bbox[0]
                line_x = (image.width - line_width) // 2
                line_y = border_position[1] + i * (line_height + border_width)
                line_position = (line_x, line_y)
                border_draw.text(
                    line_position,
                    line,
                    font=font,
                    fill=border_color,
                    align="center",
                    stroke_width=border_width + offset,
                    stroke_fill=border_color,
                )
        
        border_image = border_image.filter(ImageFilter.GaussianBlur(radius=border_width * 2.0))
        
        # Composite the blurred border onto the original image
        image = Image.alpha_composite(image.convert("RGBA"), border_image)
        
        # Draw the main text
        draw = ImageDraw.Draw(image)
        draw.multiline_text(
            text_position,
            text,
            font=font,
            fill=(255, 255, 255, 255),
            align="center",
        )

        image = image.resize((1280, 720), Image.Resampling.LANCZOS)

        # Add rectangular border
        image = image.resize((1280, 720), Image.Resampling.LANCZOS)
        
        # Create a new layer for the rectangle
        rectangle_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw_rect = ImageDraw.Draw(rectangle_layer)
        
        # Calculate rectangle dimensions with equal border spacing
        width, height = image.size
        border_width = 6  # Border thickness in pixels
        border_spacing = 60  # Fixed border spacing in pixels
        
        # Calculate rectangle position and size
        rect_width = width - (2 * border_spacing)
        rect_height = height - (2 * border_spacing)
        x1 = border_spacing
        y1 = border_spacing
        x2 = width - border_spacing
        y2 = height - border_spacing
        
        # Draw rectangle
        draw_rect.rectangle(
            [(x1, y1), (x2, y2)],
            outline=(255, 255, 255, 255),  # White color
            width=border_width
        )
        
        # Composite the rectangle layer onto the main image
        image = Image.alpha_composite(image, rectangle_layer)
        
        # Save the modified image
        file_path = os.path.join(main_path, "images/thumbnail.png")
        image.save(file_path)
        print(f"Thumbnail image saved as {file_path}")


def generate_thumbnail(prompt: str, title: str) -> None:
    """Generate and save a thumbnail image for the given prompt and title."""
    while True:
        try:
            generate_thumbnail_image(prompt, title)
            break
        except Exception as e:
            print(f"An error occurred for thumbnail prompt: {e}. Retrying...")

if __name__ == "__main__":
    # Test prompt and title for thumbnail generation
    test_prompt = "A solitary figure sits in quiet contemplation by a rain-streaked window within a softly lit, intimate room. The gentle glow of dim, ambient light casts soft shadows on vintage furnishings and worn wooden floors, while delicate droplets race down the windowpane, merging with the reflections of a distant, stormy night. Outside, the rhythmic patter of rain against glass creates a meditative backdrop, evoking inner revelations and a slow-burning spark of innovation. Each detail is rendered in smooth, brushstrokes that capture the serene, reflective mood of the scene. oil painting, professional, high detail, 4k"
    test_title = "Hidden Sparks Unleashed"
    
    print("Testing thumbnail generation...")
    try:
        generate_thumbnail(test_prompt, test_title)
        print("Thumbnail generation completed successfully!")
    except Exception as e:
        print(f"Error during thumbnail generation: {e}")
        
    # Test the image generation function
    print("\nTesting image generation...")
    image_test_prompt = "A solitary figure sits quietly on a windswept beach at twilight, bathed in the soft, fading light of a vibrant sky. The horizon is a canvas of deep purples, gentle blues, and touches of fiery orange, with the fading sun casting its last glow across the gently rolling waves. The ambient sound of the sea blends with the whisper of the breeze, creating a serene symphony of nature. The figure, silhouetted against the ethereal light, appears lost in a moment of introspective creative awakening, embracing the delicate balance between solitude and inspiration. The scene is rendered with smooth, flowing brushstrokes that capture the intimate, dreamlike quality of this quiet, transformative moment. oil painting, professional, high detail, 4k"
    try:
        generate_image(image_test_prompt, OUTPUT_DIR, 999)
        print("Image generation completed successfully!")
    except Exception as e:
        print(f"Error during image generation: {e}")