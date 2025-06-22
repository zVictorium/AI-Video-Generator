import os
import re
import json
from typing import List, Type, get_origin, get_args, TypeVar, Generic, Any
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_class_schema(cls: Type[BaseModel]) -> str:
    """
    Convert a Pydantic BaseModel class to a string representation of its schema.
    
    Args:
        cls: A class that inherits from BaseModel
        
    Returns:
        A string representation of the class schema as a dictionary
    """
    if not issubclass(cls, BaseModel):
        raise TypeError("Class must inherit from BaseModel")
    
    def process_type_hint(type_hint):
        """Process a type hint and return an appropriate schema representation"""
        # Handle list types
        origin = get_origin(type_hint)
        args = get_args(type_hint)
        
        if origin is list:
            if args:
                element_type = args[0]
                if get_origin(element_type) is not None:
                    # Nested complex type like list[list[str]]
                    return [process_type_hint(element_type)]
                elif isinstance(element_type, type) and issubclass(element_type, BaseModel):
                    # List of Pydantic models
                    return [build_schema_dict(element_type)]
                else:
                    # List of primitive types
                    return ["str"]
            return ["str"]
        
        # Handle direct BaseModel subclass
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            return build_schema_dict(type_hint)
            
        # Default case for primitive types
        return "str"
    
    def build_schema_dict(model_cls):
        """Build schema dictionary for a model class"""
        schema_dict = {}
        for field_name, field_type in model_cls.__annotations__.items():
            schema_dict[field_name] = process_type_hint(field_type)
        return schema_dict
    
    schema_dict = build_schema_dict(cls)
    return json.dumps(schema_dict).replace("]", ", ...]")

# Configuration
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com/v1"  # Add this line
MODEL = "deepseek-reasoner" # "gpt-4o-mini" # "o3-mini" # "o3-mini-2025-1-31" # "gpt-4o-mini"
MAX_TOKENS_SECTION = 5000
MAX_TOKENS = 5000
TEMPERATURE = 1
TEMPERATURE_IMAGE = 0.7
TOP_P = 1
FREQUENCY_PENALTY = 0.3
PRESENCE_PENALTY = 0.5

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL  # Add this line
)


def load_topic_history():
    try:
        with open("history.json", "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_topic_history():
    with open("history.json", "w", encoding="utf-8") as file:
        json.dump(topic_history, file, ensure_ascii=False, indent=4)


def load_topic_suggestions():
    try:
        with open("suggestions.json", "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_topic_suggestions():
    with open("suggestions.json", "w", encoding="utf-8") as file:
        json.dump(topic_suggestions, file, ensure_ascii=False, indent=4)


topic_history = load_topic_history()
topic_suggestions = load_topic_suggestions()


class Topic(BaseModel):
    title: str
    description: str


def parse_json_response(completion, model_class: Type, schema_format: str = None) -> Type:
    """
    Parse JSON response with infinite retries until valid JSON is obtained.
    
    Args:
        completion: The initial completion response
        model_class: The Pydantic model class to parse into
        schema_format: Optional schema format string to include in retry prompts
        
    Returns:
        An instance of the specified model_class
    """
    while True:
        try:
            response_text = completion.choices[0].message.content
            # Clean up the response text by removing markdown code blocks and extra whitespace
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            print(f"Response Text: {response_text}")
            
            # Try to parse the JSON
            response_dict = json.loads(response_text)
            
            # Return the parsed model
            return model_class(**response_dict)
            
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"Error parsing JSON response: {e}")
            print("Retrying with explicit request for valid JSON...")
            
            # Prepare retry message
            retry_message = f"The response was not valid JSON.\nError: {e}\nPlease provide a valid JSON response."
            if schema_format:
                retry_message += f" Follow this format: {schema_format}"
                
            # Retry the completion
            completion = client.beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "user", "content": "You must respond with valid JSON only."},
                    {"role": "assistant", "content": response_text},
                    {"role": "user", "content": retry_message},
                ],
                max_completion_tokens=MAX_TOKENS,
                top_p=TOP_P,
            )


def generate_random_topic() -> str:
    """Genera un tema aleatorio para un video de YouTube usando la IA."""
    global topic_suggestions
    try:
        topic_suggestions = load_topic_suggestions()
        if topic_suggestions:
            suggestion = topic_suggestions.pop(0)
            save_topic_suggestions()
            return suggestion.strip()
        
        history = "\n".join(["- " + topic for topic in topic_history])
        if not history:
            history = "- Nothing"
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": read_prompt_file("topic_en")},
                {
                    "role": "user",
                    "content": f"Topics previously addressed:\n{history}\n\nGenerate a random topic for a YouTube video without repeating previous topics.\nYou must follow the following response format: {get_class_schema(Topic)}",
                },
            ],
            max_completion_tokens=MAX_TOKENS,
            top_p=TOP_P,
        )
        
        # Use the utility function instead of direct parsing
        topic_class = parse_json_response(completion, Topic, get_class_schema(Topic))
        
        topic = topic_class.description
        topic_history.append(topic)
        if len(topic_history) > 5:
            topic_history.pop(0)
        save_topic_history()  # Save the updated history
        return topic

    except Exception as e:
        raise RuntimeError(f"Error generating random topic: {e}")


class Section(BaseModel):
    title: str
    description: str


class SectionContent(BaseModel):
    content: str


class Thumbnail(BaseModel):
    title: str
    image_description: str


class Video(BaseModel):
    title: str
    description: str
    hashtags: list[str]
    comment: str
    thumbnail: Thumbnail
    introduction: str
    sections: list[Section]


class Image(BaseModel):
    prompt: str


class Summary(BaseModel):
    summary: str
    scene: str


def read_prompt_file(prompt: str) -> str:
    """Read and return the contents of a file."""
    with open(f"resources/prompts/{prompt}.txt", "r", encoding="utf-8") as file:
        return file.read().strip()


def generate_brainstorm_response(topic: str, duration: int) -> Video:
    """Generates a brainstorm response video based on the given topic and duration."""
    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": read_prompt_file("brainstorm_en"),
                },
                {"role": "user", "content": f"Topic: {topic}\nYou must follow the following response format: {get_class_schema(Video)}"},
            ],
            max_completion_tokens=MAX_TOKENS,
            top_p=TOP_P,
        )
        
        # Use the utility function instead of direct parsing
        return parse_json_response(completion, Video, get_class_schema(Video))
        
    except Exception as e:
        raise RuntimeError(f"Error generating response: {e}")


def generate_image_scene_response(section: str) -> Summary:
    """Generates an image response based on the provided section text."""
    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "From a text fragment, summarize it in one sentence and imagine a scene that could represent the text in a movie in one sentence. Describe the scene as subjective as possible instead of objective, you have to imply the fragment with a creative scene that suggests specific feelings and sensations.",
                },
                {"role": "user", "content": f"{section}\nYou must follow the following response format: {get_class_schema(Summary)}"},
            ],
            max_completion_tokens=MAX_TOKENS,
            top_p=TOP_P,
        )
        
        # Use the utility function instead of direct parsing
        return parse_json_response(completion, Summary, get_class_schema(Summary))
        
    except Exception as e:
        raise RuntimeError(f"Error generating response: {e}")


def generate_image_response(section: str, type: str) -> str:
    """Generates an image response based on the provided section text."""
    try:
        scene = generate_image_scene_response(section)
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": read_prompt_file(type)},
                {"role": "user", "content": "A moonlit Venetian canal"},
                {
                    "role": "assistant",
                    "content": "The ancient Venetian canal rests under a brilliant full moon, its waters like polished obsidian reflecting the silver light. Ornate bridges arch gracefully overhead while gondolas sway gently at their moorings. The historic buildings cast long shadows across the water, their weathered facades telling stories of centuries past. A delicate mist hovers above the surface, creating an ethereal atmosphere.",
                },
                {"role": "user", "content": "A mystical forest with glowing flora"},
                {
                    "role": "assistant",
                    "content": "Deep within an ancient forest, ethereal plants pulse with an inner light, casting a blue-green glow through the misty air. Enormous mushrooms tower overhead, their caps sparkling with bioluminescent spores. Twisted tree trunks wear crowns of glowing moss, while delicate flowers emit waves of soft light that dance through the darkness like floating spirits.",
                },
                {"role": "user", "content": "A dramatic desert sunset"},
                {
                    "role": "assistant",
                    "content": "The desert stretches endlessly beneath a sky ablaze with color, where the setting sun ignites the clouds in brilliant shades of crimson and gold. Rolling dunes cast deep purple shadows across the sand, their ridges sharp against the dying light. A lone rock formation stands sentinel in the distance, its weathered face glowing amber in the last rays of day.",
                },
                {"role": "user", "content": "A man's spiritual ascension"},
                {
                    "role": "assistant",
                    "content": "A man ascends a staircase that spirals into a celestial ocean above. Each step behind him dissolves into sand, while before him, ghostly figures of his past walk in reverse. His chest glows with a wound that leaks liquid light, creating rivers that flow upward into the infinite, forming constellations in the dark.",
                },
                {"role": "user", "content": "A figure on a rooftop with fire"},
                {
                    "role": "assistant",
                    "content": "The boy stands on a rooftop under a blood-red sky, his arms outstretched as a vortex of fire spirals upward from his body. His head is no longer visible, replaced by an inferno that pulses like a living heart. The buildings around him crack and crumble under the heat, while the clouds above twist and burn, igniting the heavens.",
                },
                {"role": "user", "content": "A defeated warrior seeking redemption"},
                {
                    "role": "assistant",
                    "content": "A soldier kneels in a barren battlefield, his armor dented and smeared with blood. In his hands, he cradles a broken sword, its blade reflecting a faint glimmer of a distant sunrise. The ground is littered with the remnants of a great battle, and his tear-filled eyes gaze upward as if seeking redemption from the gray, stormy sky.",
                },
                {"role": "user", "content": f"{scene}\nYou must follow the following response format: {get_class_schema(Image)}"},
            ],
            max_completion_tokens=MAX_TOKENS,
            top_p=TOP_P,
        )
        
        # Use the utility function instead of direct parsing
        return parse_json_response(completion, Image, get_class_schema(Image)).prompt
        
    except Exception as e:
        raise RuntimeError(f"Error generating response: {e}")


def generate_section_response(
    script: str,
    video: Video,
    last_section: Section,
    section: Section,
    section_time: int,
) -> str:
    """Generates a response for a given section of a video script using a chat completion model."""
    try:
        messages = [{"role": "system", "content": f"{read_prompt_file("script_en")}\nYou must follow the following response format: {get_class_schema(SectionContent)}"}]
        title = last_section.title if last_section else "Introduction"
        description = last_section.description if last_section else video.description
        messages.append(
            {"role": "user", "content": f"Title: {title}\nTopic: {description}"}
        )
        messages.append({"role": "assistant", "content": script[-3500:]})
        messages.append(
            {
                "role": "user",
                "content": f"Title: {section.title}\nTopic: {section.description}",
            }
        )

        characters = section_time
        attempts = 0
        shortest_response = None

        while attempts < 5:
            if attempts > 0:
                print(
                    f"Attempt {attempts}: Retrying to generate a shorter response... ({len(shortest_response)}/{characters})"
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"You have written {len(shortest_response)} characters and I want it to be {characters} characters long. Make it even shorter...\nYou must follow the following response format: {get_class_schema(SectionContent)}",
                    }
                )

            # print(f"Messages: {messages}")
            completion = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_completion_tokens=MAX_TOKENS_SECTION,
                top_p=TOP_P,
            )
            # print(f"Completion: {completion}")
            
            # Use the utility function instead of direct parsing
            response = parse_json_response(completion, SectionContent, get_class_schema(SectionContent)).content
            
            if len(response) <= characters:
                return response

            if shortest_response is None or len(response) < len(shortest_response):
                shortest_response = response

            messages.append({"role": "assistant", "content": response})

            attempts += 1

        return shortest_response

    except Exception as e:
        raise RuntimeError(f"Error generating response: {e}")


def split_text(text: str, n: int) -> list[str]:
    """Split the input text into chunks of maximum length."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current_chunk = [], ""
    max_chunk_length = len(text) // n
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def generate_script(topic: str, duration: int) -> str:
    """Generate a script based on the given topic and duration."""
    brainstorm = generate_brainstorm_response(topic, duration)

    # Save the brainstorm
    content = f"Title: {brainstorm.title}\nDescription: {brainstorm.description}\n\nThumbnail Title: {brainstorm.thumbnail.title}\nThumbnail Description: {brainstorm.thumbnail.image_description}\n\n1. Introduction\n{brainstorm.introduction}"
    for i, section in enumerate(brainstorm.sections, 2):
        content += f"\n\n{i}. {section.title}\n{section.description}"
    content += f"\n\nComment: {brainstorm.comment}"
    with open("text/brainstorm.txt", "w", encoding="utf-8") as file:
        file.write(content)
    print("Brainstorm saved in brainstorm.txt")

    with open(f"text/section1.txt", "w", encoding="utf-8") as file:
        file.write(f"1. Introduction\n\n{brainstorm.introduction}")

    script = brainstorm.introduction
    last_section = None
    script_sections = [
        {
            "title": "1. Introduction",
            "description": brainstorm.introduction,
            "script": brainstorm.introduction,
        }
    ]
    section_time = (
        duration * 1000 // (len(brainstorm.sections) + 1)
    )  # Calculate time per section
    for i, section in enumerate(brainstorm.sections, 1):
        try:
            part = generate_section_response(
                script, brainstorm, last_section, section, section_time
            )
            part = re.sub(
                r"\[.*?\]|\$\$.*?\$\$|\{.*?\}|\(.*?\)|\*.*?\*|—.*?—", "", part.strip()
            )
            part = part.strip()
            with open(f"text/section{i + 1}.txt", "w", encoding="utf-8") as file:
                file.write(
                    f"{i + 1}. {section.title}\n{section.description}\n\n{' '.join(part.split())}"
                )
            script += " " + part
            script = " ".join(script.split())
            script_sections.append(
                {
                    "title": f"{i + 1}. {section.title}",
                    "description": section.description,
                    "script": part,
                }
            )
            last_section = section
        except Exception as e:
            raise RuntimeError(f"Error generating script: {e}")

    # Save the script
    with open("text/script.txt", "w", encoding="utf-8") as file:
        file.write(script)
    print("Script saved in script.txt")

    return (
        brainstorm.title,
        brainstorm.description,
        brainstorm.thumbnail.title,
        brainstorm.thumbnail.image_description,
        script,
        script_sections,
        brainstorm.comment,
        brainstorm.hashtags
    )


def generate_image_prompts(
    script: str, title: str, description: str, n: int
) -> List[str]:
    """Generate image prompts based on the script."""
    parts = split_text(script, n)

    prompts = []
    for i, phrase in enumerate(parts, 1):
        phrase = phrase.strip()
        if not phrase:
            continue
        try:
            prompt = generate_image_response(f'User: "{phrase}"', "image_en")
            prompt = prompt.strip(" \n\"'")
            prompt = " ".join(prompt.split())
            prompts.append(prompt)
            with open(f"text/prompt{i}.txt", "w", encoding="utf-8") as file:
                file.write(f"User: {phrase}\n\n{prompt}")
        except Exception as e:
            print(f"Error generating image prompt for phrase {i}: {e}")

    if not prompts:
        raise RuntimeError("No image prompts were generated.")

    return prompts


def generate_thumbnail_prompt(script: str) -> str:
    """Generate a single image prompt based on the full script."""
    # Limpiar el script
    script = script.strip()

    if not script:
        raise ValueError("El script proporcionado está vacío.")

    try:
        # Generar un único prompt basado en el script completo
        prompt = generate_image_response(f"User: {script}", "thumbnail_en")
        prompt = prompt.strip(" \n\"'")
        prompt = " ".join(prompt.split())
    except Exception as e:
        print(f"Error al generar el prompt de imagen: {e}")
        raise

    # Guardar el prompt en un archivo de texto
    with open("text/thumbnail.txt", "w", encoding="utf-8") as file:
        file.write(prompt)

    return prompt

if __name__ == "__main__":
    print(get_class_schema(Topic))
    print(get_class_schema(Video))