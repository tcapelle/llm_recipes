import os
import json
from typing import Optional
from typing import Union
import base64
import io
import time
from pydantic import BaseModel, Field, field_validator
import logging
import simple_parsing
from dataclasses import dataclass
from PIL import Image
from story_illustrator.models import FalAITextToImageGenerationModel, StoryIllustrator


logger = logging.getLogger(__name__)


def image_to_base64(image_path: Union[str, Image.Image]) -> str:
    """Converts an image to base64 encoded string to be logged and rendered on Weave dashboard.

    Args:
        image_path (Union[str, Image.Image]): Path to the image or PIL Image object.
        mimetype (Optional[str], optional): Mimetype of the image. Defaults to None.

    Returns:
        str: Base64 encoded image string.
    """
    image = Image.open(image_path) if isinstance(image_path, str) else image_path
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="PNG")
    encoded_string = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
    return str(encoded_string)


def illustrate(
    story: str,
    story_title: str = "Gift of the Magi",
    story_author: str = "O. Henry",
    story_setting: str = "the year 1905, New York City, United States of America",
    openai_model: str = "gpt-4o-mini",
    llm_seed: Optional[int] = None,
    txt2img_model_address: str = "fal-ai/flux-pro",
    image_size: str = "square",
    illustration_style: Optional[str] = None,
    use_text_encoder_2: bool = False,
):
    story_illustrator = StoryIllustrator(
        openai_model=openai_model,
        llm_seed=llm_seed,
        text_to_image_model=FalAITextToImageGenerationModel(
            model_address=txt2img_model_address
        ),
    )
    paragraphs = story.split("\n\n")
    images = story_illustrator.predict(
        story=story,
        metadata={
            "title": story_title,
            "author": story_author,
            "setting": story_setting,
        },
        paragraphs=paragraphs[:10],
        illustration_style=illustration_style,
        use_text_encoder_2=use_text_encoder_2,
        image_size=image_size,
    )

    return images


class IllustratePayload(BaseModel):
    story: str = Field(..., description="The text of the story to illustrate")
    story_title: str = Field(..., description="The title of the story")
    story_author: str = Field(..., description="The author of the story")
    story_setting: str = Field(..., description="The setting of the story")
    wandb_api_key: str = Field(
        ..., description="The wandb api key", min_length=40, max_length=40
    )
    return_images: bool = Field(
        default=False, description="Whether to return the images in the response"
    )
    dummy: bool = Field(default=False, description="Whether to use the dummy images")

    @field_validator("wandb_api_key")
    def validate_wandb_api_key(cls, v):
        if len(v) != 40:
            raise ValueError("wandb_api_key must be exactly 40 characters long")
        return v


def clean_wandb_api_key():
    if "WANDB_API_KEY" in os.environ:
        os.unsetenv("WANDB_API_KEY")
        del os.environ["WANDB_API_KEY"]

    # Delete the ~/.netrc file if it exists
    netrc_path = os.path.expanduser("~/.netrc")
    if os.path.exists(netrc_path):
        try:
            os.remove(netrc_path)
            logger.info(f"Deleted {netrc_path}")
        except OSError as e:
            logger.error(f"Error deleting {netrc_path}: {e}")
    else:
        logger.info(f"{netrc_path} does not exist")


def setup_wandb(wandb_api_key: str):
    clean_wandb_api_key()
    logger.info(f"Logging into wandb with key: {wandb_api_key}")
    import wandb

    wandb.login(key=wandb_api_key, relogin=True)
    time.sleep(1)
    print("Attemp login to Weave")
    import weave

    weave.init("illustration-project")


def generate_illustration_process(payload: IllustratePayload):
    setup_wandb(payload.wandb_api_key)
    result = illustrate(
        story=payload.story,
        story_title=payload.story_title,
        story_author=payload.story_author,
        story_setting=payload.story_setting,
    )
    clean_wandb_api_key()
    return result


def get_images_dict(stdout: str):
    dict_str = stdout.split("<img>")[-1].split("</img>")[-2]
    print(dict_str[0:100])
    print("...")
    print(dict_str[-100:])
    image_base64_dict = json.loads(dict_str)
    return image_base64_dict


def load_image(fn, mode=None):
    "Open and load a `PIL.Image` and convert to `mode`"
    im = Image.open(fn)
    im.load()
    im = im._new(im.im)
    return im


def dummy_generate_illustration_process(payload: IllustratePayload, n=2):
    "Generate n dummy PIL.Images objects"
    setup_wandb(payload.wandb_api_key)

    # dummy op to generate traces
    import weave

    @weave.op(name="dummy_generate_illustration_process")
    def dummy_gen(payload: IllustratePayload):
        return [load_image("pug.png") for _ in range(n)]

    clean_wandb_api_key()
    return dummy_gen(payload)


def generate_images(payload: IllustratePayload):
    if payload.dummy:
        images = dummy_generate_illustration_process(payload, n=2)
    else:
        images = generate_illustration_process(payload)
    if payload.return_images:
        images_dict = {
            f"image_{i}": image_to_base64(image) for i, image in enumerate(images)
        }
        print("<img>")
        print(json.dumps(images_dict))
        print("</img>")
    else:
        print("<img>")
        print("</img>")


@dataclass
class Args:
    story: str = "The text of the story to illustrate. I love chocolate cake."
    story_title: str = "The title of the story"
    story_author: str = "The author of the story"
    story_setting: str = "The setting of the story"
    wandb_api_key: str = "dummy_key"
    dummy: bool = False
    return_images: bool = False


if __name__ == "__main__":
    args = simple_parsing.parse(Args)
    args_dict = vars(args)
    payload = IllustratePayload.model_validate(args_dict)
    generate_images(payload)
