from glob import glob
import fire
import wandb
import json
from PIL import Image
from diffusers import DiffusionPipeline, AutoencoderKL
import torch
import os

GENERIC_PROMPTS = [
    "five people, beach, sunny day, playing voleyball",
    "a photo of five people sitting outside",
    "a selfie, reading a book.",
]
TEMPLATE_PROMTS = [
    "A photo of {} and his mom",
    "A photo of {} participating in a marathon.",
    "A close shot photo of {} participating in a marathon.",
    "A photo of {} reading a book.",
    "A photo of {} in international space station.",
    "A photo of {} at Red Square in Moscow.",
    "{}, riding a bicycle",
    "{} cycling in park, golden hour. Canon EOS 5D settings, f/2.8, ISO 100. Focus on realistic, human-like face, clear and detailed. Background: soft bokeh. Aim for photorealism, natural lighting, high resolution",
    "{}, standing tall, full-length photo, holding flowers",
    "{}, close shot photo, holding flowers",
]
AGES_TEMPLATE_PROMTS = [
    "{}, 7 year old, playing with wooden horse",
    "{}, 5 year old with a flower outside",
    "{}, 5 year old with a flower outside, full-length photo",
    "{}, riding a bicycle, 11 year old",
]


def prompts_generator(token):
    names = [token, f"{token} man", "man"]
    # names = [f"{token} man"]

    for p in GENERIC_PROMPTS:
        yield p, p
    for p in TEMPLATE_PROMTS:
        for name in names:
            yield p.format(name), p.format(name).replace(token, "{}")
    for p in AGES_TEMPLATE_PROMTS:
        yield p.format(token), p


def image_grid(imgs, rows, cols, resize=256):
    assert len(imgs) == rows * cols

    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]

    w, h = imgs[0].size
    grid_w, grid_h = cols * w, rows * h
    grid = Image.new("RGB", size=(grid_w, grid_h))

    for i, img in enumerate(imgs):
        x = i % cols * w
        y = i // cols * h
        grid.paste(img, box=(x, y))

    return grid


def load_pipe(LORA_FOLDER):
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    if LORA_FOLDER:
        try:
            pipe.load_lora_weights(
                LORA_FOLDER, weight_name="pytorch_lora_weights.safetensors"
            )
        except:
            pipe.load_lora_weights(
                LORA_FOLDER, weight_name="pytorch_lora_weights_kohya.safetensors"
            )
    return pipe


def get_token(prompt):
    if "zwx" in prompt:
        return "zwx"
    if "sks" in prompt:
        return "sks"
    raise NotImplementedError


def process_lora_folder(lora_folder=None):

    PROJECT = "finetune-sdxl-alex-3"
    if lora_folder:
        with open(lora_folder + "/training_params.json") as f:
            training_params = json.load(f)
        token = get_token(training_params["prompt"])
        training_params["num_images"] = len(list(glob(lora_folder + "/autotrain-data/concept*/*")))
        training_params["token"] = token

        wandb.init(
            project=PROJECT,
            id=lora_folder.replace("/", "-"),
            config=training_params,
        )
    else:
        token = "sks"
        wandb.init(
            project=PROJECT,
            id="SDXL-base",
        )

    pipe = load_pipe(lora_folder)

    for prompt, prompt_template in list(prompts_generator(token)):
        print(prompt, prompt_template)
        seed = 0
        generator = torch.Generator("cuda").manual_seed(seed)

        image = pipe(
            prompt=prompt,
            num_inference_steps=25,
            num_images_per_prompt=3,
            negative_prompt="disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w",
            generator=generator,
        )
        # import ipdb
        # ipdb.set_trace()
        wandb_images = [wandb.Image(img) for img in image.images]
        wandb.log({prompt_template: wandb_images})


if __name__ == "__main__":
    fire.Fire(process_lora_folder)
