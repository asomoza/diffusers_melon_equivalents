import torch
from diffusers import EulerAncestralDiscreteScheduler, ModularPipeline
from diffusers.pipelines.components_manager import ComponentsManager
from diffusers.pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_modular import (
    AUTO_BLOCKS,
)
from diffusers.utils import load_image

device = "cuda:0"
dtype = torch.float16
repo = "Lykon/dreamshaper-xl-v2-turbo"
guidance_scale = 2.0
num_inference_steps = 8
seed = 740578200694213
prompt = "cinematic film still, film grain, 35mm, high budget, cinemascope, epic"
height = 1024
width = 1024
image_url = "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/modular_diffusers/horse.png"
strength = 0.8

## COMPONENTS LOADER NODE
components = ComponentsManager()
components.add_from_pretrained(repo, variant="fp16", torch_dtype=dtype)
components.enable_auto_cpu_offload(device=device)

all_blocks_map = AUTO_BLOCKS.copy()
text_block = all_blocks_map.pop("text_encoder")()
decoder_block = all_blocks_map.pop("decode")()
image_encoder_block = all_blocks_map.pop("image_encoder")()


class SDXLAutoBlocks(SequentialPipelineBlocks):
    block_classes = list(all_blocks_map.values())
    block_names = list(all_blocks_map.keys())


sdxl_auto_blocks = SDXLAutoBlocks()

### TEXT ENCODER NODE
text_node = ModularPipeline.from_block(text_block)
text_node.update_states(
    **components.get(["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"])
)
text_state = text_node(prompt=prompt)

# we need the generator for the image node and the sdxl node
generator = torch.Generator(device="cuda").manual_seed(0)

### LOAD IMAGE NODE
source_image = load_image(image_url).convert("RGB")

### IMAGE NODE
image_node = ModularPipeline.from_block(image_encoder_block)
image_node.update_states(**components.get(["vae"]))
image_state = image_node(image=source_image, generator=generator)


### SDXL NODE
sdxl_node = ModularPipeline.from_block(sdxl_auto_blocks)
sdxl_node.update_states(**components.get(["unet", "scheduler", "vae"]))

sdxl_node.scheduler = EulerAncestralDiscreteScheduler.from_config(
    sdxl_node.scheduler.config, use_karras_sigmas=True
)

generator = torch.Generator(device="cuda").manual_seed(seed)
latents = sdxl_node(
    **text_state.intermediates,
    **image_state.intermediates,
    strength=strength,
    generator=generator,
    guidance_scale=guidance_scale,
    num_inference_steps=round(num_inference_steps / strength),
    height=height,
    width=width,
    output="latents",
)

### DECODE LATENTS NODE
decoder_node = ModularPipeline.from_block(decoder_block)
decoder_node.update_states(vae=components.get("vae"))
images_output = decoder_node(latents=latents, output="images")

### SAVE IMAGE
images_output.images[0].save("outputs/img2img_simple.png")
