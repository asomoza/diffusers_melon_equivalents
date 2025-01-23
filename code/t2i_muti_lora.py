import torch
from diffusers import EulerAncestralDiscreteScheduler, ModularPipeline
from diffusers.pipelines.components_manager import ComponentsManager
from diffusers.pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_modular import (
    AUTO_BLOCKS,
    StableDiffusionXLLoraStep,
)

device = "cuda:0"
dtype = torch.float16
repo = "Lykon/dreamshaper-xl-v2-turbo"
guidance_scale = 2.0
num_inference_steps = 8
seed = 851573115810490
prompt = (
    "cinematic film still of an anthropomorphic capybara, ral-dissolve, pixar style"
)
height = 1024
width = 1024

## COMPONENTS LOADER NODE
components = ComponentsManager()
components.add_from_pretrained(repo, variant="fp16", torch_dtype=dtype)
components.enable_auto_cpu_offload(device=device)

all_blocks_map = AUTO_BLOCKS.copy()
text_block = all_blocks_map.pop("text_encoder")()
decoder_block = all_blocks_map.pop("decode")()


class SDXLAutoBlocks(SequentialPipelineBlocks):
    block_classes = list(all_blocks_map.values())
    block_names = list(all_blocks_map.keys())


sdxl_auto_blocks = SDXLAutoBlocks()

### LORA NODE
lora_step = StableDiffusionXLLoraStep()
lora_node = ModularPipeline.from_block(lora_step)
lora_node.update_states(**components.get(["text_encoder", "text_encoder_2", "unet"]))
lora_node.load_lora_weights(
    "rajkumaralma/dissolve_dust_style",
    weight_name="ral-dissolve-sdxl.safetensors",
    adapter_name="ral-dissolve",
)
lora_node.load_lora_weights(
    "rajkumaralma/pixar_style",
    weight_name="PixarXL.safetensors",
    adapter_name="pixar_style",
)
lora_node.set_adapters(["ral-dissolve", "pixar_style"], [1.2, 1.0])

### TEXT ENCODER NODE
text_node = ModularPipeline.from_block(text_block)
text_node.update_states(
    **components.get(["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"])
)
text_state = text_node(prompt=prompt)


### SDXL NODE
sdxl_node = ModularPipeline.from_block(sdxl_auto_blocks)
sdxl_node.update_states(**components.get(["unet", "scheduler", "vae"]))

sdxl_node.scheduler = EulerAncestralDiscreteScheduler.from_config(
    sdxl_node.scheduler.config
)

generator = torch.Generator(device="cuda").manual_seed(seed)
latents = sdxl_node(
    **text_state.intermediates,
    generator=generator,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    height=height,
    width=width,
    output="latents",
)

### DECODE LATENTS NODE
decoder_node = ModularPipeline.from_block(decoder_block)
decoder_node.update_states(vae=components.get("vae"))
images_output = decoder_node(latents=latents, output="images")

### SAVE IMAGE
images_output.images[0].save("outputs/t2i_multi_lora.png")
