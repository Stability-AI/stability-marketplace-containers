from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from pydantic import BaseModel, Field
import torch
from typing import Optional
 

class SdxlTurboRequest(BaseModel):
    prompt: str
    num_inference_steps: int = Field(default=4)
    guidance_scale: float = Field(default=0.0)
    strength: float = Field(default=1.0)
    num_images_per_prompt: int = Field(default=1)
    image: Optional[str] = None

def setup_pipeline(model_name: str, cached_model_path):
    pipe_t2i = AutoPipelineForText2Image.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        cache_dir=cached_model_path,
        force_download=False,
        local_files_only=True
    )

    #fp16 won't work with cpu.
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()
    else:
        raise ValueError("This container must be run on a CUDA compatible architecture. Ensure this image has been run with --gpus all")
    pipe_t2i = pipe_t2i.to(device)

    pipe_t2i.unet = torch.compile(pipe_t2i.unet, mode="reduce-overhead", fullgraph=True)
    pipe_t2i.upcast_vae()

    pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i).to(device)

    return pipe_t2i, pipe_i2i
