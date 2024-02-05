from diffusers import StableVideoDiffusionPipeline
from pydantic import BaseModel, Field
import torch 

class SVDRequest(BaseModel):
    noise_aug_strength: float = Field(default=0.2)
    motion_bucket_id: int = Field(default=127)
    image: str

def setup_pipeline(model_name: str, cached_model_path):
    pipe_svd = StableVideoDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        cache_dir=cached_model_path,
        force_download=False,
        local_files_only=True,
    )

    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()
    else:
        raise ValueError("This container must be run on a CUDA compatible architecture. Ensure this image has been run with --gpus all")
    
    pipe_svd = pipe_svd.to(device)
    # If memory is low, enable forward chunking at a cost to latency
    #pipe_svd.unet.enable_forward_chunking()
    
    return pipe_svd
