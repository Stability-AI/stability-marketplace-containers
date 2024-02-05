from diffusers import StableVideoDiffusionPipeline
import os
import traceback
import torch
try:

    SAVEPATH=os.environ.get("SAVE_PATH")
    MODEL = os.environ.get("MODEL_TYPE")

    print(SAVEPATH)
    print(MODEL)
    if SAVEPATH is None:
        SAVEPATH = "./svd_weights/"
    if MODEL is None:
        MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"
    os.makedirs(SAVEPATH,exist_ok=True)
    pipe = StableVideoDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, variant="fp16",use_safetensors=True,
    cache_dir=SAVEPATH,
    force_download=True)
    del(pipe)
except:
    print(traceback.format_exc())