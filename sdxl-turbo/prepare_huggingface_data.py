from diffusers import AutoPipelineForText2Image
import os
import traceback
import torch
try:

    SAVEPATH=os.environ.get("SAVE_PATH")
    SAVEPATH = os.getenv("SAVE_PATH", "/opt/model")
    MODEL = os.getenv("MODEL_NAME", "stabilityai/sdxl-turbo")

    os.makedirs(SAVEPATH,exist_ok=True)
    pipe = AutoPipelineForText2Image.from_pretrained(MODEL, torch_dtype=torch.float16, variant="fp16",use_safetensors=True,
    cache_dir=SAVEPATH,
    force_download=True)
    del(pipe)
except:
    print(traceback.format_exc())