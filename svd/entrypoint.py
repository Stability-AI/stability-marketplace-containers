# main.py
import base64
import io
import logging
import os
import sys
import traceback
 
import torch
import uvicorn
from diffusers.utils import export_to_video

from fastapi import FastAPI, HTTPException, Response, status
from svd import SVDRequest, setup_pipeline
from PIL import Image

# for SVD
MODEL_NAME = os.environ["MODEL_NAME"]
CACHED_MODEL_PATH = os.environ['SAVE_PATH']
app = FastAPI()


@app.on_event("startup")
def load_model():
    global pipe_svd
    pipe_svd = setup_pipeline(MODEL_NAME, CACHED_MODEL_PATH)
    logging.info("Stable Video Diffusion (SVD) Image-to-Video model loaded.")

# Heartbeat endpoint
@app.get("/")
async def heartbeat():
    return {"status": "alive"}

# svd img2vid endpoint
@app.post("/svd-img2vid")
async def generate_img2vid(request: SVDRequest):
    try:
        # Load the conditioning image
        init_image = request.image
        base64_decoded = base64.b64decode(init_image)

        image = Image.frombytes("RGB", (1024, 576), base64_decoded, "raw")
        
        generator = torch.manual_seed(42)
        frames = pipe_svd(
            image=image,
            decode_chunk_size=8,
            generator=generator,
            motion_bucket_id=180,
            noise_aug_strength=0.1
        ).frames[0]

        export_to_video(frames, "generated.mp4", fps=25)

        # Read the generated video file into the buffer
        buffer = io.BytesIO()
        with open("generated.mp4", "rb") as video_file:
            buffer.write(video_file.read())

        return Response(
            content=buffer.getvalue(),
            status_code=status.HTTP_200_OK,
            media_type="video/mp4",
        )
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating video: {str(e)}")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    uvicorn.run("entrypoint:app", host="0.0.0.0", port=port, log_level="debug")