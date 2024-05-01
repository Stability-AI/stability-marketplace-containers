import base64
import io
import logging
import os
import sys
import traceback
 
import uvicorn

from fastapi import FastAPI, HTTPException, Response, status
from PIL import Image
from sdxl_turbo import SdxlTurboRequest, setup_pipeline

MODEL_NAME = os.getenv("MODEL_NAME")
CACHED_MODEL_PATH = os.getenv('SAVE_PATH')

if MODEL_NAME is None or CACHED_MODEL_PATH is None:
    logging.error("Environment variables MODEL_NAME and CACHED_MODEL_PATH must be set. See Dockerfile for values.")
    sys.exit(1) 

app = FastAPI()


@app.on_event("startup")
def load_model():
    global pipe_t2i, pipe_i2i
    pipe_t2i, pipe_i2i = setup_pipeline(MODEL_NAME, CACHED_MODEL_PATH)

    logging.info("Sdxl Turbo model loaded.")


# Heartbeat endpoint
@app.get("/")
async def heartbeat():
    return {"status": "alive"}

# sdxl turbo text to image endpoint
@app.post("/sdxl-turbo-t2i")
async def generate_t2i(request: SdxlTurboRequest):
    try:
        image = pipe_t2i(
            prompt=request.prompt,
            strength=request.strength,
            guidance_scale=request.guidance_scale,
            num_images_per_prompt=request.num_images_per_prompt,
            num_inference_steps=request.num_inference_steps,
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return Response(
            content=buffer.getvalue(),
            status_code=status.HTTP_200_OK,
            media_type="image/png",
        )
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

# sdxl turbo image to image endpoint
@app.post("/sdxl-turbo-i2i")
async def generate_i2i(request: SdxlTurboRequest):
    try:
        init_image = request.image
        base64_decoded = base64.b64decode(init_image)
        input_image = Image.frombytes("RGB", (512, 512), base64_decoded, "raw")
    
        image = pipe_i2i(
            image=input_image,
            prompt=request.prompt,
            strength=request.strength,
            guidance_scale=request.guidance_scale,
            num_images_per_prompt=request.num_images_per_prompt,
            num_inference_steps=request.num_inference_steps,
        ).images[0]

        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return Response(
                content=buffer.getvalue(),
                status_code=status.HTTP_200_OK,
                media_type="image/png",
            )
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    uvicorn.run("entrypoint:app", host="0.0.0.0", port=port)