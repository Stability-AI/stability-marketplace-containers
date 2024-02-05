# marketplace-containers
This repo will hold our universal marketplace containers to be released cross platform.

Container images are all tested on AWS EC2 instance types g5.2xl, which are equipped with Nvidia A10 GPUs.

### [SDXL Turbo](./sdxl-turbo)
SDXL Turbo is an adversarial time-distilled Stable Diffusion XL (SDXL) model capable of running inference in as little as 1 step.


This container runs the SDXL Turbo model as a web service using FastAPI. It is available on the AWS Marketplace from Stability AI.

Once you have pulled the Docker image for the Marketplace product, run the container to start the server:

Run on a `cuda` device with:
```
$ docker run --gpus all -p 8080:8080 -t <image-name>
```

To test the model:
```
cd sdxl_turbo
pip install -r client_requirements.txt

python client_request.py
```

Take a look at `sdxl_turbo/client_request.py` for an example of a text-to-image request and an image-to-image request.

### [Stable Video Diffusion Turbo](./svd)
Stable Video Diffusion (SVD) Image-to-Video is a diffusion model that takes in a still image as a conditioning frame, and generates a 25 frame video from it.

#### Request parameters:

`image`: The image to be used as a conditioning frame. Must be of size (1024, 576) pixels.

`motion_bucket_id`: An int This can be used to control the motion of the generated video. Increasing the motion bucket id will increase the motion of the generated video.
Defaults to 127.

` noise_aug_strength`: A float between [0, 1]. A multiplier, used to determine the amount of noise added to the conditioning image. The higher the values the less the video will resemble the conditioning image. Increasing this value will also increase the motion of the generated video. 
Default 0.2.


Run on a `cuda` device with:
```
$ docker run --gpus all -p 8080:8080 -t <image-name>
```

To test the model:
```
cd svd
pip install -r client_requirements.txt

python client_request.py
```

The client_request.py demo code uses the `ffmpeg` Python bindings to change the codec of the output video to `h264`, so that the resulting video can be previewed in VSCode. This is not essential to the functionality of the container and can be removed.