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