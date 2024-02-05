import base64
import json
import os
import requests

from PIL import Image


# This is an example script to show how to send a request to the SDXL Turbo endpoint
def get_t2i_payload():
    # Define your request payload
    payload = {
        "prompt": "a tree full of lemons in the Sicilian sun",
        "num_inference_steps": 1   # Turbo needs only 1 step
    }

    # Convert payload to JSON
    json_payload = json.dumps(payload)

    return json_payload


def get_i2i_payload():
    # Read the image file as binary
    image_path = os.path.join(os.getcwd(), "cat.png")

    with open(image_path, "rb") as image_file:
        original_image = Image.open(image_file)
        resized_image = original_image.resize((512, 512))
        # Convert the binary data to a base64-encoded string
        base64_encoded_image = base64.b64encode(resized_image.tobytes()).decode("utf-8")

    # Define your request payload
    payload = {
        "image": base64_encoded_image,
        "prompt":  "cat wizard, adorable",
        "strength": 0.7,
        "guidance_scale": 0.0,
        "num_inference_steps": 3,
    }

    # Convert payload to JSON
    json_payload = json.dumps(payload)

    return json_payload


def test():
    # Define the URL (replace with your actual URL)
    url = "http://0.0.0.0:8080/sdxl-turbo-t2i"

    # test text2image
    json_payload = get_t2i_payload()

    # Send the POST request
    response = requests.post(url, data=json_payload)

    # Check the response
    if response.status_code == 200:
        # Process the response (e.g., save the image, print success message)
        with open("output_image_t2i.png", "wb") as f:
            f.write(response.content)
        print(f"Image saved successfully as output_image_t2i.png.")
        pass
    else:
        print("Error:", response.status_code, response.text)

    i2i_url = "http://0.0.0.0:8080/sdxl-turbo-i2i"

    # test image2image
    json_payload = get_i2i_payload()

    # Send the POST request
    response = requests.post(i2i_url, data=json_payload)

    # Check the response
    if response.status_code == 200:
        # Process the response (e.g., save the image, print success message)
        with open("output_image_i2i.png", "wb") as f:
            f.write(response.content)
        print(f"Image saved successfully as output_image_i2i.png.")
        pass
    else:
        print("Error:", response.status_code, response.text)


if __name__ == "__main__":
    test()
