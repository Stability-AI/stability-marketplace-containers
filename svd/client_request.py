import base64
import json
import os
import requests
import ffmpeg

from PIL import Image


# This is an example script to show how to send a request to the SVD endpoint
def get_img2vid_payload(file_path: str = None):
    # Read the image file as binary
    image_path = os.path.join(os.getcwd(), file_path)

    image_data = Image.open(image_path)
    base64_encoded_image = base64.b64encode(image_data.tobytes()).decode("utf-8")

    # Define your request payload
    payload = {
        "image": base64_encoded_image,
        "noise_aug_strength": 0.5,
        "motion_bucket_id": 300
    }

    # Convert payload to JSON
    json_payload = json.dumps(payload)

    return json_payload


def test():
    # Define the URL (replace with your actual URL)
    url = "http://0.0.0.0:8080/svd-img2vid"

    for image_file in ["Waves_in_pacifica_1024.jpg", "tropical_cyclone_florence_1024.jpg"]:
        json_payload = get_img2vid_payload(image_file)

        # Send the POST request
        response = requests.post(url, data=json_payload)

        # Check the response
        if response.status_code == 200:
            # strip the extension off the filename so we can use the base to name the outputs
            image_file = os.path.splitext(image_file)[0]
            with open(f"output_video_{image_file}.mp4", "wb") as f:
                f.write(response.content)

            # Convert the codec of the video so that it can be previewed in VS code
            
            outfile = f"output_video_{image_file}_h642.mp4"
            ffmpeg.input(f"output_video_{image_file}.mp4").output(outfile, vcodec='libx264').run(overwrite_output=True)

            print(f"Video saved successfully as {outfile}.")
            pass
        else:
            print("Error:", response.status_code, response.text)


if __name__ == "__main__":
    test()
