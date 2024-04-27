import torch
import os
import time
import requests
import torchvision.transforms as transforms
from models import get_model
from PIL import Image 

def download_file(input_path):
    """
    Download a file from a given URL and save it locally if input_path is a URL.
    If input_path is a local file path and the file exists, skip the download.

    :param input_path: The URL of the file to download or a local file path.
    :return: The local filepath to the downloaded or existing file.
    """
    # Check if input_path is a URL
    if input_path.startswith(('http://', 'https://')):
        # Extract filename from the URL
        # Splits the URL by '/' and get the last part
        filename = input_path.split('/')[-1]

        # Ensure the filename does not contain query parameters if present in the URL
        # Splits the filename by '?' and get the first part
        filename = filename.split('?')[0]

        # Define the local path where the file will be saved
        local_filepath = os.path.join('.', filename)

        # Check if file already exists locally
        if os.path.isfile(local_filepath):
            print(f"The file already exists locally: {local_filepath}")
            return local_filepath

        # Start timing the download
        start_time = time.time()

        # Send a GET request to the URL
        response = requests.get(input_path, stream=True)

        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Open the local file in write-binary mode
        with open(local_filepath, 'wb') as file:
            # Write the content of the response to the local file
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # End timing the download
        end_time = time.time()

        # Calculate the download duration
        download_duration = end_time - start_time

        print(
            f"Downloaded file saved to {local_filepath} in {download_duration:.2f} seconds.")

    else:
        # Assume input_path is a local file path
        local_filepath = input_path
        # Check if the specified local file exists
        if not os.path.isfile(local_filepath):
            raise FileNotFoundError(f"No such file: '{local_filepath}'")
        print(f"Using existing file: {local_filepath}")

    return local_filepath

def is_image(img):
    return os.path.isfile(img) and img.endswith(
        tuple([".jpg", ".jpeg", ".png"])
    )

def real_or_fake_thres(probability, threshold=0.3):
    return "FAKE" if probability >= threshold else "REAL"

class CustomModel:
    """
    Wrapper class for the UniversalFakeDetect model.

    Initially designed to work for TrueMedia servers. Can be used in the future to interact
    with the model in a more flexible manner.
    """

    def __init__(self):
        model = get_model("CLIP:ViT-L/14")
        self.fc_state_dict = torch.load("checkpoints/clip_vitl14_celebahq/model_epoch_best.pth", map_location='cpu')
        model.fc.load_state_dict(self.fc_state_dict)
        model.eval()
        model.cuda()

        self.model = model

        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])


    def predict(self, inputs):
        file_path = inputs.get('file_path', None)
        image_file = download_file(file_path)

        if os.path.isfile(image_file):  
            try:
                if is_image(image_file):
                    print(f"Model is being run.")
                    return self._forward(image_file)
                else:
                    err_msg = f"Invalid media file: {image_file}. Please provide a valid image file."
                    print(err_msg)
                    return {"msg": err_msg}
            except Exception as e:
                err_msg = f"An error occurred: {str(e)}"
                print(err_msg)
                return {"msg": err_msg}

    def _forward(self, img_path, threshold=0.3):
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img).cuda().unsqueeze(0)

        with torch.no_grad():
            prob = self.model(img).sigmoid()
        
        return {"df_probability": prob.item(), "prediction": real_or_fake_thres(prob.item(), threshold)}

def main():
    # testing model setup with single file upload
    model = CustomModel()
    test_input = {'file_path': "https://uploads.civai.org/files/jhxTVhsg/b751515306e7.jpg"}
    output = model.predict(test_input)

if __name__=="__main__":
    main()