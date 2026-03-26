import os
import gdown

def download_weights(download_dir="checkpoints"):
    os.makedirs(download_dir, exist_ok=True)

    # Google Drive folder URL
    url = "https://drive.google.com/drive/folders/1BZoCcW8YzKFzHbMExrWTXE_I3ctyaidX"

    print(f"Downloading weights to {download_dir}...")

    # Download entire folder
    gdown.download_folder(url=url, output=download_dir, quiet=False, use_cookies=False)

    print("Download complete.")

if __name__ == "__main__":
    download_weights()