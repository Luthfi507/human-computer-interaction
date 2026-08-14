import os
from urllib.request import urlretrieve
from urllib.parse import urlparse

def get_model(url):
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    filename = os.path.basename(urlparse(url).path)
    path = os.path.join(model_dir, filename)

    if not os.path.exists(path):
        urlretrieve(url, path)
        print(f"{filename} downloaded")
    
    return path  