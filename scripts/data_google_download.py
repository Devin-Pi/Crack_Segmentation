import os
import urllib.request
import zipfile
import tarfile
import gdown

data_dir = "./data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
    
weights_dir = "./weights/"
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)
    

url = 'http://drive.google.com/file/d/1cWedC2ZxObZPZ-AfK5_jY7wjlvE1YhY1/view?usp=sharing'
target_path = os.path.join(data_dir) 

gdown.download(url, target_path, quiet=False,fuzzy=True)
