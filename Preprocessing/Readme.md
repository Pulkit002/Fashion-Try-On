This folder contains preprocessing files
- Download models from https://drive.google.com/drive/folders/1GCSQuKh0BtRdBh2c92U1FlcL44g7QE3O?usp=sharing and add them in "model" folder
- Download checkpoints from https://drive.google.com/drive/folders/1F6Dmw8RBAds1nQ9MdPZLOPSn2KMZz8RP?usp=sharing and add them in "Self-Correction-Human-Parsing-master\checkpoints" folder
- Download detectron2 folder from https://drive.google.com/file/d/1PKgJOLMnbnkWwuFggrIP3yQgx4p_Im-5/view?usp=sharing and extract it here.

## Instructions
- Resize the input image to 1024x768
- run openpose
- run densepose
**Run these commands (for setting up densepose) in terminal for DensePose**
python -m pip install -e detectron2

(It may ask to install visual studio build tools, follow the instructions as prompted to install the above)

pip install av>=8.0.3 opencv-python-headless>=4.5.3.56 scipy>=1.5.4 
- run humanparse
humanParse not running in local, get the result by running it in kaggle (use lip model)

kaggle link - https://www.kaggle.com/code/pulkt02/humanparsing
- run parseagnostic
- run humanAgnostic
