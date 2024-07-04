# Run these commands in terminal
#python -m pip install -e detectron2
#pip install av>=8.0.3 opencv-python-headless>=4.5.3.56 scipy>=1.5.4

import subprocess

command = "python detectron2/projects/DensePose/apply_net.py show detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl C:\\Users\\Pulkit\\Desktop\\Preprocessing\\data\\image dp_segm -v --opts MODEL.DEVICE cpu"
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
if stdout:
    print("Output:\n", stdout.decode())
if stderr:
    print("Error:\n", stderr.decode())