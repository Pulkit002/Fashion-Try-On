import subprocess
import os

os.makedirs(os.path.join('C:\\Users\\Pulkit\\Desktop\\Preprocessing\\data\\','image-parse-v3'), exist_ok=True)

command = "python Self-Correction-Human-Parsing-master\\simple_extractor.py --dataset 'atr' --gpu 'None' --model-restore 'python Self-Correction-Human-Parsing-master\\checkpoints\\exp-schp-201908301523-atr.pth' --input-dir 'C:\\Users\\Pulkit\\Desktop\\Preprocessing\\data\\image' --output-dir 'C:\\Users\\Pulkit\\Desktop\\Preprocessing\\data\\image-parse-v3'"
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
if stdout:
    print("Output:\n", stdout.decode())
if stderr:
    print("Error:\n", stderr.decode())