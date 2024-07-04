## We get image-parse-agnostic-v3.2 from this
"""

# Here is the parse label and corresponding body parts. You may need or not.
# 0 - 20
# Background
# Hat
# Hair
# Glove
# Sunglasses
# Upper-clothes
# Dress
# Coat
# Socks
# Pants
# tosor-skin
# Scarf
# Skirt
# Face
# Left-arm
# Right-arm
# Left-leg
# Right-leg
# Left-shoe
# Right-shoe

# pip install Pillow tqdm

# Need:
# 1. cloth
# 2. cloth-mask
# 3. image
# 4. image-densepose
# 5. image-parse-v3
# 6. openpose_img
# 7. openpose_json

# Upload the zip file containing these folders
"""

# unzip ForTestingParseAgnostic.zip

import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

def get_im_parse_agnostic(im_parse, pose_data, w=768, h=1024):
    label_array = np.array(im_parse)
    parse_upper = ((label_array == 5).astype(np.float32) +
                    (label_array == 6).astype(np.float32) +
                    (label_array == 7).astype(np.float32))
    parse_neck = (label_array == 10).astype(np.float32)

    r = 10
    agnostic = im_parse.copy()

    # mask arms
    for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
        mask_arm = Image.new('L', (w, h), 'black')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        i_prev = pose_ids[0]
        for i in pose_ids[1:]:
            if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
            pointx, pointy = pose_data[i]
            radius = r*4 if i == pose_ids[-1] else r*15
            mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
            i_prev = i
        parse_arm = (np.array(mask_arm) / 255) * (label_array == parse_id).astype(np.float32)
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    # mask torso & neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

    return agnostic


if __name__ =="__main__":
    data_path = 'C:\\Users\\Pulkit\\Desktop\\Preprocessing\\data'
    output_path = 'C:\\Users\\Pulkit\\Desktop\\Preprocessing\\data\\image-parse-agnostic-v3.2'

    os.makedirs(output_path, exist_ok=True)

    for im_name in tqdm(os.listdir(osp.join(data_path, 'image'))):

        # load pose image
        pose_name = im_name.replace('.jpg', '_keypoints.json')

        try:
            with open(osp.join(data_path, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]
        except IndexError:
            print(pose_name)
            continue

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(data_path, 'image-parse-v3', parse_name))
        size=im_parse.size

        agnostic = get_im_parse_agnostic(im_parse, pose_data, size[0], size[1])

        agnostic.save(osp.join(output_path, parse_name))

# import numpy as np
# from PIL import Image

# im_ori = Image.open('/content/ForTestingParseAgnostic/image-parse-v3/00008_00.png')
# im = Image.open('/content/ForTestingParseAgnostic/image-parse-agnostic-v3.2/00008_00.png')
# print(np.unique(np.array(im_ori)))
# print(np.unique(np.array(im)))

# np_im = np.array(im)
# np_im[np_im==2] = 151
# np_im[np_im==9] = 178
# np_im[np_im==13] = 191
# np_im[np_im==14] = 221
# np_im[np_im==15] = 246
# Image.fromarray(np_im)

