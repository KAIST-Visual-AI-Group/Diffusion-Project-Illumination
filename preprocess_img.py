import json 
import numpy as np 
from PIL import Image
import random 
import os 
import glob
import argparse 

# DO NOT modify the hyperparameters
RESIZE_H, RESIZE_W = 384, 384
H, W = 512, 512


def main(args):
    data_root = args.data_root
    save_root = args.save_root

    if os.path.exists(save_root):
        print("Save root already exists. Skipping...")
        exit(0)
    os.makedirs(save_root)

    with open("eval.json", "r") as f:
        metadata = json.load(f)

    for k, v in metadata.items():
        print("Processing ", k)

        # Modify this line to process other path data
        tgt_img_path = os.path.join(data_root, v["src_img_path"])

        mask_img_path = os.path.join(data_root, v["mask_path"])

        assert tgt_img_path is not None and mask_img_path is not None
            
        tgt_img = Image.open(tgt_img_path).convert("RGB")
        np_tgt_img = np.array(tgt_img)

        mask_img = Image.open(mask_img_path).convert("RGB") # Foreground mask
        # For some of the masks are given as [0, 255]
        if np.array(mask_img).max() > 1:
            np_mask_img = np.array(mask_img)
        else:
            np_mask_img = np.array(mask_img) * 255
        assert np_mask_img.max() <= 255 and np_mask_img.min() >= 0, f"{np_mask_img.min()}, {np_mask_img.max()}"
        np_tgt_img[np_mask_img == 0] = 255

        # Crop image using bbox
        y, x, r = np.where(np_mask_img == 255) # Get bbox using the mask
        x1, x2, y1, y2 = x.min(), x.max(), y.min(), y.max()

        crop_img = Image.fromarray(np_tgt_img).crop(
            (x1, y1, x2, y2)
        )

        w = x2 - x1 
        assert w > 0, f"{x2} - {x1} = {w}"
        h = y2 - y1 
        assert h > 0, f"{y2} - {y1} = {h}"

        # Resize image with respect to max length 
        max_length = max(w, h)
        ratio = RESIZE_W / max_length
        resized_w, resized_h = round(w * ratio), round(h * ratio) # Avoid float error
        assert resized_h == RESIZE_H or resized_w == RESIZE_W

        resized_mask = crop_img.resize(
            (resized_w, resized_h)
        )

        canvas = Image.new("RGB", (H, W), (255, 255, 255))
        pos_w, pos_h = resized_w - W, resized_h - H
        
        pos_w = abs(pos_w) // 2
        pos_h = abs(pos_h) // 2
        assert pos_w + resized_w <= W and pos_h + resized_h <= H

        canvas.paste(
            resized_mask, (pos_w, pos_h)
        )
        
        save_path = os.path.join(save_root, f"{k}.png")
        canvas.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--save_root", required=True)
    args = parser.parse_args()

    main(args)
