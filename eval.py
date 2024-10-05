import os
import yaml
import argparse
import json
import glob 
from pathlib import Path
from datetime import datetime
from typing import Literal, Union

from PIL import Image 
import numpy as np
import torch

import lpips
from cleanfid import fid
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def pil_to_torch(img):
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

class Evaluator:
    def __init__(
        self,
        fdir1: Union[str, os.PathLike] = None,
        fdir2: Union[str, os.PathLike] = None,
        save_dir: Union[str, os.PathLike] = None,
        device: int = 0,
    ):
        self.fdir1 = fdir1
        self.fdir2 = fdir2
        self.save_dir = save_dir

        # Load evaluation metrics
        self.device = device
        self.PSNR = PeakSignalNoiseRatio().to(self.device)
        self.SSIM = StructuralSimilarityIndexMeasure().to(self.device)
        self.VGG = lpips.LPIPS(net='vgg').to(self.device)


    def __call__(self):
        fdir1 = self.fdir1
        fdir2 = self.fdir2
        save_dir = self.save_dir

        assert Path(fdir1).exists(), f"{fdir1} not exist."
        assert Path(fdir2).exists(), f"{fdir2} not exist."

        # Compute FID: set-to-set
        fid_score = fid.compute_fid(fdir1, fdir2)
        
        # Compute PSNR, SSIM, LPIPNS. image-to-image.
        # Make sure the corresponding images have the same filename in both directories. 
        assert len(glob.glob(f"{fdir1}/*")) == len(glob.glob(f"{fdir2}/*")), f"{len(glob.glob(f'{fdir1}/*'))} != {len(glob.glob(f'{fdir2}/*'))}"

        psnr_score = []
        ssim_score = []
        lpips_score = []
        for img_path in glob.glob(f"{fdir1}/*"):
            fn = img_path.split("/")[-1]
            src_img_path = img_path
            tgt_img_path = os.path.join(fdir2, fn)
            assert os.path.exists(src_img_path), src_img_path
            assert os.path.exists(tgt_img_path), tgt_img_path

            img1 = Image.open(src_img_path).convert("RGB")
            img2 = Image.open(tgt_img_path).convert("RGB")

            torch_img1 = pil_to_torch(img1).to(self.device) # (1, 3, H, W), [0, 1]
            torch_img2 = pil_to_torch(img2).to(self.device)
            assert torch_img2.shape == torch_img1.shape, f"{torch_img1.shape}, {torch_img2.shape}"
            assert torch_img1.max() <= 1.0 and torch_img1.min() >= 0.0

            psnr_score.append(self.PSNR(torch_img1, torch_img2).item())
            ssim_score.append(self.SSIM(torch_img1, torch_img2).item())
            lpips_score.append(self.VGG(torch_img1, torch_img2).item())

        N = len(psnr_score)
        metric_dict = {
            "fdir1": fdir1,
            "fdir2": fdir2,
            "FID": fid_score,
            "PSNR": sum(psnr_score) / N,
            "SSIM": sum(ssim_score) / N,
            "LPIPS": sum(lpips_score) / N,
        }

        Path(save_dir).mkdir(exist_ok=True)
        now = get_current_time()
        metric_path = Path(save_dir) / f"eval_result_{now}.json"
        with open(metric_path, "w") as f:
            json.dump(metric_dict, f, indent=4)

        print(f"[*] Logged at {metric_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir1", type=str, required=True)
    parser.add_argument("--fdir2", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    fdir1 = args.fdir1  # imgae directory 1
    fdir2 = args.fdir2  # image directory 2
    save_dir = args.save_dir  # log directory 
    device = args.device 

    evaluator = Evaluator(
        fdir1, fdir2, save_dir, device
    )
    evaluator()
