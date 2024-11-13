<div align=center>
  <h1>
  LIT :fire: : Lighting-Conditioned Image Translation
  </h1>
  <p>
    <a href=https://mhsung.github.io/kaist-cs492d-fall-2024/ target="_blank"><b>KAIST CS492(D): Diffusion Models and Their Applications (Fall 2024)</b></a><br>
    Course Project
  </p>
</div>

<div align=center>
  <p>
    Instructor: <a href=https://mhsung.github.io target="_blank"><b>Minhyuk Sung</b></a> (mhsung [at] kaist.ac.kr)<br>
    TA: <a href=https://jh27kim.github.io/ target="_blank"><b>Jaihoon Kim</b></a>  (jh27kim [at] kaist.ac.kr)
  </p>
</div>

<div align=center>
   <img src="./assets/teaser.png">
   <figcaption>
    Large scale image dataset of real-world objects with diverse materials and geometries, captured under various illuminations.
    <i>Source: <a href="https://oppo-us-research.github.io/OpenIllumination/">OpenIllumination dataset.</a></i>
    </figcaption>
</div>

## Description
In this project, your task is to implement a conditional image diffusion model that takes a source image with its lighting conditions and generates an image with the target lighting conditions. 
You will use real-world object images captured under different lighting conditions provided by [OpenIllumination](https://huggingface.co/datasets/OpenIllumination/OpenIllumination). 

In this project, you are allowed to use only [Stable Diffusion v2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) as the pretrained diffusion model. 
Note that using any other versions of Stable Diffusion is **NOT** allowed.

## Data Specification
> :warning: Failure to meet the criteria outlined below may result in a zero score.
> 
> Do **NOT** use the pairs of images specified in `eval.json` for training, as they will be used for evaluation.
> 
> Do **NOT** use images with camera poses other than those specified below for training.

OpenIllumination dataset provides two kinds of lighting conditions: OLAT (One-Light-At-a-Time) and lighting patterns. In this task, we will focus on lighting patterns conditions (13 patterns). 
Use the following command to download `lighting_patterns` dataset in `$LOCAL_DIR`:
```
python open_illumination.py --light lighting_patterns --local_dir {$LOCAL_DIR}
```
The dataset consists of 64 objects, each captured under 13 different lighting patterns and 48 distinct camera poses. 
Additionally, the environment maps are provided [here](https://github.com/oppo-us-research/OpenIlluminationCapture/issues/2). You may use them for both training and evaluation. 
The dataset structure is outlined below:
```
./obj_01_car/
├── Lights                         # Light patterns (13)
│   ├── 001
│   │   └── raw_undistorted        # Images captured using "001" light pattern (48 per pattern)
│   │       ├── CA2.JPG
│   │       ├── CA4.JPG
│   │       ├── ...
│   │       └── NF7.JPG
│   ├── 002
└── output
    ├── com_masks                  # Object + support mask 
    ├── obj_masks                  # Object mask (May not be visible as the images are stored in [0-1])
    │   ├── CA2.png
    │   ├── CA4.png
    │   ├── ...
    │   └── NF7.png
    ├── ...
```

For training, use image pairs (condition and output) that capture the same object at the same camera pose, differing only in lighting conditions. 
For example, `./007/CA2.JPG` and `./005/CA2.JPG` form a valid pair, while `./007/CA2.JPG` and `./005/CA4.JPG` do not, as they are captured at different camera poses.
To reduce computational burden, we will limit the camera poses in the training data to 10 poses (NA3, NE7, CB5, CF8, NA7, CC7, CA2, NE1, NC3, CE2), which have approximately 0° elevation. This will scale down the dataset to roughly 100K images (64 objects x 13 x 12 light pattern combinations x 10 camera poses).

Note that the images are not square, and the objects are not center-aligned. 
Preprocess your training pairs using the `center_crop_img()` function from `preprocess_img.py`, which centers the objects and resizes the images to 128x128 pixels (do not modify the resolution). 

## Tasks
Your task is to implement a conditional diffusion model that takes a source image and its lighting condition and generates a target image with the desired lighting condition.
An example of the input and the desired output images is shown in the figure below. 

<div align="center">
  <figure>
    <img src="./assets/task.png" width="500">
    <br />
    <figcaption style="text-align: center;">
      Example of a source image (a) under one lighting condition and a target image (b) under a different lighting condition.
    </figcaption>
  </figure>
</div>

## Evaluation
Once the training is completed, generate images with the target lighting condition (specified in the `tgt_light` field of `eval.json`) using the source images (specified in the `src_img_path` field of `eval.json`) . The source and target images capture the same object with the same camera pose but under different lighting conditions. 
Note that the generated images should have the **SAME** filename as the dictionary keys (e.g., `obj_14_red_bucket_src_003_tgt_010_NA1`) to correctly identify the target images during evaluation. 

For evaluation, we will use the center-aligned ground truth images.
Run the following command to preprocess the ground truth images:
```
python preprocess_img.py --data_root {$DATA_ROOT} --save_root {$SAVE_ROOT}
```
`$DATA_ROOT` refers to the root directory of the dataset, and `$SAVE_ROOT` is the root directory where the preprocessed images will be saved.

Place the generated and ground truth images in the same directory, respectively, and ensure that each pair of images shares the same filename, as shown below:
```
├── generated_images                                          # Generated images from your model
│   ├── obj_01_car_src_009_tgt_004_CE2.png
│   ├── obj_02_egg_src_003_tgt_010_CC5.png
│   ├── ...
│   └── obj_64_greenhead_src_011_tgt_010_CD8.png
└── ground_truch_images                                       # Preprocessed ground truth images of eval.json
    ├── obj_01_car_src_009_tgt_004_CE2.png
    ├── obj_02_egg_src_003_tgt_010_CC5.png
    ├── ...
    └── obj_64_greenhead_src_011_tgt_010_CD8.png
```

We will use FID, PSNR, SSIM, and LPIPS scores to assess the quality and fidelity of the generated images compared to the ground truth target images.
First, install the following packages 
```
pip install lpips clean-fid torchmetrics
```

Then run the following command to evaluate on these metrics: 
```
python eval.py --fdir1 {$FDIR1} --fdir2 {$FDIR2} --save_dir {$SAVE_DIR}
```
`$FDIR1` and `$FDIR2` refer to the paths of the ground truth and generated images, respectively, and `$SAVE_DIR` is the path where the evaluation output file will be saved.

## Acknowledgement 
We appreciate the authors of [OpenIllumination](https://oppo-us-research.github.io/OpenIllumination/) for releasing their dataset to public. 

## References
* FID: [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
* LPIPS: [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924)
* SSIM: [Image quality assessment: from error visibility to structural similarity](https://ieeexplore.ieee.org/document/1284395)
