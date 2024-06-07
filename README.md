### MusePose

[MusePose](https://github.com/TMElyralab/MusePose) is an image-to-video generation framework for virtual human under control signal such as pose. 

`MusePose` is the last building block of **the Muse opensource serie**. Together with [MuseV](https://github.com/TMElyralab/MuseV) and [MuseTalk](https://github.com/TMElyralab/MuseTalk), we hope the community can join us and march towards the vision where a virtual human can be generated end2end with native ability of full body movement and interaction. Please stay tuned for our next milestone!


### Comfyui-MusePose


If you're running on Linux, or non-admin account on windows you'll want to ensure `/ComfyUI/custom_nodes` and `Comfyui-MusePose` has write permissions.

Followed ComfyUI's manual installation steps and do the following:
  - Navigate to your `/ComfyUI/custom_nodes/` folder
  - Run `git clone https://github.com/TMElyralab/Comfyui-MusePose.git`
  - Navigate to your `/ComfyUI/custom_nodes/Comfyui-MusePose` folder and run
  ```shell
   pip install -r requirements.txt

   pip install --no-cache-dir -U openmim 
   mim install mmengine 
   mim install "mmcv>=2.0.1" 
   mim install "mmdet>=3.1.0" 
   mim install "mmpose>=1.1.0" 
  ```
  - Start ComfyUI

#### Updates
- requirements.txt: diffusers 0.27.2 is now supported

### Download weights
You can download weights manually as follows:

1. Download our trained [weights](https://huggingface.co/TMElyralab/MusePose).

2. Download the weights of other components:
   - [sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/unet)
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   - [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
   - [yolox](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth) - Make sure to rename to `yolox_l_8x8_300e_coco.pth`
   - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)

Finally, these weights should be organized in `pretrained_weights` as follows:
```
./pretrained_weights/
|-- MusePose
|   |-- denoising_unet.pth
|   |-- motion_module.pth
|   |-- pose_guider.pth
|   └── reference_unet.pth
|-- dwpose
|   |-- dw-ll_ucoco_384.pth
|   └── yolox_l_8x8_300e_coco.pth
|-- sd-image-variations-diffusers
|   └── unet
|       |-- config.json
|       └── diffusion_pytorch_model.bin
|-- image_encoder
|   |-- config.json
|   └── pytorch_model.bin
└── sd-vae-ft-mse
    |-- config.json
    └── diffusion_pytorch_model.bin

```
### workflow demo
https://github.com/TMElyralab/Comfyui-MusePose/blob/main/musepose-workflow-demo.json

https://github.com/TMElyralab/Comfyui-MusePose/assets/114042542/9cd8b9b8-6876-4281-b7a0-a7fbcb2de7e1
