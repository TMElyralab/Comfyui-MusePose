### Build environment

```shell
pip install -r requirements.txt
```

### mmlab packages
```bash
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 
```

### Download weights
You can download weights manually as follows:

1. Download our trained [weights](https://huggingface.co/TMElyralab/MusePose).

2. Download the weights of other components:
   - [sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/unet)
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   - [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
   - [yolox](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth)
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
|   |-- config.json
|   └── diffusion_pytorch_model.bin
|-- image_encoder
|   |-- config.json
|   └── pytorch_model.bin
└── sd-vae-ft-mse
    |-- config.json
    └── diffusion_pytorch_model.bin

