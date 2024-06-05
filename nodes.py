import os,sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import av
import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
import glob
import torch.nn.functional as F
from collections import namedtuple

import copy
import cv2
import moviepy.video.io.ImageSequenceClip

PROJECT_DIR = os.path.dirname(__file__)
sys.path.insert(0,PROJECT_DIR)

from musepose.models.pose_guider import PoseGuider
from musepose.models.unet_2d_condition import UNet2DConditionModel
from musepose.models.unet_3d import UNet3DConditionModel
from musepose.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from musepose.utils.util import get_fps, read_frames, save_videos_grid

from pose.script.dwpose import DWposeDetector, draw_pose
from pose.script.util import size_calculate, warpAffine_kps

device_auto = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f'@@device:{device_auto}')


def align_img(img, pose_ori, scales, detect_resolution, image_resolution):

    body_pose = copy.deepcopy(pose_ori['bodies']['candidate'])
    hands = copy.deepcopy(pose_ori['hands'])
    faces = copy.deepcopy(pose_ori['faces'])

    # h不变，w缩放到原比例
    H_in, W_in, C_in = img.shape 
    video_ratio = W_in / H_in
    body_pose[:, 0]  = body_pose[:, 0] * video_ratio
    hands[:, :, 0] = hands[:, :, 0] * video_ratio
    faces[:, :, 0] = faces[:, :, 0] * video_ratio

    # scales of 10 body parts 
    scale_neck      = scales["scale_neck"] 
    scale_face      = scales["scale_face"]
    scale_shoulder  = scales["scale_shoulder"]
    scale_arm_upper = scales["scale_arm_upper"]
    scale_arm_lower = scales["scale_arm_lower"]
    scale_hand      = scales["scale_hand"]
    scale_body_len  = scales["scale_body_len"]
    scale_leg_upper = scales["scale_leg_upper"]
    scale_leg_lower = scales["scale_leg_lower"]

    scale_sum = 0
    count = 0
    scale_list = [scale_neck, scale_face, scale_shoulder, scale_arm_upper, scale_arm_lower, scale_hand, scale_body_len, scale_leg_upper, scale_leg_lower]
    for i in range(len(scale_list)):
        if not np.isinf(scale_list[i]):
            scale_sum = scale_sum + scale_list[i]
            count = count + 1
    for i in range(len(scale_list)):
        if np.isinf(scale_list[i]):   
            scale_list[i] = scale_sum/count

    # offsets of each part 
    offset = dict()
    offset["14_15_16_17_to_0"] = body_pose[[14,15,16,17], :] - body_pose[[0], :] 
    offset["3_to_2"] = body_pose[[3], :] - body_pose[[2], :] 
    offset["4_to_3"] = body_pose[[4], :] - body_pose[[3], :] 
    offset["6_to_5"] = body_pose[[6], :] - body_pose[[5], :] 
    offset["7_to_6"] = body_pose[[7], :] - body_pose[[6], :] 
    offset["9_to_8"] = body_pose[[9], :] - body_pose[[8], :] 
    offset["10_to_9"] = body_pose[[10], :] - body_pose[[9], :] 
    offset["12_to_11"] = body_pose[[12], :] - body_pose[[11], :] 
    offset["13_to_12"] = body_pose[[13], :] - body_pose[[12], :] 
    offset["hand_left_to_4"] = hands[1, :, :] - body_pose[[4], :]
    offset["hand_right_to_7"] = hands[0, :, :] - body_pose[[7], :]

    # neck
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_neck)

    neck = body_pose[[0], :] 
    neck = warpAffine_kps(neck, M)
    body_pose[[0], :] = neck

    # body_pose_up_shoulder
    c_ = body_pose[0]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_face)

    body_pose_up_shoulder = offset["14_15_16_17_to_0"] + body_pose[[0], :]
    body_pose_up_shoulder = warpAffine_kps(body_pose_up_shoulder, M)
    body_pose[[14,15,16,17], :] = body_pose_up_shoulder

    # shoulder 
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_shoulder)

    body_pose_shoulder = body_pose[[2,5], :] 
    body_pose_shoulder = warpAffine_kps(body_pose_shoulder, M) 
    body_pose[[2,5], :] = body_pose_shoulder

    # arm upper left
    c_ = body_pose[2]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_upper)
 
    elbow = offset["3_to_2"] + body_pose[[2], :]
    elbow = warpAffine_kps(elbow, M)
    body_pose[[3], :] = elbow

    # arm lower left
    c_ = body_pose[3]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_lower)
 
    wrist = offset["4_to_3"] + body_pose[[3], :]
    wrist = warpAffine_kps(wrist, M)
    body_pose[[4], :] = wrist

    # hand left
    c_ = body_pose[4]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_hand)
 
    hand = offset["hand_left_to_4"] + body_pose[[4], :]
    hand = warpAffine_kps(hand, M)
    hands[1, :, :] = hand

    # arm upper right
    c_ = body_pose[5]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_upper)
 
    elbow = offset["6_to_5"] + body_pose[[5], :]
    elbow = warpAffine_kps(elbow, M)
    body_pose[[6], :] = elbow

    # arm lower right
    c_ = body_pose[6]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_lower)
 
    wrist = offset["7_to_6"] + body_pose[[6], :]
    wrist = warpAffine_kps(wrist, M)
    body_pose[[7], :] = wrist

    # hand right
    c_ = body_pose[7]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_hand)
 
    hand = offset["hand_right_to_7"] + body_pose[[7], :]
    hand = warpAffine_kps(hand, M)
    hands[0, :, :] = hand

    # body len
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_body_len)

    body_len = body_pose[[8,11], :] 
    body_len = warpAffine_kps(body_len, M)
    body_pose[[8,11], :] = body_len

    # leg upper left
    c_ = body_pose[8]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_upper)
 
    knee = offset["9_to_8"] + body_pose[[8], :]
    knee = warpAffine_kps(knee, M)
    body_pose[[9], :] = knee

    # leg lower left
    c_ = body_pose[9]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_lower)
 
    ankle = offset["10_to_9"] + body_pose[[9], :]
    ankle = warpAffine_kps(ankle, M)
    body_pose[[10], :] = ankle

    # leg upper right
    c_ = body_pose[11]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_upper)
 
    knee = offset["12_to_11"] + body_pose[[11], :]
    knee = warpAffine_kps(knee, M)
    body_pose[[12], :] = knee

    # leg lower right
    c_ = body_pose[12]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_lower)
 
    ankle = offset["13_to_12"] + body_pose[[12], :]
    ankle = warpAffine_kps(ankle, M)
    body_pose[[13], :] = ankle

    # none part
    body_pose_none = pose_ori['bodies']['candidate'] == -1.
    hands_none = pose_ori['hands'] == -1.
    faces_none = pose_ori['faces'] == -1.

    body_pose[body_pose_none] = -1.
    hands[hands_none] = -1. 
    nan = float('nan')
    if len(hands[np.isnan(hands)]) > 0:
        print('nan')
    faces[faces_none] = -1.

    # last check nan -> -1.
    body_pose = np.nan_to_num(body_pose, nan=-1.)
    hands = np.nan_to_num(hands, nan=-1.)
    faces = np.nan_to_num(faces, nan=-1.)

    # return
    pose_align = copy.deepcopy(pose_ori)
    pose_align['bodies']['candidate'] = body_pose
    pose_align['hands'] = hands
    pose_align['faces'] = faces

    return pose_align


def run_align_video_with_filterPose_translate_smooth(args):
 
    vidfn=args.vidfn
    imgfn_refer=args.imgfn_refer
    outfn=args.outfn
    
    video = cv2.VideoCapture(vidfn)
    width= video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height= video.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
    total_frame= video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps= video.get(cv2.CAP_PROP_FPS)

    print("height:", height)
    print("width:", width)
    print("fps:", fps)

    H_in, W_in  = height, width
    H_out, W_out = size_calculate(H_in,W_in,args.detect_resolution) 
    H_out, W_out = size_calculate(H_out,W_out,args.image_resolution) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = DWposeDetector(
        det_config = args.yolox_config, 
        det_ckpt = args.yolox_ckpt,
        pose_config = args.dwpose_config, 
        pose_ckpt = args.dwpose_ckpt, 
        keypoints_only=False
        )    
    detector = detector.to(device)

    refer_img = cv2.imread(imgfn_refer)
    output_refer, pose_refer = detector(refer_img,detect_resolution=args.detect_resolution, image_resolution=args.image_resolution, output_type='cv2',return_pose_dict=True)
    body_ref_img  = pose_refer['bodies']['candidate']
    hands_ref_img = pose_refer['hands']
    faces_ref_img = pose_refer['faces']
    output_refer = cv2.cvtColor(output_refer, cv2.COLOR_RGB2BGR)
    

    skip_frames = args.align_frame
    max_frame = args.max_frame
    pose_list, video_frame_buffer, video_pose_buffer = [], [], []

    for i in range(max_frame):
        ret, img = video.read()
        if img is None: 
            break 
        else: 
            if i < skip_frames:
                continue           
            video_frame_buffer.append(img)

       
        # estimate scale parameters by the 1st frame in the video
        if i==skip_frames:
            output_1st_img, pose_1st_img = detector(img, args.detect_resolution, args.image_resolution, output_type='cv2', return_pose_dict=True)
            body_1st_img  = pose_1st_img['bodies']['candidate']
            hands_1st_img = pose_1st_img['hands']
            faces_1st_img = pose_1st_img['faces']

            
            # h不变，w缩放到原比例
            ref_H, ref_W = refer_img.shape[0], refer_img.shape[1]
            ref_ratio = ref_W / ref_H
            body_ref_img[:, 0]  = body_ref_img[:, 0] * ref_ratio
            hands_ref_img[:, :, 0] = hands_ref_img[:, :, 0] * ref_ratio
            faces_ref_img[:, :, 0] = faces_ref_img[:, :, 0] * ref_ratio

            video_ratio = width / height
            body_1st_img[:, 0]  = body_1st_img[:, 0] * video_ratio
            hands_1st_img[:, :, 0] = hands_1st_img[:, :, 0] * video_ratio
            faces_1st_img[:, :, 0] = faces_1st_img[:, :, 0] * video_ratio

            # scale
            align_args = dict()
            
            dist_1st_img = np.linalg.norm(body_1st_img[0]-body_1st_img[1])   # 0.078   
            dist_ref_img = np.linalg.norm(body_ref_img[0]-body_ref_img[1])   # 0.106
            align_args["scale_neck"] = dist_ref_img / dist_1st_img  # align / pose = ref / 1st

            dist_1st_img = np.linalg.norm(body_1st_img[16]-body_1st_img[17])
            dist_ref_img = np.linalg.norm(body_ref_img[16]-body_ref_img[17])
            align_args["scale_face"] = dist_ref_img / dist_1st_img

            dist_1st_img = np.linalg.norm(body_1st_img[2]-body_1st_img[5])  # 0.112
            dist_ref_img = np.linalg.norm(body_ref_img[2]-body_ref_img[5])  # 0.174
            align_args["scale_shoulder"] = dist_ref_img / dist_1st_img  

            dist_1st_img = np.linalg.norm(body_1st_img[2]-body_1st_img[3])  # 0.895
            dist_ref_img = np.linalg.norm(body_ref_img[2]-body_ref_img[3])  # 0.134
            s1 = dist_ref_img / dist_1st_img
            dist_1st_img = np.linalg.norm(body_1st_img[5]-body_1st_img[6])
            dist_ref_img = np.linalg.norm(body_ref_img[5]-body_ref_img[6])
            s2 = dist_ref_img / dist_1st_img
            align_args["scale_arm_upper"] = (s1+s2)/2 # 1.548

            dist_1st_img = np.linalg.norm(body_1st_img[3]-body_1st_img[4])
            dist_ref_img = np.linalg.norm(body_ref_img[3]-body_ref_img[4])
            s1 = dist_ref_img / dist_1st_img
            dist_1st_img = np.linalg.norm(body_1st_img[6]-body_1st_img[7])
            dist_ref_img = np.linalg.norm(body_ref_img[6]-body_ref_img[7])
            s2 = dist_ref_img / dist_1st_img
            align_args["scale_arm_lower"] = (s1+s2)/2

            # hand
            dist_1st_img = np.zeros(10)
            dist_ref_img = np.zeros(10)      
             
            dist_1st_img[0] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,1])
            dist_1st_img[1] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,5])
            dist_1st_img[2] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,9])
            dist_1st_img[3] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,13])
            dist_1st_img[4] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,17])
            dist_1st_img[5] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,1])
            dist_1st_img[6] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,5])
            dist_1st_img[7] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,9])
            dist_1st_img[8] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,13])
            dist_1st_img[9] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,17])

            dist_ref_img[0] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,1])
            dist_ref_img[1] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,5])
            dist_ref_img[2] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,9])
            dist_ref_img[3] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,13])
            dist_ref_img[4] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,17])
            dist_ref_img[5] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,1])
            dist_ref_img[6] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,5])
            dist_ref_img[7] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,9])
            dist_ref_img[8] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,13])
            dist_ref_img[9] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,17])

            ratio = 0   
            count = 0
            for i in range (10): 
                if dist_1st_img[i] != 0:
                    ratio = ratio + dist_ref_img[i]/dist_1st_img[i]
                    count = count + 1
            if count!=0:
                align_args["scale_hand"] = (ratio/count+align_args["scale_arm_upper"]+align_args["scale_arm_lower"])/3
            else:
                align_args["scale_hand"] = (align_args["scale_arm_upper"]+align_args["scale_arm_lower"])/2

            # body 
            dist_1st_img = np.linalg.norm(body_1st_img[1] - (body_1st_img[8] + body_1st_img[11])/2 )
            dist_ref_img = np.linalg.norm(body_ref_img[1] - (body_ref_img[8] + body_ref_img[11])/2 )
            align_args["scale_body_len"]=dist_ref_img / dist_1st_img

            dist_1st_img = np.linalg.norm(body_1st_img[8]-body_1st_img[9])
            dist_ref_img = np.linalg.norm(body_ref_img[8]-body_ref_img[9])
            s1 = dist_ref_img / dist_1st_img
            dist_1st_img = np.linalg.norm(body_1st_img[11]-body_1st_img[12])
            dist_ref_img = np.linalg.norm(body_ref_img[11]-body_ref_img[12])
            s2 = dist_ref_img / dist_1st_img
            align_args["scale_leg_upper"] = (s1+s2)/2

            dist_1st_img = np.linalg.norm(body_1st_img[9]-body_1st_img[10])
            dist_ref_img = np.linalg.norm(body_ref_img[9]-body_ref_img[10])
            s1 = dist_ref_img / dist_1st_img
            dist_1st_img = np.linalg.norm(body_1st_img[12]-body_1st_img[13])
            dist_ref_img = np.linalg.norm(body_ref_img[12]-body_ref_img[13])
            s2 = dist_ref_img / dist_1st_img
            align_args["scale_leg_lower"] = (s1+s2)/2

            ####################
            ####################
            # need adjust nan
            for k,v in align_args.items():
                if np.isnan(v):
                    align_args[k]=1

            # centre offset (the offset of key point 1)
            offset = body_ref_img[1] - body_1st_img[1]
        
    
        # pose align
        pose_img, pose_ori = detector(img, args.detect_resolution, args.image_resolution, output_type='cv2', return_pose_dict=True)
        video_pose_buffer.append(pose_img)
        pose_align = align_img(img, pose_ori, align_args, args.detect_resolution, args.image_resolution)
        

        # add centre offset
        pose = pose_align
        pose['bodies']['candidate'] = pose['bodies']['candidate'] + offset
        pose['hands'] = pose['hands'] + offset
        pose['faces'] = pose['faces'] + offset


        # h不变，w从绝对坐标缩放回0-1 注意这里要回到ref的坐标系
        pose['bodies']['candidate'][:, 0] = pose['bodies']['candidate'][:, 0] / ref_ratio
        pose['hands'][:, :, 0] = pose['hands'][:, :, 0] / ref_ratio
        pose['faces'][:, :, 0] = pose['faces'][:, :, 0] / ref_ratio
        pose_list.append(pose)

    # stack
    body_list  = [pose['bodies']['candidate'][:18] for pose in pose_list]
    body_list_subset = [pose['bodies']['subset'][:1] for pose in pose_list]
    hands_list = [pose['hands'][:2] for pose in pose_list]
    faces_list = [pose['faces'][:1] for pose in pose_list]
   
    body_seq         = np.stack(body_list       , axis=0)
    body_seq_subset  = np.stack(body_list_subset, axis=0)
    hands_seq        = np.stack(hands_list      , axis=0)
    faces_seq        = np.stack(faces_list      , axis=0)


    # concatenate and paint results
    H = 768 # paint height
    W1 = int((H/ref_H * ref_W)//2 *2)
    W2 = int((H/height * width)//2 *2)
    result_demo = [] # = Writer(args, None, H, 3*W1+2*W2, outfn, fps)
    result_pose_only = [] # Writer(args, None, H, W1, args.outfn_align_pose_video, fps)
    for i in range(len(body_seq)):
        pose_t={}
        pose_t["bodies"]={}
        pose_t["bodies"]["candidate"]=body_seq[i]
        pose_t["bodies"]["subset"]=body_seq_subset[i]
        pose_t["hands"]=hands_seq[i]
        pose_t["faces"]=faces_seq[i]

        ref_img = cv2.cvtColor(refer_img, cv2.COLOR_RGB2BGR)
        ref_img = cv2.resize(ref_img, (W1, H))
        ref_pose= cv2.resize(output_refer, (W1, H))
        
        output_transformed = draw_pose(
            pose_t, 
            int(H_in*1024/W_in), 
            1024, 
            draw_face=False,
            )
        output_transformed = cv2.cvtColor(output_transformed, cv2.COLOR_BGR2RGB)
        output_transformed = cv2.resize(output_transformed, (W1, H))
        
        video_frame = cv2.resize(video_frame_buffer[i], (W2, H))
        video_pose  = cv2.resize(video_pose_buffer[i], (W2, H))

        res = np.concatenate([ref_img, ref_pose, output_transformed, video_frame, video_pose], axis=1)
        result_demo.append(res)
        #result_pose_only.append(output_transformed)

        output_transformed_tensor_out = torch.tensor(np.array(output_transformed).astype(np.float32) / 255.0)  # Convert back to CxHxW
        output_transformed_tensor_out = torch.unsqueeze(output_transformed_tensor_out, 0)
        result_pose_only.append(output_transformed_tensor_out)
    res = torch.cat(tuple(result_pose_only), dim=0)
    print(res.shape)
    return (res,) 


class posealign_class:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "video": ("STRING",{"default":""}),
                "detect_resolution":("INT", {"default": 512}),
                "image_resolution":("INT", {"default": 700}),
                "max_frame":("INT", {"default": 300}),
                "align_frame":("INT", {"default": 0}),
            }
        }
 
    CATEGORY = "musepose_list"
 
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "posealign_func"
 
    def posealign_func(self, image, video, detect_resolution, image_resolution, max_frame, align_frame):
        ref_image = 255.0 * image[0].cpu().numpy()
        ref_image = Image.fromarray(np.clip(ref_image, 0, 255).astype(np.uint8))
        image_path = os.path.join(PROJECT_DIR, "data/ref_image.jpg")
        ref_image.save(image_path)
        
        Param = namedtuple('Param', [
                           'yolox_config',
                           'dwpose_config', 
                           'yolox_ckpt', 
                           'dwpose_ckpt',
                           'outfn_align_pose_video',
                           'outfn',
                           'detect_resolution',
                           'image_resolution',
                           'align_frame',
                           'max_frame', 
                           'imgfn_refer', 
                           'vidfn'])
        
        args = Param(
                     os.path.join(PROJECT_DIR,"pose/config/yolox_l_8xb8-300e_coco.py"),
                     os.path.join(PROJECT_DIR,"pose/config/dwpose-l_384x288.py"),
                     os.path.join(PROJECT_DIR,"pretrained_weights/dwpose/yolox_l_8x8_300e_coco.pth"),
                     os.path.join(PROJECT_DIR,"pretrained_weights/dwpose/dw-ll_ucoco_384.pth"), 
                     None, 
                     None,
                     detect_resolution,
                     image_resolution,
                     align_frame,
                     max_frame, 
                     image_path, 
                     video)
        print(args)

        return run_align_video_with_filterPose_translate_smooth(args)
 

class VHS_FILENAMES_STRING_MusePose:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "filenames": ("VHS_FILENAMES",),
                    }
                }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "musepose_list"
    FUNCTION = "run"

    def run(self, filenames):
        return (filenames[1][-1],)



def scale_video(video,width,height):
    video_reshaped = video.view(-1, *video.shape[2:])  # [batch*frames, channels, height, width]
    scaled_video = F.interpolate(video_reshaped, size=(height, width), mode='bilinear', align_corners=False)
    scaled_video = scaled_video.view(*video.shape[:2], scaled_video.shape[1], height, width)  # [batch, frames, channels, height, width]
    scaled_video = torch.squeeze(scaled_video)
    scaled_video = scaled_video.permute(1,2,3,0)
    
    return scaled_video


def musepose(args, image_path, video):
    config = OmegaConf.load(args.config)
    pretrained_base_model_path = os.path.join(PROJECT_DIR, config.pretrained_base_model_path)
    pretrained_vae_path = os.path.join(PROJECT_DIR, config.pretrained_vae_path)
    image_encoder_path = os.path.join(PROJECT_DIR, config.image_encoder_path)
    denoising_unet_path = os.path.join(PROJECT_DIR, config.denoising_unet_path)
    reference_unet_path = os.path.join(PROJECT_DIR, config.reference_unet_path)
    pose_guider_path = os.path.join(PROJECT_DIR, config.pose_guider_path)
    motion_module_path = os.path.join(PROJECT_DIR, config.motion_module_path)
    inference_config_path = os.path.join(PROJECT_DIR, config.inference_config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(pretrained_vae_path,
    ).to(device_auto, dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device_auto)

    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_base_model_path,
        motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device_auto)

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=device_auto
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        image_encoder_path
    ).to(dtype=weight_dtype, device=device_auto)

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(pose_guider_path, map_location="cpu"),
    )

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(device_auto, dtype=weight_dtype)
    pipe = pipe.to(device_auto, dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")

    def handle_single(ref_image_path, pose_video_path):
        print ('handle===',ref_image_path, pose_video_path)
        ref_name = Path(ref_image_path).stem
        pose_name = Path(pose_video_path).stem.replace("_kps", "")

        ref_image_pil = Image.open(ref_image_path).convert("RGB")

        pose_list = []
        pose_tensor_list = []
        pose_images = read_frames(pose_video_path)
        src_fps = get_fps(pose_video_path)
        print(f"pose video has {len(pose_images)} frames, with {src_fps} fps")
        L = min(args.L, len(pose_images))
        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
        original_width,original_height = 0,0

        pose_images = pose_images[::args.skip+1]
        print("processing length:", len(pose_images))
        src_fps = src_fps // (args.skip + 1)
        print("fps", src_fps)
        L = L // ((args.skip + 1))
        
        for pose_image_pil in pose_images[: L]:
            pose_tensor_list.append(pose_transform(pose_image_pil))
            pose_list.append(pose_image_pil)
            original_width, original_height = pose_image_pil.size
            pose_image_pil = pose_image_pil.resize((width,height))

        # repeart the last segment
        last_segment_frame_num =  (L - args.S) % (args.S - args.O) 
        repeart_frame_num = (args.S - args.O - last_segment_frame_num) % (args.S - args.O) 
        for i in range(repeart_frame_num):
            pose_list.append(pose_list[-1])
            pose_tensor_list.append(pose_tensor_list[-1])

        
        ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=L)

        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0)

        video = pipe(
            ref_image_pil,
            pose_list,
            width,
            height,
            len(pose_list),
            args.steps,
            args.cfg,
            generator=generator,
            context_frames=args.S,
            context_stride=1,
            context_overlap=args.O,
        ).videos
        print(video.shape)

        m1 = config.pose_guider_path.split('.')[0].split('/')[-1]
        m2 = config.motion_module_path.split('.')[0].split('/')[-1]

        res = scale_video(video[:,:,:L], original_width, original_height)
        print(res.shape)
        return (res,)

    return handle_single(image_path, video) 
                    


class musepose_class:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "video": ("STRING",{"default":""}),
                "Width":("INT", {"default": 512}),
                "Height":("INT", {"default": 512}),
                "frame_length":("INT", {"default": 300}),
                "slice_frame_number":("INT", {"default": 48}),
                "slice_overlap_frame_number":("INT", {"default": 4}),
                "cfg":("FLOAT", {"default": 3.5}),
                "sampling_steps":("INT", {"default": 20}),
                "fps":("INT", {"default": 12}),
            }
        }
 
    CATEGORY = "musepose_list"
 
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "musepose_func"
 
    def musepose_func(self, image, video, Width, Height, frame_length, slice_frame_number, slice_overlap_frame_number, cfg, sampling_steps, fps):
        Param = namedtuple('Param',[
                           'config',
                           'W',
                           'H',
                           'L',
                           'S',
                           'O',
                           'cfg',
                           'seed',
                           'steps',
                           'fps',
                           'skip'])
        args = Param(os.path.join(PROJECT_DIR,"configs/test_stage_2.yaml"),
                     Width,
                     Height,
                     frame_length,
                     slice_frame_number,
                     slice_overlap_frame_number,
                     cfg,
                     99,
                     sampling_steps,
                     fps,
                     1) 

        ref_image = 255.0 * image[0].cpu().numpy()
        ref_image = Image.fromarray(np.clip(ref_image, 0, 255).astype(np.uint8))
        image_path = os.path.join(PROJECT_DIR, "data/ref_image.jpg")

        return musepose(args, image_path, video)
 


NODE_CLASS_MAPPINGS = {
    "museposealign": posealign_class,
    "musepose": musepose_class,
    "filenamestring": VHS_FILENAMES_STRING_MusePose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "museposealign": "muse_pose_align",
    "musepose": "muse_pose",
    "filenamestring": "muse_pose_filenamestring",
}

