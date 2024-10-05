from accelerate import Accelerator
import torch
import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import os 

import argparse, os, sys, glob, yaml, math, random
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm
from einops import rearrange, repeat
from functools import partial
import torch
from pytorch_lightning import seed_everything
from train_utils import (visualize_two_videos_frames, normalize_tensor, check_tensor_range, save_video, 
get_phis, get_params_optimize, get_parser, check_model_gradient_flow, calculate_psnr, log_metrics)

from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
from funcs import batch_ddim_sampling
from utils.utils import instantiate_from_config
from customization import customize_vae_decoder
from attribution import MappingNetwork  
from torchvision.models import resnet50, ResNet50_Weights 
from attack_methods.attack_initializer import attack_initializer
import lpips 
import hydra 
from attribution import MappingNetwork
from torchvision import transforms
import itertools
from diffusers.optimization import get_scheduler
import cv2 
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F 
from metrics_video import tensor_to_np_img, get_img_metrics_from_tensors, save_log_stats_to_csv
import matplotlib.pyplot as plt 
# from our_decoder import * 
from pytorch_wavelets import DWTForward, DWTInverse
from our_resnet_3d import * 
from accelerate.utils.dataclasses import DistributedDataParallelKwargs
# from HVDM_based_decoder import *
from HVDM_GAP_based_decoder import *
from DWT3D.make_dwt import DWT_3D



def run_inference(args):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=None,
        kwargs_handlers=[kwargs]
    )  # Initialize Accelerator for multi-GPU support
    device = accelerator.device

    train_path = os.path.join(args.data_path, 'train')
    train_video_files = [f for f in os.listdir(train_path) if f.endswith('mp4')][:10000]
    training_video_files = train_video_files

    val_path = os.path.join(args.data_path, 'valid')
    val_video_files = [f for f in os.listdir(val_path) if f.endswith('mp4')][:1000]

    print("Number of dataset :", len(training_video_files), len(val_video_files))

    # Model config
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['unet_config']['params']['use_checkpoint'] = False

    model = instantiate_from_config(model_config)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path, strict=False)
    model.first_stage_model = customize_vae_decoder(model.first_stage_model, args.int_dimension, args.lr_mult)

    # Mapping Network
    mapping_network = MappingNetwork(args.phi_dimension, args.int_dimension, num_layers=args.mapping_layer)

    # Decoding Network 
    res = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    res.fc = torch.nn.Linear(2048, args.phi_dimension) 
    decoding_network = HVDM_with_Resnet(res, args.phi_dimension)

    # Prepare the transforms
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Optimizer and loss function
    optimizer_cls = torch.optim.AdamW
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    optimizer = optimizer_cls(
        get_params_optimize(model.first_stage_model, mapping_network, decoding_network),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.cosine_cycle * args.gradient_accumulation_steps,
    )
    model.to(device)
    # model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    # Prepare everything with Accelerator, including the DataLoader
    model, mapping_network, decoding_network, optimizer = accelerator.prepare(
        model, mapping_network, decoding_network, optimizer
    )

    train_aug = attack_initializer(args, is_train=True, device=device)
    dwt = DWT_3D(wavename='haar') 

    for epoch in tqdm(range(args.num_epochs), desc="NUM EPOCHS"):
        model.train()
        mapping_network.train()
        decoding_network.train()

        for (batch_idx, video_file) in tqdm(enumerate(training_video_files), desc="Current Video"):
            video_path = os.path.join(os.path.join(args.data_path, 'train'), video_file)
            cap = cv2.VideoCapture(video_path)
            frame_tensors = []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start_frame = max(0, total_frames // 2 - args.train_batch_size // 2)
            end_frame = min(total_frames, start_frame + args.train_batch_size)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for _ in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = train_transforms(frame)
                frame_tensors.append(frame)

            cap.release()

            try:
                frame_tensors = torch.stack(frame_tensors)
            except Exception as e:
                print(f"Error stacking frame_tensors: {e}")
                continue

            train_dataset = TensorDataset(frame_tensors)
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False)

            # Prepare the DataLoader with Accelerator
            train_dataloader = accelerator.prepare(train_dataloader)

            for idx, image_tensor in enumerate(train_dataloader):
                image_tensor[0] = image_tensor[0].to(device)
                frames = image_tensor[0].permute(1, 0, 2, 3).unsqueeze(0).to(device)

                latent_z = model.encode_first_stage(frames)
                latent_z = 1. / 0.18215 * latent_z

                phis = get_phis(args.phi_dimension, latent_z.shape[2]).to(device)

                encoded_phis = mapping_network(phis)

                b, _, t, _, _ = latent_z.shape
                z = rearrange(latent_z, 'b c t h w -> (b t) c h w')
                # generated_video = model.module.first_stage_model.decode(z, encoded_phis).sample
                generated_video = model.first_stage_model.decode(z, encoded_phis).sample
                generated_video_ = rearrange(generated_video, "(b t) c h w -> b c t h w", b=b, t=t)

                if 'AE_' in args.attack:
                    augmented_video = train_aug.forward((generated_video / 2 + 0.5).clamp(0, 1)).clamp(0, 1)
                else:
                    augmented_video = train_aug((generated_video / 2 + 0.5).clamp(0, 1))
                
                dwt_x = dwt(augmented_video.unsqueeze(0).permute(0, 2, 1, 3, 4))[0]
                reconstructed_keys = decoding_network(augmented_video, dwt_x)
                loss_key = F.binary_cross_entropy_with_logits(reconstructed_keys, phis)
                loss_lpips_reg = loss_fn_vgg(generated_video, image_tensor[0]).mean()
                loss = loss_key + loss_lpips_reg


                gt_phi = (phis > 0.5).int() 
                predicted_key = (torch.sigmoid(reconstructed_keys) > 0.5).int() 
                bit_acc = ((gt_phi == predicted_key).sum(dim=1)) / args.phi_dimension
                psnr = calculate_psnr(image_tensor[0], generated_video)
                log_metrics(loss_key, loss_lpips_reg, bit_acc, psnr, batch_idx)
                
                visualize_two_videos_frames(image_tensor[0], generated_video, save_path=f'{args.output_dir}/output_videos_frames_HVDM.png')

                optimizer.zero_grad()
                accelerator.backward(loss)
                
                optimizer.step()
                lr_scheduler.step()

                check_model_gradient_flow(model, mapping_network, decoding_network)
                

                if batch_idx % 1000 == 0:
                    # 모델을 저장하기 전에 accelerator.unwrap_model을 사용하여 원래의 모델을 추출
                    torch.save(accelerator.unwrap_model(model).first_stage_model.state_dict(), 
                            args.output_dir + f'/model_checkpoint_{args.phi_dimension}_{args.train_batch_size}_{epoch}_{batch_idx}_1003_our.ckpt')
                    
                    torch.save(mapping_network.state_dict(), 
                            args.output_dir + f"/mapping_network_{args.phi_dimension}_{args.train_batch_size}_{epoch}_{batch_idx}_1003_our.pth")
                    
                    torch.save(decoding_network.state_dict(), 
                            args.output_dir + f"/decoding_network_{args.phi_dimension}_{args.train_batch_size}_{epoch}_{batch_idx}_1003_our.pth")
                    
                    print("Model Save!!")

    print("Training complete")

if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Inference: %s" % now)
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    run_inference(args)
