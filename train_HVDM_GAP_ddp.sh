# CUDA_VISIBLE_DEVICES=1,2,3,4

version=$1 ##1024, 512, 256
seed=123
name=dynamicrafter_$1_seed${seed}

ckpt=checkpoints/dynamicrafter_$1_v1/model.ckpt
config=configs/inference_$1_v1.0.yaml


res_dir="results"


if [ "$1" = "256" ]; then
    H=256
    FS=3  ## This model adopts frame stride=3, range recommended: 1-6 (larger value -> larger motion)
elif [ "$1" = "512" ]; then
    H=320
    FS=24 ## This model adopts FPS=24, range recommended: 15-30 (smaller value -> larger motion)
elif [ "$1" = "1024" ]; then
    H=576
    FS=10 ## This model adopts FPS=10, range recommended: 15-5 (smaller value -> larger motion)
else
    echo "Invalid input. Please enter 256, 512, or 1024."
    exit 1
fi


if [ "$1" == "256" ]; then
CUDA_VISIBLE_DEVICE=4 python3 scripts/evaluation/train_HVDM_GAP_ddp.py \
  --seed ${seed} \
  --ckpt_path $ckpt \
  --config $config \
  --savedir $res_dir/$name \
  --n_samples 1 \
  --bs 1 --height ${H} --width $1 \
  --unconditional_guidance_scale 7.5 \
  --ddim_steps 50 \
  --ddim_eta 1.0 \
  --frame_stride ${FS} \
  --train_batch_size 8 \
  --phi_dimension 32 \
  --int_dimension 64 \
  --attack all \
  --output_dir model_pth_HVDM_GAP_1004
# --data_path ../video

#accelerate launch
else
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/train_HVDM_GAP_ddp.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height ${H} --width $1 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--frame_stride ${FS} \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae
fi


## multi-cond CFG: the <unconditional_guidance_scale> is s_txt, <cfg_img> is s_img
#--multiple_cond_cfg --cfg_img 7.5
#--loop
