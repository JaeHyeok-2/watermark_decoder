# from torchvision.models import resnet50, ResNet50_Weights 
from funcs import load_model_checkpoint
from DWT3D.make_dwt import DWT_3D 
import numpy as np 
import torch 
import torch.nn as nn 
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F 



class HVDM_with_Resnet(nn.Module):
    def __init__(self, resnet_model, phi_dimension):
        super().__init__()
        self.phi_dimension = phi_dimension
        self.resnet = resnet_model
    
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.message_decoder = nn.Sequential(*list(self.resnet.children())[-1:])


        self.low_freq = nn.Sequential(
        torch.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(64, 64),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=64, out_channels=192, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(192, 192),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=192, out_channels=384, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
        torch.nn.GroupNorm(384, 384),
        torch.nn.Tanh(),

        torch.nn.Conv3d(in_channels=384, out_channels=768, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        torch.nn.GroupNorm(768, 768),
        torch.nn.Tanh(),
        )

        self.low_freq_mid = nn.Sequential(
            torch.nn.Conv3d(in_channels=768, out_channels=512, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            torch.nn.GroupNorm(512, 512),
            torch.nn.Tanh(),

            torch.nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            torch.nn.GroupNorm(512, 512),
            torch.nn.Tanh())

            # FC 레이어는 필요하지 않으므로 생략
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))  # GAP 연산


    def forward(self, x, dwt_3D): 
        res_hidden_states = self.feature_extractor(x)  # [8, 2048, 8, 8]
        
        low_freq = self.low_freq(dwt_3D) 
        low_freq = low_freq.view(low_freq.shape[0], -1 ,8, 32, 32)  # [1, 512, 8, 32, 32]
        z_low_freq = self.low_freq_mid(low_freq)  # [1, 512, 8, 8, 8]

        # GAP 연산을 통해 각 프레임에 대해 [8, 1] 가중치를 얻음
        z_gap = self.gap(z_low_freq).squeeze(-1).squeeze(-1)  # [1, 512, 8] -> [1, 512, 8]
        
        # 평균 가중치를 구해 [8, 1] 형태로 변환
        z_weight = z_gap.mean(dim=1)  # [1, 8]로 변환됨

        # res_hidden_states에 가중치 적용 (t축에 브로드캐스트로 곱하기)
        z_weight = z_weight.view(z_weight.size(1), 1, 1, 1)  # [8, 1, 1, 1]로 변경
        scaled_res_hidden_states = res_hidden_states * z_weight  # [8, 2048, 8, 8]에 각 프레임(t)에 대해 가중치 적용

        hidden_states = self.pool2d(scaled_res_hidden_states) 
        hidden_states = hidden_states.view(hidden_states.size(0), -1) 
        hidden_states = self.message_decoder(hidden_states) 



        return hidden_states


# dwt = DWT_3D(wavename='haar')
# # dwt = DWTForward(wave="haar", J=1, mode="periodization")

# resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# resnet.fc = torch.nn.Linear(2048, 32) 

# model =HVDM_with_Resnet(resnet,32).cuda() 
# x=  torch.rand(8,3,256,256).cuda()
# dwt_x = dwt(x.unsqueeze(0).permute(0,2, 1, 3, 4))[0]

# message = model(x, dwt_x) 
# print(message.shape)

