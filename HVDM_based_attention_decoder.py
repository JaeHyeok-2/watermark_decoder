from torchvision.models import resnet50, ResNet50_Weights 
from funcs import load_model_checkpoint
from DWT3D.make_dwt import DWT_3D 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, low_channels, high_channels, num_heads=8):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(low_channels, high_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(high_channels, high_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(high_channels, high_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        # 각 head의 크기
        self.head_dim = high_channels // num_heads
        assert high_channels % num_heads == 0, "high_channels는 num_heads로 나누어 떨어져야 합니다."

    def forward(self, z_low_freq, video_frames):
        # z_low_freq: [1, 512, 8, 8, 8]
        # video_frames: [8, 2048, 8, 8]

        # 1. 저주파 Feature Map을 batch에 맞춰서 reshape (Cross Attention을 위해 차원 맞추기)
        z_low_freq = z_low_freq.view(8, 512, 8, 8)  # [8, 512, 8, 8]

        # 2. Query, Key, Value 계산 (Multi-Head로 나누어 계산)
        batch_size, channels, height, width = video_frames.size()

        # Query, Key, Value 계산 후 Multi-Head로 나눔
        query = self.query_conv(z_low_freq).view(batch_size, self.num_heads, self.head_dim, height * width)  # [8, num_heads, head_dim, 64]
        key = self.key_conv(video_frames).view(batch_size, self.num_heads, self.head_dim, height * width)    # [8, num_heads, head_dim, 64]
        value = self.value_conv(video_frames).view(batch_size, self.num_heads, self.head_dim, height * width) # [8, num_heads, head_dim, 64]

        # 3. Attention 가중치 계산 (각 head별로 계산)
        attn_weights = torch.einsum("bnqd,bnkd->bnqk", query, key)  # [8, num_heads, 64, 64]
        attn_weights = self.softmax(attn_weights)

        # 4. Attention 가중치를 value에 적용
        attn_output = torch.einsum("bnqk,bnvd->bnqd", attn_weights, value)  # [8, num_heads, head_dim, 64]

        # 5. Multi-Head Attention 결과를 결합 (다시 원래 형태로 복원)
        attn_output = attn_output.contiguous().view(batch_size, channels, height, width)  # [8, 2048, 8, 8]

        return attn_output


class HVDM_with_Resnet(nn.Module):
    def __init__(self, resnet_model, phi_dimension):
        super().__init__()
        self.phi_dimension = phi_dimension
        self.resnet = resnet_model
    
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.message_decoder = nn.Sequential(*list(self.resnet.children())[-1:])
        self.MHCA = MultiHeadCrossAttention(512, 2048) 

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



    def forward(self, x, dwt_3D): 
        """
        arguments:
            x : Video Diffusion으로 생성된 비디오로 (t c h w ) shape으로 입력을 받을 예정
            dwt_x : HVDM의 Low Frequency만을 가지고 만들것이기때문에 우선 입력으로 (b c t h w) 로 받고, b=1 으로 설정할 예정.
        
        
        return :
            output으로, [t,phi_dimension] shape으로 Message를 복원하는 과정으로 최종출력을 진행할 예정
            
            1) ResNet50을 기반으로 Feature-Map을 [t,c',h',w']을 뽑고, 
            2) HVDM의 Frequency Encoder를 통해서 [b, t, c h/2, w/2] 생성. 
            이후 1,2의 Output을 통해 연산을 진행하여, Decoder를 생성하는 방식으로 해야할듯 .

        """
        res_hidden_states = self.feature_extractor(x)  
        
        low_freq = self.low_freq(dwt_3D) 
        low_freq = low_freq.view(low_freq.shape[0], -1 ,8, 32, 32) 
        z_low_freq = self.low_freq_mid(low_freq) # 1 512, 8 , 8 , 8 
        
        cross_attention_featuremap = self.MHCA(z_low_freq, res_hidden_states) 
        
        hidden_states = self.pool2d(cross_attention_featuremap)
        hidden_states = hidden_states.view(hidden_states.size(0), -1)  # [8, 2048]
        hidden_states = self.message_decoder(hidden_states)


        return  hidden_states

