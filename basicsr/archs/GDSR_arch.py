import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from .arch_util import make_layer

NEG_INF = -1000000
T_MAX = 600*600

from torch.utils.cpp_extension import load
# Replace this with your own path. Please modify configs according to your needs.
wkv_cuda = load(name="wkv", sources=["basicsr/archs/cuda/wkv_op.cpp", "basicsr/archs/cuda/wkv_cuda.cu"],
                verbose=False, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])

class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = nn.Parameter(res_scale * torch.ones(1), requires_grad=True)
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.lrelu(self.conv1(x)))
        return identity + out * self.res_scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2,1,kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        u = x
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return u * self.sigmoid(x)

# Permuted Spatial Attention Module
class PSAM(nn.Module):  
    def __init__(self, kernel_size=7):
        super(PSAM, self).__init__()
        self.cw = SpatialAttention(kernel_size)
        self.hc = SpatialAttention(kernel_size)
        self.hw = SpatialAttention(kernel_size)

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()

        x_out = self.hw(x)

        x_out = 1 / 3 * (x_out + x_out11 + x_out21)

        return x_out
    
class GDRMLayer(torch.nn.Module):
    def __init__(self, dim_G=96, dim_D=96):
        super(GDRMLayer, self).__init__()

        self.shared_conv = nn.Conv2d(dim_G + dim_D, dim_G, 3, 1, 1)

        self.mul_conv2 = nn.Conv2d(dim_G, dim_G, 3, 1, 1)
        self.add_conv2 = nn.Conv2d(dim_G, dim_G, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, D_Feature, G_Feature):
        out = torch.cat((D_Feature, G_Feature), dim=1)
        shared_output = self.shared_conv(out)

        mul = torch.sigmoid(self.mul_conv2(self.lrelu(shared_output)))
        add = self.add_conv2(shared_output)

        out = (D_Feature + G_Feature) * mul + add
        
        return out

class layers_after(nn.Module):
    def __init__(self, num_feat, D_kernel_size = 5, G_kernel_size = 5):
        super(layers_after, self).__init__()

        self.attn1 = PSAM(D_kernel_size)
        self.attn2 = PSAM(G_kernel_size)

    def forward(self, conv_x, RWKV_x):

        conv_x = self.attn1(conv_x)
        RWKV_x = self.attn2(RWKV_x)

        return conv_x, RWKV_x

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

# Modified the original version of OmniShift (https://github.com/Yaziwel/Restore-RWKV/blob/main/model/Restore_RWKV.py), removing the reparam_5x5-related code.
class OmniShift(nn.Module):
    def __init__(self, dim, dilation=1):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False, dilation=dilation)
        self.conv3x3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False, dilation=dilation)
        self.conv5x5 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias=False, dilation=dilation) 

        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True) 
        
    def forward(self, x):

        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x) 

        out = self.alpha[0]*x + self.alpha[1]*out1x1 + self.alpha[2]*out3x3 + self.alpha[3]*out5x5
        return out

class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, key_norm=False):
        super().__init__()
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        
        self.recurrence = 2 
        
        self.OmniShift = OmniShift(n_embd)

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False) 

        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd))) 
            self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd))) 

    def jit_func(self, x, resolution):
        # Mix x with the previous timestep to produce xk, xv, xr

        h, w = resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.OmniShift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')    

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution) 
        
        for j in range(self.recurrence): 
            if j%2==0:
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v) 
            else:
                h, w = resolution 
                k = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w) 
                v = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w) 
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v) 
                k = rearrange(k, 'b (w h) c -> b (h w) c', h=h, w=w) 
                v = rearrange(v, 'b (w h) c -> b (h w) c', h=h, w=w) 
                
        x = v
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x

class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, hidden_rate=4, key_norm=False):
        super().__init__()
        self.n_embd = n_embd

        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False) 
        
        self.OmniShift = OmniShift(n_embd)

        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x, resolution):

        h, w = resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.OmniShift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')    

        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv 

        return x

class Block(nn.Module):
    def __init__(self, n_embd, hidden_rate=4, drop_path: float = 0,
                 key_norm=False):
        super().__init__()
         
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd) 

        self.att = VRWKV_SpatialMix(n_embd, key_norm=key_norm)

        self.ffn = VRWKV_ChannelMix(n_embd, hidden_rate, key_norm=key_norm)

        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

    def forward(self, x): 
        b, c, h, w = x.shape
        
        resolution = (h, w)

        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.gamma1 * self.att(self.ln1(x), resolution)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        x = rearrange(x, 'b c h w -> b (h w) c')    
        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution) 
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x

class BasicLayer(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_path=0.,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mlp_ratio=mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(Block(n_embd=dim, hidden_rate=mlp_ratio, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,))
            
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class ResidualGroup(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 mlp_ratio=4.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 resi_connection='1conv'):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        # self.input_resolution = [48, 48] # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,)

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x):
        return self.conv(self.residual_group(x)) + x

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

@ARCH_REGISTRY.register()
class GDSR(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 RCB_num=12,
                 drop_rate=0.,
                 mlp_ratio=4.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 **kwargs):
        super(GDSR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            # Please modify rgb_mean according to your needs.
            # rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio = mlp_ratio
        self.RCB_num = RCB_num
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        
        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.RGEG_layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # 4-layer
            layer = ResidualGroup(
                dim=embed_dim,
                depth=depths[i_layer],
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                resi_connection=resi_connection,
            )
            self.RGEG_layers.append(layer)

        self.RDEG_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            rconv = make_layer(ResidualBlockNoBN, self.RCB_num, num_feat=embed_dim, res_scale=1)
            layer = nn.Sequential(rconv)
            self.RDEG_layers.append(layer)

        self.layers_after = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = layers_after(embed_dim)
            self.layers_after.append(layers)

        self.GDRM_Layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            fuse = GDRMLayer(embed_dim,embed_dim)
            self.GDRM_Layers.append(fuse)

        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # -------------------------3. high-quality image reconstruction ------------------------ #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)

        else:
            # for image denoising
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_RWKV = self.pos_drop(x)

        for i in range(self.num_layers):
            x_RWKV = self.RGEG_layers[i](x_RWKV)
            x_CNN = self.RDEG_layers[i](x)
            CNN_conv, RWKV_conv = self.layers_after[i](x_CNN, x_RWKV)
            x = self.GDRM_Layers[i](CNN_conv , RWKV_conv)

        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)

            x = self.conv_after_body(self.forward_features(x)) + x

            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)

        else:
            # for image denoising
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x