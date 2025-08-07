import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import numbers


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):   #Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):     #与Attention代码一样
    def __init__(self, dim, num_heads=8, qkv_bias=False):  #64,8
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads   #8
        head_dim = dim // num_heads   #64//8=8
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))   #将一个不可训练的tensor转换成可以训练的类型parameter,并将这个parameter绑定到这个module里面,torch.ones(8, 1, 1)
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)   #conv2(64,192,1)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)  #conv(192,192,3,1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)   #conv(64,64,1)

    def forward(self, x):    #x.shape: torch.Size([8, 64, 128, 128])
        b, c, h, w = x.shape        # [batch_size, num_patches + 1, total_embed_dim]
        qkv = self.qkv2(self.qkv1(x))    #qkv.shape: torch.Size([8, 192, 128, 128])
        q, k, v = qkv.chunk(3, dim=1)    #torch.chunk(input, chunks, dim = 0) 函数会将输入张量（input）沿着指定维度（dim）均匀的分割成特定数量的张量块（chunks），并返回元素为张量块的元组。torch.Size([8, 64, 128, 128])
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)   #rearrange按给出的模式(注释)重组张量，其中模式中字母只是个表示，没有具体含义q.size: torch.Size([8, 8, 8, 16384])
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)    #q.size: torch.Size([8, 8, 8, 16384])
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)    #q.size: torch.Size([8, 8, 8, 16384])
        q = torch.nn.functional.normalize(q, dim=-1)      #q.shape: torch.Size([8, 8, 8, 16384])
        k = torch.nn.functional.normalize(k, dim=-1)      #k.shape: torch.Size([8, 8, 8, 16384])
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1],  @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale      #attn.shape: torch.Size([8, 8, 8, 8])
        attn = attn.softmax(dim=-1)    #attn.shape: torch.Size([8, 8, 8, 8])
        out = (attn @ v)               #out.shape: torch.Size([8, 8, 8, 16384])
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)     #out.shape: torch.Size([8, 64, 128, 128])
        out = self.proj(out)     #out.shape: torch.Size([8, 64, 128, 128])
        return out


class Mlp(nn.Module):   #与FeedForward代码相同  MLP as used in Vision Transformer, MLP-Mixer and related networks
    def __init__(self, in_features, hidden_features=None, ffn_expansion_factor=2, bias=False):   #64,1
        super(Mlp, self).__init__()
        hidden_features = int(in_features * ffn_expansion_factor)    #64*2
        self.project_in = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1, bias=bias)    #Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)    #Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        self.project_out = nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=bias)    #Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)  #原来的

    def forward(self, x):         #x.shape: torch.Size([8, 64, 128, 128])
        x = self.project_in(x)    #x.shape: torch.Size([8, 128, 128, 128])
        x1, x2 = self.dwconv(x).chunk(2, dim=1)       #x1.shape: torch.Size([8, 64, 128, 128]), x2.shape: torch.Size([8, 64, 128, 128])
        x = F.gelu(x1) * x2          #x.shape: torch.Size([8, 64, 128, 128])
        x = self.project_out(x)     #x.shape: torch.Size([8, 64, 128, 128])
        return x


class BaseFeatureExtraction(nn.Module):      #BaseFeatureExtraction(dim=64, num_heads=8)   自己进行了修改
    def __init__(self, dim, num_heads, ffn_expansion_factor=1., qkv_bias=False):     # dim=64,num_heads=8
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')       #层归一化,transform
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias)    #out.shape: torch.Size([8, 64, 128, 128])
        #self.attn = nn.Sequential(*[AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias) for i in range(2)])
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim, ffn_expansion_factor=ffn_expansion_factor)    #x.shape: torch.Size([8, 64, 128, 128])
        self.out = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))   #自己加的

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        #x = x + self.mlp(self.norm2(x))        # 原来的  x.shape: torch.Size([8, 64, 128, 128])
        x = self.out(x + self.mlp(self.norm2(x)))   #自己加的
        return x


class InvertedResidualBlock(nn.Module):   #原来的都没nn.BatchNorm2d
    def __init__(self, inp, oup, expand_ratio):     #InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)    #32*2=64
        self.bottleneckBlock = nn.Sequential(nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False),    #pw   Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            #nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),         #ReLU6(x)=min(max(0,x),6)
            nn.ReflectionPad2d(1),          #dw   对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。多2行2列
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim, bias=False),     #Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), groups=64, bias=False)
            #nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False))    #pw-linear    Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            #nn.BatchNorm2d(oup))

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self, dim):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=dim//2, oup=dim//2, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=dim//2, oup=dim//2, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=dim//2, oup=dim//2, expand_ratio=2)
        self.shffleconv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, dim, num_layers):     #原来这里是3
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode(dim) for _ in range(num_layers)]   #DetailNode()循环3次,本实验只循环1次  num_layers=1
        self.net = nn.Sequential(*INNmodules)
        #self.out = BBasicConv2d(dim, dim, 3, 1, 1)   #自己加进去的

    def forward(self, x):           #x.shape: torch.Size([8, 64, 128, 128])
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]    #z1/z2.shape: torch.Size([8, 32, 128, 128])
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        out = torch.cat((z1, z2), dim=1)
        return out              #torch.Size([8, 64, 128, 128])


class DetailFusion(nn.Module):
    def __init__(self, dim):
        super(DetailFusion, self).__init__()
        self.theta_phi = InvertedResidualBlock(inp=dim, oup=dim, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=dim, oup=dim, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=dim, oup=dim, expand_ratio=2)
        self.shffleconv = nn.Conv2d(2*dim, 2*dim, kernel_size=1, stride=1, padding=0, bias=True)    #Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.out = BBasicConv2d(2*dim, dim, 3, 1, 1)
        
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    
    def forward(self, z1, z2):   #z1.shape: torch.Size([8, 64, 128, 128])
        z1, z2 = self.separateFeature(self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)   #z2.shape: torch.Size([8, 64, 128, 128])
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)   #z1.shape: torch.Size([8, 64, 128, 128])
        z = self.out(torch.cat((z1, z2), dim=1))
        return z


class BBasicConv2d(nn.Module):    #CBR
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BBasicConv2d, self).__init__()
        self.basicconv = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
                                       nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.basicconv(x)


# Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):        #WithBias_LayerNorm(64)
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):      #判断normalized_shape是numbers.Integral类型，此类对象表示数学中整数集合的成员(包括正数和负数)，
            normalized_shape = (normalized_shape,)       #normalized_shape:(64,)
        normalized_shape = torch.Size(normalized_shape)     #normalized_shape: torch.Size([64])
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):       #x.shape: torch.Size([8, 16384, 64])
        mu = x.mean(-1, keepdim=True)       #mu.shape: torch.Size([8, 16384, 1])   128*128=16384
        sigma = x.var(-1, keepdim=True, unbiased=False)      #sigma.shape: torch.Size([8, 16384, 1])计算最后一个维度的方差，unbiased=False使用无偏估计
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):      #LayerNorm(64, LayerNorm_type='WithBias')
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)   #WithBias_LayerNorm(64)

    def forward(self, x):           #x.shape: torch.Size([8, 64, 128, 128])
        h, w = x.shape[-2:]          #h=128, w=128
        return to_4d(self.body(to_3d(x)), h, w)    #torch.Size([8, 64, 128, 128])   ,self.body(to_3d(x)) torch.Size([8, 16384, 64])


## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):      #与Mlp代码一样
    def __init__(self, dim, ffn_expansion_factor, bias):      #FeedForward(64, 2, bias=False)
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)     #128
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)     #Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)    #Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x):      #x.shape: torch.Size([8, 64, 128, 128])
        x = self.project_in(x)      #x.shape: torch.Size([8, 256, 128, 128])
        x1, x2 = self.dwconv(x).chunk(2, dim=1)     #x1.shape: torch.Size([8, 128, 128, 128])
        x = F.gelu(x1) * x2     #GELU ：高斯误差线性单元激活函数  x.shape torch.Size([8, 128, 128, 128])
        x = self.project_out(x)   #x.shape: torch.Size([8, 64, 128, 128])
        return x


## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):     #与AttentionBase代码一样
    def __init__(self, dim, num_heads, bias):    #Attention(64, 8, bias=False)
        super(Attention, self).__init__()
        self.num_heads = num_heads      #8
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))     #torch.ones(8, 1, 1)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)      #Conv2d(64, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)   #Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)     #Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x):     #x.shape: torch.Size([8, 64, 128, 128])
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))      #torch.Size([8, 192, 128, 128])
        q, k, v = qkv.chunk(3, dim=1)           #平均分成3分   torch.Size([8, 64, 128, 128])
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)       #q.shape torch.Size([8, 8, 8, 16384])
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)       #k.shape torch.Size([8, 8, 8, 16384])
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)       #v.shape torch.Size([8, 8, 8, 16384])
        q = torch.nn.functional.normalize(q, dim=-1)     #将某一个维度除以那个维度对应的范数(默认是2范数)。对行进行操作 q.shape torch.Size([8, 8, 8, 16384])
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature        #@表示常规的数学上定义的矩阵相乘；attn.shape: torch.Size([8, 8, 8, 8])
        attn = attn.softmax(dim=-1)
        out = (attn @ v)      #out.shape: torch.Size([8, 8, 8, 16384])
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)   #out.shape: torch.Size([8, 64, 128, 128])
        out = self.project_out(out)    #out.shape: torch.Size([8, 64, 128, 128])
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):     #TransformerBlock(64,8,2,bias=False, LayerNorm_type='WithBias')
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)   #LayerNorm(64, LayerNorm_type='WithBias')
        self.attn = Attention(dim, num_heads, bias)    #Attention(64, 8, bias=False) torch.Size([8, 64, 128, 128])
        self.norm2 = LayerNorm(dim, LayerNorm_type)     #torch.Size([8, 64, 128, 128])
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)     #前馈FeedForward(64, 2, bias=False),  torch.Size([8, 64, 128, 128])

    def forward(self, x):     #x.shape: torch.Size([8, 64, 128, 128])
        x = x + self.attn(self.norm1(x))     #x.shape: torch.Size([8, 64, 128, 128])
        x = x + self.ffn(self.norm2(x))    #x.shape: torch.Size([8, 64, 128, 128])
        return x


class OverlapPatchEmbed(nn.Module):     # Overlapped image patch embedding with 3x3 Conv
    def __init__(self, in_c=3, embed_dim=48, bias=False):   #OverlapPatchEmbed(1, 64)
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)    #Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x):
        x = self.proj(x)
        return x


class Restormer_Decoder(nn.Module):
    def __init__(self, dim, out_channels):
        super(Restormer_Decoder, self).__init__()
        self.encoder_level = TransformerBlock(dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')          #循环1次
        #self.encoder_level = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias') for i in range(2)])  # 循环2次
        self.output = nn.Sequential(nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3, stride=1, padding=1, bias=False), nn.LeakyReLU(), nn.Conv2d(int(dim) // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, featuer):
        out_enc_level0 = self.encoder_level(featuer)   #torch.Size([8, 32, 256, 256])
        out_enc_level1 = self.output(out_enc_level0)
        #out = self.sigmoid(out_enc_level1)
        #if inp_img is not None:
            #out_enc_level1 = self.output(out_enc_level0) + inp_img    #out_enc_level1.shape torch.Size([8, 1, 128, 128])
        #else:
           #out_enc_level1 = self.output(out_enc_level0)
        #out = (out_enc_level1 + 1) / 2
        return self.sigmoid(out_enc_level1)


if __name__ == '__main__':
    height = 128
    width = 128
    window_size = 8
    modelD = Restormer_Decoder(out_channels=1).cuda()