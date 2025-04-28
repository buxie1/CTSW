import common
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import math
# import utility
from utility import make_coord
# from data import multiscalesrdata



def make_model(args, parent=False):
    return ArbRCAN(args)



## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        # i=0 ，1
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class SA_upsample(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False):
        super(SA_upsample, self).__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(4, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale, scale2):
        # print('x.shape:', x.shape)
        b, c, h, w = x.size()

        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, round(w * scale2), 1).unsqueeze(0).float().to(x.device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale2) - (torch.floor((coor_hr[1] + 0.5) / scale2 + 1e-3)) - 0.5

        input = torch.cat((
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale2,
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale2 * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)

        # print('sa_up.input.shape:', input.shape)

        # (2) predict filters and offsets
        embedding = self.body(input)
        # print('embedding.shape:', embedding.shape)
        ## offsets
        offset = self.offset(embedding)
        # print('offset.shape:', offset.shape)

        ## filters
        routing_weights = self.routing(embedding)
        # print("routing_weights.shape:", routing_weights.shape)
        routing_weights = routing_weights.view(self.num_experts, round(scale*h) * round(scale2*w)).transpose(0, 1)      # (h*w) * n

        weight_compress = self.weight_compress.view(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.view(1, round(scale*h), round(scale2*w), self.channels//8, self.channels)

        # print('weight_compress.shape:', weight_compress.shape)

        weight_expand = self.weight_expand.view(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(1, round(scale*h), round(scale2*w), self.channels, self.channels//8)
        # print('weight_expand.shape:', weight_expand.shape)
        # (3) grid sample & spatially varying filtering
        ## grid sample
        fea0 = grid_sample(x, offset, scale, scale2)               ## b * h * w * c * 1
        # print(fea0)
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## b * h * w * c * 1
        # shape=fea.shape
        # print('fea.shape:', fea.shape)

        ## spatially varying filtering
        # print('weight_compress.expand([b, -1, -1, -1, -1]).shape', weight_compress.expand([b, -1, -1, -1, -1]).shape)
        out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)

        out = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0


class SA_adapt(nn.Module):
    def __init__(self, channels):
        super(SA_adapt, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.adapt = SA_conv(channels, channels, 3, 1, 1)

    def forward(self, x, scale, scale2):
        mask = self.mask(x)
        # print(x.shape)
        # shape = x.shape
        # print(shape)
        # shape = mask.shape
        # print(shape)
        #quick_test 用该句
        mask = F.interpolate(mask, size=x.shape[2:], mode='bilinear')
        adapted = self.adapt(x, scale, scale2)
        # shape=adapted.shape
        # print(shape)
        return x + adapted * mask


class SA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False, num_experts=4):
        super(SA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts
        self.bias = bias

        # FC layers to generate routing weights
        self.routing = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, num_experts),
            nn.Softmax(1)
        )

        # initialize experts
        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(channels_out, channels_in, kernel_size, kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

        if bias:
            self.bias_pool = nn.Parameter(torch.Tensor(num_experts, channels_out))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_pool)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, x, scale, scale2):
        # generate routing weights
        scale = torch.ones(1, 1).to(x.device) / scale
        scale2 = torch.ones(1, 1).to(x.device) / scale2
        routing_weights = self.routing(torch.cat((scale, scale2), 1)).view(self.num_experts, 1, 1)

        # fuse experts
        fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
        fused_weight = fused_weight.view(-1, self.channels_in, self.kernel_size, self.kernel_size)

        if self.bias:
            fused_bias = torch.mm(routing_weights, self.bias_pool).view(-1)
        else:
            fused_bias = None

        # convolution
        out = F.conv2d(x, fused_weight, fused_bias, stride=self.stride, padding=self.padding)

        return out


def grid_sample(x, offset, scale, scale2):
    # generate grids
    b, _, h, w = x.size()
    grid = np.meshgrid(range(round(scale2*w)), range(round(scale*h)))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    # project into LR space
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale2 - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale - 0.5

    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    # add offsets
    offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
    offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
    grid = grid + torch.cat((offset_0, offset_1), 1)
    grid = grid.permute(0, 2, 3, 1)
    # sampling
    # output = F.grid_sample(x, grid, padding_mode='zeros')
    output = F.grid_sample(x, grid, padding_mode='zeros', align_corners=True)
    # print(output.shape)
    return output

# class MLP(nn.Module):
#     def __init__(self, in_dim, out_dim, hidden_list):
#         super().__init__()
#         layers = []
#         lastv = in_dim
#         for hidden in hidden_list:
#             layers.append(nn.Linear(lastv, hidden))
#             layers.append(nn.ReLU())
#             lastv = hidden
#         layers.append(nn.Linear(lastv, out_dim))
#         self.layers = nn.Sequential(*layers)
#
#     def forward(self, x):
#         shape = x.shape[:-1]
#         x = self.layers(x.view(-1, x.shape[-1]))
#         return x.view(*shape, -1)

class LTE(nn.Module):

    def __init__(self, channels, hidden_dim=256):
        # encoder_spec 编码器
        super().__init__()
        # self.encoder = models.make(encoder_spec)
        # 编码器之后一层卷积提取振幅
        self.coef = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        # self.coef = nn.Conv2d(64, hidden_dim, 3, padding=1)
        # 编码器之后一层卷积频率
        self.freq = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        # self.freq = nn.Conv2d(64, hidden_dim, 3, padding=1)

        # 相位 全连接层 in_features, out_features, bias
        self.phase = nn.Linear(2, hidden_dim // 2, bias=False)
        # 解码器
        # self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})

    def gen_feat(self, inp):
        self.inp = inp  # input
        # self.feat = self.encoder(inp)  # 编码得到特征图
        self.coeff = self.coef(self.inp)  # 特征图对应的傅里叶系数
        self.freqq = self.freq(self.inp)  # 特征图对应的频率
        return self.inp

    def query_rgb(self, coord, cell=None):
        feat = self.inp
        coef = self.coeff
        freq = self.freqq
        h, w = feat.shape[2:]
        # print('coef:', coef.shape)
        # print('freq:', freq.shape)
        # field radius (global: [-1, 1]) 归一化
        rx = 2 / feat.shape[-2] / 2
        # print('rx:', rx)
        ry = 2 / feat.shape[-1] / 2
        # print('ry:', ry)

        # 对应特征图的坐标
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        outputs = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                # print('coord_.flip(-1).unsqueeze(1):', coord_.flip(-1).unsqueeze(1).shape)
                # print(vx, vy)
                # print(coord_.shape)
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)  ##坐标归一化
                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)

                q_freq = F.grid_sample(
                    freq, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)

                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                # prepare cell 使用形状c渲染以坐标x为中心的像素的RGB值
                rel_cell = cell.clone()  #clone()返回一个和源张量同shape、dtype和device的张量，与源张量不共享数据内存，但提供梯度的回溯。
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                bs, q = coord.shape[:2]  # batch size&q_sample
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))  # 内积 频率与local grid
                q_freq = torch.sum(q_freq, dim=-2)  # 消灭坐标信息
                q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)  # 与全连接后的相位求和
                q_freq = torch.cat((torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1)  # 正弦激活函数

                output = torch.mul(q_coef, q_freq).contiguous().view(bs, h, w, -1).permute(0, 3, 1, 2) #幅度与频率相乘
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)
                outputs.append(output)

        tot_area = torch.stack(areas).sum(dim=0)
        # 四个面积对调
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        # print(output.shape)
        # print((area / tot_area).shape)

        ret = 0

        weight = (area / tot_area).unsqueeze(-1).view(bs, h, w, -1).permute(0, 3, 1, 2 )
        # weight.view(bs, h, w, -1 )
        # print(weight.view(bs, h, w, -1).shape)

        for output, area in zip(outputs, areas):
             ret = ret + output * weight
        # print(ret.shape)
        return ret

    def forward(self, inp, coord, cell):
        # print('inp.shape:', inp.shape)
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


##OctConv is used for frequency decomposition
class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.is_dw = groups == in_channels
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."
        self.alpha_in, self.alpha_out = alpha_in, alpha_out
        self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else \
                        nn.Conv2d(int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, math.ceil(alpha_in * groups), bias)
        self.conv_l2h = None if alpha_in == 0 or alpha_out == 1 or self.is_dw else \
                        nn.Conv2d(int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 or self.is_dw else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, math.ceil(groups - alpha_in * groups), bias)

    def forward(self, x):

        # print('input size:', x.shape)
        x_h, x_l = x if type(x) is tuple else (x, None)
        x_h = self.downsample(x_h) if self.stride == 2 else x_h
        # print(x_h.shape)
        x_h2h = self.conv_h2h(x_h)
        x_h2l = self.conv_h2l(self.downsample(x_h)) if self.alpha_out > 0 and not self.is_dw else None
        if x_l is not None:
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None
            if self.is_dw:
                return x_h2h, x_l2l
            else:
                x_l2h = self.conv_l2h(x_l)
                # print(x_l2h.shape)
                x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
                # print(x_l2h.shape)
                x_h = x_l2h + x_h2h
                x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
                # print('x_h.shape:', x_h.shape)
                # print('x_l.shape:', x_l.shape)
                return x_h, x_l
        else:
            return x_h2h, x_h2l


# class Conv_BN(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
#                  groups=1, bias=False, norm_layer=nn.BatchNorm2d):
#         super(Conv_BN, self).__init__()
#         self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
#                                groups, bias)
#         self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
#         self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))
#
#     def forward(self, x):
#         x_h, x_l = self.conv(x)
#         x_h = self.bn_h(x_h)
#         x_l = self.bn_l(x_l) if x_l is not None else None
#         return x_h, x_l
#
#
# class Conv_BN_ACT(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
#                  groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
#         super(Conv_BN_ACT, self).__init__()
#         self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
#                                groups, bias)
#         self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
#         self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))
#         self.act = activation_layer(inplace=True)
#
#     def forward(self, x):
#         x_h, x_l = self.conv(x)
#         x_h = self.act(self.bn_h(x_h))
#         x_l = self.act(self.bn_l(x_l)) if x_l is not None else None
#         return x_h, x_l
#

class ArbRCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ArbRCAN, self).__init__()

        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        reduction = 16
        hidden_dim = 256
        act = nn.ReLU(True)
        self.n_resgroups = n_resgroups
        self.hidden_dim = hidden_dim
        # self.n_feats=n_feats

        # sub_mean & add_mean layers
        if args.data_train == 'Train':
        ###theta=90#######
            # print('training：Use scale=2.0 sinogram mean (0.3450, 0.3450, 0.3450)')
            # rgb_mean = (0.3450, 0.3450, 0.3450)
            print('training：Use scale=3.0 sinogram mean (0.2560, 0.2560, 0.2560)')
            rgb_mean = (0.2560, 0.2560, 0.2560)
            # print('Training：Use scale=4.0 sinogram mean (0.3450, 0.3450, 0.3450)')
            # rgb_mean = (0.3453, 0.3453, 0.3453)
            # print('Training：Use scale=4.0 sinogram mean (0.3450, 0.3450, 0.3450)')
            # rgb_mean = (0.3450, 0.3450, 0.3450)



        # elif args.data_train == 'DIVFlickr2K':
        #     print('Use DIVFlickr2K mean (0.4690, 0.4490, 0.4036)')
        #     rgb_mean = (0.4690, 0.4490, 0.4036)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)



        # head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # body module
        modules_body = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale,
                          n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail module
        modules_tail = [
            None,                                              # placeholder to match pre-trained RCAN model
            conv(n_feats, args.n_colors, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

        ##########   our plug-in module     ##########
        # scale-aware feature adaption block
        # For RCAN, feature adaption is performed after each backbone block, i.e., K=1
        self.K = 1
        sa_adapt = []
        for i in range(self.n_resgroups // self.K):
            sa_adapt.append(SA_adapt(n_feats))

        # self.CSA = Channel_Spatial_Attention_Module()
        # self.LA = Layer_Attention_Module(n_resgroups, n_feats)
        self.sa_adapt = nn.Sequential(*sa_adapt)

        #########frequency separate##########
        self.octconv = OctaveConv(128, 128, 3)

        # # LTE layer
        self.lte = LTE(64)
        self.up_conv = common.default_conv(64, hidden_dim, 3)
        # scale-aware upsampling layer
        self.sa_upsample = SA_upsample(256)
        self.down_conv = common.default_conv(256, 64, 3)

    def set_scale(self, scale, scale2):
        self.scale = scale
        self.scale2 = scale2

    # def get_cell(self, cell):
    # self.cell = cell
    def forward(self, tensor_list):
        # y=x;
        # head
        # print('x.shape:', x.shape)
        # print(len(tensor_list))
        x = tensor_list[0]
        # print(x.max())
        # y = tensor_list[3]

        # print(x.shape)
        ####quick_test
        # x = x.unsqueeze(0)
        # print(x.shape)
        x = self.sub_mean(x)
        # print(x.min())
        x = self.head(x)
        # y = self.head(y)
        # print(x.shape)
        # print(y.shape)
        # print(x.shape)
        # print(y.shape)
        # res = torch.cat((x, y), 1)
        # res = self.head(res)
        # print(res.shape)
        # res = x+y
        res = x
        for i in range(self.n_resgroups):
            res = self.body[i](res)
            # body_results.append(res)
            # scale-aware feature adaption
            if (i+1) % self.K == 0:
                res = self.sa_adapt[i](res, self.scale, self.scale2)
                # res = self.sa_adapt[i](res, 1.5, 2.0)

        # 取最后一个res
        res = self.body[-1](res)
        # long skip connection
        res += x
        # print('res:', res.shape)
        # print('res[1, :2, :3, :3]:', res[1, :2, :3, :3])

        ########frequency separation#############
        high_frequency, low_frequency = self.octconv(res)
        # print('high_frequency.shape:', high_frequency.shape)
        # print('low_frequency.shape:', low_frequency.shape)
        high_frequency = F.interpolate(high_frequency, size=res.shape[2:], mode='bilinear')
        # high_frequency = F.interpolate(high_frequency, size=[:, :, 50, 50])
        # low_frequency = F.interpolate(low_frequency, size=(25, 25))
        # print('high_frequency_interpolation.shape:', high_frequency.shape)
        # print('low_frequency_interpolation.shape:', low_frequency.shape)
        # ########LTE#############
        coord = tensor_list[1]
        cell = tensor_list[2]
        # print(coord.shape)
        # print(cell.shape)
        # print(high_frequency.shape)
        lte_output = self.lte(high_frequency, coord, cell)
        # print('LTE_after:', lte_output.shape)
        res = self.up_conv(res)

        # print(res.shape)
        # print(lte_output.shape)
        # lte_output = self.down_conv(lte_output)
        res += lte_output
        # print('res:', res.shape)
        # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # low_frequency = nn.Upsample(low_frequency,)
        #######HAN#########
        # feature_LA = self.LA(torch.stack(body_results[1:-1], dim=1))  # b, n * c, h, w
        # feature_CSA = self.CSA(body_results[-1])  # # b, c, h, w
        # res = body_results[0]+feature_LA+feature_CSA
        # #LTE
        # res = self.LTE(res)

        # res = self.body[-1](res)+feature_LA+feature_CSA
        # scale-aware upsampling
        res = self.sa_upsample(res, self.scale, self.scale2)
        # print('res:', res.shape)
        res = self.down_conv(res)
        # tail
        # 尾部卷积一次 使得输出图像的通道数与输入相等
        x = self.tail[1](res)
        x = self.add_mean(x)

        return x

# ######baseline######
# from model import common
# import torch.nn as nn
# import torch
# import numpy as np
# import torch.nn.functional as F
# import math
# import torch
# import numpy as np
#
#
# def make_model(args, parent=False):
#     return ArbRCAN(args)
#
#
# ## Channel Attention (CA) Layer
# class CALayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(CALayer, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)  #feature map
#         y = self.conv_du(y)   #统计量s
#         return x * y
#
#
# ## Residual Channel Attention Block (RCAB)
# class RCAB(nn.Module):
#     def __init__(
#             self, conv, n_feat, kernel_size, reduction,
#             bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
#
#         super(RCAB, self).__init__()
#         modules_body = []
#         for i in range(2):
#         #range（2）即：从0到2，不包含2，即0,1
#             modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#             if bn: modules_body.append(nn.BatchNorm2d(n_feat))   #去掉了两个batch norm
#             if i == 0: modules_body.append(act)
#         modules_body.append(CALayer(n_feat, reduction))
#         self.body = nn.Sequential(*modules_body)
#         self.res_scale = res_scale
#
#     def forward(self, x):
#         res = self.body(x)
#         res += x
#         return res
#
#
# ## Residual Group (RG)
# class ResidualGroup(nn.Module):
#     def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
#         super(ResidualGroup, self).__init__()
#         modules_body = [
#             RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
#             for _ in range(n_resblocks)]
#         modules_body.append(conv(n_feat, n_feat, kernel_size))
#         self.body = nn.Sequential(*modules_body)
#
#     def forward(self, x):
#         res = self.body(x)
#         res += x
#         return res
#
#
#
# #scale-aware upsampling layer
# class SA_upsample(nn.Module):
#     def __init__(self, channels, num_experts=4, bias=False):
#         super(SA_upsample, self).__init__()
#         self.bias = bias
#         self.num_experts = num_experts
#         self.channels = channels
#
#         # experts(routing weights are used to combine the experts)
#         weight_compress = []
#         for i in range(num_experts):
#             weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))  #对应论文C/8*C*k*K   k=1
#             nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5)) #根据权重设置卷积核参数
#         self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))
#
#         weight_expand = []
#         for i in range(num_experts):
#             weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1))) ##对应论文C*C/8*k*K   k=1
#             nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
#         self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))
#        # weight_compress = []&weight_expand = []  are a pair of convolutional kernels
#
#         # two FC layers
#         self.body = nn.Sequential(
#             nn.Conv2d(4, 64, 1, 1, 0, bias=True), #(in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1,bias=True, padding_mode='zeros')
#             nn.ReLU(True),
#             nn.Conv2d(64, 64, 1, 1, 0, bias=True),
#             nn.ReLU(True),
#         )
#         # routing head
#         self.routing = nn.Sequential(
#             nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
#             nn.Sigmoid()
#         )
#         # offset head
#         self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)
#
#     def forward(self, x, scale, scale2):
#         b, c, h, w = x.size()
#
#         # (1) coordinates in LR space 低到高像素的对应坐标计算
#         ## coordinates in HR space
#         #x的计算
#         coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(x.device),   #round(number, ndigits=1)将number四舍五入到给定精度，此时精度为1
#                    torch.arange(0, round(w * scale2), 1).unsqueeze(0).float().to(x.device)]   #.unsqueeze(0)增加维度 0表示，在第一个位置增加维度
#
#         ## coordinates in LR space
#         #R(X)的计算
#         coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
#         coor_h = coor_h.permute(1, 0)        #x,y转置
#         #R(y)的计算
#         coor_w = ((coor_hr[1] + 0.5) / scale2) - (torch.floor((coor_hr[1] + 0.5) / scale2 + 1e-3)) - 0.5
#
#         input = torch.cat((
#             torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale2,   #ones_like(coor_h) 输出一张与(coor_h)大小相同的全为1的张量
#             torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale,
#             coor_h.expand([-1, round(scale2 * w)]).unsqueeze(0),
#             coor_w.expand([round(scale * h), -1]).unsqueeze(0)
#         ), 0).unsqueeze(0)
#
#
#         # (2) predict filters and offsets   input is the resulting features
#         embedding = self.body(input) #对input进行两层全连接操作
#         ## offsets
#         offset = self.offset(embedding)
#
#         ## filters
#         routing_weights = self.routing(embedding)
#         routing_weights = routing_weights.view(self.num_experts, round(scale*h) * round(scale2*w)).transpose(0, 1)      # (h*w) * n
#
#         weight_compress = self.weight_compress.view(self.num_experts, -1)
#         weight_compress = torch.matmul(routing_weights, weight_compress)   #矩阵相乘
#         weight_compress = weight_compress.view(1, round(scale*h), round(scale2*w), self.channels//8, self.channels)
#
#         weight_expand = self.weight_expand.view(self.num_experts, -1)
#         weight_expand = torch.matmul(routing_weights, weight_expand)
#         weight_expand = weight_expand.view(1, round(scale*h), round(scale2*w), self.channels, self.channels//8)
#
#         # (3) grid sample & spatially varying filtering
#         ## grid sample
#         fea0 = grid_sample(x, offset, scale, scale2)               ## b * h * w * c * 1
#         fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## b * h * w * c * 1
#
#         ## spatially varying filtering
#         out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)
#         out = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), out).squeeze(-1)
#
#         return out.permute(0, 3, 1, 2) + fea0
#
# class BaseBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(BaseBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(True)
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return self.relu(x)
#
#
# class SA_adapt(nn.Module):
#     def __init__(self, channels):
#         super(SA_adapt, self).__init__()
#         # self.conv1 = nn.Conv2d(channels, 16, 3, 1, 1)
#         self.conv1 = BaseBlock(channels, 16)
#         self.avg_pool = nn.AvgPool2d(2)
#         self.conv2 = BaseBlock(16, 16)
#         self.conv3 = BaseBlock(16, 16)
#         self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#
#         self.conv4 = BaseBlock(16, channels)
#         # self.up = F.I()
#         # self.conv1 = BaseBlock(16, 16)
#
#
#         self.mask = nn.Sequential(
#             nn.Conv2d(channels, 16, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.AvgPool2d(kernel_size=1, stride=2, padding=1, ceil_mode=True),
#
#             nn.Conv2d(16, 16, 3, 1, 1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#
#             nn.Conv2d(16, 16, 3, 1, 1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#
#             nn.Conv2d(16, 1, 3, 1, 1),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#         self.adapt = SA_conv(channels, channels, 3, 1, 1)
#
#     def forward(self, x, scale, scale2):
#         # print(x.size())
#         out = self.conv1(x)
#         out = self.avg_pool(out)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.up_sample(out)
#         mask = self.conv4(out)
#         mask = F.interpolate(mask, size=x.shape[2:], mode="nearest")
#         # print(mask.size())
#         # mask = self.mask(x)
#         adapted = self.adapt(x, scale, scale2)
#         # print(adapted.size())
#         return x + adapted * mask
#
#
# # class SA_adapt(nn.Module):
# #     def __init__(self, channels):
# #         super(SA_adapt, self).__init__()
# #         self.mask = nn.Sequential(
# #             nn.Conv2d(channels, 16, 3, 1, 1),#(in_channels, out_channels, kernel_size, stride=1,padding=1）
# #             nn.BatchNorm2d(16),  #Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，一方面使得数据分布一致，另一方面避免梯度消失。 16是channel数，即特征数
# #             nn.ReLU(True),
# #             nn.AvgPool2d(2), #平均池化，kernel_size=2
# #             nn.Conv2d(16, 16, 3, 1, 1),
# #             nn.BatchNorm2d(16),
# #             nn.ReLU(True),
# #             nn.Conv2d(16, 16, 3, 1, 1),
# #             nn.BatchNorm2d(16),
# #             nn.ReLU(True),
# #             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
# #             #使用线性插值，将图片上采样为原图像大小
# #             # align_corners=False 的情况会对边角不友好。对于目标检测，鉴于少有物体中心出现在边角，所以影响不大；而 False 带来的整数倍上下采样，又方便了坐标值的计算。
# #             nn.Conv2d(16, 1, 3, 1, 1),
# #             nn.BatchNorm2d(1),
# #             nn.Sigmoid()
# #         )
# #         self.adapt = SA_conv(channels, channels, 3, 1, 1)  #F is fed to a scale-aware convolution for feature adaption, resulting in an adapted feature map F（adapt）
# #
# #     def forward(self, x, scale, scale2):
# #         mask = self.mask(x)
# #         adapted = self.adapt(x, scale, scale2)
# #
# #         return x + adapted * mask   #F（fuse）= F + F（adapt）×M
# #
# #
#
# #scale-aware convolution  Figure 4
# class SA_conv(nn.Module):
#     def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False, num_experts=4):
#         super(SA_conv, self).__init__()
#         self.channels_out = channels_out
#         self.channels_in = channels_in
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.num_experts = num_experts
#         self.bias = bias
#
#         # FC layers to generate routing weights
#         self.routing = nn.Sequential(
#             nn.Linear(2, 64),
#             #nn.Linear（）是用于设置网络中的全连接层的，需要注意在二维图像处理的任务中，全连接层的输入与输出一般都设置为二维张量，形状通常为[batch_size, size]
#             #in_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size
#             # out_features指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]，当然，它也代表了该全连接层的神经元个数。
#             #从输入输出的张量的shape角度来理解，相当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量。
#             nn.ReLU(True),
#             nn.Linear(64, num_experts),
#             nn.Softmax(1)
#         )
#
#         # initialize experts
#         weight_pool = []
#         for i in range(num_experts):
#             weight_pool.append(nn.Parameter(torch.Tensor(channels_out, channels_in, kernel_size, kernel_size)))
#             nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))    #pytorch默认使用kaiming正态分布初始化卷积层参数
#         self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))    #叠加weight_pool第一个维度
#
#         if bias:
#             self.bias_pool = nn.Parameter(torch.Tensor(num_experts, channels_out))
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_pool)   #fan_in输入神经元个数，fan_out输出神经元个数
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.bias_pool, -bound, bound)
#
#     def forward(self, x, scale, scale2):
#         # generate routing weights
#         scale = torch.ones(1, 1).to(x.device) / scale
#         scale2 = torch.ones(1, 1).to(x.device) / scale2
#         routing_weights = self.routing(torch.cat((scale, scale2), 1)).view(self.num_experts, 1, 1)
#
#         # fuse experts
#         fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
#         fused_weight = fused_weight.view(-1, self.channels_in, self.kernel_size, self.kernel_size)
#
#         if self.bias:
#             fused_bias = torch.mm(routing_weights, self.bias_pool).view(-1)
#         else:
#             fused_bias = None
#
#         # convolution
#         out = F.conv2d(x, fused_weight, fused_bias, stride=self.stride, padding=self.padding)
#
#         return out
#
#
# def grid_sample(x, offset, scale, scale2):
#     # generate grids
#     b, _, h, w = x.size()
#     grid = np.meshgrid(range(round(scale2*w)), range(round(scale*h)))
#     grid = np.stack(grid, axis=-1).astype(np.float64)
#     grid = torch.Tensor(grid).to(x.device)
#
#     # project into LR space
#     grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale2 - 0.5
#     grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale - 0.5
#
#     # normalize to [-1, 1]
#     grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
#     grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
#     grid = grid.permute(2, 0, 1).unsqueeze(0)
#     grid = grid.expand([b, -1, -1, -1])
#
#     # add offsets
#     offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
#     offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
#     grid = grid + torch.cat((offset_0, offset_1),1)
#     grid = grid.permute(0, 2, 3, 1)
#
#     # sampling
#     output = F.grid_sample(x, grid, padding_mode='zeros')
#
#     return output
#
#
# class ArbRCAN(nn.Module):
#     def __init__(self, args, conv=common.default_conv):
#         # global rgb_mean
#         super(ArbRCAN, self).__init__()
#
#         n_resgroups = 10
#         n_resblocks = 20
#         n_feats = 64
#         kernel_size = 3
#         reduction = 16
#         act = nn.ReLU(True)
#         self.n_resgroups = n_resgroups
#
#         # sub_mean & add_mean layers
#         # if args.data_train == 'SIN':
#         print('Use sinogram mean (0.2565, 0.2565, 0.2565)')
#         rgb_mean = (0.2565, 0.2565, 0.2565)
#         # elif args.data_train == 'DIVFlickr2K':
#         #     print('Use DIVFlickr2K mean (0.4690, 0.4490, 0.4036)')
#         #     rgb_mean = (0.4690, 0.4490, 0.4036)
#         rgb_std = (1.0, 1.0, 1.0)
#         self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
#         self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
#
#
#         # head module
#         modules_head = [conv(args.n_colors, n_feats, kernel_size)]
#         self.head = nn.Sequential(*modules_head)
#
#         # body module
#         modules_body = [
#             ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale,
#                           n_resblocks=n_resblocks) \
#             for _ in range(n_resgroups)]
#         modules_body.append(conv(n_feats, n_feats, kernel_size))
#         self.body = nn.Sequential(*modules_body)
#
#         # tail module
#         modules_tail = [
#             None,                                              # placeholder(占位符) to match pre-trained RCAN model
#             conv(n_feats, args.n_colors, kernel_size)]
#         self.tail = nn.Sequential(*modules_tail)
#
#         ##########   our plug-in module     ##########
#         # scale-aware feature adaption block
#         # For RCAN, feature adaption is performed after each backbone block, i.e., K=1
#         self.K = 1
#         sa_adapt = []
#         for i in range(self.n_resgroups // self.K):
#             sa_adapt.append(SA_adapt(64))
#         self.sa_adapt = nn.Sequential(*sa_adapt)
#
#         # scale-aware upsampling layer
#         self.sa_upsample = SA_upsample(64)
#
#     def set_scale(self, scale, scale2):
#         self.scale = scale
#         self.scale2 = scale2
#
#     def forward(self, x):
#         # head
#         # x = self.sub_mean(x)
#         # x = torch.Tensor(x)
#         x = self.sub_mean(x)
#         x = self.head(x)
#
#         # body
#         res = x
#         for i in range(self.n_resgroups):
#             res = self.body[i](res)
#             # scale-aware feature adaption
#             if (i+1) % self.K == 0:
#                 res = self.sa_adapt[i](res, self.scale, self.scale2)
#
#         res = self.body[-1](res)
#         res += x
#
#         # scale-aware upsampling
#         res = self.sa_upsample(res, self.scale, self.scale2)
#
#         # tail
#         x = self.tail[1](res)
#         x = self.add_mean(x)
#
#         return x
