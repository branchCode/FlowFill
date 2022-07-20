import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.conv2d import GatedConv2dWithActivation, GatedDeConv2dWithActivation, get_pad


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation, with_attn=False):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        if self.with_attn:
            return out, attention
        else:
            return out


class InpaintSANet(torch.nn.Module):
    """
    Inpaint generator, input should be 4*256*256,
    where 3*256*256 is the masked image, 1*256*256 for mask
    """

    def __init__(self, n_in_channel=4):
        super(InpaintSANet, self).__init__()
        cnum = 32
        self.coarse_net1 = nn.Sequential(
            GatedConv2dWithActivation(n_in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1)),
            # downsample 128
            GatedConv2dWithActivation(cnum, 2 * cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1))
        )
        self.coarse_net2 = nn.Sequential(
            # downsample to 64
            GatedConv2dWithActivation(2 * cnum, 4 * cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
        )
        self.coarse_net3 = nn.Sequential(
            # atrous convlution
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            # Self_Attn(4*cnum, 'relu'),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
        )
        self.coarse_net4 = nn.Sequential(
            # upsample
            GatedDeConv2dWithActivation(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
            # Self_Attn(2*cnum, 'relu'),
            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
        )
        self.coarse_net5 = nn.Sequential(
            GatedDeConv2dWithActivation(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum // 2, 3, 1, padding=get_pad(256, 3, 1)),
            # Self_Attn(cnum//2, 'relu'),
            GatedConv2dWithActivation(cnum // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)
        )

        self.refine_conv_net1 = nn.Sequential(
            GatedConv2dWithActivation(n_in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1)),
            # downsample
            GatedConv2dWithActivation(cnum, cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
        )
        self.refine_conv_net2 = nn.Sequential(
            # downsample
            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(2 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
        )
        self.refine_conv_net3 = nn.Sequential(
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            # Self_Attn(4*cnum, 'relu'),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),

            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
        )
        self.refine_attn = Self_Attn(4 * cnum, 'relu', with_attn=False)
        self.refine_upsample_net1 = nn.Sequential(
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedDeConv2dWithActivation(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
        )
        self.refine_upsample_net2 = nn.Sequential(
            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),
            GatedConv2dWithActivation(cnum, cnum // 2, 3, 1, padding=get_pad(256, 3, 1)),
            # Self_Attn(cnum, 'relu'),
            GatedConv2dWithActivation(cnum // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None),
        )

        self.conv_half = GatedConv2dWithActivation(3, 3, 4, 2, padding=get_pad(256, 4, 2))
        self.conv_double = GatedDeConv2dWithActivation(2, 4 * cnum, 4 * cnum, 3, 1, padding=get_pad(256, 3, 1))

    def forward(self, imgs, masks, get_steps=True):
        # Coarse
        masked_imgs = imgs
        ft = []
        x = self.coarse_net1(torch.cat([masked_imgs, masks], dim=1))
        ft.append(x)
        x = self.coarse_net2(x)
        # ft.append(x)
        x = self.coarse_net3(x)
        ft.append(self.conv_double(x))
        x = self.coarse_net4(x)
        ft.append(x)
        x = self.coarse_net5(x)
        x = torch.clamp(x, 0, 1.)
        ft.append(self.conv_half(x))

        # Refine
        masked_imgs = imgs * (1 - masks) + x * masks
        x = self.refine_conv_net1(torch.cat([masked_imgs, masks], dim=1))
        ft.append(x)
        x = self.refine_conv_net2(x)
        # ft.append(x)
        x = self.refine_conv_net3(x)
        # ft.append(x)

        x = self.refine_attn(x)
        ft.append(self.conv_double(x))

        x = self.refine_upsample_net1(x)
        ft.append(x)
        x = self.refine_upsample_net2(x)
        x = torch.clamp(x, 0, 1.)
        ft.append(self.conv_half(x))

        if get_steps:
            return torch.cat(ft, dim=1), x
        else:
            return x


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = GatedConv2dWithActivation(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = GatedConv2dWithActivation(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = GatedConv2dWithActivation(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = GatedConv2dWithActivation(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = GatedConv2dWithActivation(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class EncoderBlock(nn.Module):
    def __init__(self, n_channels, size, n_layers=3):
        super(EncoderBlock, self).__init__()

        self.first_conv = GatedConv2dWithActivation(n_channels, n_channels * 2, 4, 2, padding=get_pad(size, 4, 2))

        conv_layers = []
        for _ in range(n_layers):
            conv_layers.append(RRDB(nf=n_channels * 2, gc=32))
        self.layers = nn.Sequential(*conv_layers)

        self.attn = Self_Attn(n_channels * 2, 'relu', with_attn=False)
        self.last_conv = GatedConv2dWithActivation(n_channels * 2, n_channels * 2, 3, 1, padding=1)

    def forward(self, x):
        x = self.first_conv(x)
        return x + self.last_conv(self.attn(self.layers(x)))


class Encoder(nn.Module):
    def __init__(self, in_channels=4, size=128, n_level=4):
        super(Encoder, self).__init__()

        cnum = 32
        self.first_conv = GatedConv2dWithActivation(in_channels, cnum, 3, 1, padding=1)
        self.blocks = nn.ModuleList()
        for _ in range(n_level):
            self.blocks.append(EncoderBlock(cnum, size=size))
            size /= 2
            cnum *= 2

    def forward(self, x):
        x = self.first_conv(x)

        ft = []
        for block in self.blocks:
            x = block(x)
            ft.append(x)

        return ft


class InpaintSADirciminator(nn.Module):
    def __init__(self):
        super(InpaintSADirciminator, self).__init__()
        cnum = 32
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(4, 2*cnum, 4, 1, padding=get_pad(128, 5, 1)),
            SNConvWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 5, 2)),
            SNConvWithActivation(4*cnum, 8*cnum, 4, 2, padding=get_pad(64, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(32, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(16, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(8, 5, 2)),
            Self_Attn(8*cnum, 'relu'),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(4, 5, 2)),
        )
        self.linear = nn.Linear(8 * cnum * 2 * 2, 1)

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0), -1))
        x = self.linear(x)
        return x