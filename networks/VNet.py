import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling = 1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res
    
    
class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization, mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization, mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization, mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization, mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        
        return out_seg
class DecoderDiceCE(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(DecoderDiceCE, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization, mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization, mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization, mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization, mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        
        return out_seg,x8_up 
class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(VNet, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        return out_seg1

class MCNet3d_v1(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(MCNet3d_v1, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters,normalization,  has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        return out_seg1, out_seg2
    
class MCNet3d_v2(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(MCNet3d_v2, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.decoder3 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 2)
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        out_seg3 = self.decoder3(features)
        return out_seg1, out_seg2, out_seg3
class ECNet3d(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(ECNet3d, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)

    def forward(self, input):
        B = input.size(0)
        B = torch.tensor(B / 2, dtype=torch.int)
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        evidence = F.softplus(out_seg2)
        alpha = evidence+1
        return out_seg1, evidence, alpha
class DiceCENet3d(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(DiceCENet3d, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)

    def forward(self, input):
        B = input.size(0)
        B = torch.tensor(B / 2, dtype=torch.int)
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        evidence1 = F.softplus(out_seg1)
        alpha1 = evidence1+1
        evidence2 = F.softplus(out_seg2)
        alpha2 = evidence2 + 1
        return  alpha1,alpha2
def DS_Combin_two(alpha1, alpha2,class_num):
        """
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1,alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            #print(b[v].shape)
            u[v] = class_num / S[v]#B*C*1

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, class_num, 1), b[1].view(-1, 1, class_num ))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[1].shape) #B*C*1
        #print(uv1_expand.shape,'uv1')
        bu = torch.mul(b[0], uv1_expand)#B*C*1

        # b^1 * u^0
        uv2_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv2_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)#B
        #print(bb.shape, 'bb_sum',torch.diagonal(bb, dim1=-2, dim2=-1))
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1,1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1,1).expand(u[0].shape))

        # calculate new S
        S_a = class_num / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a
class DiceCENet3d_fuse(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(DiceCENet3d_fuse, self).__init__()
        convBlock = ConvBlock if not has_residual else ResidualConvBlock
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = DecoderDiceCE(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = DecoderDiceCE(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.1, inplace=False)  
    def forward(self, input):
        B = input.size(0)
        B = torch.tensor(B / 2, dtype=torch.int)
        features = self.encoder(input)
        out_seg1,x_up1 = self.decoder1(features)
        out_seg2,x_up2 = self.decoder2(features)
        evidence1 = F.softplus(out_seg1)
        alpha1 = evidence1+1
        evidence2 = F.softplus(out_seg2)
        alpha2 = evidence2 + 1
        prob2 = alpha2/torch.sum(alpha2,dim=1,keepdim=True)
        resize_alpha1 = alpha1.view(alpha1.size(0), alpha1.size(1), -1)  # [N, C, HW]
        resize_alpha1 = resize_alpha1.transpose(1, 2)  # [N, HW, C]
        resize_alpha1 = resize_alpha1.contiguous().view(-1, resize_alpha1.size(2))
        resize_alpha2 = alpha2.view(alpha2.size(0), alpha2.size(1), -1)  # [N, C, HW]
        resize_alpha2 = resize_alpha2.transpose(1, 2)  # [N, HW, C]
        resize_alpha2 = resize_alpha2.contiguous().view(-1, resize_alpha2.size(2))
        fuse_out_sup = DS_Combin_two( resize_alpha1, resize_alpha2,2)
        fuse_out_sup = fuse_out_sup/torch.sum(fuse_out_sup,dim=1,keepdim=True)
        fuse_out =self.out_conv(self.dropout(self.block_nine(x_up1+x_up2))) 
        fuse_out = F.softplus(fuse_out)
        #fuse_out = fuse_out.view(fuse_out.size(0), fuse_out.size(1), -1)  # [N, C, HW]
        #fuse_out = fuse_out.transpose(1, 2)  # [N, HW, C]
        #fuse_out = fuse_out.contiguous().view(-1, fuse_out.size(2))
        fuse_out = fuse_out/torch.sum(fuse_out,dim=1,keepdim=True)
        return  alpha1,alpha2,prob2,fuse_out,fuse_out_sup
class DiceCENet3d_fuse_2(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(DiceCENet3d_fuse_2, self).__init__()
        convBlock = ConvBlock if not has_residual else ResidualConvBlock
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = DecoderDiceCE(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = DecoderDiceCE(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder3 = DecoderDiceCE(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.1, inplace=False)  
    def forward(self, input):
        B = input.size(0)
        B = torch.tensor(B / 2, dtype=torch.int)
        features = self.encoder(input)
        out_seg1,x_up1 = self.decoder1(features)
        out_seg2,x_up2 = self.decoder2(features)
        out_seg3,x_up3 = self.decoder3(features)
        evidence1 = F.softplus(out_seg1)
        alpha1 = evidence1+1
        evidence2 = F.softplus(out_seg2)
        alpha2 = evidence2 + 1
        evidence3 = F.softplus(out_seg3)
        alpha3 = evidence3 + 1
        prob2 = alpha2/torch.sum(alpha2,dim=1,keepdim=True)
        resize_alpha1 = alpha1.view(alpha1.size(0), alpha1.size(1), -1)  # [N, C, HW]
        resize_alpha1 = resize_alpha1.transpose(1, 2)  # [N, HW, C]
        resize_alpha1 = resize_alpha1.contiguous().view(-1, resize_alpha1.size(2))
        resize_alpha2 = alpha2.view(alpha2.size(0), alpha2.size(1), -1)  # [N, C, HW]
        resize_alpha2 = resize_alpha2.transpose(1, 2)  # [N, HW, C]
        resize_alpha2 = resize_alpha2.contiguous().view(-1, resize_alpha2.size(2))
        fuse_out_sup = DS_Combin_two( resize_alpha1, resize_alpha2,2)
        fuse_out_sup = fuse_out_sup/torch.sum(fuse_out_sup,dim=1,keepdim=True)
        #fuse_out =self.out_conv(self.dropout(self.block_nine(x_up1+x_up2))) 
        #fuse_out = F.softplus(fuse_out)
        #fuse_out = fuse_out.view(fuse_out.size(0), fuse_out.size(1), -1)  # [N, C, HW]
        #fuse_out = fuse_out.transpose(1, 2)  # [N, HW, C]
        #fuse_out = fuse_out.contiguous().view(-1, fuse_out.size(2))
        fuse_out = alpha3/torch.sum(alpha3,dim=1,keepdim=True)
        return  alpha1,alpha2,prob2,fuse_out,fuse_out_sup,alpha3
if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from ptflops import get_model_complexity_info
    model = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False)
    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model, (1, 112, 112, 80), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model, (1, 96, 96, 96), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    import ipdb; ipdb.set_trace()
