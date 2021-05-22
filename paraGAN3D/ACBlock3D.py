import torch.nn as nn
import torch.nn.init as init


class CropLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[1]
        self.cols_to_crop = - crop_set[2]
        self.heis_to_crop = -crop_set[0]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0
        assert self.heis_to_crop >= 0

    def forward(self, input):
        if self.rows_to_crop == 0 and self.cols_to_crop == 0 and self.heis_to_crop==0:
            return input
        elif self.rows_to_crop > 0 and self.cols_to_crop == 0 and self.heis_to_crop==0:
            return input[:, :, :, self.rows_to_crop:-self.rows_to_crop, :]
        elif self.rows_to_crop == 0 and self.cols_to_crop > 0 and self.heis_to_crop==0:
            return input[:, :, :, :, self.cols_to_crop:-self.cols_to_crop]
        elif self.rows_to_crop == 0 and self.cols_to_crop == 0 and self.heis_to_crop>0:
            return input[:, :, self.heis_to_crop:-self.heis_to_crop,: , :]
        else:
            return input[:, :, self.heis_to_crop:-self.heis_to_crop, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


class ACBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', deploy=False,
                 use_affine=True, reduce_gamma=False, use_last_bn=False, gamma_init=None):
        super(ACBlock3D, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size,kernel_size), stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=True,
                                        padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size,kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm3d(num_features=out_channels, affine=use_affine)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border,center_offset_from_origin_border+1)
            hor_pad_or_crop = (center_offset_from_origin_border+1, center_offset_from_origin_border + 1,center_offset_from_origin_border)
            hei_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border+1 ,center_offset_from_origin_border+1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
                self.hei_conv_padding = nn.Identity()
                hei_conv_padding = hei_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0, 0)
                self.hei_conv_crop_layer = CropLayer(crop_set=hei_pad_or_crop)
                hei_conv_padding = (0, 0, 0)
            self.ver_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,1, 3),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3, 1),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hei_conv = nn.Conv3d(in_channels=in_channels,out_channels = out_channels,kernel_size=(1, 3, 3),
                                      stride=stride,
                                      padding=hei_conv_padding,dilation=dilation,groups=groups,bias=False,
                                      padding_mode = padding_mode)

            self.ver_bn = nn.BatchNorm3d(num_features=out_channels, affine=use_affine)
            self.hor_bn = nn.BatchNorm3d(num_features=out_channels, affine=use_affine)
            self.hei_bn = nn.BatchNorm3d(num_features=out_channels, affine=use_affine)
            if reduce_gamma:
                assert not use_last_bn
                init.constant_(self.square_bn.weight, 1.0 / 3)
                init.constant_(self.ver_bn.weight, 1.0 / 3)
                init.constant_(self.hor_bn.weight, 1.0 / 3)
                init.constant_(self.hei_bn.weight,1.0/3)

            if use_last_bn:
                assert not reduce_gamma
                self.last_bn = nn.BatchNorm3d(num_features=out_channels, affine=True)

            if gamma_init is not None:
                assert not reduce_gamma
                init.constant_(self.square_bn.weight, gamma_init)
                init.constant_(self.ver_bn.weight, gamma_init)
                init.constant_(self.hor_bn.weight, gamma_init)
                init.constant_(self.hei_bn.weight, gamma_init)

    def init_gamma(self, gamma_value):
        init.constant_(self.square_bn.weight, gamma_value)
        init.constant_(self.ver_bn.weight, gamma_value)
        init.constant_(self.hor_bn.weight, gamma_value)
        init.constant_(self.hei_bn.weight, gamma_value)
        print('init gamma of square, ver and hor as ', gamma_value)

    def single_init(self):
        init.constant_(self.square_bn.weight, 1.0)
        init.constant_(self.ver_bn.weight, 0.0)
        init.constant_(self.hor_bn.weight, 0.0)
        print('init gamma of square as 1, ver and hor as 0')

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            # square_outputs = self.square_bn(square_outputs)

            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            # vertical_outputs = self.ver_bn(vertical_outputs)
            # print(vertical_outputs.size())
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            # horizontal_outputs = self.hor_bn(horizontal_outputs)

            height_outputs = self.hei_conv_crop_layer(input)
            height_outputs = self.hei_conv(height_outputs)
            # height_outputs = self.hei_bn(height_outputs)

            # print(horizontal_outputs.size())
            result = square_outputs + vertical_outputs + horizontal_outputs+ height_outputs
            if hasattr(self, 'last_bn'):
                return self.last_bn(result)
            return result

