import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # deconv2 and conv2, height, width: 20*20
        self.deconv2 = self.deconv_layer(769, 384)
        self.conv2_1 = self.conv_layer(384 + 384, 384)
        self.conv2_2 = self.conv_layer(384, 384)
        
        # deconv3 and conv3, height, width: 40*40
        self.deconv3 = self.deconv_layer(384, 192)
        self.conv3_1 = self.conv_layer(192 + 192, 192)
        self.conv3_2 = self.conv_layer(192, 192)
        
        # deconv4 and conv4, height, width: 80*80
        self.deconv4 = self.deconv_layer(192, 96)
        self.conv4_1 = self.conv_layer(96 + 96, 96)
        self.conv4_2 = self.conv_layer(96, 96)
        
        # deconv5 and conv5, height, width: 160*160
        self.deconv5 = self.deconv_layer(96, 48)
        self.conv5_1 = self.conv_layer(48, 48)
        self.conv5_2 = self.conv_layer(48, 48)
        
        # deconv6 and conv6, height, width: 320*320
        self.deconv6 = self.deconv_layer(48, 24)
        self.conv6_1 = self.conv_layer(24 + 3, 9)
        self.conv6_2 = self.conv_layer(9, 9)

        # deconv7 and conv7, height, width: 640*640
        self.deconv7 = self.deconv_layer(9, 3)
        self.conv7_1 = self.conv_layer(3, 3)
        self.conv7_2 = self.conv_layer(3, 1, activated=False)
        

    def conv_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activated=True):
        if activated:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU(inplace=True)
        )
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        return layer

    def deconv_layer(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, activated=True):
        if activated:
            layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
                nn.ReLU(inplace=True)
        )
        else:
            layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding))
        return layer

    def forward(self, x, sat320, sat80, sat40, sat20):
        x = self.deconv2(x)
        x = self.conv2_1(torch.cat([x, sat20], dim=1))
        x = self.conv2_2(x)
        
        # deconv3 and conv3, height, width: 40*40
        x = self.deconv3(x)
        x = self.conv3_1(torch.cat([x, sat40], dim=1))
        x = self.conv3_2(x)
        
        # deconv4 and conv4, height, width: 80*80
        x = self.deconv4(x)
        x = self.conv4_1(torch.cat([x, sat80], dim=1))
        x = self.conv4_2(x)
        # deconv5 and conv5, height, width: 160*160
        x = self.deconv5(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        
        # deconv6 and conv6, height, width: 320*320
        x = self.deconv6(x)
        x = self.conv6_1(torch.cat([x, sat320], dim=1))
        x = self.conv6_2(x)

        # deconv7 and conv7, height, width: 640*640
        x = self.deconv7(x) 
        x = self.conv7_1(x) 
        x = self.conv7_2(x) 

        return x