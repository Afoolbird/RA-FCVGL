import numpy as np
import torch
import torch.nn as nn

class Aggregator(nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()
        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))   #    AdaptiveAvgPool2d   AdaptiveMaxPool2d
    
    def forward(self, x):
        # 输入 x 的形状是 [batch_size, channels, height, width]
        x = self.global_avg_pool(x)
        return x



def load_initialize_model(model):
    params = np.load('initial_model_params.npz')
    with torch.no_grad():  #module
        # query_net
        # model.query_net.encoder.vgg.conv1_1[0].weight.copy_(torch.tensor(params['VGG_grd/conv1_1/weights']).permute(3, 2, 0, 1))
        # model.query_net.encoder.vgg.conv1_1[0].bias.copy_(torch.tensor(params['VGG_grd/conv1_1/biases']))

        # model.query_net.encoder.vgg.conv1_2[0].weight.copy_(torch.tensor(params['VGG_grd/conv1_2/weights']).permute(3, 2, 0, 1))
        # model.query_net.encoder.vgg.conv1_2[0].bias.copy_(torch.tensor(params['VGG_grd/conv1_2/biases']))

        # model.query_net.encoder.vgg.conv2_1[0].weight.copy_(torch.tensor(params['VGG_grd/conv2_1/weights']).permute(3, 2, 0, 1))
        # model.query_net.encoder.vgg.conv2_1[0].bias.copy_(torch.tensor(params['VGG_grd/conv2_1/biases']))

        # model.query_net.encoder.vgg.conv2_2[0].weight.copy_(torch.tensor(params['VGG_grd/conv2_2/weights']).permute(3, 2, 0, 1))
        # model.query_net.encoder.vgg.conv2_2[0].bias.copy_(torch.tensor(params['VGG_grd/conv2_2/biases']))

        # model.query_net.encoder.vgg.conv3_1[0].weight.copy_(torch.tensor(params['VGG_grd/conv3_1/weights']).permute(3, 2, 0, 1))
        # model.query_net.encoder.vgg.conv3_1[0].bias.copy_(torch.tensor(params['VGG_grd/conv3_1/biases']))

        # model.query_net.encoder.vgg.conv3_2[0].weight.copy_(torch.tensor(params['VGG_grd/conv3_2/weights']).permute(3, 2, 0, 1))
        # model.query_net.encoder.vgg.conv3_2[0].bias.copy_(torch.tensor(params['VGG_grd/conv3_2/biases']))

        # model.query_net.encoder.vgg.conv3_3[0].weight.copy_(torch.tensor(params['VGG_grd/conv3_3/weights']).permute(3, 2, 0, 1))
        # model.query_net.encoder.vgg.conv3_3[0].bias.copy_(torch.tensor(params['VGG_grd/conv3_3/biases']))

        # model.query_net.encoder.vgg.conv4_1[0].weight.copy_(torch.tensor(params['VGG_grd/conv4_1/weights']).permute(3, 2, 0, 1))
        # model.query_net.encoder.vgg.conv4_1[0].bias.copy_(torch.tensor(params['VGG_grd/conv4_1/biases']))

        # model.query_net.encoder.vgg.conv4_2[0].weight.copy_(torch.tensor(params['VGG_grd/conv4_2/weights']).permute(3, 2, 0, 1))
        # model.query_net.encoder.vgg.conv4_2[0].bias.copy_(torch.tensor(params['VGG_grd/conv4_2/biases']))

        # model.query_net.encoder.vgg.conv4_3[0].weight.copy_(torch.tensor(params['VGG_grd/conv4_3/weights']).permute(3, 2, 0, 1))
        # model.query_net.encoder.vgg.conv4_3[0].bias.copy_(torch.tensor(params['VGG_grd/conv4_3/biases']))

        # model.query_net.encoder.vgg.conv5_1[0].weight.copy_(torch.tensor(params['VGG_grd/conv5_1/weights']).permute(3, 2, 0, 1))
        # model.query_net.encoder.vgg.conv5_1[0].bias.copy_(torch.tensor(params['VGG_grd/conv5_1/biases']))

        # model.query_net.encoder.vgg.conv5_2[0].weight.copy_(torch.tensor(params['VGG_grd/conv5_2/weights']).permute(3, 2, 0, 1))
        # model.query_net.encoder.vgg.conv5_2[0].bias.copy_(torch.tensor(params['VGG_grd/conv5_2/biases']))

        # model.query_net.encoder.vgg.conv5_3[0].weight.copy_(torch.tensor(params['VGG_grd/conv5_3/weights']).permute(3, 2, 0, 1))
        # model.query_net.encoder.vgg.conv5_3[0].bias.copy_(torch.tensor(params['VGG_grd/conv5_3/biases']))




        # reference_net
        model.reference_net.encoder.vgg.conv1_1[0].weight.copy_(torch.tensor(params['VGG_sat/conv1_1/weights']).permute(3, 2, 0, 1))
        model.reference_net.encoder.vgg.conv1_1[0].bias.copy_(torch.tensor(params['VGG_sat/conv1_1/biases']))

        model.reference_net.encoder.vgg.conv1_2[0].weight.copy_(torch.tensor(params['VGG_sat/conv1_2/weights']).permute(3, 2, 0, 1))
        model.reference_net.encoder.vgg.conv1_2[0].bias.copy_(torch.tensor(params['VGG_sat/conv1_2/biases']))

        model.reference_net.encoder.vgg.conv2_1[0].weight.copy_(torch.tensor(params['VGG_sat/conv2_1/weights']).permute(3, 2, 0, 1))
        model.reference_net.encoder.vgg.conv2_1[0].bias.copy_(torch.tensor(params['VGG_sat/conv2_1/biases']))

        model.reference_net.encoder.vgg.conv2_2[0].weight.copy_(torch.tensor(params['VGG_sat/conv2_2/weights']).permute(3, 2, 0, 1))
        model.reference_net.encoder.vgg.conv2_2[0].bias.copy_(torch.tensor(params['VGG_sat/conv2_2/biases']))

        model.reference_net.encoder.vgg.conv3_1[0].weight.copy_(torch.tensor(params['VGG_sat/conv3_1/weights']).permute(3, 2, 0, 1))
        model.reference_net.encoder.vgg.conv3_1[0].bias.copy_(torch.tensor(params['VGG_sat/conv3_1/biases']))

        model.reference_net.encoder.vgg.conv3_2[0].weight.copy_(torch.tensor(params['VGG_sat/conv3_2/weights']).permute(3, 2, 0, 1))
        model.reference_net.encoder.vgg.conv3_2[0].bias.copy_(torch.tensor(params['VGG_sat/conv3_2/biases']))

        model.reference_net.encoder.vgg.conv3_3[0].weight.copy_(torch.tensor(params['VGG_sat/conv3_3/weights']).permute(3, 2, 0, 1))
        model.reference_net.encoder.vgg.conv3_3[0].bias.copy_(torch.tensor(params['VGG_sat/conv3_3/biases']))

        model.reference_net.encoder.vgg.conv4_1[0].weight.copy_(torch.tensor(params['VGG_sat/conv4_1/weights']).permute(3, 2, 0, 1))
        model.reference_net.encoder.vgg.conv4_1[0].bias.copy_(torch.tensor(params['VGG_sat/conv4_1/biases']))

        model.reference_net.encoder.vgg.conv4_2[0].weight.copy_(torch.tensor(params['VGG_sat/conv4_2/weights']).permute(3, 2, 0, 1))
        model.reference_net.encoder.vgg.conv4_2[0].bias.copy_(torch.tensor(params['VGG_sat/conv4_2/biases']))

        model.reference_net.encoder.vgg.conv4_3[0].weight.copy_(torch.tensor(params['VGG_sat/conv4_3/weights']).permute(3, 2, 0, 1))
        model.reference_net.encoder.vgg.conv4_3[0].bias.copy_(torch.tensor(params['VGG_sat/conv4_3/biases']))

        model.reference_net.encoder.vgg.conv5_1[0].weight.copy_(torch.tensor(params['VGG_sat/conv5_1/weights']).permute(3, 2, 0, 1))
        model.reference_net.encoder.vgg.conv5_1[0].bias.copy_(torch.tensor(params['VGG_sat/conv5_1/biases']))

        model.reference_net.encoder.vgg.conv5_2[0].weight.copy_(torch.tensor(params['VGG_sat/conv5_2/weights']).permute(3, 2, 0, 1))
        model.reference_net.encoder.vgg.conv5_2[0].bias.copy_(torch.tensor(params['VGG_sat/conv5_2/biases']))

        model.reference_net.encoder.vgg.conv5_3[0].weight.copy_(torch.tensor(params['VGG_sat/conv5_3/weights']).permute(3, 2, 0, 1))
        model.reference_net.encoder.vgg.conv5_3[0].bias.copy_(torch.tensor(params['VGG_sat/conv5_3/biases']))
        
        print('Initialize_weights load finish!')
    
    return model


# def load_checkpoint_model(model, load_model_path):
#     return 