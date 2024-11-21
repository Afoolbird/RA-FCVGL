import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

from model.Decoder_output_conxt import Decoder
from model.Network_TransMEF_conxt import convnext_tiny
from model.Utils import Aggregator
from timm.models.registry import register_model

# from Decoder import Decoder
# from Utils import Aggregator
# from Network_TransMEF import TransNet
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# from .Deit import deit_small_distilled_patch16_224

def l2norm(x):
    """L2-normalize columns of x"""
    norm = torch.pow(x, 2).sum(dim=1, keepdim=True).sqrt()
    return torch.div(x, norm)

def cosine_sim(x, y):
#   print('cosine_sim')
  """Cosine similarity between all the image and sentence pairs. Assumes x and y are l2 normalized"""
  return x.mm(y.t())

def elu_plus(x):
    # return torch.nn.ELU(x)+1.0 
    return F.elu(x)+1.0

def get_uncertainty(L):
    L=elu_plus(L)
    zero_column = torch.zeros((L.shape[0], 1)).cuda()
    tensor_b = torch.cat([L[:, :1], zero_column, L[:, 1:]], dim=1)
    L_mat = tensor_b.view(-1, 2, 2)

    Sigma = torch.matmul(L_mat, L_mat.transpose(1, 2))
    det_Sigma = Sigma[:, 0, 0] * Sigma[:, 1, 1] - Sigma[:, 0, 1] * Sigma[:, 1, 0]

    result = torch.sqrt(det_Sigma)
    return result

# def relu_plus(x):
#     return torch.where(x <= 0, torch.exp(x), x)


class TransGeo(nn.Module):
    """
    Simple Siamese baseline with avgpool
    """
    def __init__(self,  args, mode='', pretrained=True):
        """
        dim: feature dimension (default: 512)
        """
        super(TransGeo, self).__init__()
        self.dim = args.dim

        # create the encoders
        # num_classes is the output fc dimension

        if args.dataset == 'vigor':
            self.size_sat = [320, 320]
            self.size_sat_default = [320, 320]
            self.size_grd = [320, 640]
        elif args.dataset == 'cvusa':
            self.size_sat = [256, 256]
            self.size_sat_default = [256, 256]
            self.size_grd = [112, 616]
        elif args.dataset == 'cvact':
            self.size_sat = [256, 256]
            self.size_sat_default = [256, 256]
            self.size_grd = [112, 616]

        # if args.sat_res != 0:
        #     self.size_sat = [args.sat_res, args.sat_res]
        if args.fov != 0:
            self.size_grd[1] = int(args.fov / 360. * self.size_grd[1])

        # self.ratio = self.size_sat[0]/self.size_sat_default[0]
        # base_model = deit_small_distilled_patch16_224

        # self.query_net = base_model(crop=False, img_size=self.size_grd, num_classes=args.dim)
        # self.reference_net = base_model(crop=False, img_size=self.size_sat, num_classes=args.dim)
        
        # self.polar = None

        # self.query_net = ViT_grd(image_size=self.size_grd, patch_size=16, dim=256, depth=12, heads=16, mlp_dim=1024, dropout=0.1,
        #                        emb_dropout=0.1)

        # self.query_net = TransNet(image_size=self.size_grd)
        # self.reference_net = TransNet(image_size=self.size_sat)
        # self.reference_net = ViT_sat(image_size=self.size_sat, patch_size=16, dim=256, depth=12, heads=16, mlp_dim=1024, dropout=0.1,
        #                        emb_dropout=0.1)


        self.mode = mode
        self.ratio = self.size_sat[0]/self.size_sat_default[0]
        base_model = convnext_tiny

        self.query_net = base_model(pretrained=pretrained, in_22k=False)
        self.reference_net = base_model(pretrained=pretrained, in_22k=False)
        self.polar = None
        self.decoder_net = Decoder()
        self.aggregator = Aggregator()
        self.l2norm = l2norm
        self.get_uncertainty = get_uncertainty
        # self.relu_plus = relu_plus
        if self.mode == 'uncertainty':
            self.tem_mlp = nn.Sequential(nn.Linear(768*2, 384),
                                         nn.ReLU(),
                                         nn.Linear(384, 96),
                                         nn.ReLU(),
                                         nn.Linear(96, 1)             
                                         )




            # with torch.no_grad():
            #     # self.temperatures = nn.Parameter(torch.ones(batch_size, 1, 1, 1), requires_grad=False)
        
    def shared_modules(self):
        return [self.query_net,self.reference_net,self.aggregator]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def forward(self, im_q, im_k, delta=None, atten=None, indexes=None):
        # if atten is not None:
        #     return self.query_net(im_q), self.reference_net(x=im_k, atten=atten)
        # else:
        #     return self.query_net(im_q), self.reference_net(x=im_k, indexes=indexes)
        # batch_size = im_q.size(0)
        x_grd_features, x_grd_mil = self.query_net(im_q)
        x_grd_10=x_grd_features[4]
        # x_grd, x_grd_5, x_grd_4, x_grd_3, x_grd_2, x_grd_1 = self.query_net(im_q)
        

        x_sat_features, x_sat_mil=self.reference_net(im_k)
        
        x_sat_320 = x_sat_features[0]
        x_sat_80 = x_sat_features[1]
        x_sat_40 = x_sat_features[2]
        x_sat_20 = x_sat_features[3]
        x_sat_10 = x_sat_features[4]
        # x_sat=x_sat_10

        # x_sat_reshaped = x_sat.view(-1, 512, 100)
        # x_grd_global_reshaped = x_grd_global.view(-1, 512, 1)

        x_grd_norm = self.l2norm(x_grd_10)
        x_sat_norm = self.l2norm(x_sat_10)
        # print(x_sat_norm.shape)
        x_grd_1_global = self.aggregator(x_grd_norm)

        # x_grd_global_norm = self.l2norm(x_grd_global)

        x_grd_global_broadcasted = x_grd_1_global.expand(-1, -1, x_sat_10.size(2), x_sat_10.size(3))
        # cosine_sim = F.cosine_similarity(x_grd_global_reshaped, x_sat_reshaped, dim=1)
        matching_score = torch.sum(x_grd_global_broadcasted * x_sat_norm, dim=1, keepdim=True)
        # cosine_sim_reshaped = cosine_sim.view(-1, 1, 10, 10)
        dec_input = torch.cat([x_sat_10, matching_score], dim=1)
        logits = self.decoder_net(dec_input, x_sat_320, x_sat_80, x_sat_40, x_sat_20)

        




        # logits_reshaped = torch.reshape(logits, [logits.shape[0], 640*640])
        # logits_reshaped_norm = self.l2norm(logits_reshaped)
        # logits = torch.reshape(logits_reshaped_norm, logits.shape) 

        if self.mode == 'uncertainty':
            x_sat_global = self.aggregator(x_sat_10)
            x_grd_global = self.aggregator(x_grd_10)
            x_sat_grd_global = torch.cat([x_sat_global, x_grd_global], dim=1)
            x_sat_grd_global = torch.reshape(x_sat_grd_global,[-1,x_sat_grd_global.size(1)])
            # temperatures = self.get_uncertainty(self.tem_mlp(x_sat_grd_global))
            # temperatures = self.relu_plus(self.tem_mlp(x_sat_grd_global))
            temperatures = self.tem_mlp(x_sat_grd_global)
            # print(temperatures.size())
            # temperatures = torch.reshape(temperatures,[-1, 1, 1, 1])
            
            # if temperatures.shape[0] != logits.shape[0]:
            #     temperatures_resized = temperatures[:logits.shape[0]]  # 截取 self.temperatures
            #     logits = logits / torch.pow(temperatures_resized,2)
            #     temperature = temperatures_resized
            # # print(logits.shape,self.temperatures.size())
            # else:
            #     logits = logits / torch.pow(temperatures,2)
            #     temperature = temperatures
            return x_grd_10, x_grd_1_global, x_sat_10, logits, matching_score, x_grd_mil, x_sat_mil, temperatures
        else:
            temperatures = {}
            return x_grd_10, x_grd_1_global, x_sat_10, logits, matching_score, x_grd_mil, x_sat_mil, temperatures
    
    # logits_reshaped =logits.view(logits.shape[0], 640*640)
    # gt_reshaped = gt.view(logits.shape[0], 640*640)
    # gt_reshaped = gt_reshaped / torch.sum(gt_reshaped, dim=1, keepdims=rue)