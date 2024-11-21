import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.Utils import Aggregator

def l2norm(x):
  """L2-normalize columns of x"""
  norm = torch.pow(x, 2).sum(dim=1, keepdim=True).sqrt()
  return torch.div(x, norm)


def cosine_sim(x, y):
  """Cosine similarity between all the image and sentence pairs. Assumes x and y are l2 normalized"""
  # x_sat_reshaped = x_sat.view(-1, 512, 100)
  # x_grd_global_reshaped = x_grd_global.view(-1, 512, 1)
  # cosine_sim = F.cosine_similarity(x_grd_global_reshaped, x_sat_reshaped, dim=1)
  # cosine_sim_reshaped = cosine_sim.view(-1, 10, 10)
  # return cosine_sim_reshaped


  return x.mm(y.t())

def two_d_softmax(x):
    exp_y = torch.exp(x)
    return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)


class TransGeoLoss(nn.Module):
    def __init__(self, opt, mode='', reduction='mean'):
        super(TransGeoLoss, self).__init__()

        self.margin = opt.margin if hasattr(opt, 'margin') else 1.0
        self.max_violation = opt.max_violation if hasattr(opt, 'max_violation') else False
        self.reduction = reduction
        self.mode = mode
        self.temperature = opt.temperature
        self.sim_fn = cosine_sim #if hasattr(opt, 'order') and opt.order #else cosine_sim
        self.sim_weight = opt.sim_weight
        self.out_weight = opt.out_weight
        self.mil_weight = opt.mil_weight
        self.sat_ori_size = [640,640]
        self.aggregator = Aggregator()
        self.l2norm = l2norm

    def L_mil(self, A, B, I, max_dim):
        loss = (self.margin + A - B).clamp(min=0.0)
        loss.masked_fill_(I, 0.0)
        if self.max_violation:
            print('max_violation')
            loss = loss.max(max_dim)[0]
        return loss.mean() if self.reduction=='mean' else loss.sum()
    

    def L_sim(self, scores, labels, temperature=1.0):
        """Contrastive loss over matching score. Adapted from https://arxiv.org/pdf/2004.11362.pdf Eq.2
        We extraly weigh the positive samples using the ground truth likelihood on those positions
        匹配得分的对比损失。改编自 https://arxiv.org/pdf/2004.11362.pdf Eq.2 我们使用这些位置上的地面实况似然对正样本进行外部权衡
        
        loss = - 1/sum(weights) * sum(inner_element*weights)
        inner_element = log( exp(score_pos/temperature) / sum(exp(score/temperature)) )
        """
        temperature = self.temperature
        exp_scores = torch.exp(scores / temperature) 

        # max_values, _ = labels.view(labels.size(0), -1).max(dim=1, keepdim=True)
        # max_values = max_values.view(labels.size(0), labels.size(1), 1, 1)

        bool_mask = labels > 1e-2 # only keep positive samples, we set a threshod on the likelihood in GT
        # bool_mask = (labels == max_values)
        
        denominator = torch.sum(exp_scores, dim=[1, 2, 3], keepdim=True)# 计算分母
        
        inner_element = torch.log(torch.masked_select(exp_scores / denominator, bool_mask))
  
        loss = -torch.sum(inner_element * torch.masked_select(labels, bool_mask)) / torch.sum(torch.masked_select(labels, bool_mask))
        return loss
        
    def L_out(self, gt_reshaped, logits_reshaped, tem):
        
        loss_fn = torch.nn.CrossEntropyLoss()
        
        loss_heatmap = loss_fn(logits_reshaped, gt_reshaped)/torch.pow(tem,2) + torch.log(tem)
        loss_heatmap = torch.mean(loss_heatmap)
        # loss_heatmap = sum(loss_heatmap)/len(loss_heatmap)
        return loss_heatmap
    
    def L_out_uncertainty(self, gt_onehot, logits):
        
        # logits=logits.cpu()
        # print(logits)
        nll = -gt_onehot * torch.log(logits.double())
        
        return torch.mean(torch.sum(nll, dim=(2, 3)))
                                                                                                                                                                                                                                    

    def forward(self, x_grd, x_grd_global, x_sat, matching_score, gt, logits, delta, x_grd_mil, x_sat_mil, tem):
        loss, losses = 0, dict()
       
        gt_bottleneck = F.max_pool2d(gt, kernel_size=(64, 64), stride=(64, 64))  #padding='SAME'  'SAME' padding 等效于 kernel_size // 2
        gt_bottleneck = gt_bottleneck.cuda()
        
       
        x_grd_mil=self.l2norm(x_grd_mil)
        x_sat_mil=self.l2norm(x_sat_mil)
        
        #Batch图像之间的对比学习
        scores = self.sim_fn(x_grd_mil, x_sat_mil)
        diagonal = scores.diag().view(x_grd_mil.size(0), 1)  #提取对角线元素
        # print('diagonal shape:',diagonal.shape)
        d1 = diagonal.expand_as(scores)   #将对角线沿行方向进行扩展
        d2 = diagonal.t().expand_as(scores)  #将对角线沿列方向进行扩展

        mask = torch.eye(scores.size(0)) > .5
        # print('mask shape:',mask.shape)
        I = torch.autograd.Variable(mask)
        if torch.cuda.is_available():
          I = I.cuda()

        # compare every diagonal score to scores in its column (image-to-text retrieval)
        i2t_loss = self.L_mil(scores, d1, I, 1)
        # compare every diagonal score to scores in its row (text-to-image retrieval)
        t2i_loss = self.L_mil(scores, d2, I, 0)
        Lmil_loss = i2t_loss + t2i_loss




        loss += self.mil_weight * Lmil_loss
        losses['L_mil'] = Lmil_loss

        # Lsim loss 
        Lsim_loss = self.L_sim(matching_score, gt_bottleneck)
        loss += self.sim_weight * Lsim_loss
        losses['L_sim'] = Lsim_loss

        # Lout loss
        # if self.mode == 'uncertainty':
        #   Lout_loss = self.L_out_uncertainty(gt_onehot, logits)
        # else:
        logits_reshaped = torch.reshape(logits, [logits.shape[0], 640*640])
        gt_reshaped = torch.reshape(gt, [logits.shape[0], 640*640])
        gt_reshaped = gt_reshaped / torch.sum(gt_reshaped, dim=1, keepdim=True)
        gt_reshaped = gt_reshaped.cuda()
        Lout_loss = self.L_out(gt_reshaped, logits_reshaped, tem)
        # Lout_loss = self.L_out_uncertainty(gt_onehot, logits)
        loss += self.out_weight * Lout_loss
        losses['L_out'] = Lout_loss



        return loss, losses
