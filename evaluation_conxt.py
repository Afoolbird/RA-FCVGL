import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["WORLD_SIZE"] = "1"

from model.FCVGL_model_conxt import FCVGL
# from dataset.global_sampler import DistributedMiningSampler, DistributedMiningSamplerVigor

# import cv2
import numpy as np
import torch
import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
import torch.optim


from dataset.VIGOR import VIGOR

# from dataset.CVACT import CVACT

from parsers import get_parser
from tqdm import tqdm
import torch.utils.data
args = get_parser()
area = args.area


# load_model_path = './result_vigor_mso/240/checkpoint.pth.tar'  # path to the trained model  model_best.pth.tar
load_model_path = './result_vigor_cagrad_3_192i/model_best.pth.tar'

input_data = VIGOR(args=args)
dataset = VIGOR
val_query_reference_dataset = dataset(mode='test_q_r', same_area=(not args.cross), args=args)
val_query_reference_loader = torch.utils.data.DataLoader(
        val_query_reference_dataset, batch_size=8, shuffle=False,
        num_workers=1, pin_memory=True)

# Define the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FCVGL(args=args, pretrained=False,mode='uncertainty').to(device)

checkpoint = torch.load(load_model_path)
model_state_dict = checkpoint['state_dict']
print('epoch:',checkpoint['epoch'])
new_state_dict = {}
for k, v in model_state_dict.items():
    new_key = k.replace('module.', '')
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict, strict=False)
model.eval()

mean_error_list=[]
with tqdm(total=len(val_query_reference_loader) , desc="Img_idx Loop", ncols=100, ascii=True) as pbar:
    for i, (images_q, images_k, indexes, labels) in enumerate(val_query_reference_loader):
        images_q = images_q.cuda(0, non_blocking=True)
        images_k = images_k.cuda(0, non_blocking=True)
        indexes = indexes.cuda(0, non_blocking=True)
        labels = labels.cuda(0, non_blocking=True)
        
        with torch.no_grad():
            _, _,  _, logits, _, _, _, _ = model(im_q =images_q, im_k=images_k)
            heatmap = torch.softmax(logits.view(logits.size(0), -1), dim=1).view_as(logits)
            len_heatmap = heatmap.shape[0]
            heatmap = heatmap.cpu().numpy()

        # loc_pred = np.unravel_index(heatmap[:,0,:,:].argmax(), heatmap[:,0,:,:].shape)
        # loc_gt_x = 320.0+labels[:][0].cpu().numpy()
        # loc_gt_y = 320.0-labels[:][1].cpu().numpy()
        # mean_error = 0.114 * np.sqrt(np.square((loc_gt_x[:]-loc_pred[:][0]))+np.square((loc_gt_y[:]-loc_pred[:][1])))
        # mean_error_list.extend(mean_error)
        for j in range(len_heatmap):
            # print('@@@@@@@@@@@@')
            loc_pred = np.unravel_index(heatmap[j,0,:,:].argmax(), heatmap[j,0,:,:].shape)
            loc_gt_x = 320.0+labels[j][0] 
            loc_gt_y = 320.0-labels[j][1] 
            # print('===============================')
            mean_error = 0.114 * np.sqrt(np.square((loc_gt_x-loc_pred[0]).cpu().numpy())+np.square((loc_gt_y-loc_pred[1]).cpu().numpy()))
            mean_error_list.append(mean_error)

        del logits
        del images_q
        del images_k
        del indexes
        del labels

        pbar.update(1)

print('mean_error_on_test_set:',sum(mean_error_list)/len(mean_error_list))
