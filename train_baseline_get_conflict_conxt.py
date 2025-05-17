import argparse
import builtins
import math
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from Loss_CAGrad_conxt import TransGeoLoss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
# os.environ["WORLD_SIZE"] = "1"

import random
import shutil
import time
import warnings
import torch.distributed as dist
# dist.init_process_group('nccl', init_method='file:///tmp/somefile', rank=0, world_size=1)
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torch.nn.functional as F
from datetime import datetime

import numpy as np
from dataset.VIGOR import VIGOR
from dataset.CVUSA import CVUSA
from dataset.CVACT import CVACT
from model.TransGeo_conxt import FCVGL
# from criterion.soft_triplet import SoftTripletBiLoss
from dataset.global_sampler import DistributedMiningSampler,DistributedMiningSamplerVigor
from criterion.sam import SAM
from ptflops import get_model_complexity_info
from model.Utils import load_initialize_model
from parsers import get_parser
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import minimize


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# args = get_parser()
# args.log_path = train_baseline_s1_mso


def cagrad(grads, alpha=0.5, rescale=1):
        GG = grads.t().mm(grads).cpu() # [num_tasks, num_tasks]
        g0_norm = (GG.mean()+1e-8).sqrt() # norm of the average gradient

        x_start = np.ones(2) / 2
        bnds = tuple((0,1) for x in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha*g0_norm+1e-8).item()
        def objfn(x):
            return (x.reshape(1,2).dot(A).dot(b.reshape(2, 1)) + c * np.sqrt(x.reshape(1,2).dot(A).dot(x.reshape(2,1))+1e-8)).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm+1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale== 0:
            return g
        elif rescale == 1:
            return g / (1+alpha**2)
        else:
            return g / (1 + alpha)

def grad2vec(m, grads, grad_dims, task):
    # store the gradients
    grads[:, task].fill_(0.0)
    cnt = 0
    for mm in m.shared_modules():
        for p in mm.parameters():
            grad = p.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

def overwrite_grad(m, newgrad, grad_dims):
    newgrad = newgrad * 2 # to match the sum loss
    cnt = 0
    for mm in m.shared_modules():
        for param in mm.parameters():
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1


def compute_complexity(model, args):
    if args.dataset == 'vigor':
        size_sat = [320, 320]  # [512, 512]
        size_sat_default = [320, 320]  # [512, 512]
        size_grd = [320, 640]
    elif args.dataset == 'cvusa':
        size_sat = [256, 256]  # [512, 512]
        size_sat_default = [256, 256]  # [512, 512]
        size_grd = [112, 616]  # [224, 1232]
    elif args.dataset == 'cvact':
        size_sat = [256, 256]  # [512, 512]
        size_sat_default = [256, 256]  # [512, 512]
        size_grd = [112, 616]  # [224, 1232]

    if args.sat_res != 0:
        size_sat = [args.sat_res, args.sat_res]

    if args.fov != 0:
        size_grd[1] = int(args.fov /360. * size_grd[1])

    with torch.cuda.device(0):
        macs_1, params_1 = get_model_complexity_info(model.module.query_net, (3, size_grd[0], size_grd[1]), as_strings=False,
                                                 print_per_layer_stat=True, verbose=True)
        macs_2, params_2 = get_model_complexity_info(model.module.reference_net, (3, size_sat[0] , size_sat[1] ),
                                                     as_strings=False,
                                                     print_per_layer_stat=True, verbose=True)

        print('flops:', (macs_1+macs_2)/1e9, 'params:', (params_1+params_2)/1e6)



def main():  
    args = get_parser()
    # print(args)

    if args.seed is not None:  #X
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:  #X
        args.world_size = int(os.environ["WORLD_SIZE"])
    # print('args.world_size:',args.world_size)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    # print('ngpus_per_node',ngpus_per_node)

    if args.multiprocessing_distributed: #X
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # print('args.world_size:',args.world_size)
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        # print('args.gpu:',args.gpu)
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # global best_acc1
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:  #X
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:  #X
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
      
        print(f"Process {args.rank} is using GPU {torch.cuda.current_device()}")
    # create model
    print("=> creating model '{}'")

    model = FCVGL(args=args, pretrained=False)

    # model = load_initialize_model(model)

    if args.distributed:  #X
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
          
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            print(f"DDP model device_ids: {model.device_ids}")

        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        # print('单GPU训练')
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:   #DP 
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
 
    #Define contrast loss
    # criterion = SoftTripletBiLoss().cuda(args.gpu)
    criterion = TransGeoLoss(args, mode = 'uncertainty').cuda(args.gpu)

    #Defining the Optimizer
    if args.op == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.op == 'adam':
        optimizer = torch.optim.Adam(parameters, args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    elif args.op == 'adamw':
        optimizer = torch.optim.AdamW(parameters, args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    elif args.op == 'sam':
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(parameters, base_optimizer,  lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False, rho=args.rho, adaptive=args.asam)

    # current_time = datetime.now()
    # print("Current Time:", current_time)
    # logs_path=f'./logs/{current_time}'train_baseline_s1_so
    # args.log_path = train_baseline_s1_mso
    logs_path=f'./logs/{args.log_path}/'
    #Define the tensorboar object
    if args.rank == 0 or args.gpu is not None:
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        writer = SummaryWriter(f'{logs_path}')
        # dummy_grd = torch.randn(args.batch_size, 3, 320, 640).cuda()
        # dummy_sat = torch.randn(args.batch_size, 3, 320, 320).cuda()
        # writer.add_graph(model, (dummy_grd, dummy_sat))
    else: writer = None
    

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
                # if 'temperatures' in checkpoint:
                #     del checkpoint['temperatures']
            # if not args.crop:
            #     args.start_epoch = checkpoint['epoch']

            # best_acc1 = checkpoint['best_acc1']
            # if args.crop and args.sat_res != 0:
            #     pos_embed_reshape = checkpoint['state_dict']['module.reference_net.pos_embed'][:, 2:, :].reshape(
            #         [1,
            #          np.sqrt(checkpoint['state_dict']['module.reference_net.pos_embed'].shape[1] - 2).astype(int),
            #          np.sqrt(checkpoint['state_dict']['module.reference_net.pos_embed'].shape[1] - 2).astype(int),
            #          model.module.reference_net.embed_dim]).permute((0, 3, 1, 2))
            #     checkpoint['state_dict']['module.reference_net.pos_embed'] = \
            #         torch.cat([checkpoint['state_dict']['module.reference_net.pos_embed'][:, :2, :],
            #                    torch.nn.functional.interpolate(pos_embed_reshape, (
            #                    args.sat_res // model.module.reference_net.patch_embed.patch_size[0],
            #                    args.sat_res // model.module.reference_net.patch_embed.patch_size[1]),
            #                                                    mode='bilinear').permute((0, 2, 3, 1)).reshape(
            #                        [1, -1, model.module.reference_net.embed_dim])], dim=1)

            model.load_state_dict(checkpoint['state_dict'], strict=False)
            # args.global_step = checkpoint['global_step']
            # if args.op == 'sam' and args.dataset != 'cvact':    # Loading the optimizer status gives better result on CVUSA, but not on CVACT.
            #     optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True



    # Data loading code
    if not args.multiprocessing_distributed or args.gpu == 0:
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
            os.mkdir(os.path.join(args.save_path, 'attention'))
            os.mkdir(os.path.join(args.save_path, 'attention','train'))
            os.mkdir(os.path.join(args.save_path, 'attention','val'))

    if args.dataset == 'vigor':
        dataset = VIGOR
        mining_sampler = DistributedMiningSamplerVigor
    elif args.dataset == 'cvusa':
        dataset = CVUSA
        mining_sampler = DistributedMiningSampler
    elif args.dataset == 'cvact':
        dataset = CVACT
        mining_sampler = DistributedMiningSampler

    train_dataset = dataset(mode='train', print_bool=True, same_area=(not args.cross),args=args)
    # train_scan_dataset = dataset(mode='scan_train' if args.dataset == 'vigor' else 'train', print_bool=True, same_area=(not args.cross), args=args)
    # val_scan_dataset = dataset(mode='scan_val', same_area=(not args.cross), args=args)
    # val_query_dataset = dataset(mode='test_query', same_area=(not args.cross), args=args)
    # val_reference_dataset = dataset(mode='test_reference', same_area=(not args.cross), args=args)
    val_query_reference_dataset = dataset(mode='test_q_r', same_area=(not args.cross), args=args)

    if args.distributed:   #X
        if args.mining:
            train_sampler = mining_sampler(train_dataset, batch_size=args.batch_size, dim=args.dim, save_path=args.save_path)
            if args.resume:
                train_sampler.load(args.resume.replace(args.resume.split('/')[-1],''))
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None  #####如果单GPU训练，train_sampler应该是None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),  
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
        #数据集对象，包含了训练数据。通常是 torch.utils.data.Dataset 的一个实例，例如自定义数据集或 torchvision.datasets 提供的预定义数据集。
        #shuffle为true,表示在每个epoch开始时将数据打乱
        #num_workers用于数据加载的子进程数。更多的子进程可以加速数据加载。args.workers 指定了数据加载时使用的子进程数量。
        #pin_memory=True如果为 True，数据加载器会将数据加载到固定内存（Pinned memory）中，这样可以加速将数据转移到GPU上的过程。固定内存在GPU和CPU之间的数据传输更高效。
        #train_sampler自定义数据采样器，用于定义数据加载顺序。如果 train_sampler 被指定，shuffle 参数通常应为 False，因为采样器已经定义了数据加载顺序。
        #drop_last如果为 True，则在每个 epoch 中，如果剩余的数据不足以构成一个完整的批次，就将其丢弃。这样可以保证所有批次大小一致，有助于稳定训练过程。

    # train_scan_loader = torch.utils.data.DataLoader(
    #     train_scan_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True, sampler=torch.utils.data.distributed.DistributedSampler(train_scan_dataset), drop_last=False)

    # val_scan_loader = torch.utils.data.DataLoader(
    #     val_scan_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True,
    #     sampler=torch.utils.data.distributed.DistributedSampler(val_scan_dataset), drop_last=False)

    # val_query_loader = torch.utils.data.DataLoader(
    #     val_query_dataset,batch_size=32, shuffle=False,
    #     num_workers=args.workers, pin_memory=True) # 512, 64
    # val_reference_loader = torch.utils.data.DataLoader(
    #     val_reference_dataset, batch_size=64, shuffle=False,
    #     num_workers=args.workers, pin_memory=True) # 80, 128
    
    val_query_reference_loader = torch.utils.data.DataLoader(
        val_query_reference_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    dataset_length = len(train_dataset)
    print(f"Number of samples in dataset: {dataset_length}")

    best_mean_error = 100.0

    grad_dims = []
    for mm in model.shared_modules():
        for param in mm.parameters():
            grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), 2).cuda()

    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):

        print('start epoch:{}, date:{}'.format(epoch, datetime.now()))
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.mining:
                train_sampler.update_epoch()
        adjust_learning_rate(optimizer, epoch, args)

        
        train(train_loader, model, criterion, optimizer, epoch, args, train_sampler, grads, grad_dims, writer = writer)
       

        # if not args.multiprocessing_distributed or args.gpu == 0:
        mean_error_val = validate(val_query_reference_loader, model, args)
        # remember best acc@1 and save checkpoint
        is_best = mean_error_val <= best_mean_error
        best_mean_error = min(mean_error_val, best_mean_error)
        
        # dist.barrier()
        # torch.distributed.barrier()

        #Save model
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'global_step': args.global_step,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'mean_error_val': mean_error_val,
                'optimizer': optimizer.state_dict(),
            }, epoch+1, is_best, filename='checkpoint.pth.tar', args=args) # 'checkpoint_{:04d}.pth.tar'.format(epoch)






###########################################################################################################################





def train(train_loader, model, criterion, optimizer, epoch, args ,train_sampler, grads, grad_dims, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # temperature = AverageMeter('Temperatures', ':.4e')
    losses_dict = dict()
    # mean_error_list=[]
    losses_dict['L_mil'] = AverageMeter('L_mil')
    losses_dict['L_sim'] = AverageMeter('L_sim')
    losses_dict['L_out'] = AverageMeter('L_out')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    
    end = time.time()
    temperatures_dict = np.load('./result_vigor_mso_19_2_i/temperatures_dict_best.npy', allow_pickle=True).item()
    min_value = min(temperatures_dict.values())
    max_value = max(temperatures_dict.values())
    mean_value = sum(temperatures_dict.values()) / len(temperatures_dict)
    normalized_temperatures_dict = {k: (v - min_value) / (max_value - min_value) for k, v in temperatures_dict.items()}
    normalized_mean_value = sum(normalized_temperatures_dict.values()) / len(normalized_temperatures_dict)
    
    for i, (images_q, images_k, indexes, _, delta, atten, gt) in enumerate(train_loader):
        temperatures = []
        normalized_temperatures = []
     
        # measure data loading time
        data_time.update(time.time() - end)
        for k in range(len(indexes)):
            temperatures_dict.setdefault(indexes[k].item(), mean_value)
            normalized_temperatures_dict.setdefault(indexes[k].item(), normalized_mean_value)

            temperatures.append(temperatures_dict[indexes[k].item()])
            normalized_temperatures.append(normalized_temperatures_dict[indexes[k].item()])

        temperatures = torch.tensor(normalized_temperatures).cuda()
        # normalized_temperatures = torch.tensor(normalized_temperatures)
        alpha = np.mean(normalized_temperatures)
        if args.gpu is not None:
            images_q = images_q.cuda(args.gpu, non_blocking=True)
            images_k = images_k.cuda(args.gpu, non_blocking=True)
            # indexes = indexes.cuda(args.gpu, non_blocking=True)

     

        # compute output
        embed_grd, embed_grd_global,  embed_sat, logits, matching_score, x_grd_mil, x_sat_mil, _ = model(im_q =images_q, im_k=images_k, delta=delta)

        loss, loss_dict, train_loss_tmp = criterion(embed_grd, embed_grd_global, embed_sat, matching_score, gt, logits, delta, x_grd_mil, x_sat_mil, temperatures)

       
        losses.update(loss.item())
        for key, val in loss_dict.items():
            losses_dict[key].update(val.item())
       
      

        # compute gradient and do SGD step
        optimizer.zero_grad()
        for j in range(2):
        
            train_loss_tmp[j].backward(retain_graph=True)
            
            optimizer.first_step(zero_grad=True) 
        
            embed_grd, embed_grd_global,  embed_sat, logits, matching_score, x_grd_mil, x_sat_mil, _ = model(im_q =images_q, im_k=images_k, delta=delta)
            
            loss, loss_dict, train_loss_tmp  = criterion(embed_grd, embed_grd_global, embed_sat, matching_score, gt, logits, delta,x_grd_mil, x_sat_mil, temperatures)

            train_loss_tmp[j].backward(retain_graph=True)   #retain_graph=True
            grad2vec(model, grads, grad_dims, j)
            model.zero_grad_shared_modules()
        # Compute the gradient update direction
        g = cagrad(grads, alpha, rescale=1)

        # Update Gradient
        overwrite_grad(model, g, grad_dims)
        optimizer.second_step(zero_grad=True)
     



        
        if args.rank == 0 or args.gpu is not None:
            writer.add_scalar('Loss/train', loss.item(), args.global_step)
            writer.add_scalar('Loss/mil', args.mil_weight*loss_dict['L_mil'].item(), args.global_step)
            writer.add_scalar('Loss/sim', args.sim_weight*loss_dict['L_sim'].item(), args.global_step)
            writer.add_scalar('Loss/out', args.out_weight*loss_dict['L_out'].item(), args.global_step)
            
        args.global_step+=1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        

        # if i % args.print_freq == 0:
        progress.display(i)
            # print('Tem:',temperature_mean.item())

        del loss
        del embed_grd
        del embed_grd_global
        del embed_sat
        del logits
        del matching_score




# save all the attention map
def scan(loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time],
        prefix="Scan:")

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images_q, images_k, _, indexes , delta, _) in enumerate(loader):

            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images_q = images_q.cuda(args.gpu, non_blocking=True)
                images_k = images_k.cuda(args.gpu, non_blocking=True)
                indexes = indexes.cuda(args.gpu, non_blocking=True)

            # compute output
            embed_q, embed_k = model(im_q =images_q, im_k=images_k, delta=delta, indexes=indexes)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)


def validate(val_query_reference_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
  
    

    # switch to evaluate mode
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            

    model.eval()
    print('model validate on cuda', args.gpu)
    mean_error_list = []
    with torch.no_grad():
        end = time.time()
        
        for i, (images_q, images_k, indexes, labels) in enumerate(val_query_reference_loader):
            if args.gpu is not None:
                images_q = images_q.cuda(args.gpu, non_blocking=True)
                images_k = images_k.cuda(args.gpu, non_blocking=True)
                indexes = indexes.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)

            # compute output
            embed_grd, embed_grd_global,  embed_sat, logits, matching_score, x_grd_mil, x_sat_mil, _ = model(im_q =images_q, im_k=images_k, delta=labels)
           
            heatmap = torch.softmax(logits.view(logits.size(0), -1), dim=1).view_as(logits)
            len_heatmap = heatmap.shape[0]
            heatmap = heatmap.cpu().numpy()
           
            for j in range(len_heatmap):
                
                loc_pred = np.unravel_index(heatmap[j,0,:,:].argmax(), heatmap[j,0,:,:].shape)
                loc_gt_x = 320.0+labels[j][0] 
                loc_gt_y = 320.0-labels[j][1]
               
                mean_error = 0.114 * np.sqrt(np.square((loc_gt_x-loc_pred[0]).cpu().numpy())+np.square((loc_gt_y-loc_pred[1]).cpu().numpy()))
                mean_error_list.append(mean_error)
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print(f'{i}/{len(val_query_reference_loader)} batch_time:{batch_time}')
                
            del embed_grd
            del embed_grd_global
            del embed_sat
            del logits
            del matching_score

    print('mean_error:',sum(mean_error_list)/len(mean_error_list))
    return sum(mean_error_list)/len(mean_error_list)


def save_checkpoint(state, epoch, is_best, filename='checkpoint.pth.tar', args=None):  #is_best
    save_dir = os.path.join(args.save_path,f'{epoch}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir,filename))
    if is_best:
        # np.save(os.path.join(args.save_path,'temperatures_dict_best.npy'), temperatures_dict)
        shutil.copyfile(os.path.join(save_dir,filename), os.path.join(args.save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(query_features, reference_features, query_labels, topk=[1,5,10]):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    ts = time.time()
    N = query_features.shape[0]
    M = reference_features.shape[0]
    topk.append(M//100)
    results = np.zeros([len(topk)])
    # for CVUSA, CVACT
    if N < 80000:
        query_features_norm = np.sqrt(np.sum(query_features**2, axis=1, keepdims=True))
        reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
        similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).transpose())

        for i in range(N):
            ranking = np.sum((similarity[i,:]>similarity[i,query_labels[i]])*1.)

            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.
    else:
        # split the queries if the matrix is too large, e.g. VIGOR
        assert N % 4 == 0
        N_4 = N // 4
        for split in range(4):
            query_features_i = query_features[(split*N_4):((split+1)*N_4), :]
            query_labels_i = query_labels[(split*N_4):((split+1)*N_4)]
            query_features_norm = np.sqrt(np.sum(query_features_i ** 2, axis=1, keepdims=True))
            reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
            similarity = np.matmul(query_features_i / query_features_norm,
                                   (reference_features / reference_features_norm).transpose())
            for i in range(query_features_i.shape[0]):
                ranking = np.sum((similarity[i, :] > similarity[i, query_labels_i[i]])*1.)
                for j, k in enumerate(topk):
                    if ranking < k:
                        results[j] += 1.

    results = results/ query_features.shape[0] * 100.
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(results[0], results[1], results[2], results[-1], time.time() - ts))
    return results[:2]

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    # print(torch.cuda.current_device())
    main()
