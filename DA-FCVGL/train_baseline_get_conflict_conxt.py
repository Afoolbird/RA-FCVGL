import argparse
import builtins
import math
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from Loss_conxt import TransGeoLoss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# os.environ["WORLD_SIZE"] = "1"

import random
import shutil
import time
import warnings
import torch.distributed as dist
# dist.init_process_group('nccl', init_method='file:///tmp/somefile', rank=0, world_size=1)
import torch
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
from dataset.CVACT import CVACT
from model.FCVGL_model_conxt import TransGeo
# from criterion.soft_triplet import SoftTripletBiLoss
from dataset.global_sampler import DistributedMiningSampler,DistributedMiningSamplerVigor

from ptflops import get_model_complexity_info
from model.Utils import load_initialize_model
from parsers import get_parser
from torch.utils.tensorboard import SummaryWriter


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# args = get_parser()
# args.log_path = train_baseline_s1_mso





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



def main():   #这里的作用是对是否用多GPU训练进行判断，进而执行相应操作
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
    global best_acc1
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
        # print('###########################################')
        # print('args.world_size:',args.world_size)
        # print('args.gpu:',args.gpu)
        # print('args.rank:',args.rank)
        print(f"Process {args.rank} is using GPU {torch.cuda.current_device()}")
    # create model
    print("=> creating model '{}'")

    model = TransGeo(args=args, pretrained=False, mode='uncertainty', batch_size= int(args.batch_size / ngpus_per_node))

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
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            print(f"DDP model device_ids: {model.device_ids}")

        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:   #DP 模式
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)


    # print(next(model.parameters()).is_cuda)
    # compute_complexity(model, args)  # uncomment to see detailed computation cost
    #定义对比损失
    # criterion = SoftTripletBiLoss().cuda(args.gpu)
    criterion = TransGeoLoss(args, mode = 'uncertainty').cuda(args.gpu)

    #定义优化器
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
    #定义tensorboar对象
    if args.rank == 0:
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


    #以上都是有关模型和loss的。
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

    # if args.evaluate:
    #     if not args.multiprocessing_distributed or args.gpu == 0:
    #         validate(val_query_loader, val_reference_loader, model, args)
    #     return

    
    # mean_error_val = validate(val_query_reference_loader, model, args)
        
    
    # print('train_dataset:',type(train_dataset))
    dataset_length = len(train_dataset)
    print(f"Number of samples in dataset: {dataset_length}")
    # first_sample = dataset[0]
    # print(first_sample)

    # if args.resume:
    #     best_path= os.path.join(args.save_path,'model_best.pth.tar')
    #     loc = 'cuda:{}'.format(args.gpu)
    #     best_checkpoint = torch.load(best_path, map_location=loc)
    #     best_mean_error = best_checkpoint['mean_error_val']
    # else:
    #     best_mean_error = 100.0
    best_mean_error = 100.0
        
    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):

        print('start epoch:{}, date:{}'.format(epoch, datetime.now()))
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.mining:
                train_sampler.update_epoch()
        adjust_learning_rate(optimizer, epoch, args)

        # print('train_loader.shape:',train_loader.shape)
        # train for one epoch    目前为止train_sampler为None
        
        temperatures_dict = train(train_loader, model, criterion, optimizer, epoch, args, train_sampler, writer = writer)
       
        # for name, param in model.named_parameters():
        #    writer.add_histogram(name, param, epoch)
        # if args.rank == 0:
        #     writer.close()
        
        # print(f"Process {args.rank} reached the barrier.")

        # # evaluate on validation set   这是在验证检索的精度，直接注释，对我们没帮助目前  
        # 或者说得改一个对我们有帮助的验证方式  先注释，训起来看看效果
        # if not args.multiprocessing_distributed or args.gpu == 0:
        #     acc1 = validate(val_query_loader, val_reference_loader, model, args)
        #     # remember best acc@1 and save checkpoint
        #     is_best = acc1 > best_acc1
        #     best_acc1 = max(acc1, best_acc1)   validate(val_query_reference_loader, model, args)
        # dist.barrier()
        
        torch.cuda.empty_cache()

        # if not args.multiprocessing_distributed or args.gpu == 0:
        mean_error_val = validate(val_query_reference_loader, model, args)
        # remember best acc@1 and save checkpoint
        is_best = mean_error_val <= best_mean_error
        best_mean_error = min(mean_error_val, best_mean_error)
        
        dist.barrier()
        torch.distributed.barrier()

        #保存模型
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'global_step': args.global_step,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'mean_error_val': mean_error_val,
                'optimizer': optimizer.state_dict(),
            }, epoch+1, is_best, filename='checkpoint.pth.tar', temperatures_dict = temperatures_dict, args=args) # 'checkpoint_{:04d}.pth.tar'.format(epoch)

    ##这里是展示训练或加载数据的相关时间，目前没必要
    # if not args.crop:
    #     model.module.reference_net.save = os.path.join(args.save_path, 'attention', 'train')
    #     scan(train_scan_loader, model, args)
    #     model.module.reference_net.save = os.path.join(args.save_path, 'attention', 'val')
    #     scan(val_scan_loader, model, args)





###########################################################################################################################





def train(train_loader, model, criterion, optimizer, epoch, args ,train_sampler=None, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # temperature = AverageMeter('Temperatures', ':.4e')
    losses_dict = dict()
    # mean_error_list=[]
    losses_dict['L_mil'] = AverageMeter('L_mil')
    losses_dict['L_sim'] = AverageMeter('L_sim')
    losses_dict['L_out'] = AverageMeter('L_out')
    # mean_ps = AverageMeter('Mean-P', ':6.2f')
    # mean_ns = AverageMeter('Mean-N', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    temperatures_dict={}
    for i, (images_q, images_k, indexes, _, delta, atten, gt) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images_q = images_q.cuda(args.gpu, non_blocking=True)
            images_k = images_k.cuda(args.gpu, non_blocking=True)
            indexes = indexes.cuda(args.gpu, non_blocking=True)

        # compute output
        # if args.crop:
        #     embed_q, embed_k = model(im_q=images_q, im_k=images_k, delta=delta, atten=atten)
        # else:
        #     embed_q, embed_k = model(im_q =images_q, im_k=images_k, delta=delta)

        # compute output
        embed_grd, embed_grd_global,  embed_sat, logits, matching_score, x_grd_mil, x_sat_mil, temperatures = model(im_q =images_q, im_k=images_k, delta=delta)
        # logits_reshaped = torch.reshape(logits, [logits.shape[0], 640*640])
        # gt_reshaped = torch.reshape(gt, [logits.shape[0], 640*640])
        # gt_reshaped = gt_reshaped / torch.sum(gt_reshaped, dim=1, keepdim=True)
        # gt_bottleneck = F.max_pool2d(gt, kernel_size=(64, 64), stride=(64, 64))  #padding='SAME'  'SAME' padding 等效于 kernel_size // 2
        # gt_bottleneck = gt_bottleneck.cuda()
        # gt_reshaped = gt_reshaped.cuda()
        loss, loss_dict  = criterion(embed_grd, embed_grd_global, embed_sat, matching_score, gt, logits, delta, x_grd_mil, x_sat_mil, temperatures)

        if args.mining:
            train_sampler.update(concat_all_gather(embed_sat).detach().cpu().numpy(),concat_all_gather(embed_grd).detach().cpu().numpy(),concat_all_gather(indexes).detach().cpu().numpy())
        losses.update(loss.item())
        for key, val in loss_dict.items():
            losses_dict[key].update(val.item())
       
      

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if args.op != 'sam':
            optimizer.step()
        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param, epoch)

        else:
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass, only for ASAM
            # if args.crop:
            #     embed_q, embed_k = model(im_q=images_q, im_k=images_k, delta=delta, atten=atten)
            # else:
            #     embed_q, embed_k = model(im_q=images_q, im_k=images_k, delta=delta)
            # loss, mean_p, mean_n = criterion(embed_q, embed_k)
            embed_grd, embed_grd_global,  embed_sat, logits, matching_score, x_grd_mil, x_sat_mil, temperatures = model(im_q =images_q, im_k=images_k, delta=delta)
           
            loss, loss_dict  = criterion(embed_grd, embed_grd_global, embed_sat, matching_score, gt, logits, delta,x_grd_mil, x_sat_mil, temperatures)

            loss.backward()
            optimizer.second_step(zero_grad=True)
            # _, _,  _, _, _, _, _, temperatures= model(im_q =images_q, im_k=images_k, delta=delta)
            # temperature_mean = torch.mean(temperatures)
            # temperature.update(temperature_mean)
            for j in range(temperatures.size(0)):
                temperatures_dict[indexes[j].item()] = temperatures[j].item()
                # print(temperatures_dict[indexes[j].item()],':',temperatures[j].item())



        
        if args.rank == 0:
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

    np.save(os.path.join(args.save_path,'temperatures_dict.npy'), temperatures_dict)
    return temperatures_dict


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


# # query features and reference features should be computed separately without correspondence label
# def validate(val_query_loader, val_reference_loader, model, args):
#     batch_time = AverageMeter('Time', ':6.3f')
#     progress_q = ProgressMeter(
#         len(val_query_loader),
#         [batch_time],
#         prefix='Test_query: ')
#     progress_k = ProgressMeter(
#         len(val_reference_loader),
#         [batch_time],
#         prefix='Test_reference: ')

#     # switch to evaluate mode
#     model_query = model.module.query_net
#     model_reference = model.module.reference_net
#     if args.distributed:
#         # For multiprocessing distributed, DistributedDataParallel constructor
#         # should always set the single device scope, otherwise,
#         # DistributedDataParallel will use all available devices.
#         if args.gpu is not None:
#             torch.cuda.set_device(args.gpu)
#             model_query.cuda(args.gpu)
#             model_reference.cuda(args.gpu)

#     model_query.eval()
#     model_reference.eval()
#     print('model validate on cuda', args.gpu)

#     query_features = np.zeros([len(val_query_loader.dataset), args.dim])
#     query_labels = np.zeros([len(val_query_loader.dataset)])
#     reference_features = np.zeros([len(val_reference_loader.dataset), args.dim])

#     with torch.no_grad():
#         end = time.time()
#         # reference features
#         for i, (images, indexes, atten) in enumerate(val_reference_loader):
#             if args.gpu is not None:
#                 images = images.cuda(args.gpu, non_blocking=True)
#                 indexes = indexes.cuda(args.gpu, non_blocking=True)

#             # compute output
#             if args.crop:
#                 reference_embed = model_reference(x=images, atten=atten)
#             else:
#                 reference_embed = model_reference(x=images, indexes=indexes)  # delta

#             reference_features[indexes.cpu().numpy().astype(int), :] = reference_embed.detach().cpu().numpy()

#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#             if i % args.print_freq == 0:
#                 progress_k.display(i)

#         end = time.time()

#         # query features
#         for i, (images, indexes, labels) in enumerate(val_query_loader):
#             if args.gpu is not None:
#                 images = images.cuda(args.gpu, non_blocking=True)
#                 indexes = indexes.cuda(args.gpu, non_blocking=True)
#                 labels = labels.cuda(args.gpu, non_blocking=True)

#             # compute output
#             query_embed = model_query(images)

#             query_features[indexes.cpu().numpy(), :] = query_embed.cpu().numpy()
#             query_labels[indexes.cpu().numpy()] = labels.cpu().numpy()

#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#             if i % args.print_freq == 0:
#                 progress_q.display(i)

#         [top1, top5] = accuracy(query_features, reference_features, query_labels.astype(int))

#     if args.evaluate:
#         np.save(os.path.join(args.save_path, 'grd_global_descriptor.npy'), query_features)
#         np.save('sat_global_descriptor.npy', reference_features)

#     return top1

# query features and reference features should be computed separately without correspondence label
def validate(val_query_reference_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    # progress_qk = ProgressMeter(
    #     len(val_query_reference_loader),
    #     [batch_time],
    #     prefix='Test_query: ')
    

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
            # logits_reshaped = torch.reshape(logits, [logits.shape[0], 640*640])
            # logits_reshaped = logits_reshaped.cpu().numpy()
            heatmap = torch.softmax(logits.view(logits.size(0), -1), dim=1).view_as(logits)
            len_heatmap = heatmap.shape[0]
            heatmap = heatmap.cpu().numpy()
            # print('##########')
            for j in range(len_heatmap):
                # print('@@@@@@@@@@@@')
                loc_pred = np.unravel_index(heatmap[j,0,:,:].argmax(), heatmap[j,0,:,:].shape)
                loc_gt_x = 320.0+labels[j][0] 
                loc_gt_y = 320.0-labels[j][1]
                # print('===============================')
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


def save_checkpoint(state, epoch, is_best, filename='checkpoint.pth.tar', temperatures_dict={}, args=None):  #is_best
    save_dir = os.path.join(args.save_path,f'{epoch}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir,filename))
    if is_best:
        np.save(os.path.join(args.save_path,'temperatures_dict_best.npy'), temperatures_dict)
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
