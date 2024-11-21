import argparse

def get_parser():
     parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

     parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                         help='number of data loading workers (default: 32)')
     parser.add_argument('--epochs', default=200, type=int, metavar='N',
                         help='number of total epochs to run')
     parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')
     parser.add_argument('-b', '--batch_size', default=256, type=int,
                         metavar='N',
                         help='mini-batch size (default: 256), this is the total '
                              'batch size of all GPUs on the current node when '
                              'using Data Parallel or Distributed Data Parallel')
     parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                         metavar='LR', help='initial learning rate', dest='lr')
     parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                         help='learning rate schedule (when to drop lr by 10x)')
     parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                         help='momentum of SGD solver')
     parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                         metavar='W', help='weight decay (default: 1e-4)',
                         dest='weight_decay')
     parser.add_argument('-p', '--print-freq', default=10, type=int,
                         metavar='N', help='print frequency (default: 10)')
     parser.add_argument('--resume', default='', type=str, metavar='PATH',
                         help='path to latest checkpoint (default: none)')
     parser.add_argument('--save_path', default='./model_save', type=str, metavar='PATH',
                         help='path to save checkpoint (default: none)')
     parser.add_argument('--world-size', default=1, type=int,
                         help='number of nodes for distributed training')
     parser.add_argument('--rank', default=-1, type=int,
                         help='node rank for distributed training')
     parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                         help='url used to set up distributed training')
     parser.add_argument('--dist-backend', default='nccl', type=str,
                         help='distributed backend')
     parser.add_argument('--seed', default=None, type=int,
                         help='seed for initializing training. ')
     parser.add_argument('--gpu', default=None, type=int,
                         help='GPU id to use.')
     parser.add_argument('--multiprocessing-distributed', action='store_true',
                         help='Use multi-processing distributed training to launch '
                              'N processes per node, which has N GPUs. This is the '
                              'fastest way to use PyTorch for either single node or '
                              'multi node data parallel training')
     parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                         help='evaluate model on validation set')

     # moco specific configs:
     parser.add_argument('--dim', default=1000, type=int,
                         help='feature dimension (default: 128)')


     parser.add_argument('--cos', action='store_true',
                         help='use cosine lr schedule')

     parser.add_argument('--cross', action='store_true',
                         help='use cross area')

     parser.add_argument('--dataset', default='vigor', type=str,
                         help='vigor, cvusa, cvact')
     parser.add_argument('--op', default='adam', type=str,
                         help='sgd, adam, adamw')

     parser.add_argument('--share', action='store_true',
                         help='share fc')

     parser.add_argument('--mining', action='store_true',
                         help='mining')
     parser.add_argument('--asam', action='store_true',
                         help='asam')

     parser.add_argument('--rho', default=0.05, type=float,
                         help='rho for sam')
     parser.add_argument('--sat_res', default=0, type=int,
                         help='resolution for satellite')

     parser.add_argument('--crop', action='store_true',
                         help='nonuniform crop')

     parser.add_argument('--fov', default=0, type=int,
                         help='Fov')

     parser.add_argument('--margin', default=0.1, type=float, help='Rank loss margin')

     parser.add_argument('--max_violation', action='store_true', help='Use max instead of sum in the rank loss')

     parser.add_argument('--temperature', default=1, help='temperature')
     
     parser.add_argument('--mil_weight', default=0.1, type=float, help='Weight term for the mil loss')
     parser.add_argument('--sim_weight', default=0.1, type=float, help='Weight term for the sim loss')
     parser.add_argument('--out_weight', default=1/30, type=float, help='Weight term for the out loss')
     parser.add_argument('--global_step', default=0, type=int, help='global step of the training')
     parser.add_argument('--log_path', default='', type=str, help='')


     # parser for visualize
     parser.add_argument('-a', '--area', type=str, help='same or cross area testing', default='same')
     parser.add_argument('-i', '--img_idx', type=int, help='image index', default=110)



     
     args = parser.parse_args()
     return args