
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from losses.losses import fusion_loss_vif
from dataloader.fuse_data_vsm import GetDataset_type3
import pathlib
import logging.config
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn as nn
import torch.cuda as cuda
first_execution = True
from torchvision.transforms import Resize
def hyper_args():
    """
    get hyper parameters from args
    """
    parser = argparse.ArgumentParser(description='RobF Net train process')
    # dataset
    parser.add_argument('--ir_path', default='/public/home/Feecuin1/feecuin/xiaobo_mamba/datasets/Compound_weather/ir', type=str)
    parser.add_argument('--vi_path', default='/public/home/Feecuin1/feecuin/xiaobo_mamba/datasets/Compound_weather/vi', type=str)
    parser.add_argument('--gt_path', default='/public/home/Feecuin1/feecuin/xiaobo_mamba/datasets/Compound_weather/gt_vi', type=str)
    parser.add_argument('--gt_ir_path',default='/public/home/Feecuin1/feecuin/xiaobo_mamba/datasets/Compound_weather/gt_ir',type=str)
    # parser.add_argument('--img_size', default=[96,144,160,192] , type=int, help='裁剪图片的大小')
    parser.add_argument('--img_size', default=[128,128,128,96] , type=int, help='裁剪图片的大小')
    parser.add_argument('--batchsize', default=[7,7,7,12], type=int, help='mini-batch size')  # 32
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--nEpochs', default=4000, type=int, help='number of total epochs to run')
    parser.add_argument("--step", type=int, default=400,
                        help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument('--interval', default=20, help='record interval')
    # checkpoint
    parser.add_argument("--load_model_fuse", default='/public/home/Feecuin1/feecuin/xiaobo_mamba/checkpoint/Compound_weather/xiaobo_mamba_0680.pth', help="path to pretrained model (default: none)")
    parser.add_argument('--ckpt', default='/public/home/Feecuin1/feecuin/xiaobo_mamba/checkpoint/Compound_weather-0680', help='checkpoint cache folder')

    args = parser.parse_args()
    return args


# def main(args):

#     torch.backends.cudnn.benchmark = True
#     log = logging.getLogger()
#     interval = args.interval
#     torch.distributed.init_process_gr
#     local_rank = int(os.environ['LOCAL_RANK'])
#     torch.cuda.set_device(local_rank)
#     device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

#     print("===> Creating Save Path of Checkpoints")
#     cache = pathlib.Path(args.ckpt)

#     print("===> Loading datasets")
#     data_train = GetDataset_type3('train', ir_path=args.ir_path, vi_path=args.vi_path,
#                                   gt_path=args.gt_path, gt_ir_path=args.gt_ir_path,
#                                   img_size=args.img_size)
#     train_sampler = DistributedSampler(data_train)
#     training_data_loader = torch.utils.data.DataLoader(data_train, args.batchsize, sampler=train_sampler,
#                                                        pin_memory=True,
#                                                        num_workers=12)

#     print("===> Building models")
#     from model.AWM_Fuse import WaveMamba
#     AWMFuse = WaveMamba().to(device)
#     AWMFuse = torch.nn.parallel.DistributedDataParallel(AWMFuse, device_ids=[local_rank],
#                                                              find_unused_parameters=True)


#     print("===> Setting Optimizers")
#     optimizer = torch.optim.Adam(params=AWMFuse.parameters(), lr=args.lr)

#     # TODO: optionally copy weights from a checkpoint
#     if args.load_model_fuse is not None:
#         print('Loading pre-trained FuseNet checkpoint %s' % args.load_model_fuse)
#         log.info(f'Loading pre-trained checkpoint {str(args.load_model_fuse)}')
#         state = torch.load(str(args.load_model_fuse),map_location = 'cpu')
#         AWMFuse.load_state_dict(state)
#     else:
#         print("=> no model found at '{}'".format(args.load_model_fuse))

#     print("===> Starting Training")
#     for epoch in range(args.start_epoch, args.nEpochs + 1):
#         prev_time = time.time()
#         train_step1(args, training_data_loader, optimizer, AWMFuse,
#                     epoch, device, prev_time)

#         # TODO: save checkpoint
#         if local_rank == 0:
#             save_checkpoint(AWMFuse, epoch, cache) if epoch % interval == 0 else None

def main(args):
    torch.backends.cudnn.benchmark = True
    log = logging.getLogger()
    interval = args.interval
    torch.distributed.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    print("===> Creating Save Path of Checkpoints")
    cache = pathlib.Path(args.ckpt)

    print("===> Loading datasets")
    # 初始设置
    current_stage = 0
    max_stages = len(args.img_size)
    current_img_size = args.img_size[current_stage]
    current_batchsize = args.batchsize[current_stage]
    
    data_train = GetDataset_type3('train', ir_path=args.ir_path, vi_path=args.vi_path,
                                 gt_path=args.gt_path, gt_ir_path=args.gt_ir_path,
                                 img_size=current_img_size)
    train_sampler = DistributedSampler(data_train)
    training_data_loader = torch.utils.data.DataLoader(data_train, current_batchsize, 
                                                     sampler=train_sampler,
                                                     pin_memory=True,
                                                     num_workers=24)

    print("===> Building models")
    from model.ori_version_526_version import WaveMamba
    AWMFuse = WaveMamba().to(device)
    AWMFuse = torch.nn.parallel.DistributedDataParallel(AWMFuse, device_ids=[local_rank],
                                                      find_unused_parameters=False)

    print("===> Setting Optimizers")
    optimizer = torch.optim.Adam(params=AWMFuse.parameters(), lr=args.lr)

    if args.load_model_fuse is not None:
        print('Loading pre-trained FuseNet checkpoint %s' % args.load_model_fuse)
        log.info(f'Loading pre-trained checkpoint {str(args.load_model_fuse)}')
        state = torch.load(str(args.load_model_fuse), map_location='cpu')
        AWMFuse.load_state_dict(state)
    else:
        print("=> no model found at '{}'".format(args.load_model_fuse))

    print("===> Starting Training")
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        # 每100个epoch检查是否需要切换阶段
        if epoch % 1000 == 1 and epoch > 1:
            current_stage = min(current_stage + 1, max_stages - 1)
            current_img_size = args.img_size[current_stage]
            current_batchsize = args.batchsize[current_stage]
            
            print(f"\n=== Switching to Stage {current_stage}: img_size={current_img_size}, batchsize={current_batchsize} ===")
            
            # 重新创建数据集和DataLoader
            data_train = GetDataset_type3('train', ir_path=args.ir_path, vi_path=args.vi_path,
                                        gt_path=args.gt_path, gt_ir_path=args.gt_ir_path,
                                        img_size=current_img_size)
            train_sampler = DistributedSampler(data_train)
            training_data_loader = torch.utils.data.DataLoader(data_train, current_batchsize, 
                                                             sampler=train_sampler,
                                                             pin_memory=True,
                                                             num_workers=24)
        prev_time = time.time()
        train_step1(args, training_data_loader, optimizer, AWMFuse,
                    epoch, device, prev_time)

        if local_rank == 0:
            save_checkpoint(AWMFuse, epoch, cache) if epoch % interval == 0 else None
def train_step1(args, tqdm_loader, optimizer1, AWMFuse,
                 epoch, device, prev_time):

    AWMFuse.train()
    # TODO: update learning rate of the optimizer
    lr_F = adjust_learning_rate(args, optimizer1, epoch - 1)
    print("Epoch={}, lr_F={} ".format(epoch, lr_F))
    for i, (data_IR, data_VIS, data_GT, data_gt_ir) in (enumerate(tqdm_loader)):
        data_VIS_rgb, data_IR, data_GT, data_gt_ir = (data_VIS.cuda(non_blocking=True), data_IR.cuda(non_blocking=True), data_GT.cuda(non_blocking=True), data_gt_ir.cuda(non_blocking=True))

        AWMFuse.train()
        AWMFuse.zero_grad()
    
        rgb_Fuse = AWMFuse(data_VIS_rgb,data_IR)

        loss_ = fusion_loss_vif()
        loss__= loss_(data_GT, data_gt_ir, rgb_Fuse)

        total_loss =  loss__

        torch.distributed.barrier()
        optimizer1.zero_grad()
        total_loss.backward()
        optimizer1.step()


        # Determine approximate time left
        batches_done = epoch * len(tqdm_loader) + i
        batches_left = args.nEpochs * len(tqdm_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss_vi: %f]  ETA: %.10s "
            % (
                epoch,
                args.nEpochs,
                i,
                len(tqdm_loader),
                loss__.item(),
                time_left

            )
        )

def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(net, epoch, cache):
    model_folder = cache
    model_out_path = str(model_folder / f'xiaobo_mamba_{epoch:04d}.pth')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(net.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))




first_execution = True

if __name__ == "__main__":
    args = hyper_args()
    main(args)
