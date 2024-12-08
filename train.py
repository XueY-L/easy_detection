'''
python train.py --config configs/faster_rcnn/faster_rcnn_coco.yml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./dist_train.sh 8 --config configs/faster_rcnn/faster_rcnn_coco_dist.yml

python -m torch.distributed.launch --nproc_per_node=8 train.py --config configs/faster_rcnn/faster_rcnn_coco_dist.yml
'''

# encoding = utf-8
import os
import time
from pprint import pprint

import torch
import torch.distributed as dist

from options.helper import is_distributed, is_first_gpu, setup_multi_processes
from options import opt, config

# 设置多卡训练
if is_distributed():
    setup_multi_processes()

from dataloader.dataloaders import train_dataloader, val_dataloader

from network import get_model
from eval import eval_mAP

from options.helper import init_log, load_meta, save_meta
from utils import seed_everything
from scheduler import schedulers

from mscv.summary import create_summary_writer, write_meters_loss
from misc_utils import progress_bar, format_time

# 初始化
with torch.no_grad():
    # 设置随机种子
    if 'RANDOM_SEED' in config.MISC:
        seed_everything(config.MISC.RANDOM_SEED)
    
    # 初始化路径
    save_root = os.path.join('checkpoints', opt.tag)
    log_root = os.path.join('logs', opt.tag)

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    # dataloader
    train_dataloader = train_dataloader
    val_dataloader = val_dataloader

    if is_first_gpu():
        # 初始化日志
        logger = init_log(log_dir=log_root)

        # 初始化训练的meta信息
        meta = load_meta(new=True)
        save_meta(meta)
        
    # 初始化模型
    Model = get_model(config.MODEL.NAME)
    model = Model(config)  # 这一步里构造了opt.epochs

    # 转到GPU
    model = model.to(device=opt.device)

    if opt.load:
        load_epoch = model.load(opt.load)
        start_epoch = load_epoch + 1 if opt.resume or 'RESUME' in config.MISC else 1
    elif 'LOAD' in config.MODEL:
        load_epoch = model.load(config.MODEL.LOAD)
        start_epoch = load_epoch + 1 if opt.resume or 'RESUME' in config.MISC else 1
    else:
        start_epoch = 1

    model.train()

    if is_first_gpu():
        # 开始训练
        print('Start training...')
    
    start_step = (start_epoch - 1) * len(train_dataloader)
    global_step = start_step
    total_steps = opt.epochs * len(train_dataloader)
    start = time.time()

    # 定义scheduler
    scheduler = model.scheduler

    if is_first_gpu():
        # tensorboard日志
        writer = create_summary_writer(log_root)
    else:
        writer = None

    # 在日志记录transforms
    if is_first_gpu():
        logger.info('train_trasforms: ' +str(train_dataloader.dataset.transforms))
        logger.info('===========================================')
        if val_dataloader is not None:
            logger.info('val_trasforms: ' +str(val_dataloader.dataset.transforms))
        logger.info('===========================================')

        # 在日志记录scheduler
        if config.OPTIMIZE.SCHEDULER in schedulers:
            logger.info('scheduler: (Lambda scheduler)\n' + str(schedulers[config.OPTIMIZE.SCHEDULER]))
            logger.info('===========================================')

# 训练循环
#try:
if __name__ == '__main__':
    eval_result = ''
    
    # for epoch in range(start_epoch, opt.epochs + 1):
    for epoch in range(1):
        if is_distributed():
            train_dataloader.sampler.set_epoch(epoch)
        for iteration, sample in enumerate(train_dataloader):
            global_step += 1

            base_lr = config.OPTIMIZE.BASE_LR
            if global_step < 500:
                # 500个step从0.001->1.0
                lr = (0.001 + 0.999 / 499 * (global_step - 1)) * base_lr
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = lr

            elif global_step == 500:
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = base_lr

            # 计算剩余时间
            rate = (global_step - start_step) / (time.time() - start)
            remaining = (total_steps - global_step) / rate

            # --debug模式下只训练10个batch
            if opt.debug and iteration > 10:
                break

            sample['global_step'] = global_step
     
            #  更新网络参数
            updated = model.update(sample)
            predicted = updated.get('predicted')

            pre_msg = 'Epoch:%d' % epoch

            lr = model.optimizer.param_groups[0]['lr']
            # 显示进度条
            msg = f'lr:{round(lr, 6) : .6f} (loss) {str(model.avg_meters)} ETA: {format_time(remaining)}'
            if is_first_gpu():
                progress_bar(iteration, len(train_dataloader), pre_msg, msg)  # 打印训练进度
                # print(pre_msg, msg)

                if global_step % 1000 == 0:  # 每1000个step将loss写到tensorboard
                    write_meters_loss(writer, 'train', model.avg_meters, global_step)
                # 训练时每100个step记录一下loss
                if global_step % 100 == 0:
                    logger.info(f'step: {global_step} (loss) '+ str(model.avg_meters))
            break

        if is_first_gpu():
            # 记录训练日志
            logger.info(f'Train epoch: {epoch}, lr: {round(scheduler.get_lr()[0], 6) : .6f}, (loss) ' + str(model.avg_meters))

        if epoch % config.MISC.SAVE_FREQ == 0 or epoch == opt.epochs:  # 最后一个epoch要保存一下
            if is_first_gpu():
                model.save(epoch)

        # 训练时验证
        if not opt.no_val and epoch % config.MISC.VAL_FREQ == 0:
            if is_first_gpu():
                model.eval()
                eval_mAP(model, val_dataloader, epoch, writer, logger, log_root, data_name='val')
                model.train()

        if scheduler is not None:
            scheduler.step()

        # if is_distributed():
        #    dist.barrier()

    if is_distributed():
        dist.destroy_process_group()


