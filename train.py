import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from test import eval_psnr
from scheduler import GradualWarmupScheduler
from adaptive_sampling import full_hr_collate, recovery_scores, sample_batch, reconstruction_loss


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    full_hr = spec['wrapper'].get('args', {}).get('return_full_hr', False)
    workers = spec.get('num_workers', 8)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=workers, pin_memory=True,
        persistent_workers=(workers > 0), collate_fn=full_hr_collate if full_hr else None)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    if (config.get('resume') is not None) and os.path.exists(config.get('resume')):
        sv_file = torch.load(config['resume'],map_location=torch.device('cpu'))
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        optimizer.param_groups[0]['lr'] = config['optimizer']['args']['lr']
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            cosine = CosineAnnealingLR(optimizer, config['epoch_max']-config['warmup_step_lr']['total_epoch'])
            lr_scheduler = GradualWarmupScheduler(optimizer,**config['warmup_step_lr'],after_scheduler=cosine)
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for e in range(1,epoch_start):
            lr_scheduler.step(e)
            #lr_scheduler.step()
        print(epoch_start,optimizer.param_groups[0]['lr'])
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            cosine = CosineAnnealingLR(optimizer, config['epoch_max']-config['warmup_step_lr']['total_epoch'])
            lr_scheduler = GradualWarmupScheduler(optimizer,**config['warmup_step_lr'],after_scheduler=cosine)
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer, \
         epoch):
    model.train()
    train_loss = utils.Averager()
    metric_fn = utils.calc_psnr

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    adaptive = config.get('adaptive_sampling', {})
    
    #num_dataset = 800 # DIV2K
    #iter_per_epoch = int(num_dataset / config.get('train_dataset')['batch_size'] \
    #                    * config.get('train_dataset')['dataset']['args']['repeat'])
    iteration = 0
    pbar = tqdm(train_loader, leave=False, desc='train')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = ([image.cuda(non_blocking=True) for image in v]
                        if isinstance(v, list) else v.cuda(non_blocking=True))

        inp = (batch['inp'] - inp_sub) / inp_div
        weight = None
        if adaptive.get('enabled', False):
            block_grid = tuple(adaptive.get('block_grid', [8, 8]))
            warmup = epoch <= adaptive.get('warmup_epochs', 10)
            uniform_fraction = adaptive.get('uniform_fraction', 0.5)
            if warmup or uniform_fraction == 1:
                scores = inp.new_zeros((inp.shape[0], *block_grid))
            else:
                preview_shape = tuple(adaptive.get('preview_size', [32, 32]))
                coord = utils.make_coord(preview_shape, flatten=False).to(inp)
                coord = coord.unsqueeze(0).expand(inp.shape[0], -1, -1, -1)
                # Preserve target pixel footprint: the preview samples the same SR task.
                # Do not cache gen_feat here: the training pass needs its autograd graph.
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    preview = model(inp, coord, batch['cell'])
                    scores = recovery_scores(preview, block_grid)
                model.train(was_training)
            selected = sample_batch(batch['gt'], scores,
                sample_q=adaptive['sample_q'],
                uniform_fraction=1.0 if warmup else uniform_fraction,
                power=adaptive.get('power', 1.0))
            batch.update(selected)
            weight = batch['sample_weight']
        elif isinstance(batch['gt'], list):
            raise ValueError('return_full_hr requires adaptive_sampling.enabled.')

        pred = model(inp, batch['coord'], batch['cell'], sample_weight=weight)

        gt = (batch['gt'] - gt_sub) / gt_div
        loss = reconstruction_loss(pred, gt, weight,
                                   mode=adaptive.get('loss_measure', 'uniform'))
        #psnr = metric_fn(pred, gt)
        
        # tensorboard
        #writer.add_scalars('loss', {'train': loss.item()}, (epoch-1)*iter_per_epoch + iteration)
        #writer.add_scalars('psnr', {'train': psnr}, (epoch-1)*iter_per_epoch + iteration)
        iteration += 1
        
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None
        pbar.set_description('train {:.4f}'.format(train_loss.item()))
        
    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=True)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer, \
                           epoch)
        if lr_scheduler is not None:
            # lr_scheduler.step()
            lr_scheduler.step(epoch)

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if ((epoch_save is not None) and (epoch % epoch_save == 0)):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch == 1) or ((epoch_val is not None) and (epoch % epoch_val == 0)):
            if n_gpus > 1: #and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_psnr(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))

            log_info.append('val: psnr={:.4f}'.format(val_res))
#             writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='./configs/train_edsr-sronet.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0,1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)

