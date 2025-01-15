# -*-coding:utf-8-*-


import time
import os

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lib.dataset.dataietr import AlaskaDataIter

from train_config import config as cfg
# from lib.dataset.dataietr import DataIter


from lib.core.base_trainer.metric import *
from lib.core.base_trainer.model import Net



class KLDivLossWithLogits(nn.KLDivLoss):

    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, y, t,weight=None, trainning=False):
        y = nn.functional.log_softmax(y,  dim=1)


        loss = super().forward(y, t)
        if weight is not None:
            weight = weight.unsqueeze(-1)
            loss=loss*weight

        if not trainning:
            loss=loss.sum() / (torch.sum(weight)+1e-6)
        else:
            loss=loss.sum()/loss.size(0)
        return loss
class Train(object):
    """Train class.
    """

    def __init__(self,
                 train_df,
                 val_df,
                 fold):

        self.ddp=False


        if self.ddp:
            torch.distributed.init_process_group(backend="nccl")
            self.train_generator = AlaskaDataIter(train_df, training_flag=True, shuffle=False)
            self.train_ds = DataLoader(self.train_generator,
                                       cfg.TRAIN.batch_size,
                                       num_workers=cfg.TRAIN.process_num,
                                       sampler=DistributedSampler(self.train_generator,
                                                                  shuffle=True))

            self.val_generator = AlaskaDataIter(val_df, training_flag=False, shuffle=False)

            self.val_ds = DataLoader(self.val_generator,
                                     cfg.TRAIN.validatiojn_batch_size,
                                     num_workers=cfg.TRAIN.process_num,
                                     sampler=DistributedSampler(self.val_generator,
                                                                shuffle=False))
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)


        else:
            self.train_generator = AlaskaDataIter(train_df, training_flag=True, shuffle=False)
            self.train_ds = DataLoader(self.train_generator,
                                       cfg.TRAIN.batch_size,
                                       num_workers=cfg.TRAIN.process_num,shuffle=True)

            self.val_generator = AlaskaDataIter(val_df, training_flag=False, shuffle=False)

            self.val_ds = DataLoader(self.val_generator,
                                     cfg.TRAIN.validatiojn_batch_size,
                                     num_workers=cfg.TRAIN.process_num,shuffle=False)

            self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.fold = fold

        self.init_lr = cfg.TRAIN.init_lr
        self.warup_step = cfg.TRAIN.warmup_step
        self.epochs = cfg.TRAIN.epoch
        self.batch_size = cfg.TRAIN.batch_size
        self.l2_regularization = cfg.TRAIN.weight_decay_factor

        self.early_stop = cfg.MODEL.early_stop

        self.accumulation_step = cfg.TRAIN.accumulation_batch_size // cfg.TRAIN.batch_size


        self.gradient_clip = cfg.TRAIN.gradient_clip

        self.save_dir=cfg.MODEL.model_path
        #### make the device



        self.model = Net().to(self.device)
        self.load_weight()

        if 'Adamw' in cfg.TRAIN.opt:

            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=self.init_lr, eps=1.e-5,
                                               weight_decay=self.l2_regularization)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.init_lr,
                                             momentum=0.9,
                                             weight_decay=self.l2_regularization)



        if self.ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[local_rank],
                                                                   output_device=local_rank,
                                                                   find_unused_parameters=True)
        else:
            self.model=nn.DataParallel(self.model)

        ###control vars
        self.iter_num = 0

        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='max', patience=5,
        #                                                             min_lr=1e-6,factor=0.5,verbose=True)

        if cfg.TRAIN.lr_scheduler=='cos':
            logger.info('lr_scheduler.CosineAnnealingLR')
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        self.epochs,
                                                                        eta_min=1.e-7)
        else:
            logger.info('lr_scheduler.ReduceLROnPlateau')
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        mode='max',
                                                                        patience=5,
                                                                        min_lr=1e-7,
                                                                        factor=cfg.TRAIN.lr_scheduler_factor,
                                                                        verbose=True)

        self.criterion = KLDivLossWithLogits()

        self.scaler = torch.cuda.amp.GradScaler()

    def custom_loop(self):
        """Custom training and testing loop.
        Args:
          train_dist_dataset: Training dataset created using strategy.
          test_dist_dataset: Testing dataset created using strategy.
          strategy: Distribution strategy.
        Returns:
          train_loss, train_accuracy, test_loss, test_accuracy
        """

        def distributed_train_epoch(epoch_num):

            summary_loss = AverageMeter()

            self.model.train()

            for waves,weight, label in self.train_ds:

                if epoch_num < 10:
                    ###excute warm up in the first epoch
                    if self.warup_step > 0:
                        if self.iter_num < self.warup_step:
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = self.iter_num / float(self.warup_step) * self.init_lr
                                lr = param_group['lr']

                            logger.info('warm up with learning rate: [%f]' % (lr))

                start = time.time()

                data = waves.to(self.device).float()
                weight= weight.to(self.device).float()
                label = label.to(self.device).float()


                batch_size = data.shape[0]


                with torch.cuda.amp.autocast(enabled=cfg.TRAIN.mix_precision):
                    predictions = self.model(data)
                    current_loss = self.criterion(predictions, label,weight,trainning=True)

                summary_loss.update(current_loss.detach().item(), batch_size)

                self.scaler.scale(current_loss).backward()


                if ((self.iter_num + 1) % self.accumulation_step) == 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.gradient_clip>0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip, norm_type=2)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                self.iter_num += 1
                time_cost_per_batch = time.time() - start

                images_per_sec = cfg.TRAIN.batch_size / time_cost_per_batch

                if self.iter_num % cfg.TRAIN.log_interval == 0:
                    log_message = '[fold %d], ' \
                                  'Train Step %d, ' \
                                  'summary_loss: %.6f, ' \
                                  'time: %.6f, ' \
                                  'speed %d images/persec' % (
                                      self.fold,
                                      self.iter_num,
                                      summary_loss.avg,
                                      time.time() - start,
                                      images_per_sec)
                    logger.info(log_message)


            return summary_loss

        def distributed_test_epoch(epoch_num):


            summary_loss = AverageMeter()
            self.model.eval()


            oof_pre=[]
            oof_weight = []
            oof_gt=[]
            with torch.no_grad():
                for  (waves,weight, labels) in tqdm(self.val_ds):

                    data = waves.to(self.device).float()
                    weight= weight.to(self.device).float()
                    labels = labels.to(self.device).float()
                    batch_size = data.shape[0]

                    predictions = self.model(data)

                    current_loss = self.criterion(predictions, labels,weight,trainning=False)

                    summary_loss.update(current_loss.detach().item(), batch_size)

                    oof_pre.append(torch.softmax(predictions,-1).detach().cpu().numpy())
                    oof_gt.append(labels.detach().cpu().numpy())
                    oof_weight.append(weight.detach().cpu().numpy())
            return  summary_loss,oof_pre,oof_gt,oof_weight



        best_distance = 9999999
        not_improvement = 0
        for epoch in range(self.epochs):

            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']
            logger.info('learning rate: [%f]' % (lr))
            t = time.time()

            summary_loss = distributed_train_epoch(epoch)
            train_epoch_log_message = '[fold %d], ' \
                                      '[RESULT]: TRAIN. Epoch: %d,' \
                                      ' summary_loss: %.5f,' \
                                      ' time:%.5f' % (
                                          self.fold,
                                          epoch,
                                          summary_loss.avg,
                                          (time.time() - t))
            logger.info(train_epoch_log_message)

           



            if epoch % cfg.TRAIN.test_interval == 0:
                summary_loss,oof_pre,oof_gt,oof_weight = distributed_test_epoch(epoch)

                val_epoch_log_message = '[fold %d], ' \
                                        '[RESULT]: VAL. Epoch: %d,' \
                                        ' val_loss: %.5f,' \
                                        ' time:%.5f' % (
                                            self.fold,
                                            epoch,
                                            summary_loss.avg,
                                            (time.time() - t))
                logger.info(val_epoch_log_message)



            if cfg.TRAIN.lr_scheduler=='cos':
                self.scheduler.step()
            else:
                self.scheduler.step(roc_auc_score.avg)

            #### save model
            if not os.access(cfg.MODEL.model_path, os.F_OK):
                os.mkdir(cfg.MODEL.model_path)
            ###save the best auc model

            #### save the model every end of epoch
            current_model_saved_name = self.save_dir+'/fold%d_epoch_%d_val_loss_%.6f.pth' % (self.fold,
                                                                                                epoch,
                                                                                                summary_loss.avg)

            logger.info('A model saved to %s' % current_model_saved_name)
            torch.save(self.model.module.state_dict(), current_model_saved_name)


            # save_checkpoint({
            #           'state_dict': self.model.state_dict(),
            #           },iters=epoch,tag=current_model_saved_name)


            if summary_loss.avg < best_distance:
                best_distance = summary_loss.avg
                logger.info(' best metric score update as %.6f' % (best_distance))
                logger.info(' bestmodel update as %s' % (current_model_saved_name))
                not_improvement=0
                self.oof_pre=np.concatenate(oof_pre,axis=0)
                self.oof_gt=np.concatenate(oof_gt,axis=0)
                self.oof_weight = np.concatenate(oof_weight, axis=0)

            else:
                not_improvement += 1

            if not_improvement >= self.early_stop and self.early_stop>-1:
                logger.info(' best metric score not improvement for %d, break' % (self.early_stop))
                break

            torch.cuda.empty_cache()

    def load_weight(self):
        if cfg.TRAIN.stage ==2:
            stage1_weight='avg_fold%d.pth'%self.fold
            state_dict = torch.load(stage1_weight, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)




