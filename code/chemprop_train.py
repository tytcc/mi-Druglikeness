#!/Anaconda3/envs/ccjpython/python3
# Created on 2021/6/23 13:31
# Author:tyty
import logging
from typing import Callable

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch import optim
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR
# from chemprop.utils import get_metric_func
from chemprop.train.metrics import get_metric_func

def train(model: nn.Module,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          sou=False,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data_loader: A MoleculeDataLoader.
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print

    train_loss=[]
    param_norm=[]
    gradient_norm=[]
    l_n_iter=[]

    model.train()
    loss_sum, iter_count = 0, 0
    if args.num_tasks == 2:
        cls0loss_sum,cls1loss_sum=0,0

    for batch in tqdm(data_loader, total=len(data_loader)):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, target_batch = batch.batch_graph(), batch.features(), batch.targets()  ##在MoleculeDataset的分类中把graph算好了
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch]) # add a mask for None target
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]) # process the targets chanege the "None" to 0

        # Run model
        model.zero_grad()
        preds = model(mol_batch, features_batch) # no softmax

        # Move tensors to correct device
        mask = mask.to(preds.device)  #去除nan
        targets = targets.to(preds.device)
        class_weights = torch.ones(targets.shape, device=preds.device)

        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
        else:
            loss = loss_func(preds, targets) * class_weights * mask

        if sou == False:
            if args.num_tasks ==2:
                cls0loss_sum+=(loss[:,0].sum() / mask[:,0].sum()).item()
                cls1loss_sum+=(loss[:,1].sum() / mask[:,1].sum()).item()
        else:
            if args.sou_num_tasks ==2:
                cls0loss_sum+=(loss[:,0].sum() / mask[:,0].sum()).item()
                cls1loss_sum+=(loss[:,1].sum() / mask[:,1].sum()).item()

        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        # iter_count += len(batch)
        iter_count += 1

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)
        l_n_iter.append(n_iter)
        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0: # default log_frequency = 10
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                if sou==False:
                    if args.num_tasks == 2:
                        writer.add_scalar('cls0loss', cls0loss_sum/iter_count, n_iter)
                        writer.add_scalar('cls1loss', cls1loss_sum/iter_count, n_iter)
                    writer.add_scalar('train_loss', loss_avg, n_iter)

                    writer.add_scalar('param_norm', pnorm, n_iter)
                    writer.add_scalar('gradient_norm', gnorm, n_iter)
                    for i, lr in enumerate(lrs):
                        writer.add_scalar(f'learning_rate_{i}', lr, n_iter)
                else:
                    if args.sou_num_tasks == 2:
                        writer.add_scalar('sou cls0loss', cls0loss_sum / iter_count, n_iter)
                        writer.add_scalar('sou cls1loss', cls1loss_sum / iter_count, n_iter)
                    writer.add_scalar('sou train_loss', loss_avg, n_iter)

                    writer.add_scalar('sou param_norm', pnorm, n_iter)
                    writer.add_scalar('sou gradient_norm', gnorm, n_iter)
                    for i, lr in enumerate(lrs):
                        writer.add_scalar(f'sou learning_rate_{i}', lr, n_iter)
            loss_sum, iter_count = 0, 0
            train_loss.append(loss_avg)
            param_norm.append(pnorm)
            gradient_norm.append(gnorm)

    return n_iter, l_n_iter,train_loss,param_norm,gradient_norm

def T_scaling(logits, temperature):
  temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
  return logits / temperature

def temperature_train(model: nn.Module,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          args: TrainArgs) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data_loader: A MoleculeDataLoader.
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """

    temperature = nn.Parameter(torch.ones(1).cuda())
    optimizer_tem = optim.LBFGS([temperature], lr=0.1, max_iter=50) #line_search_fn='strong_wolfe'
    l_n_iter=[]


    preds_list=[]
    targets_list=[]
    class_weights_list=[]
    mask_list=[]
    temps=[]
    losses=[]
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            # Prepare batch
            batch: MoleculeDataset
            # mol_batch, features_batch, target_batch = batch.batch_graph(), batch.features(), batch.targets()  ##在MoleculeDataset的分类中把graph算好了
            mol_batch, features_batch, target_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch, data_weights_batch = \
            batch.batch_graph(), batch.features(), batch.targets(), batch.atom_descriptors(), \
            batch.atom_features(), batch.bond_features(), batch.data_weights()
            mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch]) # add a mask for None target
            targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]) # process the targets chanege the "None" to 0

            # Run model
            preds = model(mol_batch, features_batch,train_phase=False) # no softmax
            # Move tensors to correct device
            mask = mask.to(preds.device)  #去除nan
            targets = targets.to(preds.device)
            class_weights = torch.ones(targets.shape, device=preds.device)

            preds_list.append(preds)
            targets_list.append(targets)
            class_weights_list.append(class_weights)
            mask_list.append(mask)

    def _eval():
        count_list=[]
        loss_list=[]
        optimizer_tem.zero_grad()
        loss_sum, count = 0, 0
        for m in range(len(preds_list)):
            preds=preds_list[m]
            targets=targets_list[m]
            class_weights=class_weights_list[m]
            mask=mask_list[m]
            if args.dataset_type == 'multiclass':
                targets = targets.long()
                loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
            else:
                preds=T_scaling(preds,temperature)
                loss = loss_func(preds, targets) * class_weights * mask
            loss_list.append(loss)
            count_list.append(mask)


        loss_avg = torch.cat(loss_list).sum()/torch.cat(count_list).sum()
        loss_avg.backward()
        temps.append(temperature.item())
        # losses.append(loss_avg.item())
        return loss_avg
    optimizer_tem.step(_eval)

    return temps


def clswt_train(model: nn.Module,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          class_weights = None,
          sou=False,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch. add the classweight

    :param model: Model.
    :param data_loader: A MoleculeDataLoader.
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print

    train_loss=[]
    param_norm=[]
    gradient_norm=[]
    l_n_iter=[]

    model.train()
    loss_sum, iter_count = 0, 0
    if args.num_tasks == 2:
        cls0loss_sum,cls1loss_sum=0,0

    for batch in tqdm(data_loader, total=len(data_loader)):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, target_batch = batch.batch_graph(), batch.features(), batch.targets()  ##在MoleculeDataset的分类中把graph算好了
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch]) # add a mask for None target
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]) # process the targets chanege the "None" to 0

        # Run model
        model.zero_grad()
        preds = model(mol_batch, features_batch) # no softmax

        # Move tensors to correct device
        mask = mask.to(preds.device)  #去除nan
        targets = targets.to(preds.device)
        if class_weights is not None:
            target_onehot =nn.functional.one_hot(targets.long(),num_classes=2).float().squeeze(dim=1)
            weights=torch.tensor(class_weights).to(preds.device)
            # print(target_onehot.shape,weights.shape)
            # print(target_onehot)
            # print(weights)

            weights= torch.mm(target_onehot, weights.unsqueeze(dim=1)).squeeze(dim=1)
        else:
            weights = torch.ones(targets.shape, device=preds.device)

        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * weights * mask
        else:
            # loss,pt= loss_func(preds, targets)
            loss=loss_func(preds,targets)
            loss=loss * weights * mask

        if sou == False:
            if args.num_tasks ==2:
                cls0loss_sum+=(loss[:,0].sum() / mask[:,0].sum()).item()
                cls1loss_sum+=(loss[:,1].sum() / mask[:,1].sum()).item()
        else:
            if args.sou_num_tasks ==2:
                cls0loss_sum+=(loss[:,0].sum() / mask[:,0].sum()).item()
                cls1loss_sum+=(loss[:,1].sum() / mask[:,1].sum()).item()

        loss = loss.sum() / (mask*weights).sum()

        loss_sum += loss.item()
        # iter_count += len(batch)
        iter_count += 1

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)
        l_n_iter.append(n_iter)
        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0: # default log_frequency = 10
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                if sou==False:
                    if args.num_tasks == 2:
                        writer.add_scalar('cls0loss', cls0loss_sum/iter_count, n_iter)
                        writer.add_scalar('cls1loss', cls1loss_sum/iter_count, n_iter)
                    writer.add_scalar('train_loss', loss_avg, n_iter)

                    writer.add_scalar('param_norm', pnorm, n_iter)
                    writer.add_scalar('gradient_norm', gnorm, n_iter)
                    for i, lr in enumerate(lrs):
                        writer.add_scalar(f'learning_rate_{i}', lr, n_iter)
                else:
                    if args.sou_num_tasks == 2:
                        writer.add_scalar('sou cls0loss', cls0loss_sum / iter_count, n_iter)
                        writer.add_scalar('sou cls1loss', cls1loss_sum / iter_count, n_iter)
                    writer.add_scalar('sou train_loss', loss_avg, n_iter)

                    writer.add_scalar('sou param_norm', pnorm, n_iter)
                    writer.add_scalar('sou gradient_norm', gnorm, n_iter)
                    for i, lr in enumerate(lrs):
                        writer.add_scalar(f'sou learning_rate_{i}', lr, n_iter)
            loss_sum, iter_count = 0, 0
            train_loss.append(loss_avg)
            param_norm.append(pnorm)
            gradient_norm.append(gnorm)

    return n_iter, l_n_iter,train_loss,param_norm,gradient_norm



def clsandreg_train(model: nn.Module,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data_loader: A MoleculeDataLoader.
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print

    train_loss=[]
    param_norm=[]
    gradient_norm=[]
    l_n_iter=[]

    model.train()
    loss_sum, iter_count = 0, 0

    for batch in tqdm(data_loader, total=len(data_loader)):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, target_batch = batch.batch_graph(), batch.features(), batch.targets()  ##在MoleculeDataset的分类中把graph算好了
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch]) # add a mask for None target
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]) # process the targets chanege the "None" to 0

        #remove the regression task targets
        if mask.shape[1] > 1:
            mask[:,1:]=0
        # Run model
        model.zero_grad()
        preds = model(mol_batch, features_batch) # no softmax

        # Move tensors to correct device
        mask = mask.to(preds.device)
        targets = targets.to(preds.device)
        class_weights = torch.ones(targets.shape, device=preds.device)

        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
        else:
            loss = loss_func(preds, targets) * class_weights * mask
        cls_loss = loss.sum() / mask.sum()

        if mask.shape[1] > 1:
            reg_loss_func=nn.MSELoss(reduction='none')
            reg_loss=[]
            for i in range(1,preds.shape[1]):
                reg_loss.append((reg_loss_func(preds[:,i],targets[:,i])).sum()/len(preds))

            reg_loss=torch.tensor(reg_loss).sum()
            loss=0.2* cls_loss + 1 * reg_loss
            loss_sum +=loss.item()
            loss.backward()
        else:
            loss_sum+=cls_loss.item()
            cls_loss.backward()
        # iter_count += len(batch)
        iter_count += 1


        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)
        l_n_iter.append(n_iter)
        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0: # default log_frequency = 10
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                writer.add_scalar('cls_loss',cls_loss.item(),n_iter)
                writer.add_scalar('reg_loss',reg_loss.item(),n_iter)
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

            train_loss.append(loss_avg)
            param_norm.append(pnorm)
            gradient_norm.append(gnorm)

    return n_iter, l_n_iter,train_loss,param_norm,gradient_norm


def multitrain(model: nn.Module,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          sou: bool = True,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data_loader: A MoleculeDataLoader.
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print

    model.train()
    loss_sum, iter_count = 0, 0

    for batch in tqdm(data_loader, total=len(data_loader)):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, target_batch = batch.batch_graph(), batch.features(), batch.targets()  ##在MoleculeDataset的分类中把graph算好了
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])  # add a mask for None target
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])  # process the targets chanege the "None" to 0

        # Run model
        model.zero_grad()
        preds = model(mol_batch, features_batch,sou=sou)  # no softmax

        # Move tensors to correct device
        mask = mask.to(preds.device)
        targets = targets.to(preds.device)
        class_weights = torch.ones(targets.shape, device=preds.device)

        if sou == True:
            if args.sou_dataset_type == 'multiclass':
                targets = targets.long()
                loss = torch.cat(
                    [loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in
                     range(preds.size(1))], dim=1) * class_weights * mask
            else:
                loss = loss_func(preds, targets) * class_weights * mask
        else:
            if args.dataset_type == 'multiclass':
                targets = targets.long()
                loss = torch.cat(
                    [loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in
                     range(preds.size(1))], dim=1) * class_weights * mask
            else:
                loss = loss_func(preds, targets) * class_weights * mask

        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += len(batch)

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0

            if sou == False:
                lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
                debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')
                if writer is not None:
                        writer.add_scalar('train_loss', loss_avg, n_iter)
                        writer.add_scalar('param_norm', pnorm, n_iter)
                        writer.add_scalar('gradient_norm', gnorm, n_iter)
                        for i, lr in enumerate(lrs):
                            writer.add_scalar(f'learning_rate_{i}', lr, n_iter)
            else:
                lrs_str = ', '.join(f'sou_lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
                debug(f'sou_Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')
                if writer is not None:
                    writer.add_scalar('sou_train_loss', loss_avg, n_iter)
                    writer.add_scalar('sou_param_norm', pnorm, n_iter)
                    writer.add_scalar('sou_gradient_norm', gnorm, n_iter)
                    for i, lr in enumerate(lrs):
                        writer.add_scalar(f'sou_learning_rate_{i}', lr, n_iter)
    return n_iter