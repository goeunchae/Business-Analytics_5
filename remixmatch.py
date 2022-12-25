import os, math, sys, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from tqdm import tqdm
from colorama import Fore
from torchvision import datasets 
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from mixmatch import BasicBlock, NetworkBlock, WideResNet, accuracy, get_tqdm_config
from fixmatch import RandAugmentMC, CIFAR10_SSL 

PARAMETER_MAX = 10

mean_cifar10 = (0.4914, 0.4822, 0.4465)
std_cifar10 = (0.2471, 0.2345, 0.2616)

class TransformReMixMatch(object):
    
    def __init__(self, mean=mean_cifar10, std=std_cifar10):
        
        self.weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                padding=int(32*0.125),
                                padding_mode='reflect')
        ])

        self.strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                padding=int(32*0.125),
                                padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)
        ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]) 
    
    
    def __call__(self, x):
        
        weak = self.weak_transform(x)
        strong = self.strong_transform(x)

        return self.normalize(weak), self.normalize(strong)
    

def split_labeled_unlabeled(args, labels):
    
    label_per_class = args.n_labeled // args.n_classes
    labels = np.array(labels, dtype=int)
    indice_labeled, indice_unlabeled, indice_val = [], [], []

    for i in range(10):
        indice_tmp = np.where(labels==i)[0]

        indice_labeled.extend(indice_tmp[: label_per_class])
        indice_unlabeled.extend(indice_tmp[label_per_class: -500])
        indice_val.extend(indice_tmp[-500: ])
    
    for i in [indice_labeled, indice_unlabeled, indice_val]:
        np.random.shuffle(i)
    
    return np.array(indice_labeled), np.array(indice_unlabeled), np.array(indice_val)
    
def get_cifar10(args, data_dir):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_cifar10, std=std_cifar10)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_cifar10, std=std_cifar10)
    ])

    base_dataset = datasets.CIFAR10(data_dir, train=True, download=True)

    indice_labeled, indice_unlabeled, indice_val = split_labeled_unlabeled(args, base_dataset.targets)
    
    labeled_dataset = CIFAR10_SSL(
        data_dir, indice_labeled, train=True,
        transform=transform_labeled
    )

    unlabeled_dataset = CIFAR10_SSL(
        data_dir, indice_unlabeled, train=True,
        transform=TransformReMixMatch(mean=mean_cifar10, std=std_cifar10)
    )

    val_dataset = CIFAR10_SSL(
        data_dir, indice_val, train=True, transform=transform_val, download=False
    )

    test_dataset = datasets.CIFAR10(
        data_dir, train=False, transform=transform_val, download=False
    )
    
    return labeled_dataset, unlabeled_dataset, val_dataset, test_dataset

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

class ReMixMatchTrainer():
    def __init__(self, args):
        self.args = args

        root_dir = '/content/ReMixMatch' # PROJECT directory
        data_dir = os.path.join(root_dir, 'data') ### Data Directory
        self.experiment_dir = os.path.join(root_dir, 'results') # 학습된 모델을 저장할 폴더 경로 정의 및 폴더 생성
        os.makedirs(self.experiment_dir, exist_ok=True)

        name_exp = "_".join([str(self.args.n_labeled), str(self.args.T)]) # 주요 하이퍼 파라미터로 폴더 저장 경로 지정 
        self.experiment_dir = os.path.join(self.experiment_dir, name_exp)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Data
        print("==> Preparing CIFAR10 dataset")
        labeled_set, unlabeled_set, val_set, test_set = get_cifar10(self.args, data_dir=data_dir)
                 
        self.labeled_loader = DataLoader(
            labeled_set,
            sampler=RandomSampler(labeled_set), ### RandomSampler: DataLoader(shuffle=True) 와 동일한 역할
            batch_size=self.args.batch_size,
            num_workers=0,
            drop_last=True
        )

        self.unlabeled_loader = DataLoader(
            unlabeled_set,
            sampler=RandomSampler(unlabeled_set),
            batch_size=self.args.batch_size,
            num_workers=0,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_set,
            sampler=SequentialSampler(val_set), ### SequentialSampler: DataLoader(shuffle=False) 와 동일한 역할
            batch_size=self.args.batch_size,
            num_workers=0,
            drop_last=True
        )

        self.test_loader = DataLoader(
            test_set,
            sampler=SequentialSampler(test_set),
            batch_size=self.args.batch_size,
            num_workers=0
        )


        # Build WideResNet
        print("==> Preparing WideResNet")
        self.model = WideResNet(self.args.n_classes).to(self.args.cuda)

        # Define loss functions
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.cuda)

        # Define optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        # 학습 결과를 저장할 Tensorboard 정의
        self.writer = SummaryWriter(self.experiment_dir)

    def create_model(self, ema=False):
        # Build WideResNet & EMA model
        model = WideResNet(num_classes=10)
        model = model.to(self.args.cuda)

        if ema:
            for param in model.parameters():
                param.detach_()
            
        return model
    
    def train(self, epoch):
        # 모델 학습 함수
        losses_t, losses_x, losses_u, ws = 0.0, 0.0, 0.0, 0.0
        self.model.train()

        iter_labeled = iter(self.labeled_loader)
        iter_unlabeled = iter(self.unlabeled_loader)

        with tqdm(**get_tqdm_config(total=self.args.num_iter,
                leave=True, color='blue')) as pbar:
            for batch_idx in range(self.args.num_iter):
              
                try:
                    inputs_x, targets_x = iter_labeled.next()
                except:
                    iter_labeled = iter(self.labeled_loader)
                    inputs_x, targets_x = iter_labeled.next()
                real_B = inputs_x.size(0)

                try:
                    (inputs_u_w, inputs_u_s), _ = iter_unlabeled.next()
                except:
                    iter_unlabeled = iter(self.unlabeled_loader)
                    (inputs_u_w, inputs_u_s), _ = iter_unlabeled.next()
                
                inputs_x, inputs_u_w, inputs_u_s = inputs_x.to(self.args.cuda), inputs_u_w.to(self.args.cuda), inputs_u_s.to(self.args.cuda)
                
                with torch.no_grad():
                    logits_x = self.model(inputs_x)
                    logits_tmp = self.model(inputs_u_w)
                    q = torch.softmax(logits_tmp,dim=1)
                    q = q * (torch.softmax(logits_x, dim=1)).mean()/q.mean()
            
                    pt = q**(1/0.5)
                    targets_u = pt / pt.sum(dim=1, keepdim=True)
                    pseudo_label = targets_u.detach()
                    
                _, targets_u = torch.max(pseudo_label, dim=-1)
                targets_u = targets_u.long()
                
                logits_u_s = self.model(inputs_u_s)
                
                inputs_m = torch.cat((inputs_x,inputs_u_s),dim=0).to(self.args.cuda)
                
                l_mixup = np.random.beta(self.args.alpha, self.args.alpha)
                l_mixup = max(l_mixup, 1-l_mixup)
                
                B = inputs_m.size(0)
                random_idx = torch.randperm(B)

                inputs_a, inputs_b = inputs_m, inputs_m[random_idx]

                mixed_input = l_mixup*inputs_a + (1-l_mixup)*inputs_b

                
                mixed_input = list(torch.split(mixed_input, real_B))
                mixed_input = interleave(mixed_input, real_B)

                logits = [self.model(mixed_input[0])] # for labeled
                for input in mixed_input[1:]:
                    logits.append(self.model(input)) # for unlabeled

                logits = interleave(logits, real_B)
                logits_x = logits[0]
                logits_u = torch.cat(logits[1:], dim=0)
                
                targets_x = targets_x.type(torch.LongTensor)
                targets_x = targets_x.to(self.args.cuda)
                
            
                loss_x = F.cross_entropy(logits_x, targets_x, reduction='mean')
                pseudo_labels = torch.softmax(logits_u.detach()/self.args.T, dim=-1)
                max_prob, targets_u = torch.max(pseudo_labels, dim=-1)
                
                loss_u = (F.cross_entropy(logits_u_s,targets_u,reduction='mean'))
                
                loss = loss_x + self.args.lambda_u * loss_u

                # Backpropagation and Model parameter update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses_x += loss_x.item()
                losses_u += loss_u.item()
                losses_t += loss.item()

                self.writer.add_scalars(
                    'Training steps', {
                        'Total_loss': losses_t/(batch_idx+1),
                        'Labeled_loss':losses_x/(batch_idx+1),
                        'Unlabeled_loss':losses_u/(batch_idx+1),
                        'W values': ws/(batch_idx+1)
                    }, global_step=epoch*self.args.batch_size+batch_idx
                )

                pbar.set_description(
                    '[Train(%4d/ %4d)-Total: %.3f|Labeled: %.3f|Unlabeled: %.3f]'%(
                        (batch_idx+1), self.args.num_iter,
                        losses_t/(batch_idx+1), losses_x/(batch_idx+1), losses_u/(batch_idx+1)
                    )
                )
                pbar.update(1)

            pbar.set_description(
                '[Train(%4d/ %4d)-Total: %.3f|Labeled: %.3f|Unlabeled: %.3f]'%(
                    epoch, self.args.epochs,
                    losses_t/(batch_idx+1), losses_x/(batch_idx+1), losses_u/(batch_idx+1)
                )
            )
        
        return losses_t/(batch_idx+1), losses_x/(batch_idx+1), losses_u/(batch_idx+1)

    @torch.no_grad()
    def validate(self, epoch, phase):

        if phase == 'Train':
            data_loader = self.labeled_loader
            c = 'blue'
        elif phase == 'Valid':
            data_loader = self.val_loader
            c = 'green'
        elif phase == 'Test ':        
            data_loader = self.test_loader
            c = 'red'

        losses = 0.0
        top1s, top5s = [], []

        with tqdm(**get_tqdm_config(total=len(data_loader),
                leave=True, color=c)) as pbar:
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.args.cuda), targets.to(self.args.cuda)
                targets = targets.type(torch.LongTensor).to(self.args.cuda)
        
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)


                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                losses += loss.item()
                top1s.append(prec1)
                top5s.append(prec5)

                self.writer.add_scalars(
                    f'{phase} steps', {
                        'Total_loss': losses/(batch_idx+1),
                        'Top1 Acc': np.mean(top1s),
                        'Top5 Acc': np.mean(top5s)
                    }, global_step=epoch*self.args.batch_size+batch_idx
                )

                pbar.set_description(
                    '[%s-Loss: %.3f|Top1 Acc: %.3f|Top5 Acc: %.3f]'%(
                        phase,
                        losses/(batch_idx+1), np.mean(top1s), np.mean(top5s)
                    )
                )
                pbar.update(1)

            pbar.set_description(
                '[%s(%4d/ %4d)-Loss: %.3f|Top1 Acc: %.3f|Top5 Acc: %.3f]'%(
                    phase,
                    epoch, self.args.epochs,
                    losses/(batch_idx+1), np.mean(top1s), np.mean(top5s)
                )
            )

        return losses/(batch_idx+1), np.mean(top1s), np.mean(top5s)
    
def ReMixMatch_parser():
    parser = argparse.ArgumentParser()
    
    # method arguments
    parser.add_argument('--n-labeled', type=int, default=1024)
    parser.add_argument('--n-classes', type=int, default=10) # Class의 수
    parser.add_argument('--num-iter', type=int, default=1024,
                        help="The number of iteration per epoch")
    parser.add_argument('--alpha', type=float, default=0.75)
    parser.add_argument('--lambda-u', type=float, default=75)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--ema-decay', type=float, default=0.999)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.002)

    return parser

def main():
    parser = ReMixMatch_parser()
    args = parser.parse_args([])
    args.cuda = torch.device("cuda:0")

    trainer = ReMixMatchTrainer(args)
    
    best_loss = np.inf

    losses, losses_x, losses_u = [], [], []
    
    train_losses, train_top1s, train_top5s = [], [], []
    val_losses, val_top1s, val_top5s = [], [], []
    test_losses, test_top1s, test_top5s = [], [], []
    results = {'loss': [], 'test_acc_top1': [], 'test_acc_top5': []}
    
    for epoch in range(1, args.epochs+1, 1):
        loss, loss_x, loss_u = trainer.train(epoch)
        losses.append(loss)
        losses_x.append(loss_x)
        losses_u.append(loss_u)

        loss, top1, top5 = trainer.validate(epoch, 'Train')
        train_losses.append(loss)
        train_top1s.append(top1)
        train_top5s.append(top5)

        loss, top1, top5 = trainer.validate(epoch, 'Valid')
        val_losses.append(loss)
        val_top1s.append(top1)
        val_top5s.append(top5)
        
        results['loss'].append(loss)
        results['test_acc_top1'].append(top1)
        results['test_acc_top5'].append(top5)
        
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        savepath = 'results/remixmatch'
        os.makedirs(savepath, exist_ok=True)
        data_frame.to_csv(os.path.join(savepath,'statistics.csv'), index_label='epoch')

        if loss < best_loss:
            best_loss = loss
            torch.save(trainer.model, os.path.join(trainer.experiment_dir, 'model.pth'))

        loss, top1, top5 = trainer.validate(epoch, 'Test ')
        test_losses.append(loss)
        test_top1s.append(top1)
        test_top5s.append(top5)

        torch.save(trainer.model, os.path.join(trainer.experiment_dir, 'checkpooint_model.pth'))
        
if __name__=="__main__":
    main()