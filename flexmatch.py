import sys, os, copy, random, argparse, math
import numpy as np

import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets
from torchvision import transforms
from colorama import Fore
from tqdm import tqdm

from fixmatch import * 
PARAMETER_MAX = 10

mean_cifar10 = (0.4914, 0.4822, 0.4465)
std_cifar10 = (0.2471, 0.2345, 0.2616)

def flexmatch_augment_pool():
    
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs


class RandAugmentMC(object):
    
    def __init__(self, n, m):

        assert n >= 1
        assert 1 <= m <= 10
        
        self.n = n
        self.m = m
        self.augment_pool = flexmatch_augment_pool()
    
    def __call__(self, img):

        ops = random.choices(self.augment_pool, k=self.n)
        
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)

        img = CutoutAbs(img, int(32*0.5))
        
        return img
    
# Weak augmentation & Strong augmentation

class TransformFlexMatch(object):
    
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

# trainer를 정의
class FlexMatchTrainer():

    def __init__(self, args, c_threshold):

        self.args = args
        self.c_threshold = c_threshold
        
        root_dir = '/content/FlexMatch' 
        data_dir = os.path.join(root_dir, 'data') 
        
        self.experiment_dir = os.path.join(root_dir, 'results')
        os.makedirs(self.experiment_dir, exist_ok=True)

        name_exp = "_".join([str(self.args.n_labeled), str(self.args.T)]) 
        self.experiment_dir = os.path.join(self.experiment_dir, name_exp)
        os.makedirs(self.experiment_dir, exist_ok=True)

        print("==> Preparing CIFAR10 dataset")
        labeled_set, unlabeled_set, val_set, test_set = get_cifar10(self.args, data_dir=data_dir)
              
        self.labeled_loader = DataLoader(
            labeled_set,
            sampler=RandomSampler(labeled_set), 
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
            sampler=SequentialSampler(val_set), 
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

        print("==> Preparing WideResNet")
        self.model = WideResNet(self.args.n_classes).to(self.args.cuda)
        
        self.model.zero_grad()
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.cuda)

        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 
        self.optimizer = torch.optim.SGD(grouped_parameters, lr=self.args.lr,
                            momentum=0.9, nesterov=self.args.nesterov)

        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                    self.args.warmup,
                                                    self.args.total_steps)
        
        if self.args.use_ema:  
            self.ema_model = WeightEMA(self.model, self.args.ema_decay)
        
        self.writer = SummaryWriter(self.experiment_dir)

        
    def train(self, epoch):
    
        losses_t, losses_x, losses_u, mask_probs = 0.0, 0.0, 0.0, 0.0
        
        self.model.train()

        iter_labeled = iter(self.labeled_loader)
        iter_unlabeled = iter(self.unlabeled_loader)
        
        over_threshold_count = [0] * 10 
        under_threshold_count = 0 
        with tqdm(**get_tqdm_config(total=self.args.eval_step,
                leave=True, color='blue')) as pbar:
            
            for batch_idx in range(self.args.eval_step): 

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
                
                inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s), dim=0).to(self.args.cuda)
                targets_x = targets_x.type(torch.LongTensor)
                targets_x = targets_x.to(self.args.cuda)
                
                logits = self.model(inputs) 
                
                
                logits_x = logits[:real_B]
                logits_u_w, logits_u_s = logits[real_B:].chunk(2)
                del(logits)
                loss_x = F.cross_entropy(logits_x, targets_x, reduction='mean')

                pseudo_labels = torch.softmax(logits_u_w.detach()/self.args.T, dim=-1) 
                max_prob, targets_u = torch.max(pseudo_labels, dim=-1)
                mask = torch.tensor([max_prob[idx].ge(self.c_threshold[idx]).float() for idx in targets_u])
                
                for mask_value, class_idx in zip(mask, targets_u):
                    if mask_value == 0:
                        under_threshold_count += 1

                    elif mask_value == 1:
                        over_threshold_count[class_idx] += 1

                logits_u_s = logits_u_s.to(self.args.cuda)
                targets_u = targets_u.to(self.args.cuda)
                mask = mask.to(self.args.cuda)
                loss_u = (F.cross_entropy(logits_u_s, targets_u, reduction='none')*mask).mean()
                
                loss = loss_x + self.args.lambda_u * loss_u
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.args.use_ema:
                    self.ema_model.step(self.model)
                
                self.model.zero_grad()

                if max(over_threshold_count) < under_threshold_count: ### Warmup
                    for idx in range(10):
                        beta = over_threshold_count[idx] / max(max(over_threshold_count), under_threshold_count)
                        self.c_threshold[idx] = (beta/(2-beta)) * self.args.threshold

                else:
                    for idx in range(10):
                        beta = over_threshold_count[idx] / max(over_threshold_count) 
                        self.c_threshold[idx] = (beta/(2-beta)) * self.args.threshold
                
                losses_x += loss_x.item()
                losses_u += loss_u.item()
                losses_t += loss.item()
                mask_probs += max_prob.mean().item()
                
                self.writer.add_scalars(
                    'Training steps', {
                        'Total_loss': losses_t/(batch_idx+1),
                        'Labeled_loss':losses_x/(batch_idx+1),
                        'Unlabeled_loss':losses_u/(batch_idx+1),
                        'Mask probs': mask_probs/(batch_idx+1)
                    }, global_step=epoch*self.args.batch_size+batch_idx
                )

                pbar.set_description(
                    '[Train(%4d/ %4d)-Total: %.3f|Labeled: %.3f|Unlabeled: %.3f]'%(
                        (batch_idx+1), self.args.eval_step,
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
            
        print(f'Threshold_per_class: {self.c_threshold}')
        
        return losses_t/(batch_idx+1), losses_x/(batch_idx+1), losses_u/(batch_idx+1), self.c_threshold

    
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
                
                targets = targets.type(torch.LongTensor)
                inputs, targets = inputs.to(self.args.cuda), targets.to(self.args.cuda)
                targets = targets.type(torch.LongTensor).to(self.args.cuda)
                
                outputs = self.ema_model.ema(inputs)
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
    
def FlexMatch_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n-labeled', type=int, default=4000) # labeled dat의 수
    parser.add_argument('--n-classes', type=int, default=10) # Class의 수
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")

    parser.add_argument('--batch-size', type=int, default=64) # 배치 사이즈
    parser.add_argument('--total-steps', default=2**14, type=int) # iteration마다 Scheduler가 적용되기에, Epoch가 아닌, Total-step을 정의
    parser.add_argument('--eval-step', type=int, default=1024) # Evaluation Step의 수
    parser.add_argument('--lr', type=float, default=0.03) # Learning rate
    parser.add_argument('--weight-decay', type=float, default=5e-4) # Weight Decay 정도
    parser.add_argument('--nesterov', action='store_true', default=True) # Nesterov Momentum
    parser.add_argument('--warmup', type=float, default=0.0) # Warmup 정도

    parser.add_argument('--use-ema', action='store_true', default=True) # EMA 사용여부
    parser.add_argument('--ema-decay', type=float, default=0.999) # EMA에서 Decay 정도

    parser.add_argument('--mu', type=int, default=7) # Labeled data의 mu배를 Unlabeled 데이터의 개수로 정의하기 위한 함수 (근데 위 Trainer에서는 안 쓰임)
    parser.add_argument('--T', type=float, default=1.0) # Sharpening 함수에 들어가는 하이퍼 파라미터

    parser.add_argument('--threshold', type=float, default=0.95) # Pseudo-Labeling이 진행되는 Threshold 정의
    parser.add_argument('--lambda-u', type=float, default=1.0) # Loss 가중치 정도
    return parser


def main():
    parser = FlexMatch_parser()
    args = parser.parse_args([])
    args.cuda = torch.device("cuda:0")
    args.epochs = math.ceil(args.total_steps/args.eval_step)
    
    class_threshold = [args.threshold] * 10 ### 각 Class 별 Threshold를 저장할 공간 형성

    trainer = FlexMatchTrainer(args, class_threshold)

    best_loss = np.inf
    losses, losses_x, losses_u = [], [], []
    
    train_losses, train_top1s, train_top5s = [], [], []
    val_losses, val_top1s, val_top5s = [], [], []
    test_losses, test_top1s, test_top5s = [], [], []
    results = {'loss': [], 'test_acc_top1': [], 'test_acc_top5': []}       

    for epoch in range(1, args.epochs+1, 1):
        loss, loss_x, loss_u, class_threshold = trainer.train(epoch)
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
        
        results['loss'].append(loss)
        results['test_acc_top1'].append(top1)
        results['test_acc_top5'].append(top5)
        
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        savepath = 'results/flexmatch'
        os.makedirs(savepath, exist_ok=True)
        data_frame.to_csv(os.path.join(savepath,'statistics.csv'), index_label='epoch')

        if loss < best_loss:
            best_loss = loss
            torch.save(trainer.model, os.path.join(trainer.experiment_dir, 'model.pth'))
            torch.save(trainer.ema_model, os.path.join(trainer.experiment_dir, 'ema_model.pth'))

        loss, top1, top5 = trainer.validate(epoch, 'Test ')
        test_losses.append(loss)
        test_top1s.append(top1)
        test_top5s.append(top5)

        torch.save(trainer.model, os.path.join(trainer.experiment_dir, 'checkpooint_model.pth'))
        torch.save(trainer.ema_model, os.path.join(trainer.experiment_dir, 'checkpoint_ema_model.pth'))
        
# 실행
if __name__=="__main__":
    main()