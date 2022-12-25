import sys, os, copy, random, argparse, math
import numpy as np
import pandas as pd 

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


from mixmatch import BasicBlock, NetworkBlock, WideResNet, accuracy, get_tqdm_config


PARAMETER_MAX = 10

mean_cifar10 = (0.4914, 0.4822, 0.4465)
std_cifar10 = (0.2471, 0.2345, 0.2616)

def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)

# Augmentation 함수들을 정의

def AutoContrast(img, **kwargs):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def CutoutAbs(img, v, **kwargs):
    w, h = img.size
    x0, y0 = np.random.uniform(0, w), np.random.uniform(0, h)
    x0, y0 = int(max(0, x0 - v / 2.)), int(max(0, y0 - v / 2.))

    x1, y1 = int(min(w, x0 + v)), int(min(h, y0 + v))

    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def Equalize(img, **kwargs):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwargs):
    return img


def Invert(img, **kwargs):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

# Augmentation list for RandAugment
def fixmatch_augment_pool():
    
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

# 위에서 구현된 Augmentpool에서 랜덤으로 선정하여 실제 Augmentation을 구현

class RandAugmentMC(object):
    
    def __init__(self, n, m):

        assert n >= 1
        assert 1 <= m <= 10
        
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()
    
    def __call__(self, img):
        
        ops = random.choices(self.augment_pool, k=self.n)
        
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)

        img = CutoutAbs(img, int(32*0.5))
        
        return img

# Generate train data
class CIFAR10_SSL(datasets.CIFAR10):
    
    def __init__(self, root, indexs, train=True,
                transform=None, target_transform=None,
                download=False):
        
        
        super(CIFAR10_SSL, self).__init__(
            root, train=train, transform=transform,
            target_transform=target_transform, download=download
        )

        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
    
    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
# Weak augmentation & Strong augmentation

class TransformFixMatch(object):
    
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
    
# Labeled data와 Unlabeled data를 분리

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
        transform=TransformFixMatch(mean=mean_cifar10, std=std_cifar10)
    )

    val_dataset = CIFAR10_SSL(
        data_dir, indice_val, train=True, transform=transform_val, download=False
    )

    test_dataset = datasets.CIFAR10(
        data_dir, train=False, transform=transform_val, download=False
    )
    
    return labeled_dataset, unlabeled_dataset, val_dataset, test_dataset

# Parameter update with weightEMA
class WeightEMA(object): 

    def __init__(self, model, decay):
        
        self.ema = copy.deepcopy(model)
        self.ema.eval()

        self.decay = decay

        self.ema_has_module = hasattr(self.ema, 'module')

        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def step(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])
                
# Learning rate scheduler
def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps,
    num_cycles=7.0/16.0, last_epoch=-1
    ):
    
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step)/float(max(1, num_warmup_steps))
        
        no_progress = float(current_step-num_warmup_steps)/\
            (float(max(1, num_training_steps-num_warmup_steps)))
        return max(0.0, math.cos(math.pi*num_cycles*no_progress))
    
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

# define trainer
class FixMatchTrainer():
    
    def __init__(self, args):

        self.args = args
        
        root_dir = '/content/FixMatch' ### Project Directory
        data_dir = os.path.join(root_dir, 'data') ### Data Directory
        
        self.experiment_dir = os.path.join(root_dir, 'results') ### 학습된 모델을 저장할 큰 폴더
        os.makedirs(self.experiment_dir, exist_ok=True)

        name_exp = "_".join([str(self.args.n_labeled), str(self.args.T)]) ### 학습된 모델을 저장할 세부 폴더 (하이퍼파라미터로 지정)
        self.experiment_dir = os.path.join(self.experiment_dir, name_exp)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
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

        with tqdm(**get_tqdm_config(total=self.args.eval_step,
                leave=True, color='blue')) as pbar:
            
            for batch_idx in range(self.args.eval_step): ### eval_step: 1024 // batch_size: 64
                
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
                mask = max_prob.ge(self.args.threshold).float() ##### mask: Threshold보다 크면 True, 작으면 False를 반환
                loss_u = (F.cross_entropy(logits_u_s, targets_u, reduction='none')*mask).mean()

                loss = loss_x + self.args.lambda_u * loss_u
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.args.use_ema:
                    self.ema_model.step(self.model)
                
                self.model.zero_grad()
                
                ### Tensorboard를 위해 loss값들을 기록
                losses_x += loss_x.item()
                losses_u += loss_u.item()
                losses_t += loss.item()
                mask_probs += max_prob.mean().item()
                
                ### Print log
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
        return losses_t/(batch_idx+1), losses_x/(batch_idx+1), losses_u/(batch_idx+1)

    
    @torch.no_grad()
    def validate(self, epoch, phase):
        if phase == 'Train': ### Train Loss
            data_loader = self.labeled_loader
            c = 'blue'
        elif phase == 'Valid': ### Valid Loss
            data_loader = self.val_loader
            c = 'green'
        elif phase == 'Test ': ### Test Loss
            data_loader = self.test_loader
            c = 'red'
        
        losses = 0.0
        top1s, top5s = [], []
        
        with tqdm(**get_tqdm_config(total=len(data_loader),
                leave=True, color=c)) as pbar:
            for batch_idx, (inputs, targets) in enumerate(data_loader):
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
    
# Argument 
def FixMatch_parser():
    parser = argparse.ArgumentParser()
    
    # method arguments
    parser.add_argument('--n-labeled', type=int, default=4000) # labeled data의 수
    parser.add_argument('--n-classes', type=int, default=10) # Class의 수
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")

    # training hyperparameters
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
   
    parser = FixMatch_parser()
    args = parser.parse_args([])
    args.cuda = torch.device("cuda:0")
    args.epochs = math.ceil(args.total_steps/args.eval_step)

    trainer = FixMatchTrainer(args)

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
        savepath = 'results/fixmatch'
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
        
if __name__=="__main__":
    main()