import os, math, sys, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from tqdm import tqdm
from colorama import Fore
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# Two kinds of Augmentation 
class Transform_Twice:
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, img):
        out1 = self.transform(img)
        out2 = self.transform(img)
        
        return out1, out2
    
# Generate labeled data
class Labeled_CIFAR10(torchvision.datasets.CIFAR10):
    
    def __init__(self, root, indices=None,
                train=True, transform=None,
                target_transform=None, download=False):
        
        super(Labeled_CIFAR10, self).__init__(root,
                                        train=train,
                                        transform=transform,
                                        target_transform=target_transform,
                                        download=download)

        if indices is not None:
            self.data = self.data[indices]
            self.targets = np.array(self.targets)[indices]
        
        self.data = Transpose(Normalize(self.data))
    
    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
# Generate unlabeled data

class Unlabeled_CIFAR10(Labeled_CIFAR10):
    
    def __init__(self, root, indices, train=True, transform=None, target_transform=None, download=False):
        
        super(Unlabeled_CIFAR10, self).__init__(root, indices, train,
                                            transform=transform,
                                            target_transform=target_transform,
                                            download=download)
        
        self.targets = np.array([-1 for i in range(len(self.targets))])
        

def split_datasets(labels, n_labeled_per_class):
    
    labels = np.array(labels, dtype=int) 
    indice_labeled, indice_unlabeled, indice_val = [], [], [] 
    
    for i in range(10): 
        indice_tmp = np.where(labels==i)[0]
        indice_labeled.extend(indice_tmp[: n_labeled_per_class])
        indice_unlabeled.extend(indice_tmp[n_labeled_per_class: -500])
        indice_val.extend(indice_tmp[-500: ])
    
    for i in [indice_labeled, indice_unlabeled, indice_val]:
        np.random.shuffle(i)
    
    return indice_labeled, indice_unlabeled, indice_val

# Load CIFAR10 data
def get_cifar10(data_dir: str, n_labeled: int,
                transform_train=None, transform_val=None,
                download=True):
    
    base_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=download)
    
    indice_labeled, indice_unlabeled, indice_val = split_datasets(base_dataset.targets, int(n_labeled/10)) ### n_labeled는 아래 MixMatch_argparser 함수에서 정의

    train_labeled_set = Labeled_CIFAR10(data_dir, indice_labeled, train=True, transform=transform_train) 
    train_unlabeled_set = Unlabeled_CIFAR10(data_dir, indice_unlabeled, train=True, transform=Transform_Twice(transform_train))
    val_set = Labeled_CIFAR10(data_dir, indice_val, train=True, transform=transform_val, download=True) 
    test_set = Labeled_CIFAR10(data_dir, train=False, transform=transform_val, download=True) 

    return train_labeled_set, train_unlabeled_set, val_set, test_set

# Normalize data
def Normalize(x, m=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2345, 0.2616)):

    x, m, std = [np.array(a, np.float32) for a in (x, m, std)] 

    x -= m * 255 
    x *= 1.0/(255*std)
    return x

def Transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')


class RandomPadandCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, x):
        x = pad(x, 4)
        
        old_h, old_w = x.shape[1: ]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, old_h-new_h)
        left = np.random.randint(0, old_w-new_w)
        
        x = x[:, top:top+new_h, left:left+new_w]
        return x

# Random flip
class RandomFlip(object):
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]
        
        return x.copy()
    
    
# Gaussian noise 
class GaussianNoise(object):
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w)*0.15
        return x
    
class ToTensor(object):
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x
    
# WideResNet
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual
    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
    
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)
    
class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
# Define loss function 
class Loss_Semisupervised(object):
    def __call__(self, args, outputs_x, target_x, outputs_u, targets_u, epoch):
        self.args = args
        probs_u = torch.softmax(outputs_u, dim=1)

        loss_x = -torch.mean(
            torch.sum(F.log_softmax(outputs_x, dim=1)*target_x, dim=1)
        )

        loss_u = torch.mean((probs_u-targets_u)**2)

        return loss_x, loss_u, self.args.lambda_u*linear_rampup(epoch, self.args.epochs)
    
def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current/rampup_length, 0.0, 1.0)
        return float(current)
    
class WeightEMA(object): # EMA=Exponential Moving Average

    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model

        self.alpha = alpha

        self.params = list(self.model.state_dict().items())
        self.ema_params = list(self.ema_model.state_dict().items())

        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param[1].data.copy_(ema_param[1].data)
    
    def step(self):
        inverse_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param[1].dtype == torch.float32:
                ema_param[1].mul_(self.alpha) # ema_params_new = self.alpha * ema_params_old
                ema_param[1].add_(param[1]*inverse_alpha) # ema_params_Double_new = (1-self.alpha)*params

                param[1].mul_(1-self.wd)
                
def interleave_offsets(batch_size, nu):
    
    groups = [batch_size//(nu+1)]*(nu+1)
    for x in range(batch_size-sum(groups)):
        groups[-x-1] += 1

    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1]+g)
    
    assert offsets[-1] == batch_size
    return offsets

def interleave(xy, batch_size):
    
    nu = len(xy) - 1
    offsets = interleave_offsets(batch_size, nu)

    xy = [[v[offsets[p]:offsets[p+1]] for p in range(nu+1)] for v in xy]
    for i in range(1, nu+1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def get_tqdm_config(total, leave=True, color='white'):
    fore_colors = {
        'red': Fore.LIGHTRED_EX,
        'green': Fore.LIGHTGREEN_EX,
        'yellow': Fore.LIGHTYELLOW_EX,
        'blue': Fore.LIGHTBLUE_EX,
        'magenta': Fore.LIGHTMAGENTA_EX,
        'cyan': Fore.LIGHTCYAN_EX,
        'white': Fore.LIGHTWHITE_EX,
    }
    return {
        'file': sys.stdout,
        'total': total,
        'desc': " ",
        'dynamic_ncols': True,
        'bar_format':
            "{l_bar}%s{bar}%s| [{elapsed}<{remaining}, {rate_fmt}{postfix}]" % (fore_colors[color], Fore.RESET),
        'leave': leave
    }
    
def accuracy(output, target, topk=(1, )):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        if k == 1:
            correct_k = correct[:k].view(-1).float().sum(0)
        if k > 1:
            correct_k = correct[:k].float().sum(0).sum(0)
        acc = correct_k.mul_(100.0 / batch_size)
        acc = acc.detach().cpu().numpy()
        res.append(acc)
    return res

class MixMatchTrainer():
    def __init__(self, args):
        self.args = args

        root_dir = '/content/MixMatch' # PROJECT directory
        self.experiment_dir = os.path.join(root_dir, 'results') # 학습된 모델을 저장할 폴더 경로 정의 및 폴더 생성
        os.makedirs(self.experiment_dir, exist_ok=True)

        name_exp = "_".join([str(self.args.n_labeled), str(self.args.T)]) # 주요 하이퍼 파라미터로 폴더 저장 경로 지정 
        self.experiment_dir = os.path.join(self.experiment_dir, name_exp)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Data
        print("==> Preparing CIFAR10 dataset")
        transform_train = transforms.Compose([
            RandomPadandCrop(32),
            RandomFlip(),
            ToTensor()
        ]) # 학습에 사용할 data augmentation 정의

        transform_val = transforms.Compose([
            ToTensor()
        ]) # validation, test dataset에 대한 data augmentation 정의
           # 합성곱 신경망에 입력 될 수 있도록만 지정(Augmentation 사용하지 않는 것과 동일)

        train_labeled_set, train_unlabeled_set, val_set, test_set = \
            get_cifar10(
                data_dir=os.path.join(root_dir, 'data'),
                n_labeled=self.args.n_labeled,
                transform_train=transform_train,
                transform_val=transform_val
            ) # 앞에서 정의한 (def) get_cifar10 함수에서 train_labeled, train_unlabeled, validation, test dataset
        
        # DataLoader 정의
        self.labeled_loader = DataLoader(
            dataset=train_labeled_set,
            batch_size=self.args.batch_size,
            shuffle=True, num_workers=0, drop_last=True
        )

        self.unlabeled_loader = DataLoader(
            dataset=train_unlabeled_set,
            batch_size=self.args.batch_size,
            shuffle=True, num_workers=0, drop_last=True
        )

        self.val_loader = DataLoader(
            dataset=val_set, shuffle=False, num_workers=0, drop_last=False
        )

        self.test_loader = DataLoader(
            dataset=test_set, shuffle=False, num_workers=0, drop_last=False
        )

        # Build WideResNet
        print("==> Preparing WideResNet")
        self.model = self.create_model(ema=False)
        self.ema_model = self.create_model(ema=True)

        # Define loss functions
        self.criterion_train = Loss_Semisupervised()
        self.criterion_val = nn.CrossEntropyLoss().to(self.args.cuda)

        # Define optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.ema_optimizer = WeightEMA(self.model, self.ema_model, lr=self.args.lr, alpha=self.args.ema_decay)

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

                # Transform label to one-hot
                targets_x = torch.zeros(real_B, 10).scatter_(1, targets_x.view(-1,1).long(), 1)
                inputs_x, targets_x = inputs_x.to(self.args.cuda), targets_x.to(self.args.cuda)

                try:
                    tmp_inputs, _ = iter_unlabeled.next()
                except:
                    iter_unlabeled = iter(self.unlabeled_loader)
                    tmp_inputs, _ = iter_unlabeled.next()

                inputs_u1, inputs_u2 = tmp_inputs[0], tmp_inputs[1]
                inputs_u1, inputs_u2 = inputs_u1.to(self.args.cuda), inputs_u2.to(self.args.cuda)

                with torch.no_grad():
                    outputs_u1 = self.model(inputs_u1)
                    outputs_u2 = self.model(inputs_u2)

                    pt = (torch.softmax(outputs_u1, dim=1)+torch.softmax(outputs_u2, dim=1)) / 2
                    pt = pt**(1/self.args.T)

                    targets_u = pt / pt.sum(dim=1, keepdim=True)
                    targets_u = targets_u.detach()
                
                inputs = torch.cat([inputs_x, inputs_u1, inputs_u2], dim=0)
                targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

                l_mixup = np.random.beta(self.args.alpha, self.args.alpha)
                l_mixup = max(l_mixup, 1-l_mixup)

                B = inputs.size(0)
                random_idx = torch.randperm(B)

                inputs_a, inputs_b = inputs, inputs[random_idx]
                targets_a, targets_b = targets, targets[random_idx]

                mixed_input = l_mixup*inputs_a + (1-l_mixup)*inputs_b
                mixed_target = l_mixup*targets_a + (1-l_mixup)*targets_b

                
                mixed_input = list(torch.split(mixed_input, real_B))
                mixed_input = interleave(mixed_input, real_B)

                logits = [self.model(mixed_input[0])] # for labeled
                for input in mixed_input[1:]:
                    logits.append(self.model(input)) # for unlabeled

                logits = interleave(logits, real_B) # interleave: 정확히 섞이었는지 확인
                logits_x = logits[0]
                logits_u = torch.cat(logits[1:], dim=0)

                loss_x, loss_u, w = \
                    self.criterion_train(self.args,
                                    logits_x, mixed_target[:real_B],
                                    logits_u, mixed_target[real_B:],
                                    epoch+batch_idx/self.args.num_iter) # Semi-supervised loss 계산

                loss = loss_x + w * loss_u

                # Backpropagation and Model parameter update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_optimizer.step()

                losses_x += loss_x.item()
                losses_u += loss_u.item()
                losses_t += loss.item()
                ws += w

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
        self.ema_model.eval()

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
        
                outputs = self.ema_model(inputs)
                loss = self.criterion_val(outputs, targets)


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
    
def MixMatch_parser():
    parser = argparse.ArgumentParser()
    
    # method arguments
    parser.add_argument('--n-labeled', type=int, default=1024)
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
    parser = MixMatch_parser()
    args = parser.parse_args([])
    args.cuda = torch.device("cuda:0")

    trainer = MixMatchTrainer(args)
    
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
        savepath = 'results/mixmatch'
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