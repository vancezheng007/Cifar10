import torch
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import argparse
from models import cifar10 as C
from models import lenet as L
from models import vgg as V
import os
import Input
import visdom
import time
from datetime import datetime

viz = visdom.Visdom()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def create_vis_plot(_xlabel, _ylabel, _title, _legend):

    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 2)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )

def update_vis_plot(iteration, loss, window1, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 1)).cpu() * iteration,
        Y=torch.Tensor([loss]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )

    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 1)).cpu(),
            Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
            win=window1,
            update=True
        )

def Train(para, net):

    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     assert os.path.isdir('cifar10_weights'), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load(args.resume)
    #     net.load_state_dict(checkpoint)

    net.train()

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        net.load_weights(args.resume)
    # else:
    #     vgg_weights = torch.load(args.save_folder + args.basenet)
    #     print('Loading base network...')
    #     net._make_layers.load_state_dict(vgg_weights)

    if args.visdom:
        import visdom
        viz = visdom.Visdom()
    if args.visdom:
       vis_title = args.dataset
       vis_legend = ['Epoch loss', 'Iteration loss']
       iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
       epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), args.lr, args.momentum)

    try:
        current_iteration = 0
        for epoch in range(500):  # loop over the dataset multiple times
            for iteration, data in enumerate(para, 0):
                running_loss = 0.0
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                if args.cuda:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                # Caculate the time difference
                start_time = time.time()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                current_iteration += 1
                running_loss += loss.item() * inputs.size(0)
                duration = time.time() - start_time

                if current_iteration % 1000 == 0:
                    num_examples_per_step = args.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), current_iteration, running_loss,
                                        examples_per_sec, sec_per_batch))

                if (current_iteration - 1) != 0 and current_iteration % 5000 == 0:
                    print('Saving state, iter:', current_iteration)
                    torch.save(net.state_dict(), args.save_folder +
                               repr(current_iteration) + '.pth')
                if args.visdom:
                    update_vis_plot(current_iteration, loss, iter_plot, 'append')

            if args.visdom:
                update_vis_plot(epoch, loss, epoch_plot, 'append')

            if epoch % 20 == 0:
                print('Saving state, epoch:', epoch)
                torch.save(net.state_dict(), args.save_folder + 'epoch_' +
                           repr(epoch) + '.pth')

        print('Finished Training')

    except Exception as e:
        print('Exception happended')

    finally:
        torch.save(net.state_dict(), args.save_folder + 'current_' +
           repr(current_iteration) + '.pth')
        print('save model')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')
    train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset', default='Cifar10',
                        type=str, help='Cifar10')
    parser.add_argument('--basenet', default='',
                        help='Pretrained base model')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default='', type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='cifar10_weights_0.1_VGG_data_aug/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use cuda to train model')
    parser.add_argument('--visdom', default=True, type=str2bool,
                        help='Use visdom for loss visualization')
    # parser.add_argument('--start_iter', default=0, type=int,
    #                     help='Resume training at this iter')
    # parser.add_argument('--dataset_root', default='/media/disk/Backup/ZhengFeng/SSD/data/WIDER',
    #                     help='Dataset root directory path')

    args = parser.parse_args()

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    # Load the data
    Cifar10 = Input.Cifar10()
    trainset = Cifar10.trainset
    trainloader = Cifar10.trainloader
    classes = Cifar10.classes

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # net = C.Net()
    # net = L.LeNet()
    net = V.VGG(vgg_name='VGG16')

    if args.cuda:
        # GPU selection
        net = torch.nn.DataParallel(net)
        net = net.cuda()
    else:
        net = torch.nn.DataParallel(net)

    Train(para=trainloader, net=net)
