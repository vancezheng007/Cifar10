import torch
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
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

# def conv_ResNet_model():
#
#     model_conv = torchvision.models.resnet18(pretrained=True)
#     for param in model_conv.parameters():
#         param.requires_grad = False
#     # Parameters of newly constructed modules have requires_grad=True by default
#     num_ftrs = model_conv.fc.in_features
#     model_conv.fc = nn.Linear(num_ftrs, 2)
#     model_conv = model_conv.to(device)
#
#     return model_conv

def ft_ResNet_model():
    model_ft = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_conv = model_ft.to(device)
    return model_ft

def Train(para, net):

    net.train()
    if args.visdom:
        import visdom
        viz = visdom.Visdom()
    if args.visdom:
       vis_title = args.dataset
       vis_legend = ['Epoch loss', 'Iteration loss']
       iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
       epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.fc.parameters(), args.lr, args.momentum)
    optimizer = optim.SGD(net.parameters(), args.lr, args.momentum)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    try:
        current_iteration = 0
        for epoch in range(24):  # loop over the dataset multiple times
            for iteration, data in enumerate(para, 0):
                # Caculate the time difference
                start_time = time.time()

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
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                exp_lr_scheduler.step()
                optimizer.step()
                current_iteration += 1
                running_loss += loss.item()
                duration = time.time() - start_time

                if current_iteration % 10 == 0:
                    num_examples_per_step = args.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), current_iteration, running_loss,
                                        examples_per_sec, sec_per_batch))

                if args.visdom:
                    update_vis_plot(current_iteration, loss, iter_plot, 'append')

            if args.visdom:
                update_vis_plot(epoch, loss, epoch_plot, 'append')

        print('Finished Training')

    except Exception as e:
        print('Exception happended')
        print(e)

    finally:
        torch.save(net.state_dict(), args.save_folder + 'current_' +
           repr(current_iteration) + '.pth')
        print('save model')

# ######################################################################
# # Visualizing the model predictions
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# #
# # Generic function to display predictions for a few images
# #
#
# def visualize_model(model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()
#
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#
#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 imshow(inputs.cpu().data[j])
#
#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)
#
# ######################################################################
# # Finetuning the convnet
# # ----------------------
# #
# # Load a pretrained model and reset final fully connected layer.
# #
#
# model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 2)
#
# model_ft = model_ft.to(device)
#
# criterion = nn.CrossEntropyLoss()
#
# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
#
# ######################################################################
# # Train and evaluate
# # ^^^^^^^^^^^^^^^^^^
# #
# # It should take around 15-25 min on CPU. On GPU though, it takes less than a
# # minute.
# #
#
# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=25)
#
# ######################################################################
# #
#
# visualize_model(model_ft)

# ######################################################################
# # ConvNet as fixed feature extractor
# # ----------------------------------
# #
# # Here, we need to freeze all the network except the final layer. We need
# # to set ``requires_grad == False`` to freeze the parameters so that the
# # gradients are not computed in ``backward()``.
# #
# # You can read more about this in the documentation
# # `here <http://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
# #
#
# model_conv = torchvision.models.resnet18(pretrained=True)
# for param in model_conv.parameters():
#     param.requires_grad = False
#
# # Parameters of newly constructed modules have requires_grad=True by default
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, 10)
#
# model_conv = model_conv.to(device)
#
# criterion = nn.CrossEntropyLoss()
#
# # Observe that only parameters of final layer are being optimized as
# # opoosed to before.
# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
#
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
#
#
# ######################################################################
# # Train and evaluate
# # ^^^^^^^^^^^^^^^^^^
# #
# # On CPU this will take about half the time compared to previous scenario.
# # This is expected as gradients don't need to be computed for most of the
# # network. However, forward does need to be computed.
# #
#
# model_conv = train_model(model_conv, criterion, optimizer_conv,
#                          exp_lr_scheduler, num_epochs=25)
#
# ######################################################################
# #
#
# visualize_model(model_conv)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')
    train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset', default='Cifar10',
                        type=str, help='Cifar10')
    parser.add_argument('--basenet', default='',
                        help='Pretrained base model')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default='', type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='hymenoptera_data/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use cuda to train model')
    parser.add_argument('--visdom', default=True, type=str2bool,
                        help='Use visdom for loss visualization')
    # parser.add_argument('--start_iter', default=0, type=int,
    #                     help='Resume training at this iter')
    # parser.add_argument('--dataset_root', default='/media/disk/Backup/ZhengFeng/SSD/data/WIDER',
    #                     help='Dataset root directory path')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    # # Load the data
    # Cifar10 = Input.Cifar10()
    # trainset = Cifar10.trainset
    # trainloader = Cifar10.trainloader
    # classes = Cifar10.classes
    #
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Load the Hymeno data
    Hymeno = Input.Hymeno()
    trainset = Hymeno.image_datasets['train']
    trainloader = Hymeno.dataloaders['train']
    classes = Hymeno.class_names

    net = ft_ResNet_model()
    # net = conv_ResNet_model()

    # if args.cuda:
    #     # GPU selection
    #     net = torch.nn.DataParallel(net)
    #     net = net.cuda()
    # else:
    #     net = torch.nn.DataParallel(net)

    Train(para=trainloader, net=net)
