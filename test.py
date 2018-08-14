import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from models import cifar10 as C
from models import lenet as L
from models import vgg as V
import argparse
import os
import Input

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def Test(para, net, cuda):

    # # print images
    # dataiter = iter(testloader)
    # images, labels = dataiter.next()
    # # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    #
    # # Test only one picture
    #
    # if args.cuda:
    #     images = Variable(images.cuda())
    #     labels = Variable(labels.cuda())
    # else:
    #     images = Variable(images)
    #     labels = Variable(labels)
    #
    # outputs = net(images)
    #
    # _, predicted = torch.max(outputs.data, 1)
    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
    #                           for j in range(4)))

    # Test all the test dataset
    # correct = 0
    # total = 0
    # for data in para:
    #     images, labels = data
    #
    #     if args.cuda:
    #         images = Variable(images.cuda())
    #         labels = Variable(labels.cuda())
    #     else:
    #         images = Variable(images)
    #         labels = Variable(labels)
    #
    #     outputs = net(images)
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()
    #
    # print('Accuracy of the network on the 10000 test images: %d %%' % (
    #     100 * correct / total))

    # Test on all kinds of classes
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in para:
        images, labels = data

        if args.cuda:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()

        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %4d %%' % (
            classes[i], 100 * (class_correct[i] / class_total[i])))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cifar10')
    parser.add_argument('--trained_model', default='/media/disk/Backup/ZhengFeng/Image_forensics/cifar10/cifar10_weights_0.1_VGG_data_aug/current_144830.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use cuda to train model')
    args = parser.parse_args()

    # Load the data
    Cifar10 = Input.Cifar10()
    testset = Cifar10.testset
    testloader = Cifar10.testloader
    classes = Cifar10.classes

    # net = C.Net()
    # net = L.LeNet()
    net = V.VGG(vgg_name='VGG16')

    # GPU selection
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    # print(net.parameters())
    # print(net)

    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')

    # evaluation
    Test(para=testloader, net=net, cuda=args.cuda)
