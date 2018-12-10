import argparse
import os
import sys
sys.path.append('../')
sys.path.append('./')
import torch.optim as optim
import torchvision.utils as utils
from dataset import prepare_data, Dataset
from models import DMCN_prelu
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import *

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--debug", type=str, default='Y', help='whether to use 5 training pictures [Y N]')
parser.add_argument("--resume", type=bool, default=False, help="whether to load model file")
parser.add_argument("--model_name", type=str, default='net_final.pth', help='load this model')
parser.add_argument("--start_epoch", type=int, default=0, help="the start of training epochs")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone1", type=int, default=10, help="first time to decay learning rate; should be less than epochs")
parser.add_argument("--milestone2", type=int, default=20, help="second time to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="../logs/", help='path of log files')
parser.add_argument("--train_id", type=str, default="00", help='path of log files')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--gpu_id", type=str, default="0", help='use which gpu to train')

opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
opt.outf += opt.train_id
def main():
    # Load dataset
    print('Denoising_Blind <==> Part0 : Loading dataset  <==> Begin')
    # processing dataset_train

    if opt.debug == 'Y':
        dataset_train = Dataset(train=True, set='debug')
    else:
        dataset_train = Dataset(train=True)
    # print("dataset_train:", opt.data_mode)
    # processing dataset_val
    dataset_val = Dataset(train=False)
    # dataset_val_68 = Dataset(train=False, set='68')
    psnr_list = list()
    if not os.path.exists(opt.outf):
        os.mkdir(opt.outf)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    # print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    print('Denoising_Blind <==> Part1 :Build model  <==> Begin')

    net = DMCN_prelu()
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # model = torch.nn.DataParallel(net).cuda()

    if opt.resume:
        model_path = os.path.join(opt.outf, opt.model_name)
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            model.load_state_dict(torch.load(os.path.join(model_path)))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B=[0,55] # ingnored when opt.mode=='S'

    for epoch in range(opt.start_epoch, opt.epochs):
        if epoch < opt.milestone1:  # epoch大一些的时候，学习率变为十分之一
            current_lr = opt.lr
        elif epoch < opt.milestone2:
            current_lr = opt.lr / 10.
        else:
            current_lr = opt.lr / 100.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('\tlearning rate %f' % current_lr)
        # train
        print('Denoising_Blind <==> Part2 :train  <==> Begin')

        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            imgn_train = img_train + noise  # 加入noise之后的
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            # noise = Variable(noise.cuda())
            out_train = model(imgn_train)
            loss = criterion(out_train, img_train) / (imgn_train.size()[0] * 2)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            out_img = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_img, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
            ## the end of each epoch
        print('Denoising_Blind <==> Part3 :test  <==> Begin')

        model.eval()
        dataset_val = dataset_val
        psnr_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL / 255.)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
            out_val = torch.clamp(model(imgn_val), 0., 1.)  # 保证图片像素的范围
            psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        psnr_val = psnr_val
        psnr_list.append(psnr_val)
        print("[epoch %d] PSNR_val: %.4f, Best PSNR till now: %.4f"
              % (epoch + 1, psnr_val, max(psnr_list)))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        out_train = torch.clamp(model(imgn_train), 0., 1.)
        try:
            Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
            Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
            Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
            writer.add_image('clean image', Img, epoch)
            writer.add_image('noisy image', Imgn, epoch)
            writer.add_image('reconstructed image', Irecon, epoch)
        except:
            print('[{}] Get error when log the images ...'.format(epoch))
        # save model
        if not max(psnr_list) > psnr_val:
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net_best.pth'))
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net_final.pth'))

if __name__ == "__main__":
    # print('begin to run ...')
    if opt.preprocess:
        prepare_data(data_path='../data', patch_size=40, stride=10,  aug_times=1, debug=opt.debug)
    main()
