import torch.nn as nn
import torch
import numpy as np
import math
import os

class CnnModel(nn.Module):
    
    def __init__(self, opt):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opt = opt

        super(CnnModel, self).__init__()
        input_nc = opt.input_nc
        num_classes = opt.num_classes
        norm_layer = nn.BatchNorm2d

        kw = 3
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.nf

        # 32 * 32 * input_nc

        sequence = [nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        # 16 * 16 * 32

        sequence += [nn.Conv2d(nf, nf*2, kernel_size=kw, stride=2, padding=padw), norm_layer(nf*2), nn.LeakyReLU(0.2, True)]
        # 8 * 8 * 64

        sequence += [nn.Conv2d(nf*2, nf*4, kernel_size=kw, stride=2, padding=padw), norm_layer(nf*4), nn.LeakyReLU(0.2, True)]
        # 4 * 4 * 128

        sequence += [nn.Conv2d(nf*4, nf*8, kernel_size=kw, stride=2, padding=padw), norm_layer(nf*8), nn.LeakyReLU(0.2, True)]
        # 2 * 2 * 256

        sequence += [nn.Conv2d(nf*8, nf*16, kernel_size=kw, stride=2, padding=1), norm_layer(nf*16), nn.LeakyReLU(0.2, True)]
        # 1 * 1 * 512

        self.denseLayer = nn.Linear(nf*16, num_classes)
        self.model = nn.Sequential(*sequence)

        params = list(self.denseLayer.parameters()) + list(self.model.parameters())

        if opt.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(params, momentum=0.9, lr=opt.lr, weight_decay=1e-1)
        elif opt.optimizer == "adam":
            self.optimizer = torch.optim.Adam(params, betas=[opt.beta1, opt.beta2], lr=opt.lr, weight_decay=1e-1)

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        self._initialize_weights()
        self.to(device)

    def _initialize_weights(self):
        # initialize layer weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def save_network(self, epoch_label):
        # save the network to disk
        save_filename = '{}_{}_model.pth'.format(self.opt.model_name, epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.cpu().state_dict(), save_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.denseLayer(x)
        return x

    def predict(self, x):
        return self.forward(x)

    def load(self, save_path):
        # restore the network from a checkpoint
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
        else:
            self.load_state_dict(torch.load(save_path))