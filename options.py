import argparse
import os
import torch
import utils

class CnnModelOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default='experiment_name',
                                 help='name of the experiment. It decides where to store samples and models')

        # network arch
        self.parser.add_argument('--dataset_name', type=str, default='FashionMNIST0.5.npz', help='the name of the dataset')
        self.parser.add_argument('--dataset_root', type=str, default='./datasets', help='the root folder of the dataset')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='the root folder of checkpoints')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--nf', type=int, default=32, help='# of gen filters in first conv layer')
        self.parser.add_argument('--num_classes', type=int, default=3, help='# of classes')
        self.parser.add_argument('--is_testing', action='store_true', help='if it is testing phase')

        # train
        self.parser.add_argument('--epoch', type=int, default=20, help='training epochs')
        self.parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
        self.parser.add_argument('--loss_name', type=str, default='ce', help='name of the loss function used to train the model bootstrap|ce|mpe|symmetric.')
        self.parser.add_argument('--sigma', type=float, default=0.2, help='sigma value in the loss function.')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        self.parser.add_argument('--val_split_rate', type=float, default=0.0, help='the split rate of data for validating')
        self.parser.add_argument('--optimizer', type=str, default="sgd", help='the optimizer method sgd|adam')
        self.parser.add_argument('--print_loss', type=bool, default=True, help='If loss should be printed during the training phase')
        self.parser.add_argument('--save_model', type=bool, default=False, help='If model need to be saved after training')
        self.parser.add_argument('--num_trained_model', type=int, default=1, help='How many model do we need to train')

        # noise handling
        self.parser.add_argument('--trans_matrix', type=str, default='', help='transition matrix if known')
        self.parser.add_argument('--pretrained_model_path', type=str, default='', help='pretrained_model_path')
        self.parser.add_argument('--pretrained_model_path_format', type=str, default='', help='pretrained_model_path_format')
        self.parser.add_argument('--log_file_path', type=str, default='log.txt', help='the path of the log file')

        self.initialized = True

    def parse_str(self, ids):
        str_ids = ids.split(',')
        ids_list = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                ids_list.append(id)
        return ids_list

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.is_training = not self.opt.is_testing  # train or test

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        self.opt.model_name = self.opt.name

        # transition matrix
        if len(self.opt.trans_matrix)>0:
            import ast
            import numpy as np
            self.opt.trans_matrix = np.array(ast.literal_eval(self.opt.trans_matrix), dtype=np.float).T
        else:
            self.opt.trans_matrix = None

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        utils.mkdirs(expr_dir)
        if save:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
