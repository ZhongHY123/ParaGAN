import datetime
import math

import dateutil.tz
import os
import os.path as osp
from shutil import copyfile, copytree
import glob
import time
import random
import torch

import paraGAN.myglobal as myglobal
import paraGAN.config as config

from paraGAN.training_generation import *




# noinspection PyInterpreter
def gogo():
    parser = config.get_arguments()
    parser.add_argument('--input_name', help='input image name for training', default="")
    parser.add_argument('--naive_img', help='naive input image  (harmonization or editing)', default="")
    parser.add_argument('--gpu', type=int, help='which GPU to use', default=0)
    parser.add_argument('--train_mode', default='generation', help="generation, retarget, harmonization, editing")
    parser.add_argument('--lr_scale', type=float, help='scaling of learning rate for lower stages', default=0.1)
    parser.add_argument('--train_stages', type=int, help='how many stages to use for training', default=6)

    parser.add_argument('--model_dir', help='model to be used for fine tuning (harmonization or editing)', default="")
    parser.add_argument('--groups', type=int, help='', default=1)
    opt = parser.parse_args()
    opt = functions.post_config(opt)



    if not os.path.exists(opt.input_name):
        print("Image does not exist: {}".format(opt.input_name))
        print("Please specify a valid image.")
        exit()

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)

    dir2save = functions.generate_dir2save(opt)

    if osp.exists(dir2save):
        print('Trained model already exist: {}'.format(dir2save))
        exit()

    # create log dir
    try:
        os.makedirs(dir2save)
    except OSError:
        pass

    # save hyperparameters and code files
    with open(osp.join(dir2save, 'parameters.txt'), 'w') as f:
        for o in opt.__dict__:
            f.write("{}\t-\t{}\n".format(o, opt.__dict__[o]))
    current_path = os.path.dirname(os.path.abspath(__file__))
    for py_file in glob.glob(osp.join(current_path, "*.py")):
        copyfile(py_file, osp.join(dir2save, py_file.split("/")[-1]))
    copytree(osp.join(current_path, "paraGAN"), osp.join(dir2save, "paraGAN"))
    opt.num_layer =3
    myglobal.num_layer_list = []
    myglobal.currentdepth = 3
    myglobal.updownload = {}
    myglobal.priorStageFactor = 0.1
    myglobal.WhenEspcn = 100
    # train model

    print("Training model ({})".format(dir2save))
    opt.niter=2000
    opt.min_size = 25
    opt.max_size = 250
    opt.nfc = 64

    start = time.time()
    train(opt)
    end = time.time()
    elapsed_time = end - start
    print("Time for training: {} seconds".format(elapsed_time))


if __name__ == '__main__':
        gogo()