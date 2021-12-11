# Copyright (c) Xiao Wu, LJ Deng (UESTC-MMHCISP). All rights reserved.
from UDL.Basis.config import Config
from UDL.pansharpening.common.main_pansharpening import main
from UDL.Basis.auxiliary import set_random_seed
from UDL.pansharpening.models.DiCNN.option_dicnn import cfg as args
from UDL.pansharpening.models.DiCNN.dicnn_main import build_dicnn as builder

if __name__ == '__main__':
    # cfg = Config.fromfile("../pansharpening/DCFNet/option_DCFNet.py")
    set_random_seed(args.seed)
    # print(cfg.builder)
    args.builder = builder
    main(args)