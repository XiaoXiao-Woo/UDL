# Copyright (c) Xiao Wu, LJ Deng (UESTC-MMHCISP). All rights reserved.
from UDL.Basis.config import Config
from UDL.pansharpening.common.main_pansharpening import main
from UDL.Basis.auxiliary import set_random_seed
from UDL.pansharpening.models.DCFNet.option_DCFNet import cfg as args
from UDL.pansharpening.models.DCFNet.model_fcc_dense_head import build_DCFNet

if __name__ == '__main__':
    # cfg = Config.fromfile("../pansharpening/DCFNet/option_DCFNet.py")
    set_random_seed(args.seed)
    # print(cfg.builder)
    args.builder = build_DCFNet
    main(args)