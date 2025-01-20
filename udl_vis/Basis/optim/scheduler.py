import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingLR,
    MultiStepLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)


class CosineAnnealingWarmRestartsReduce(CosineAnnealingWarmRestarts):
    """
    Cosine annealing restart learning rate scheduler with reducing learning rate
    in a fixed ratio after each restart.

    Args:
        opt (optim.Optimizer): optimizer
        T_0 (int): number of epochs for the first restart
        T_mult (int, optional): factor to increase T_i after a restart. Defaults to 1.
        lr_mult (float, optional): learning rate multiplier after each restart. Defaults to 1.
        eta_min (float, optional): minimum learning rate. Defaults to 0.
        last_epoch (int, optional): index of the last epoch. Defaults to -1.
        warmup_epochs (int, optional): number of epochs for linear warmup. Defaults to 0.
    """

    def __init__(
        self,
        opt: optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        lr_mult: float = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
        warmup_epochs: int = 0,
    ):
        self.lr_mult = lr_mult
        self.warmup_epochs = warmup_epochs
        self._warmed_up = False if warmup_epochs > 0 else True
        super().__init__(opt, T_0, T_mult, eta_min, last_epoch)

    def step(self, ep: int = None):
        if (
            self.warmup_epochs > 0
            and self.T_cur <= self.warmup_epochs
            and not self._warmed_up
            and self.T_cur >= 0
        ):
            self._last_lr = []
            for i in range(len(self.optimizer.param_groups)):
                # from eta_min to base_lr
                _ratio = self.T_cur / self.warmup_epochs
                _curr_lr = _ratio * self.base_lrs[i] + (1 - _ratio) * self.eta_min
                self.optimizer.param_groups[i]["lr"] = _curr_lr
                self._last_lr.append(_curr_lr)
            self.T_cur += 1
        elif self.T_cur > self.warmup_epochs and not self._warmed_up:
            self._warmed_up = True
            self.T_cur = 0
        elif self._warmed_up and self.T_cur == self.T_i - 1 and self.last_epoch != 0:
            # reduce the base lr
            for i in range(len(self.base_lrs)):
                self.base_lrs[i] *= self.lr_mult
                self.base_lrs[i] = max(self.base_lrs[i], self.eta_min)
            # step the scheduler by super().step() in cosine annealing way
            super().step()
        elif self.T_cur < 0:
            self.T_cur = 0
            self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        else:
            # neither warmup nor reduce the base_lr,
            # step the scheduler by super().step() in cosine annealing way
            super().step()


class LRScheduler(object):

    def __init__(self, optimizer, **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = kwargs.pop('scheduler_type')
        self.set_scheduler(self.scheduler_type, **kwargs)

    # 六大学习率调整策略，lr = lr * gamma
    """
    ReduceLROnPlateau:
        mode(str)- 模式选择，有 min 和 max 两种模式， min 表示当指标不再降低(如监测loss)， max 表示当指标不再升高(如监测 accuracy)。
        factor(float)- 学习率调整倍数(等同于其它方法的 gamma)，即学习率更新为 lr = lr * factor
        patience(int)- 忍受该指标多少个 step 不变化，当忍无可忍时，调整学习率。
        verbose(bool)- 是否打印学习率信息， print(‘Epoch {:5d}: reducing learning rate of group {} to {:.4e}.’.format(epoch, i, new_lr))
        threshold_mode(str)- 选择判断指标是否达最优的模式，有两种模式， rel 和 abs。
        当 threshold_mode == rel, 并且 mode == max 时， dynamic_threshold = best * ( 1 +threshold );
        当 threshold_mode == rel, 并且 mode == min 时， dynamic_threshold = best * ( 1 -threshold );
        当 threshold_mode == abs, 并且 mode== max 时， dynamic_threshold = best + threshold;
        当 threshold_mode == rel, 并且 mode == max 时， dynamic_threshold = best - threshold;
        threshold(float)- 配合 threshold_mode 使用。
        cooldown(int)- “冷却时间“，当调整学习率之后，让学习率调整策略冷静一下，让模型再训练一段时间，再重启监测模式。
        min_lr(float or list)- 学习率下限，可为 float，或者 list，当有多个参数组时，可用 list 进行设置
    """

    def set_scheduler(self, scheduler_type, **kwargs):
        # self.lr_scheduler = lr_scheduler
        # self.scheduler = []
        optimizer = self.optimizer
        if scheduler_type == "StepLR":
            # 等间距阶段式衰减
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer, **kwargs#step_size=10, gamma=0.95
            )
        elif scheduler_type == "ReduceLROnPlateau":
            # Reduce learning rate when validation accuarcy plateau.
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **kwargs#mode="max", patience=5, verbose=True
            )
        elif scheduler_type == "MultiStepLR":
            # milestones=[epoch1,epoch2,...] 阶段式衰减
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, **kwargs,#[100, 200, 300], gamma=0.5
            )  # [50, 100, 150, 200, 250, 300, 350, 400], gamma=0.5)
        elif scheduler_type == "ExponentialLR":
            # 指数衰减x, 0.1,0.01,0.001,...
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, **kwargs,#gamma=0.1
            )
        elif scheduler_type == "CosineAnnealingLR":
            # Cosine annealing learning rate.
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **kwargs,#T_max=30, eta_min=1e-7
            )
        elif scheduler_type == "CyclicLR":
            self.lr_scheduler = optim.lr_scheduler.CyclicLR(
                optimizer,
                **kwargs,
                # base_lr=1e-3,
                # max_lr=1e-4,
                # step_size_down=30,
                # step_size_up=150,
                # cycle_momentum=False,
            )
        elif scheduler_type == "LambdaLR":
            # 学习率 = 初始学习率 * lr_lambda(last_epoch）
            curves = lambda epoch: epoch // 30
            # lambda2 = lambda epoch: 0.95 ** epoch
            # lr_lambda对应optimizer中的keys，model.parameters()就只有一个lambda函数
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, **kwargs#lr_lambda=[curves]
            )
        elif scheduler_type == "CosineAnnealingWarmRestartsReduce":
            self.lr_scheduler = CosineAnnealingWarmRestartsReduce(
                optimizer, **kwargs #T_0=20, T_mult=2, lr_mult=0.7, eta_min=8e-5, warmup_epochs=0
            )
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            # To 初始周期
            # T_mult 每次循环 周期改变倍数  T_0 = T_0*T_mult
            # Learning rate warmup by 10 epochs.
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, **kwargs #T_0=10, T_mult=2, eta_min=0
            )
        else:
            raise NotImplementedError(f"{__file__}: scheduler_type: {scheduler_type} not in pytorch")

    def adjust_2_learning_rate(self, epoch):
        """编写2种形式的学习率衰减策略的组合"""
        param_groups = self.optimizer.param_groups
        if epoch <= 5:
            lr = [param_groups[0]["lr"] * 0.9]
            for param_group, val in zip(param_groups, lr):
                param_group["lr"] = val
        else:
            for param_group in param_groups:
                if epoch % 5 == 0:
                    # 0.09 0.009 0.0009
                    param_group["lr"] *= 0.9
        # print(param_group['lr'])

    def adjust_1_learning_rate(self, epoch, mini_lr=1e-6):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if self.optimizer.param_groups[0]["lr"] < mini_lr:
            lr = 1e-5
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            return
        if epoch <= 40:  # 40 20 80
            # lr = self.lr
            lr = self.lr * (0.1 ** (epoch // 20))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            return
        elif epoch == 81:  # 41
            self.lr = self.optimizer.param_groups[0]["lr"]
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = 1e-4
        # if epoch >= 42 and epoch % 5 ==0:
        if epoch >= 81:
            lr = self.lr * (0.9 ** (epoch // 20))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            return
        elif epoch == 81:
            lr = 1e-5
        else:
            lr = 1e-5
            # self.lr = self.lr * (0.9 ** (epoch // 50))
        # #if epoch <= 120:
        #     lr = self.lr * (0.9 ** (epoch // 50))
        # elif epoch == 121:
        #    self.lr = self.optimizer.param_groups[0]["lr"]
        #    lr = self.lr * (0.9 ** (epoch // 50))
        # else:
        #    lr = 0.01
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self, epoch):
        # if not end:
        #     self.optimizer.step()
        # else:
        if self.scheduler_type == "manual":
            # self.optimizer.step()
            self.adjust_1_learning_rate(epoch)
        else:
            # self.optimizer.step()
            self.lr_scheduler.step()

    # preprint lr_map


def get_lr_map(optimizer, lr_scheduler, epochs, out_file=None, viz=False):
    plt.figure()
    lr = []
    print(f"[get_lr_map] plot lr_scheduler curves")
    tmp = optimizer.param_groups[0]["lr"]
    if lr_scheduler is None:
        for epoch in range(epochs):
            optimizer.step()
            # TODO:按层绘制
            # print(self.optimizer.param_groups[0]['lr'])
            lr.append(optimizer.param_groups[0]["lr"])
    else:
        for epoch in range(epochs):

            try:
                lr.append(lr_scheduler.get_last_lr())
                # lr.append(self.lr_scheduler.get_lr())
            except:
                # ReduceLROnPlateau没有get_lr方法
                lr.append(optimizer.param_groups[0]["lr"])
            lr_scheduler.step()
    plt.plot(list(range(epochs)), lr)
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.title(lr_scheduler.__class__.__name__)
    if out_file is not None:
        plt.savefig(out_file + f"/{lr_scheduler.__class__.__name__}.png")
    if viz:
        plt.show()
    optimizer.param_groups[0]["lr"] = tmp
    # lr = tmp
    print(np.array(lr).flatten().tolist()[:10])


if __name__ == "__main__":
    import os



    model = nn.Linear(10, 10).cuda()
    optimizer = optim.SGD(params=model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingWarmRestartsReduce(
            optimizer, T_0=20, T_mult=2, lr_mult=0.7, eta_min=8e-5, warmup_epochs=0
        )
    
    epochs = 3000
    # 构造一个带warmup小学习率的optimizer，再上升到标准值，再正常周期下降
    lrs = lr_scheduler(0.1, epochs)

    # lrs.set_optimizer(optimizer, optim.lr_scheduler.MultiStepLR)
    # lrs.get_lr_map("MultiStepLR")
    # lrs.set_optimizer(optimizer, optim.lr_scheduler.ExponentialLR)
    # lrs.get_lr_map("ExponentialLR")
    lrs.set_optimizer(optimizer, scheduler)
    get_lr_map(optimizer, lrs.lr_scheduler, epochs, out_file=os.path.dirname(__file__), viz=True)
    print(lrs.lr)
    # lrs.set_optimizer(optimizer, optim.lr_scheduler.CyclicLR)
    # lrs.get_lr_map("CyclicLR")
    # # lrs.set_optimizer(optimizer, optim.lr_scheduler.ReduceLROnPlateau)
    # # lrs.get_lr_map("ReduceLROnPlateau")
    # lrs.set_optimizer(optimizer, None)
    # lrs.get_lr_map("LambdaLR")
    # lrs.set_optimizer(optimizer, optim.lr_scheduler.CosineAnnealingLR)
    # lrs.get_lr_map("CosineAnnealingLR")
    # lrs.set_optimizer(optimizer, optim.lr_scheduler.CosineAnnealingWarmRestarts)
    # lrs.get_lr_map("CosineAnnealingWarmRestarts")
