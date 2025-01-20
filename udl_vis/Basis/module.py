import torch
from torch import nn
from torch.nn import functional as F
import math
from torch.utils.data import DataLoader
from udl_vis.Basis.auxiliary import args_to_dict
import inspect
from udl_vis.Basis.distributed import DistributedSampler, RandomSampler


def set_dataloader(
    dataset,
    seed,
    distributed=False,
    state_dataloader=None,
    return_dataset=True,
    kwargs=None,
):
    sampler = None

    # if not return_gen:
    #     return dataset, sampler

    generator = torch.Generator()
    generator.manual_seed(seed)
    if state_dataloader is not None:
        generator.set_state(state_dataloader.cpu())

    if distributed:
        sampler = DistributedSampler(dataset, generator=generator)
    else:
        sampler = RandomSampler(dataset, generator=generator)

    if return_dataset:
        return dataset, sampler, generator
    else:
        # parameters = list(inspect.signature(DataLoader.__init__).parameters.keys())[1:]
        # _kwargs = args_to_dict(kwargs, parameters)
        # _kwargs["batch_size"] = kwargs["samples_per_gpu"]
        # _kwargs["num_workers"] = kwargs["workers_per_gpu"]
        # _kwargs.pop("dataset")
        # _kwargs["dataset"] = dataset
        loader = DataLoader(
            dataset,
            batch_size=kwargs["samples_per_gpu"],
            persistent_workers=(True if kwargs["workers_per_gpu"] > 0 else False),
            pin_memory=True,
            generator=generator,
            shuffle=(sampler is None),
            num_workers=kwargs["workers_per_gpu"],
            drop_last=True,
            sampler=sampler,
        )
        return loader, sampler, generator


class PatchMergeModule(nn.Module):
    def __init__(
        self,
        crop_batch_size=1,
        patch_size_list=[],
        scale=1,
        # device='cuda:0',
        bs_axis_merge=True,
    ):
        super().__init__()
        if bs_axis_merge:
            self.split_func = torch.split
        else:
            self.split_func = lambda x, _, dim: [x]
        self.crop_batch_size = crop_batch_size
        self.patch_size_list = patch_size_list
        print(f"it has patch_size_list: {patch_size_list} in PatchMergeModule")
        self.scale = scale[0] if isinstance(scale, list) else scale
        # self.device = device

    def forward(self, data, mode, **kwargs):
        if mode == "train":
            return self.train_step(data, **kwargs)
        else:
            return self.val_step(data, **kwargs)

    def _forward(self, *x):
        # TODO: Only support one output in `forward` function
        assert len(x) == 2, print(len(x))
        return x[0]

    @torch.no_grad()
    def forward_chop(self, data, shave=12, **kwargs):
        '''
        data: a dict composed of multiple data, each data is a tensor
        kwargs: transfer other arguments into model.forward without patch chop
        '''

        # 输出存在维度不一样的情况, 因为网络有多层并且每层尺度不同, 进行分开处理: final_output, intermediate_output
        # TODO: bug: 1. 输出的张量尺寸不一样的时候 无法进行fold操作, 一是因为参数, 二是可能还原不回去
        #            2. unfold的在重叠切的时候是共享原始数据的，所以后者会覆盖前者

        # x = torch.cat(x, dim=0)
        # split_func = self.split_func

        # device = self.device

        patch_size_list = self.patch_size_list
        batchsize = self.crop_batch_size
        scale = self.scale
        # self.axes[1][0].imshow(x[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        # x = x.cpu()
        if len(patch_size_list) == 0:
            raise RuntimeError('patch_size_list is empty') from None
        if len(patch_size_list.keys()) != len(data.keys()):
            raise RuntimeError(
                f"patch_size_list: {patch_size_list} should have {len(data.keys())} variants"
            ) from None

        data_keys = list(data.keys())
        x_unfold_dict = {}
        x_hw_cut_dict = {}
        h_cut_list = []
        w_cut_list = []
        hw_padsize_list = []
        hw_shave_list = []
        x_size_list = []
        x_range_list = []
        for k in data_keys:
            patch_size = patch_size_list[k]
            x = data[k]
            b, c, h, w = x.size()
            x_size_list.append(x.size())
            # assert b == 3, 'batch size should be 1 when doing inference'

            if isinstance(patch_size, (list, tuple)):
                (h_padsize, w_padsize) = patch_size
                hshave, wshave = h_padsize // 2, w_padsize // 2
            else:
                h_padsize = w_padsize = int(patch_size)
                hshave = wshave = patch_size // 2

            h_cut = (h - h_padsize) % (int(hshave / 2))
            w_cut = (w - w_padsize) % (int(wshave / 2))
            h_cut_list.append(h_cut)
            w_cut_list.append(w_cut)
            hw_padsize_list.append((h_padsize, w_padsize))
            hw_shave_list.append((hshave, wshave))

            x_unfold = (
                F.unfold(
                    x, (h_padsize, w_padsize), stride=(int(hshave / 2), wshave // 2)
                )
                .permute(2, 0, 1)
                .contiguous()
            )
            x_unfold = x_unfold.view(b * x_unfold.size(0), c, h_padsize, w_padsize)
            x_unfold_dict[k] = x_unfold
            x_range_list.append(
                x_unfold.size(0) // batchsize + (x_unfold.size(0) % batchsize != 0)
            )
            ################################################
            # 最后一块patch单独计算
            ################################################

            x_hw_cut = x[..., (h - h_padsize) :, (w - w_padsize) :]
            x_hw_cut_dict[k] = x_hw_cut.cuda()

        # print("x_range_list: ", x_range_list)
        # NOTE: input bs should be 1
        # y_hw_cut = self.forward(*[s.to(device) for s in split_func(x_hw_cut, [1, 1, 1], dim=0)], **kwargs).cpu()
        y_hw_cut = self._forward(**x_hw_cut_dict, **kwargs).cpu()

        x_hw_cut_dict.clear()
        torch.cuda.empty_cache()

        # for x, h_padsize, w_padsize, h_cut, w_cut, hshave, wshave in zip(x_list, h_padsize_list, w_padsize_list, h_cut_list, w_cut_list, hshave_list, wshave_list):
        #     b, c, h, w = x.size()
        #     x_h_cut = x[..., (h - h_padsize):, :]
        #     x_w_cut = x[..., :, (w - w_padsize):]
        x_h_cut_list = []
        x_w_cut_list = []
        x_h_top_list = []
        x_w_top_list = []
        for k, (h_padsize, w_padsize) in zip(data_keys, hw_padsize_list):
            x = data[k]
            b, c, h, w = x.size()
            x_h_cut_list.append(x[..., (h - h_padsize) :, :])
            x_w_cut_list.append(x[..., :, (w - w_padsize) :])
            x_h_top_list.append(x[..., :h_padsize, :])
            x_w_top_list.append(x[..., :, :w_padsize])

        y_h_cut = self.cut_h(
            data_keys,
            x_h_cut_list,
            x_size_list,
            h_cut_list,
            w_cut_list,
            hw_padsize_list,
            hw_shave_list,
            scale,
            batchsize,
            **kwargs,
        )
        x_h_cut_list.clear()
        torch.cuda.empty_cache()

        y_w_cut = self.cut_w(
            data_keys,
            x_w_cut_list,
            x_size_list,
            h_cut_list,
            w_cut_list,
            hw_padsize_list,
            hw_shave_list,
            scale,
            batchsize,
            **kwargs,
        )
        x_w_cut_list.clear()
        torch.cuda.empty_cache()

        # self.axes[0][0].imshow(y_h_cut[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        ################################################
        # 左上patch单独计算，不是平均而是覆盖
        ################################################

        y_h_top = self.cut_h(
            data_keys,
            x_h_top_list,
            x_size_list,
            h_cut_list,
            w_cut_list,
            hw_padsize_list,
            hw_shave_list,
            scale,
            batchsize,
            **kwargs,
        )
        y_w_top = self.cut_w(
            data_keys,
            x_w_top_list,
            x_size_list,
            h_cut_list,
            w_cut_list,
            hw_padsize_list,
            hw_shave_list,
            scale,
            batchsize,
            **kwargs,
        )
        x_h_top_list.clear(), x_w_top_list.clear()
        torch.cuda.empty_cache()

        # self.axes[0][1].imshow(y_h_top[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        ################################################
        # img->patch，最大计算crop_s个patch，防止bs*p*p太大
        ################################################

        # x_unfold = x_unfold.view(x_unfold.size(0), -1, c, h_padsize, w_padsize)

        # x_range = x_unfold.size(0) // batchsize + (x_unfold.size(0) % batchsize != 0)
        # x_unfold = x_unfold.to(device)

        # for i in range(x_range):
        #     s_input = [s[:, 0, ...] for s in
        #                split_func(x_unfold[i * batchsize:(i + 1) * batchsize, ...], [1, 1, 1], dim=1)]
        #     iter_outs = []
        #     for j in range(s_input[0].size(0)):
        #         iter_input = [s[j:j + 1] for s in s_input]
        #         iter_res = self.forward(*iter_input, **kwargs)
        #         iter_outs.append(iter_res)
        #     res = torch.cat(iter_outs, dim=0)
        #     del iter_outs, s_input
        #     torch.cuda.empty_cache()
        #     y_unfold.append(res)

        y_unfold = []
        for i in range(x_range_list[0]):
            res = self._forward(
                **{k:
                    s[i * batchsize : (i + 1) * batchsize, ...].cuda()
                    for k, s in x_unfold_dict.items()
                },
                **kwargs,
            )
            torch.cuda.empty_cache()
            y_unfold.append(res)

        if isinstance(y_unfold[0], tuple):
            y_unfold_out = [None] * len(y_unfold[0])
            for i, s in enumerate(zip(*y_unfold)):
                y_unfold_out[i] = s
            y_unfold = y_unfold_out
            y_hw_cut = [s.cpu() for s in y_hw_cut]
        else:
            y_unfold = [y_unfold]
            y_hw_cut = [y_hw_cut.cpu()]

        # =================mix overlapped patches together================
        out = []
        x_size, h_cut, w_cut, (h_padsize, w_padsize), (hshave, wshave) = (
            x_size_list[0],
            h_cut_list[0],
            w_cut_list[0],
            hw_padsize_list[0],
            hw_shave_list[0],
        )
        b, c, h, w = x_size
        for s_unfold, s_h_top, s_w_top, s_h_cut, s_w_cut, s_hw_cut in zip(
            y_unfold, y_h_top, y_w_top, y_h_cut, y_w_cut, y_hw_cut
        ):
            s_unfold = torch.cat(s_unfold, dim=0).cpu()

            y = F.fold(
                s_unfold.view(b, s_unfold.size(0) // b, -1)
                .transpose(1, 2)
                .contiguous(),
                ((h - h_cut) * scale, (w - w_cut) * scale),
                (h_padsize * scale, w_padsize * scale),
                stride=(int(hshave / 2 * scale), int(wshave / 2 * scale)),
            )
            # 312， 480
            # self.axes[0][2].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
            ################################################
            # 第一块patch->y
            ################################################
            y[..., : h_padsize * scale, :] = s_h_top
            y[..., :, : w_padsize * scale] = s_w_top
            # self.axes[0][3].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)

            # central patches
            s_unfold = s_unfold[
                ...,
                int(hshave / 2 * scale) : h_padsize * scale - int(hshave / 2 * scale),
                int(wshave / 2 * scale) : w_padsize * scale - int(wshave / 2 * scale),
            ].contiguous()

            s_inter = F.fold(
                s_unfold.view(b, s_unfold.size(0) // b, -1)
                .transpose(1, 2)
                .contiguous(),
                ((h - h_cut - hshave) * scale, (w - w_cut - wshave) * scale),
                (
                    h_padsize * scale - hshave * scale,
                    w_padsize * scale - wshave * scale,
                ),
                stride=(int(hshave / 2 * scale), int(wshave / 2 * scale)),
            )
            # 1，3，750，540
            #
            s_ones = torch.ones(s_inter.shape, dtype=s_inter.dtype)
            divisor = F.fold(
                F.unfold(
                    s_ones,
                    (
                        h_padsize * scale - hshave * scale,
                        w_padsize * scale - wshave * scale,
                    ),
                    stride=(int(hshave / 2 * scale), int(wshave / 2 * scale)),
                ),
                ((h - h_cut - hshave) * scale, (w - w_cut - wshave) * scale),
                (
                    h_padsize * scale - hshave * scale,
                    w_padsize * scale - wshave * scale,
                ),
                stride=(int(hshave / 2 * scale), int(wshave / 2 * scale)),
            )

            s_inter = s_inter.cpu() / divisor
            # self.axes[1][1].imshow(y_inter[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
            ################################################
            # 第一个半patch
            ################################################
            y[
                ...,
                int(hshave / 2 * scale) : (h - h_cut) * scale - int(hshave / 2 * scale),
                int(wshave / 2 * scale) : (w - w_cut) * scale - int(wshave / 2 * scale),
            ] = s_inter
            # self.axes[1][2].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
            y = torch.cat(
                [
                    y[..., : y.size(2) - int((h_padsize - h_cut) / 2 * scale), :],
                    s_h_cut[..., int((h_padsize - h_cut) / 2 * scale + 0.5) :, :],
                ],
                dim=2,
            )
            # 图分为前半和后半
            # x->y_w_cut
            # model->y_hw_cut TODO:check
            y_w_cat = torch.cat(
                [
                    s_w_cut[
                        ..., : s_w_cut.size(2) - int((h_padsize - h_cut) / 2 * scale), :
                    ],
                    s_hw_cut[..., int((h_padsize - h_cut) / 2 * scale + 0.5) :, :],
                ],
                dim=2,
            )
            y = torch.cat(
                [
                    y[..., :, : y.size(3) - int((w_padsize - w_cut) / 2 * scale)],
                    y_w_cat[..., :, int((w_padsize - w_cut) / 2 * scale + 0.5) :],
                ],
                dim=3,
            )
            out.append(y.cuda())
            # self.axes[1][3].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
            # plt.show()

        return out  # y.cuda()

    def cut_h(
        self,
        data_keys,
        x_h_cut_list,
        x_size_list,
        h_cut_list,
        w_cut_list,
        padsize_list,
        shave_list,
        scale,
        batchsize,
        **kwargs,
    ):
        x_h_cut_unfold_dict = {}
        x_range_list = []
        for idx, (b, c, h, w) in enumerate(x_size_list):
            k = data_keys[idx]
            x_h_cut_unfold = (
                F.unfold(
                    x_h_cut_list[idx],
                    padsize_list[idx],
                    stride=(int(shave_list[idx][0] / 2), int(shave_list[idx][1] / 2)),
                )
                .permute(0, 2, 1)
                .contiguous()
            )  # transpose(0, 2)
            x_h_cut_unfold = x_h_cut_unfold.view(
                b * x_h_cut_unfold.size(1), c, *padsize_list[idx]
            )  # x_h_cut_unfold.size(0), -1, padsize, padsize
            x_h_cut_unfold_dict[k] = x_h_cut_unfold
            x_range = x_h_cut_unfold.size(0) // batchsize + (
                x_h_cut_unfold.size(0) % batchsize != 0
            )
            x_range_list.append(x_range)
        # print("cut_h: x_range_list: ", x_range_list)

        # 多个输入得到一个输出，跟上一版是一样的
        y_h_cut_unfold = []
        for i in range(x_range_list[0]):
            res = self._forward(
                **{k:
                    x_h_cut_unfold_dict[k][i * batchsize : (i + 1) * batchsize, ...].cuda()
                    for k in x_h_cut_unfold_dict.keys()
                },
                **kwargs,
            )
            y_h_cut_unfold.append(res)

        # [[a0, b0, c0], [a1, b1, c1], ...] -> [[a0, a1], [b0, b1], ...] -> [cat() for s in [[a0, a1], [b0, b1], ...]]
        if isinstance(y_h_cut_unfold[0], tuple):
            y_h_cut_unfold_out = [None] * len(y_h_cut_unfold[0])
            for i, s in enumerate(zip(*y_h_cut_unfold)):
                y_h_cut_unfold_out[i] = s
            y_h_cut_unfold = y_h_cut_unfold_out
        else:
            y_h_cut_unfold = [y_h_cut_unfold]
        y_h_cut = []

        padsize = padsize_list[0]
        shave = shave_list[0]
        h_cut = h_cut_list[0]
        w_cut = w_cut_list[0]
        b, c, h, w = x_size_list[0]
        for s_h_cut_unfold in y_h_cut_unfold:
            s_h_cut_unfold = torch.cat(s_h_cut_unfold, dim=0).cpu()
            # s_h_cut_unfold = s_h_cut_unfold.reshape(-1, c, *s_h_cut_unfold.size()[1:])
            # nH*nW, c, k, k: 3, 3, 100, 100 (17, 3, 30, 120)
            # out_size=(30, 600), k=(30, 120)
            s_h_cut = F.fold(
                s_h_cut_unfold.view(b, s_h_cut_unfold.size(0) // b, -1)
                .transpose(1, 2)
                .contiguous(),
                (padsize[0] * scale, (w - w_cut) * scale),
                (padsize[0] * scale, padsize[1] * scale),
                stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)),
            )
            s_h_cut_unfold = s_h_cut_unfold[
                ...,
                :,
                int(shave[1] / 2 * scale) : padsize[1] * scale
                - int(shave[1] / 2 * scale),
            ].contiguous()
            # 17, 3, 30, 60
            # out_size=(30, 540), k=(30, 90)
            s_h_cut_inter = F.fold(
                s_h_cut_unfold.view(b, s_h_cut_unfold.size(0) // b, -1)
                .transpose(1, 2)
                .contiguous(),
                (padsize[0] * scale, (w - w_cut - shave[1]) * scale),
                (padsize[0] * scale, padsize[1] * scale - shave[1] * scale),
                stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)),
            )

            s_ones = torch.ones(s_h_cut_inter.shape, dtype=s_h_cut_inter.dtype)
            divisor = F.fold(
                F.unfold(
                    s_ones,
                    (padsize[0] * scale, padsize[1] * scale - shave[1] * scale),
                    stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)),
                ),
                (padsize[0] * scale, (w - w_cut - shave[1]) * scale),
                (padsize[0] * scale, padsize[1] * scale - shave[1] * scale),
                stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)),
            )
            s_h_cut_inter = s_h_cut_inter.cpu() / divisor

            s_h_cut[
                ...,
                :,
                int(shave[1] / 2 * scale) : (w - w_cut) * scale
                - int(shave[1] / 2 * scale),
            ] = s_h_cut_inter
            y_h_cut.append(s_h_cut)

        return y_h_cut

    def cut_w(
        self,
        data_keys,
        x_w_cut_list,
        x_size_list,
        h_cut_list,
        w_cut_list,
        padsize_list,
        shave_list,
        scale,
        batchsize,
        **kwargs,
    ):
        x_w_cut_unfold_dict = {}
        x_range_list = []
        # 1, 3, 792, 120 -> (30, 120), s=(7, 30) -> 109, 1, 3, 30, 120
        for idx, (b, c, h, w) in enumerate(x_size_list):
            k = data_keys[idx]
            x_w_cut_unfold = (
                F.unfold(
                    x_w_cut_list[idx],
                    padsize_list[idx],
                    stride=(int(shave_list[idx][0] / 2), int(shave_list[idx][1] / 2)),
                )
                .permute(0, 2, 1)
                .contiguous()
            )

            x_w_cut_unfold = x_w_cut_unfold.view(
                b * x_w_cut_unfold.size(1), c, *padsize_list[idx]
            )
            x_range = x_w_cut_unfold.size(0) // batchsize + (
                x_w_cut_unfold.size(0) % batchsize != 0
            )
            x_w_cut_unfold = x_w_cut_unfold.cuda()
            x_w_cut_unfold_dict[k] = x_w_cut_unfold
            x_range_list.append(x_range)
        # print("cut_w: x_range_list: ", x_range_list)
        # TODO: [[a0, b0], [a1, b1], ...] -> [[a0, a1], [b0, b1], ...] -> [cat() for s in [[a0, a1], [b0, b1], ...]]
        # for i in range(x_range):
        #     s_input = [s[:, 0, ...].to(self.device) for s in
        #                split_func(x_w_cut_unfold[i * batchsize:(i + 1) * batchsize, ...], [1, 1, 1], dim=1)]
        #     iter_outs = []
        #     for j in range(s_input[0].size(0)):
        #         # reduce every input bs to 1
        #         iter_input = [s[j:j + 1] for s in s_input]  # [1, c, p, p]
        #         iter_res = self.forward(*iter_input, **kwargs)
        #         iter_outs.append(iter_res.cpu())
        #     res = torch.cat(iter_outs, dim=0)
        #     del iter_outs, s_input
        #     torch.cuda.empty_cache()
        #     y_w_cut_unfold.append(res)

        y_w_cut_unfold = []
        for i in range(x_range_list[0]):
            # res = self.forward(*[s[:, 0, ...] for s in split_func(x_w_cut_unfold[i * batchsize:(i + 1) * batchsize,
            #                                                       ...], [1, 1, 1], dim=1)], **kwargs)
            res = self._forward(
                **{k:
                    x_w_cut_unfold_dict[k][i * batchsize : (i + 1) * batchsize, ...].cuda()
                    for k in x_w_cut_unfold_dict.keys()
                },
                **kwargs,
            )
            torch.cuda.empty_cache()
            y_w_cut_unfold.append(res)

        if isinstance(y_w_cut_unfold[0], tuple):
            y_w_cut_unfold_out = [None] * len(y_w_cut_unfold[0])
            for i, s in enumerate(zip(*y_w_cut_unfold)):
                y_w_cut_unfold_out[i] = s
            y_w_cut_unfold = y_w_cut_unfold_out
        else:
            y_w_cut_unfold = [y_w_cut_unfold]

        y_w_cut = []
        padsize = padsize_list[0]
        shave = shave_list[0]
        h_cut = h_cut_list[0]
        w_cut = w_cut_list[0]
        b, c, h, w = x_size_list[0]
        for s_w_cut_unfold in y_w_cut_unfold:
            s_w_cut_unfold = torch.cat(s_w_cut_unfold, dim=0).cpu()
            # s_w_cut_unfold = s_w_cut_unfold.reshape(-1, c, *s_w_cut_unfold.size()[1:])
            # 109,3,30,120
            # out_size=(786, 120), k=(30, 120)
            s_w_cut = F.fold(
                s_w_cut_unfold.view(b, s_w_cut_unfold.size(0) // b, -1)
                .transpose(1, 2)
                .contiguous(),
                ((h - h_cut) * scale, padsize[1] * scale),
                (padsize[0] * scale, padsize[1] * scale),
                stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)),
            )
            s_w_cut_unfold = s_w_cut_unfold[
                ...,
                int(shave[0] / 2 * scale) : padsize[0] * scale
                - int(shave[0] / 2 * scale),
                :,
            ].contiguous()
            # 109, 3, 16, 120
            # out_size=(771, 120), k=(15, 120)
            s_w_cut_inter = F.fold(
                s_w_cut_unfold.view(b, s_w_cut_unfold.size(0) // b, -1)
                .transpose(1, 2)
                .contiguous(),
                ((h - h_cut - shave[0]) * scale, padsize[1] * scale),
                (padsize[0] * scale - shave[0] * scale, padsize[1] * scale),
                stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)),
            )

            s_ones = torch.ones(s_w_cut_inter.shape, dtype=s_w_cut_inter.dtype)
            divisor = F.fold(
                F.unfold(
                    s_ones,
                    (padsize[0] * scale - shave[0] * scale, padsize[1] * scale),
                    stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)),
                ),
                ((h - h_cut - shave[0]) * scale, padsize[1] * scale),
                (padsize[0] * scale - shave[0] * scale, padsize[1] * scale),
                stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)),
            )
            s_w_cut_inter = s_w_cut_inter.cpu() / divisor

            s_w_cut[
                ...,
                int(shave[0] / 2 * scale) : (h - h_cut) * scale
                - int(shave[0] / 2 * scale),
                :,
            ] = s_w_cut_inter
            y_w_cut.append(s_w_cut)

        return y_w_cut
