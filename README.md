# UDL
UDL is a practicable framework used in Deep Learning (computer vision).
## Benchmark

codes, results and models are available in UDL, please contact [@Liang-Jian Deng](https://liangjiandeng.github.io/) (corresponding author)

<details open>
<summary>Pansharpening model zoo:
</summary>

* PNN (RS'2016)
* PanNet (CVPR'2017)
* DiCNN1 (JSTAR'2019)
* FusionNet (TGRS'2020)
* DCFNet (ICCV'2021)

</details>



## Results of DCFNet 


### Quantitative results


|      wv3       |        SAM        |        ERGAS       |
| :------------: | :---------------: | :----------------: |
|   new_data10   |       3.934       |       2.531        |
|   new_data11   |       4.133       |       2.630        |
| new_data12_512 |       4.108       |       2.712        |
|   new_data6    |       2.638       |       1.461        |
|   new_data7    |       3.866       |       2.820        |
|   new_data8    |       3.257       |       2.210        |
|   new_data9    |       4.154       |       2.718        |
|    Avg(std)    | 3.727(0.571) | 2.440(0.474)  |
|  Ideal Value   |         0         |         0          |

|  wv3_1258   |        SAM         |      ERGAS         |
| :---------: | :----------------: | :----------------: |
|  Avg(std)   |    3.377(1.200)    |    2.257(0.910)    |
| Ideal Value |         0          |         0          |

### Visual results

please see the paper and the sub-directory: **./UDL/results/DCFNet**



## Install [Option]

please run ```python setup.py develop```

## Usage

open UDL/panshaprening/tests, run the following code:

```
python run_DCFNet.py
```

Note that default configures don't fit other environments, you can modify configures in **pansharpening/models/DCFNet/option_DCFNet.py**.

Benefit from mmcv/config.py, the project has the global configures in Basis/option.py, option_DCFNet  inherits directly from Basis/option.py.

### 1. Data preparation

You need to download WorldView-3 datasets.

The directory tree should be look like this:

```
|-$ROOT/datasets
├── pansharpening
│   ├── training_data
│   │   ├── train_wv3_10000.h5
│   │   ├── valid_wv3_10000.h5
│   ├── test_data
│   │   ├── WV3_Simu
│   │   │   ├── new_data6.mat
│   │   │   ├── new_data7.mat
│   │   │   ├── ...
│   │   ├── WV3_Simu_mulExm
│   │   │   ├── test1_mulExm1258.mat
```

### 2. Training

```args.eval = False, args.dataset='wv3'```

### 3. Inference

```args.eval = True, args.dataset='wv3_singleMat'```

## Plannings

Please expect more tasks and models

- [x] pansharpening
  - [ ] models

- [ ] derain
  - [ ] models

- [ ] HISR
  - [ ] models

## Contribution
We appreciate all contributions to improve UDL. Looking forward to your contribution to UDL.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.
```
@InProceedings{Wu_2021_ICCV,
    author    = {Wu, Xiao and Huang, Ting-Zhu and Deng, Liang-Jian and Zhang, Tian-Jing},
    title     = {Dynamic Cross Feature Fusion for Remote Sensing Pansharpening},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14687-14696}
}
```
## Acknowledgement
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [HRNet ](https://github.com/HRNet/HRNet-Semantic-Segmentation): High-resolution networks and Segmentation Transformer for Semantic Segmentation

## License & Copyright

This project is open sourced under GNU General Public License v3.0
