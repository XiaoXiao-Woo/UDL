# UDL (Make Available on [PyPI](https://pypi.org/project/udl-vis/) :tada:)

UDL is a unified Pytorch framework for vision research:

* UDL has a faster library loading speed and a more convenient reflection mechanism to call different models and methods.
* UDL is based on MMCV which provides the following functionalities.
* UDL is based on NNI to perform automatic machine learning.



[English](https://github.com/XiaoXiao-Woo/UDL/edit/dev/README.md) | [简体中文](https://github.com/XiaoXiao-Woo/UDL/edit/dev/README_zh.md)

See the [repo](https://github.com/liangjiandeng/PanCollection) for more detailed descriptions. 

## Note

For the implementation of DCFNet as described in the ICCV paper "Dynamic Cross Feature Fusion for Remote Sensing Pansharpening," please refer to the [branch](https://github.com/XiaoXiao-Woo/UDL/blob/UDL_DCFNet) in the this repository.

## Requirements
* Python3.7+, Pytorch>=1.9.0
* NVIDIA GPU + CUDA
* Run `python setup.py develop`

Note: Our project is based on MMCV, but you needn't to install it currently.

## Quick Start

> pip install -i pancollection https://pypi.org/simple

PanCollection is used for remote sensing pansharpening, which is one of our applications.

### Quick Start for developer

**Step0.** We use UDL in PanCollection, first please set your Python environment.

>git clone https://github.com/XiaoXiao-Woo/UDL
> 
> git clone https://github.com/XiaoXiao-Woo/PanCollection

Then, 

> python setup.py develop


## Plannings

Please expect more tasks and models

- [x] [pansharpening](https://github.com/XiaoXiao-Woo/PanCollection)
  - [x] models

- [x] [derain](https://github.com/XiaoXiao-Woo/derain) (not available on PyPI)
  - [x] models

- [ ] HISR
  - [ ] models
 
 - [ ] Improve MMCV repo to simplify expensive hooks.

## Contribution
We appreciate all contributions to improving PanCollection. Looking forward to your contribution to PanCollection.


## Citation
Please cite this project if you use datasets or the toolbox in your research.
```
@misc{PanCollection,
    author = {Xiao Wu, Liang-Jian Deng and Ran Ran},
    title = {"PanCollection" for Remote Sensing Pansharpening},
    url = {https://github.com/XiaoXiao-Woo/PanCollection/},
    year = {2022},
}

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

## License & Copyright
This project is open sourced under GNU General Public License v3.0.

