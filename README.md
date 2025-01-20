# Unified Deep Learning Framework for vision tasks (Release v1.0.0 [PyPI](https://pypi.org/project/udl-vis/) :tada:)


[[UDL](https://github.com/XiaoXiao-Woo/UDL)] [[PanCollection](https://github.com/XiaoXiao-Woo/PanCollection)] [[HyperSpectralCollection]()] [[MSIF]()] 

UDL is a unified Pytorch framework for vision research:

* UDL has a faster library loading speed and a more convenient reflection mechanism to call different methods.
* UDL is based on [accelerate](https://github.com/huggingface/accelerate)/[transformers](https://github.com/huggingface/transformers)/[lightning](https://github.com/LightningAI/lightning)/[MMCV1](https://github.com/open-mmlab/mmcv) which provides more functionalities.
* ~~UDL is based on [NNI](https://nni.readthedocs.io/en/stable/) to perform automatic machine learning.~~ We will release UDL-CIL to perform automatic experimental management.
* UDL is based on [Hydra](https://hydra.cc/docs/intro/) to manage the configuration of the experiment.
* UDL is based on [Huggingface](https://huggingface.co/) to download and upload datasets and models.


## News🔥🔥🔥
* **2025-?**: We will Release **UDL-CIL** for experimental management. 🎉
* **2025.1**: Release **UDL v1.0.0**. 🎉
* :art: The FC-Former convers multiple multi-source image fusion scenes:
  * Multispectral and hyperspectral image fusion;
  * Remote sensing pansharpening;
  * Visible and infrared image fusion (VIS-IR); 
  * Digital photographic image fusion: Multi-focus image fusion (MFF) and multi-exposure image fusion (MEF). 
* 🎁 We will release a new version of [UDL](https://github.com/XiaoXiao-Woo/UDL), [PanCollection](https://github.com/XiaoXiao-Woo/PanCollection). Furthermore, we also release repositories of [HyperSpectralCollection, comming soon](https://github.com/XiaoXiao-Woo/HyperSpectralCollection) and [MSIF, comming soon](https://github.com/XiaoXiao-Woo/MSIF).
* **2025.1**: Release **PanCollection v1.0.0**. 🎉
* **2024.12**: *Fully-connected Transformer for Multi-source Image Fusion.*  IEEE T-PAMI 2025. ([Paper](coming soon), [Code](https://github.com/XiaoXiao-Woo/FC-Former)) 📖
* **2024.12**: *Deep Learning in Remote Sensing Image Fusion: Methods, Protocols, Data, and Future Perspectives.* IEEE GRSM 2024. ([Paper](https://ieeexplore.ieee.org/abstract/document/10778974)) 📖
* **2024.10**: *SSDiff: Spatial-spectral Integrated Diffusion Model for Remote Sensing Pansharpening.* NeurIPS 2024. ([Paper](https://openreview.net/pdf?id=QMVydwvrx7), [Code](https://github.com/Z-ypnos/SSDiff_main)) 🚀
* **2024.5**: *Content-Adaptive Non-Local Convolution for Remote Sensing Pansharpening.* CVPR 2024. ([Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Duan_Content-Adaptive_Non-Local_Convolution_for_Remote_Sensing_Pansharpening_CVPR_2024_paper.html), [Code](https://github.com/YuleDuan/CANConv)) 🚀
* **2022.5**: We released PanCollection. 🎉
* **2022.5**: We released UDL. 🎉

## Features

| **Features**                                                                     | **Value** |
| -------------------------------------------------------------------------------- | --------- |
| Automatic experimental configuration                                             | ✅         |
| Lightning, transformers, accelerate, mmcv, FSDP, DeepSpeed, etc.                 | ✅         |
| Downstream tasks, including pansharpening, hyperspectral image fusion, etc.       | ✅         |
| Download and upload huggingface models                                           | ✅         |


## Recommendations

We recommend users utilize this code toolbox alongside our other open-source datasets for optimal results, such as [PanCollection](https://github.com/XiaoXiao-Woo/PanCollection), [HyperSpectralCollection](https://github.com/XiaoXiao-Woo/HyperSpectralCollection), [MSIF](https://github.com/XiaoXiao-Woo/MSIF), etc.




## Quick Start 🤗
1. Install our basic training/inference repo with multiple Pytorch framework support. You can select one of backends: accelerate, lightning, transformers, and mmcv1.
```
pip install udl_vis --upgrade
```
2. Install the [Task] repo by populating one of the following repos: ``pancollection``, ``mhif`` ``msif``.

``` 
pip install [Task] --upgrade
```

3. Then in this repository:
```
pip install -e .
```



## Train in PanCollection
Here, we take Huggingface ``accelerate`` as the backend, thus
```
sh  run_accelerate_ddp_pansharpening.sh
```
The script will call ``python_scripts/accelerate_pansharpening.py`` to run the FC-Former. More information you can see ``model.yaml``


## Inference 
You only need to update the `fusionnet.yaml` as follows:
```yaml
  eval : true # change false to true
  workflow:
    - ["test", 1]
    # - ["train", 10] # comment it
```




## Test Your Metrics
Finally, we provide the correspoinding Matlab toolboxes to test the metrics on those tasks. Please check them in our repo.

* MHIF
  * [HyperSpectralToolbox, comming soon](https://github.com/XiaoXiao-Woo/HyperSpectralToolbox)
  * `run_hisr.m`
  
* Pansharpening
  * [PanCollection](https://github.com/XiaoXiao-Woo/PanCollection)
  * [DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox)
  
* VIS-IR, MEF, MFF
  * [Image Fusion Toolbox, comming soon]()



## Experiments
In this part, we conduct experiments in the following cases. All settings can be changed in `dataset.yaml`.
* MHIF: CAVE and Harvard. 
  * See our repo [MHIF, comming soon]() for more details;
* VIS-IR image fusion: TNO and RoadScene datasets. 
  * See our repo [MSIF, comming soon]() for more details;
* Pansharpening: WorldView-3, GaoFen-2, QuickBird datasets.
  * See our repo [PanCollection, comming soon]() for more details;
* Digital photographic image fusion: MFF-WHU, MEF-Lytro,MEF-SLICE, MEFB. 
  * See our repo [MSIF, comming soon]() for more details;




## Citation
Please cite this project if you use datasets or the toolbox in your research.
```bibtex
@article{FCFormer,
  title={Fully-connected Transformer for Multi-source  Image Fusion},
  author={Xiao Wu, Zi-Han Cao, Ting-Zhu Huang, Liang-Jian Deng, Jocelyn Chanussot, and Gemine Vivone}
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}

```bibtex
@inproceedings{Wu_2021_ICCV,
    author    = {Wu, Xiao and Huang, Ting-Zhu and Deng, Liang-Jian and Zhang, Tian-Jing},
    title     = {Dynamic Cross Feature Fusion for Remote Sensing Pansharpening},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14687-14696}
}
```

```bibitex
@article{vivone2024deep,
  title={Deep Learning in Remote Sensing Image Fusion: Methods, protocols, data, and future perspectives},
  author={Vivone, Gemine and Deng, Liang-Jian and Deng, Shangqi and Hong, Danfeng and Jiang, Menghui and Li, Chenyu and Li, Wei and Shen, Huanfeng and Wu, Xiao and Xiao, Jin-Liang and others},
  journal={IEEE Geoscience and Remote Sensing Magazine},
  year={2024},
  publisher={IEEE}
}
```

```bibtex
@article{zhong2024ssdiff,
  title={SSDiff: Spatial-spectral Integrated Diffusion Model for Remote Sensing Pansharpening},
  author={Zhong, Yu and Wu, Xiao and Deng, Liang-Jian and Cao, Zihan},
  journal={arXiv preprint arXiv:2404.11537},
  year={2024}
}
```

```bibtex   
@ARTICLE{duancvpr2024,
    title={Content-Adaptive Non-Local Convolution for Remote Sensing Pansharpening},
    author={Yule Duan, Xiao Wu, Haoyu Deng, Liang-Jian Deng*},
    journal={IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
    year={2024}
}
```

```bibtex
@ARTICLE{dengijcai2023,
    title={Bidirectional Dilation Transformer for Multispectral and Hyperspectral Image Fusion},
    author={Shang-Qi Deng, Liang-Jian Deng*, Xiao Wu, Ran Ran, Rui Wen},
    journal={International Joint Conference on Artificial Intelligence (IJCAI)},
    year={2023}
}
```

```bibtex
@misc{PanCollection,
    author = {Xiao Wu, Liang-Jian Deng and Ran Ran},
    title = {"PanCollection" for Remote Sensing Pansharpening},
    url = {https://github.com/XiaoXiao-Woo/PanCollection/},
    year = {2022}
}
```


## License
This project is open sourced under GNU General Public License v3.0.


## Contact
If you have any questions, please contact us at:
* Xiao Wu: xiao.wu@mbzuai.ac.ae
