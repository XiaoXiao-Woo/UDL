import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Trainer, TrainingArguments
from transformers import Trainer, TrainingArguments, AutoModel, BeitForMaskedImageModeling
from transformers.trainer_utils import get_last_checkpoint
from torchvision.transforms import Compose, ToTensor
from einops.layers.torch import Rearrange
from pancollection import getDataSession
from udl_vis.Basis.auxiliary import set_random_seed
# import datasets
import numpy as np
from udl_vis.Basis.auxiliary import MetricLogger
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_utils import EvalLoopOutput, has_length
from transformers.trainer_pt_utils import find_batch_size, IterableDatasetShard
from transformers.utils import logging
import time

logger = logging.get_logger(__name__)

def inspect_inputs(func):
    import inspect
    print(inspect.signature(func).parameters)
    # print(inspect.getfullargspec(func))



class CustomTrainer(Trainer):
    def __init__(self, cfg, training_args, task_model, build_model, train_dataset=None, val_dataset=None, eval_dataset=None):
        
        self.cfg = cfg

        model, criterion, optimizer, self.scheduler = build_model(device=cfg.device)(cfg)
        self.task_model = task_model(cfg.device, model, criterion)
        # test_log_iter_interval = cfg.test_log_iter_interval
        
        super().__init__(model, training_args, 
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         data_collator=self.data_collator)
        
        self.log_buffer = MetricLogger(logger=logger, delimiter="  ")
        self.optimizer = optimizer

    def data_collator(self, features):
        batch = {}
        for k in features[0].keys():
            batch[k] = torch.stack([f[k] for f in features])
        
        return batch
    
    # Trainer's training_step
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        
        
        with self.compute_loss_context_manager():
            outputs = self.train_step(inputs, mode="train")    
        del inputs
        torch.cuda.empty_cache()
        
        kwargs = {}
        
        self.accelerator.backward(outputs['loss'], **kwargs)
        # Finally we need to normalize the loss for reporting
        if num_items_in_batch is None:
            return outputs['loss'].detach() / self.args.gradient_accumulation_steps
        
        return outputs
    
    def train_step(self, batch, mode="train"):
        try:
            loss = self.task_model.train_step(batch, mode=mode)
        except Exception as e:
            print(e)
            inspect_inputs(self.task_model.train_step)
            raise e
        return loss


    def val_step(self, batch, mode="val"):
        self.mode = "val"
        outputs = self.prediction_step(self.task_model.mdoel, batch)
    
    
    # Trainer's evaluation_loop. It's very, very ugly codes
    def evaluation_loop(self, dataloader,
                            description: str,
                            prediction_loss_only = None,
                            ignore_keys = None,
                            metric_key_prefix: str = "eval",):
        
        args = self.args
        
        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
            
        model = self._wrap_model(self.task_model.model, training=False, dataloader=dataloader)
        
        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or self.is_fsdp_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size
        
        observed_num_examples = 0
        
        
        self.log_buffer.clear()
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size
                    
            outputs = self.task_model.val_step(inputs, mode="test",
                                            test=self.cfg.test,
                                            save_fmt=self.cfg.save_fmt,
                                            idx=step,
                                            filename=self.cfg.filename,
                                            img_range=self.cfg.img_range,
                                            test_mode=True)
        
        
        self.log_buffer.update_dict(outputs["log_vars"])
        metrics = {k: meter.val if not hasattr(meter, "image") else meter.image for k, meter in self.log_buffer.meters.items()}
        
        # Number of samples
        if has_length(self.eval_dataset):
            num_samples = len(self.eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(self.eval_dataset, IterableDatasetShard) and getattr(self.eval_dataset, "num_examples", 0) > 0:
            num_samples = self.eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples
        
        
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)
    
    
    def create_optimizer(self):
        return self.optimizer
    
    
    # def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    #     ...
        
    def compute_loss(self, model, inputs, 
                     return_outputs=False, 
                     num_items_in_batch=None):
        
        outputs = None
        loss = self.train_step(inputs)

        return (loss, outputs) if return_outputs else loss

# class MyDatasetConfig(datasets.BuilderConfig):


#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

# class MyDataset(datasets.GeneratorBasedBuilder):



#     BUILDER_CONFIGS = [
#         MyDatasetConfig(name="default", version=datasets.Version("1.0.0"), description="My custom dataset.")
#     ]

#     def _info(self):
#         return datasets.DatasetInfo(
#             features=datasets.Features({
#                 "ms": datasets.Array3D(dtype="float32", shape=(8, 64, 64)),
#                 "gt": datasets.Array3D(dtype="float32", shape=(8, 256, 256)),
#                 "lms": datasets.Array3D(dtype="float32", shape=(8, 256, 256)),
#                 "pan": datasets.Array3D(dtype="float32", shape=(1, 256, 256)),
#             })
#         )

#     def _split_generators(self, dl_manager):

#         return [
#             datasets.SplitGenerator(
#                 name=datasets.Split.TRAIN,
#                 gen_kwargs={"start": 0, "end": 9714}
#             ),
#         ]

#     def _generate_examples(self, start, end):
#         # ms = torch.randn(100, 8, 64, 64).numpy()
#         # gt = torch.rand(100, 8, 256, 256).numpy()
#         # lms = torch.randn(100, 8, 256, 256).numpy()
#         # pan = torch.rand(100, 1, 256, 256).numpy()
#         # data = {"ms": ms, "gt": gt, "lms": lms, "pan": pan}
#         # data = h5py.File("/Data/Datasets/pansharpening_2/PanCollection/training_data/train_wv3_9714.h5")
#         data = {"ms": torch.from_numpy(np.asarray(data["ms"])).float(),
#                   "gt": torch.from_numpy(np.asarray(data["gt"])).float(), 
#                   "lms": torch.from_numpy(np.asarray(data["lms"])).float(),
#                   "pan": torch.from_numpy(np.asarray(data["pan"])).float()}


#         for idx in range(start, end):
#             yield idx, {k: v[idx].cuda() for k, v in data.items()}




def train_transforms(example_batch):
        
    _train_transforms = Compose(
            [
                # RandomResizedCrop(size),
                # RandomHorizontalFlip(),
                ToTensor(),
                Rearrange("b h w c -> b c h w"),
                # normalize,
            ]
        )
    for k, v in example_batch.items():
        print(k, v)
        example_batch[k] = [_train_transforms(v)]
        print(example_batch[k].shape)
    return example_batch



def val_test_transforms(example_batch):
        
    _train_transforms = Compose(
            [
                ToTensor(),
                Rearrange("b h w c -> b c h w"),
            ]
        )
    for k, v in example_batch.items():
            print(v.shape)
            example_batch[k] = [
                _train_transforms(v)
            ]
            print(example_batch[k].shape)
    return example_batch

    



def run_transformers_engine(cfg,
    logger,
    task_model,
    build_model,
    getDataSession,
    **kwargs):
    
    
    # print("Note that: getDataSession is replaced because transformers use a new dataset definition, ")
    # dataset_builder = MyDataset()
    # dataset_builder.download_and_prepare()
    # dataset = dataset_builder.as_dataset()
    # print(dataset)
    
    
    sess = getDataSession(cfg)
    train_loader, train_sampler, generator = sess.get_dataloader(
        cfg.dataset.train_name, cfg.distributed, None
    )
    
    
    eval_loader, eval_sampler = sess.get_eval_dataloader(
        cfg.dataset.test_name, cfg.distributed
    )
    
    
    # dataset["train"].set_transform(train_transforms)
    
    # model = AutoModel.from_pretrained("microsoft/dit-base")

    
    # model = TransformersEngine(cfg, task_model, build_model)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=cfg.work_dir,
        logging_dir=f"{cfg.work_dir}/logs",
        logging_steps=cfg.train_log_iter_interval,
        per_device_train_batch_size=cfg.samples_per_gpu,
        per_device_eval_batch_size=cfg.test_samples_per_gpu,
        num_train_epochs=cfg.epochs,
        save_steps=3,
        dataloader_num_workers=cfg.workers_per_gpu,
        save_total_limit=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        metric_for_best_model="SAM",
        greater_is_better=False,
        remove_unused_columns=False
    )

    # Initialize the Trainer
    trainer = CustomTrainer(
        cfg,
        training_args=training_args,
        build_model=build_model,
        task_model=task_model,
        train_dataset=train_loader.dataset,#dataset.get("train"),
        eval_dataset=eval_loader.dataset
    )

    # Start training
    trainer.train()



def test_pansharpening():
    from rich.traceback import install
    install()
    import hydra
    from omegaconf import OmegaConf, DictConfig
    from udl_vis.mmcv.utils.logging import print_log, create_logger
    from udl_vis.Basis.option import Config
    from pancollection.models.FusionNet.model_fusionnet import build_fusionnet
    from pancollection.models.base_model import PanSharpeningModel
    from hydra.core.hydra_config import HydraConfig

    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    set_random_seed(42)

    @hydra.main(config_path="/home/dsq/NIPS/PanCollection/pancollection/configs", config_name="model")
    def main(cfg: DictConfig):
        
        if isinstance(cfg, DictConfig):
            cfg = Config(OmegaConf.to_container(cfg, resolve=True))
            cfg.merge_from_dict(cfg.args)
            hydra_cfg = HydraConfig.get()
        cfg.work_dir = hydra_cfg.runtime.output_dir
        print(cfg.pretty_text)
  
        cfg.distributed = False
        cfg.dataset_type = "Dummy"
        
        logger = create_logger(cfg, work_dir=cfg.work_dir)
        run_transformers_engine(cfg, logger, PanSharpeningModel, build_fusionnet, getDataSession)

    
    return main()




if __name__ == "__main__":
    test_pansharpening()