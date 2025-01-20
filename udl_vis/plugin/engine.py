import warnings
warnings.filterwarnings("ignore", module="torch.utils")
warnings.filterwarnings("ignore", module="deepspeed.accelerator")
warnings.filterwarnings("ignore", module="torch.cuda.amp", category=FutureWarning)

def get_engine(backend):

    if backend == "mmcv1":
        from .mmcv1_engine import run_mmcv1_engine as run_engine

    if backend == "naive":
        from .naive_engine import run_naive_engine as run_engine

    elif backend == "lightning":
        from .lightning_engine import run_lightning_engine as run_engine

    elif backend == "accelerate":
        from .accelerate_engine import run_accelerate_engine as run_engine

    else:
        raise NotImplementedError(f"Engine: {backend} does not exist.")

    return run_engine
