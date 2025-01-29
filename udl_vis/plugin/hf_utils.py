import os
from huggingface_hub import HfApi, HfFolder, hf_hub_download, snapshot_download

os.environ["HF_HUB_CACHE"] = "/home/yutong.xie/xiaowu/huggingface/hf_cache"
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"
os.makedirs(os.getenv("HF_HUB_CACHE"), exist_ok=True)


def download_single_model(
    repo_id="elsting/PanCollection",
    filename="PanCollection/wv3/FusionNet/FusionNet.pth.tar",
    local_dir=None,
    cache_dir=None,
    token=None,
):

    model = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=os.environ.get("HF_TOKEN", token),
        local_dir=os.path.dirname(os.environ.get("HF_HUB_CACHE", local_dir)),
        cache_dir=os.environ.get("HF_HUB_CACHE", cache_dir),
    )
    print(model, type(model))


def snapshot_download_models(
    repo_id="elsting/PanCollection", local_dir=None, cache_dir=None, token=None
):

    snapshot_download(
        repo_id=repo_id,
        library_name="PanCollection",
        revision="main",
        token=os.environ.get("HF_TOKEN", token),
        local_dir=os.path.dirname(os.environ.get("HF_HUB_CACHE", local_dir)),
        cache_dir=os.environ.get("HF_HUB_CACHE", cache_dir),
    )


if __name__ == "__main__":
    snapshot_download_models()
