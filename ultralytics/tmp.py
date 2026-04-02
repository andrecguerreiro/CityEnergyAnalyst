# smoke_test_model.py
import torch
from pathlib import Path
from ultralytics.nn.tasks import RoofModel
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import (
    DEFAULT_CFG,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)

def print_info(m):
    head = m.model[-1]
    print("Head attributes:")
    for k in ("nc", "ne", "no", "reg_max", "nl", "stride"):
        print(f"  {k}: {getattr(head, k, None)}")
    print("Model summary done.")

def smoke(cfg="yolo11n-roof.yaml", imgsz=640, device="cpu"):
    m = RoofModel(cfg=cfg, ch=3, verbose=False)

    m.to(device)
    m.eval()
    print("Model loaded.")
    print_info(m)

    img = torch.randn(1, 3, imgsz, imgsz, device=device)
    # Try forward in eval (should set extras)
    out = m(img)
    print("Forward (eval) returned type:", type(out))
    # If extras stored on head, print shape
    head = m.model[-1]
    if hasattr(head, "extras"):
        print("extras shape:", head.extras.shape)
    else:
        print("No `extras` attribute on head after eval forward.")

    # Try training forward (for training route)
    m.train()
    try:
        out_train = m(img)
        print("Forward (train) OK. returned type:", type(out_train))
    except Exception as e:
        print("Train forward failed:", e)

def test_loss(cfg="yolo11n-roof.yaml", imgsz=640, device="cpu"):
    from ultralytics.utils.loss import v8RoofLoss
    from ultralytics.nn.tasks import RoofModel

    m = RoofModel(cfg=cfg, ch=3, verbose=False)
    m.args = DEFAULT_CFG  # or a custom dict with required hyperparameters
    m.to(device)
    m.train()
    criterion = v8RoofLoss(m)

    # Mock batch: should include "extras" key for target_extras
    tmpbatch = {
        "img": torch.randn(1, 3, imgsz, imgsz, device=device),
        "cls": torch.randint(0, 1, (1, 100, 1), device=device),
        "bboxes": torch.randn(1, 100, 4, device=device),
        "extras": torch.randn(1, 100, 3, device=device),  # area, height, inclination
    }

    bs = tmpbatch["img"].shape[0]      # 1 in your mock
    nt = tmpbatch["bboxes"].shape[1]   # 100 in your mock
    # For general batch size:
    batch_idx = torch.arange(bs, device=tmpbatch["img"].device).unsqueeze(1).repeat(1, nt).view(-1)

    batch = {
        "img": torch.randn(1, 3, imgsz, imgsz, device=device),
        "cls": torch.randint(0, 1, (1, 100, 1), device=device),
        "bboxes": torch.randn(1, 100, 4, device=device),
        "extras": torch.randn(1, 100, 3, device=device),  # area, height, inclination
    }

    # For your (bs=1) case this will be all zeros of length 100
    batch["batch_idx"] = batch_idx
    # Also flatten cls and bboxes to match Ultralytics expectation:
    batch["cls"] = tmpbatch["cls"].view(-1)          # (M,)
    batch["bboxes"] = tmpbatch["bboxes"].view(-1, 4) # (M,4)

    feats, extras = m(batch["img"])         # returns same structure as training forward()
    preds = (feats, extras)

    # Mock predictions: (detect_out, extras)
    #preds = (
    #    torch.randn(1, 100, 8, device=device),  # detect_out: xywh, conf, cls, area, height, inclination
    #    torch.randn(1, 3, 100, device=device),  # extras: area, height, inclination
    #)

    try:
        total_loss, loss_details = criterion(preds, batch)
        print("Loss computation successful!")
        print(f"Total loss: {total_loss.item()}")
        print(f"Loss details: {loss_details}")
    except Exception as e:
        print(f"Loss computation failed: {e}")

def test_roof_dataset_loading():
    import yaml, torch
    from ultralytics.data import build_dataloader
    from ultralytics.data.augment import v8_transforms
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.cfg import get_cfg, get_save_dir
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    cfg = yaml.safe_load(open("datasets/roof_dataset.yaml"))
    img_root = cfg.get("path") or cfg.get("train") or str(Path("datasets").resolve())

    overrides = {
        "data": "datasets/roof_dataset.yaml",
        "model": "yolo11n-roof.yaml",   # or path to model config / weights
        "epochs": 1,
        "batch": 2
    }
    
    data = check_det_dataset(cfg)
    
    args = get_cfg(DEFAULT_CFG, overrides)  # or a custom dict with required hyperparameters
    dataset = build_yolo_dataset(args , img_path= img_root, batch = 2 , data = cfg, mode="train", rect=False, stride=32)



    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils import IterableSimpleNamespace

    augmented_data = dataset.transforms(dataset[0])
    print("i am god")

def test_trainer():
    from ultralytics.models.yolo.roof import RoofTrainer
    overrides = {
        "data": "datasets/roof_dataset.yaml",
        "model": "yolo11n-roof.yaml",   # or path to model config / weights
        "epochs": 1,
        "batch": 2
    }
    trainer = RoofTrainer(overrides=overrides)
    trainer.train()

if __name__ == "__main__":
    #smoke()
    #test_loss()
    test_roof_dataset_loading()
    #test_trainer()