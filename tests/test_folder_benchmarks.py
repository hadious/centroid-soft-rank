from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from datasets.folder_benchmarks import FolderBenchmark
from datasets.vision import build_datasets
from models import build_model


def image(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.new("RGB",(80,64),color=(100,150,200)).save(path)


@pytest.fixture
def tiny(tmp_path):
    root=tmp_path/"tiny-imagenet-200";root.mkdir()
    (root/"wnids.txt").write_text("n002\nn001\n")
    lines=[]
    for c in ["n001","n002"]:
        for i in range(6):image(root/"train"/c/"images"/f"{i}.JPEG")
        for i in range(2):
            name=f"{c}_{i}.JPEG";image(root/"val/images"/name);lines.append(f"{name}\t{c}\t0\t0\t64\t64")
    (root/"val/val_annotations.txt").write_text("\n".join(lines))
    return tmp_path


def test_tiny_labels_and_split_isolation(tiny):
    b=build_datasets("tinyimagenet",root=tiny,train_size=6,val_size=4,include_test=True)
    assert b["metadata"]["num_classes"]==2
    assert not set(b["ssl_train"].indices)&set(b["val_eval"].indices)
    assert b["metadata"]["class_to_idx"]=={"n001":0,"n002":1}
    assert len(b["test_eval"])==4
    assert b["test_eval"].base.root==tiny/"tiny-imagenet-200"
    assert b["test_eval"].base.split=="val"
    assert tuple(b["val_eval"][0][0].shape)==(3,64,80)  # original tiny data is natively 64x64
    c=build_datasets("tinyimagenet",root=tiny,train_size=8,val_size=4,include_test=False)
    assert np.array_equal(b["val_eval"].indices,c["val_eval"].indices)
    assert c["test_eval"] is None


def test_pilot_does_not_require_official_validation(tiny):
    (tiny/"tiny-imagenet-200/val/val_annotations.txt").unlink()
    b=build_datasets("tinyimagenet",root=tiny,train_size=6,val_size=4,include_test=False)
    assert b["test_eval"] is None


def test_imagenette_canonical_labels_and_shape(tmp_path):
    root=tmp_path/"imagenette2-160"
    for split in ["train","val"]:
        for c in ["n002","n001"]:
            for i in range(6):image(root/split/c/f"{i}.JPEG")
    train=FolderBenchmark(tmp_path,"imagenette",True);val=FolderBenchmark(tmp_path,"imagenette",False)
    assert train.class_to_idx==val.class_to_idx=={"n001":0,"n002":1}
    b=build_datasets("imagenette",root=tmp_path,train_size=6,val_size=4,include_test=True)
    assert tuple(b["ssl_train"][0][0].shape)==tuple(b["val_eval"][0][0].shape)==(3,128,128)


@pytest.mark.parametrize("name,size",[("tinyimagenet",64),("imagenette",128)])
def test_encoder_accepts_resolution(name,size):
    import torch
    torch.set_num_threads(2)
    model=build_model(name).eval()
    with torch.no_grad():h,z=model(torch.zeros(2,3,size,size))
    assert h.shape==(2,512) and z.shape==(2,256)
