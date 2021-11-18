import pytest


def test_installation():
    import torch
    import torchvision

    assert torch.__version__
    assert torchvision.__version__

    import mmcv
    from mmcv.ops import get_compiling_cuda_version, get_compiler_version

    assert mmcv.__version__
    assert get_compiling_cuda_version()
    assert get_compiler_version()

    import mmdet

    assert mmdet.__version__
    import mmocr

    assert mmocr.__version__


@pytest.mark.xfail(reason="only passes if cuda is available")
def test_installation_gpu():
    import torch

    assert torch.cuda.is_available()
