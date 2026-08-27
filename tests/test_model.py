"""
test_model.py - Unit tests for model, dataset, and training utilities.
"""

import sys
from pathlib import Path

import pytest
import torch

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import get_model, LightweightCNN, count_parameters, CIFAR10_CLASSES  # noqa: E402
from dataset import get_transforms, get_single_image_transform  # noqa: E402


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestGetModel:
    """Tests for the get_model factory function."""

    def test_resnet18_output_shape(self):
        model = get_model(architecture="resnet18", num_classes=10, pretrained=False)
        model.eval()
        dummy = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"

    def test_cnn_output_shape(self):
        model = get_model(architecture="cnn", num_classes=10, pretrained=False)
        model.eval()
        dummy = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_resnet18_custom_num_classes(self):
        model = get_model(architecture="resnet18", num_classes=5, pretrained=False)
        model.eval()
        dummy = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 5)

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            get_model(architecture="vgg16", num_classes=10)

    def test_count_parameters_positive(self):
        model = get_model(architecture="cnn", num_classes=10, pretrained=False)
        params = count_parameters(model)
        assert params > 0

    def test_resnet18_has_more_params_than_cnn(self):
        resnet = get_model(architecture="resnet18", num_classes=10, pretrained=False)
        cnn = get_model(architecture="cnn", num_classes=10, pretrained=False)
        assert count_parameters(resnet) > count_parameters(cnn)


class TestLightweightCNN:
    """Tests for the custom LightweightCNN architecture."""

    def test_forward_pass(self):
        model = LightweightCNN(num_classes=10)
        model.eval()
        x = torch.randn(8, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 10)

    def test_batch_size_one(self):
        model = LightweightCNN(num_classes=10)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 10)

    def test_output_is_logits_not_probabilities(self):
        """Logits should not sum to 1 (that's softmax's job)."""
        model = LightweightCNN(num_classes=10)
        model.eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        row_sums = out.sum(dim=1)
        # Logits won't sum to 1.0 (unlike softmax output)
        assert not torch.allclose(row_sums, torch.ones(4), atol=1e-3)


# ---------------------------------------------------------------------------
# CIFAR-10 class list tests
# ---------------------------------------------------------------------------

class TestCIFAR10Classes:
    def test_ten_classes(self):
        assert len(CIFAR10_CLASSES) == 10

    def test_known_classes_present(self):
        for cls in ["airplane", "automobile", "cat", "dog", "ship", "truck"]:
            assert cls in CIFAR10_CLASSES


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------

class TestTransforms:
    def test_train_transform_returns_tensor(self):
        from PIL import Image
        import numpy as np

        transform = get_transforms(train=True)
        img = Image.fromarray(
            (np.random.rand(32, 32, 3) * 255).astype("uint8")
        )
        tensor = transform(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 32, 32)

    def test_val_transform_returns_tensor(self):
        from PIL import Image
        import numpy as np

        transform = get_transforms(train=False)
        img = Image.fromarray(
            (np.random.rand(32, 32, 3) * 255).astype("uint8")
        )
        tensor = transform(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 32, 32)

    def test_inference_transform_resizes(self):
        from PIL import Image
        import numpy as np

        transform = get_single_image_transform()
        # Input is 64x64, should be resized to 32x32
        img = Image.fromarray(
            (np.random.rand(64, 64, 3) * 255).astype("uint8")
        )
        tensor = transform(img)
        assert tensor.shape == (3, 32, 32)

    def test_train_transform_normalizes(self):
        """After normalization, pixel values should not be in [0, 1]."""
        from PIL import Image
        import numpy as np

        transform = get_transforms(train=False)
        # All-white image
        img = Image.fromarray(
            (np.ones((32, 32, 3)) * 255).astype("uint8")
        )
        tensor = transform(img)
        # After normalization with CIFAR-10 stats, values won't be in [0,1]
        assert tensor.max().item() > 1.0 or tensor.min().item() < 0.0


# ---------------------------------------------------------------------------
# Training utility tests
# ---------------------------------------------------------------------------

class TestTrainingUtilities:
    """Smoke tests for train/evaluate functions."""

    def _make_loader(self, n_samples=16, batch_size=8):
        """Create a tiny synthetic DataLoader."""
        from torch.utils.data import DataLoader, TensorDataset

        images = torch.randn(n_samples, 3, 32, 32)
        labels = torch.randint(0, 10, (n_samples,))
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=batch_size)

    def test_train_one_epoch_returns_loss_and_accuracy(self):
        from train import train_one_epoch

        model = get_model(architecture="cnn", num_classes=10, pretrained=False)
        loader = self._make_loader()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")

        loss, acc = train_one_epoch(model, loader, optimizer, criterion, device)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
        assert loss >= 0.0

    def test_evaluate_returns_loss_and_accuracy(self):
        from train import evaluate

        model = get_model(architecture="cnn", num_classes=10, pretrained=False)
        loader = self._make_loader()
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")

        loss, acc = evaluate(model, loader, criterion, device)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
