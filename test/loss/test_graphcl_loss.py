"""Tests for the GraphCL NT-Xent loss function."""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from topobench.loss.dataset.graphcl_loss import GraphCLLoss


class TestGraphCLLoss:
    """Tests for the GraphCLLoss (NT-Xent) implementation."""

    def setup_method(self):
        self.loss_fn = GraphCLLoss(temperature=0.5)

    def test_repr(self):
        assert "GraphCLLoss" in repr(self.loss_fn)
        assert "0.5" in repr(self.loss_fn)

    def test_output_is_scalar(self):
        z1 = torch.randn(8, 32)
        z2 = torch.randn(8, 32)
        model_out = {"z1_proj": z1, "z2_proj": z2}
        batch = Data()
        loss = self.loss_fn(model_out, batch)
        assert loss.dim() == 0

    def test_loss_is_nonnegative(self):
        """NT-Xent is a cross-entropy loss so it should be >= 0."""
        z1 = torch.randn(16, 64)
        z2 = torch.randn(16, 64)
        model_out = {"z1_proj": z1, "z2_proj": z2}
        loss = self.loss_fn(model_out, Data())
        assert loss.item() >= 0

    def test_identical_views_give_low_loss(self):
        """When both views are the same, positives have similarity 1
        and the loss should be low (close to log(2N-1)/tau baseline)."""
        z = torch.randn(16, 64)
        model_out = {"z1_proj": z, "z2_proj": z.clone()}
        loss_identical = self.loss_fn(model_out, Data())

        z2_random = torch.randn(16, 64)
        model_out_random = {"z1_proj": z, "z2_proj": z2_random}
        loss_random = self.loss_fn(model_out_random, Data())

        assert loss_identical.item() < loss_random.item()

    def test_loss_uses_cross_view_n_by_n_matrix(self):
        """Verify the N x N cross-view formulation matching official GraphCL.

        For each anchor z1_i the denominator should sum over the N-1
        cross-view negatives (z2_j for j != i), excluding the positive.
        """
        batch_size = 4
        z1 = torch.randn(batch_size, 8)
        z2 = torch.randn(batch_size, 8)

        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)
        sim = torch.mm(z1_norm, z2_norm.t()) / self.loss_fn.temperature

        pos_sim = sim.diag()
        diag_mask = torch.eye(batch_size, dtype=torch.bool)
        neg_sim = sim.masked_fill(diag_mask, float('-inf'))
        expected_loss = (-pos_sim + torch.logsumexp(neg_sim, dim=1)).mean()

        model_out = {"z1_proj": z1, "z2_proj": z2}
        actual_loss = self.loss_fn(model_out, Data())

        assert torch.allclose(actual_loss, expected_loss, atol=1e-5)

    def test_loss_is_asymmetric(self):
        """N x N cross-view loss is asymmetric: swapping z1/z2 changes loss."""
        z1 = torch.randn(8, 32)
        z2 = torch.randn(8, 32)

        loss_12 = self.loss_fn({"z1_proj": z1, "z2_proj": z2}, Data())
        loss_21 = self.loss_fn({"z1_proj": z2, "z2_proj": z1}, Data())

        assert not torch.allclose(loss_12, loss_21, atol=1e-5)

    def test_gradient_flows(self):
        z1 = torch.randn(8, 32, requires_grad=True)
        z2 = torch.randn(8, 32, requires_grad=True)
        loss = self.loss_fn({"z1_proj": z1, "z2_proj": z2}, Data())
        loss.backward()
        assert z1.grad is not None
        assert z2.grad is not None
        assert not torch.all(z1.grad == 0)

    def test_temperature_effect(self):
        """Lower temperature should produce higher loss for random embeddings."""
        z1 = torch.randn(16, 32)
        z2 = torch.randn(16, 32)
        model_out = {"z1_proj": z1, "z2_proj": z2}

        loss_low_temp = GraphCLLoss(temperature=0.1)(model_out, Data())
        loss_high_temp = GraphCLLoss(temperature=1.0)(model_out, Data())

        assert loss_low_temp.item() > loss_high_temp.item()

    def test_batch_size_two(self):
        """Smallest valid batch: N=2 gives 1 cross-view negative per anchor."""
        z1 = torch.randn(2, 16)
        z2 = torch.randn(2, 16)
        loss = self.loss_fn({"z1_proj": z1, "z2_proj": z2}, Data())
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert torch.isfinite(loss)
