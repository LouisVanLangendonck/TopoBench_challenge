"""Tests for the GraphCL readout (projection head)."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.nn.readouts.graphcl_readout import GraphCLReadOut


class TestGraphCLReadOut:
    """Tests for the GraphCLReadOut projection head."""

    def test_mlp_projection(self):
        readout = GraphCLReadOut(
            hidden_dim=32,
            out_channels=16,
            projection_type="mlp",
            task_level="graph",
            num_cell_dimensions=1,
        )
        z1 = torch.randn(4, 32)
        z2 = torch.randn(4, 32)
        model_out = {
            "z1": z1, "z2": z2,
            "x_0": torch.randn(20, 32),
            "batch_0": torch.zeros(20, dtype=torch.long),
        }
        result = readout(model_out, Data())
        assert result["z1_proj"].shape == (4, 16)
        assert result["z2_proj"].shape == (4, 16)
        assert "logits" in result

    def test_linear_projection(self):
        readout = GraphCLReadOut(
            hidden_dim=32,
            out_channels=16,
            projection_type="linear",
            task_level="graph",
            num_cell_dimensions=1,
        )
        model_out = {
            "z1": torch.randn(4, 32), "z2": torch.randn(4, 32),
            "x_0": torch.randn(20, 32),
            "batch_0": torch.zeros(20, dtype=torch.long),
        }
        result = readout(model_out, Data())
        assert result["z1_proj"].shape == (4, 16)

    def test_identity_projection(self):
        readout = GraphCLReadOut(
            hidden_dim=32,
            out_channels=32,
            projection_type="none",
            task_level="graph",
            num_cell_dimensions=1,
        )
        z1 = torch.randn(4, 32)
        model_out = {
            "z1": z1, "z2": torch.randn(4, 32),
            "x_0": torch.randn(20, 32),
            "batch_0": torch.zeros(20, dtype=torch.long),
        }
        result = readout(model_out, Data())
        assert torch.equal(result["z1_proj"], z1)

    def test_shared_projection_head(self):
        """Both views should pass through the same projection head parameters."""
        readout = GraphCLReadOut(
            hidden_dim=32,
            out_channels=16,
            projection_type="mlp",
            task_level="graph",
            num_cell_dimensions=1,
        )
        z = torch.randn(4, 32)
        model_out = {
            "z1": z, "z2": z.clone(),
            "x_0": torch.randn(20, 32),
            "batch_0": torch.zeros(20, dtype=torch.long),
        }
        result = readout(model_out, Data())
        assert torch.allclose(result["z1_proj"], result["z2_proj"], atol=1e-6)

    def test_invalid_projection_type_raises(self):
        with pytest.raises(ValueError, match="Unknown projection type"):
            GraphCLReadOut(
                hidden_dim=32,
                out_channels=16,
                projection_type="wrong",
                task_level="graph",
                num_cell_dimensions=1,
            )

    def test_repr(self):
        readout = GraphCLReadOut(
            hidden_dim=32, out_channels=16,
            projection_type="mlp", task_level="graph",
            num_cell_dimensions=1,
        )
        r = repr(readout)
        assert "GraphCLReadOut" in r
        assert "mlp" in r
