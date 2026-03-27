"""Tests for the GraphCL evaluator."""

import pytest
import torch

from topobench.evaluator.graphcl_evaluator import GraphCLEvaluator


class TestGraphCLEvaluator:
    """Tests for the GraphCLEvaluator."""

    def test_default_metrics(self):
        evaluator = GraphCLEvaluator()
        assert "contrastive_loss" in evaluator.metric_names
        assert "alignment" in evaluator.metric_names
        assert "cosine_sim" in evaluator.metric_names

    def test_custom_metrics(self):
        evaluator = GraphCLEvaluator(metrics=["uniformity", "cosine_sim"])
        assert "uniformity" in evaluator.metric_names
        assert "cosine_sim" in evaluator.metric_names

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            GraphCLEvaluator(metrics=["nonexistent_metric"])

    def test_update_and_compute(self):
        evaluator = GraphCLEvaluator(
            metrics=["contrastive_loss", "alignment", "cosine_sim"]
        )
        z1 = torch.randn(8, 32)
        z2 = torch.randn(8, 32)
        model_out = {
            "z1_proj": z1,
            "z2_proj": z2,
            "loss": torch.tensor(1.5),
        }
        evaluator.update(model_out)
        result = evaluator.compute()
        assert "contrastive_loss" in result
        assert "alignment" in result
        assert "cosine_sim" in result

    def test_alignment_decreases_for_similar_views(self):
        evaluator = GraphCLEvaluator(metrics=["alignment"])

        z = torch.randn(16, 32)
        evaluator.update({"z1_proj": z, "z2_proj": z.clone()})
        result_same = evaluator.compute()
        evaluator.reset()

        evaluator.update({"z1_proj": z, "z2_proj": torch.randn(16, 32)})
        result_diff = evaluator.compute()

        assert result_same["alignment"] < result_diff["alignment"]

    def test_cosine_sim_range(self):
        evaluator = GraphCLEvaluator(metrics=["cosine_sim"])
        z1 = torch.randn(16, 32)
        z2 = torch.randn(16, 32)
        evaluator.update({"z1_proj": z1, "z2_proj": z2})
        result = evaluator.compute()
        assert -1.0 <= result["cosine_sim"].item() <= 1.0

    def test_reset(self):
        evaluator = GraphCLEvaluator(metrics=["cosine_sim"])
        evaluator.update({
            "z1_proj": torch.randn(8, 32),
            "z2_proj": torch.randn(8, 32),
        })
        evaluator.reset()
        evaluator.update({
            "z1_proj": torch.randn(8, 32),
            "z2_proj": torch.randn(8, 32),
        })
        result = evaluator.compute()
        assert "cosine_sim" in result

    def test_repr(self):
        evaluator = GraphCLEvaluator()
        assert "GraphCLEvaluator" in repr(evaluator)
