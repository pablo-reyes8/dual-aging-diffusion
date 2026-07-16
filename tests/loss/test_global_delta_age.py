import torch

from src.loss.global_loss import GlobalAgingLoss


class _AgeOnlyAux:
    def predict_age(self, images, grad_to_input):
        return images.mean(dim=(1, 2, 3))


def _minimal_loss(delta_mode):
    loss = GlobalAgingLoss.__new__(GlobalAgingLoss)
    torch.nn.Module.__init__(loss)
    loss.device = torch.device("cpu")
    loss.lambda_age = 1.0
    loss.lambda_delta_age = 1.0
    loss.lambda_id = 0.0
    loss.lambda_perc = 0.0
    loss.age_loss_scale = 100.0
    loss.delta_age_target_mode = delta_mode
    loss.global_loss_bundle = _AgeOnlyAux()
    return loss


def test_chronological_delta_is_not_duplicate_of_absolute_age():
    loss = _minimal_loss("chronological_gap")
    output = loss.compute_semantic_losses(
        source_images=torch.full((1, 3, 2, 2), 10.0),
        generated_images=torch.full((1, 3, 2, 2), 20.0),
        source_ages=torch.tensor([30.0]),
        target_ages=torch.tensor([50.0]),
        semantic_weights=torch.ones(1),
        semantic_components=("age", "delta_age"),
    )
    assert torch.isclose(output["loss_age"], torch.tensor(0.30))
    assert torch.isclose(output["loss_delta_age"], torch.tensor(0.10))


def test_legacy_estimator_anchor_reproduces_duplicate_objective():
    loss = _minimal_loss("estimator_anchor")
    output = loss.compute_semantic_losses(
        source_images=torch.full((1, 3, 2, 2), 10.0),
        generated_images=torch.full((1, 3, 2, 2), 20.0),
        source_ages=torch.tensor([30.0]),
        target_ages=torch.tensor([50.0]),
        semantic_weights=torch.ones(1),
        semantic_components=("age", "delta_age"),
    )
    assert torch.isclose(output["loss_delta_age"], output["loss_age"])
