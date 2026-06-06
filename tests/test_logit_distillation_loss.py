import torch

from qwen_clt.training.losses import last_token_logit_distillation_loss


def test_last_token_logit_distillation_loss_is_zero_for_identical_logits():
    logits = torch.randn(2, 4, 16)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ]
    )

    loss = last_token_logit_distillation_loss(
        student_logits=logits,
        teacher_logits=logits.clone(),
        attention_mask=attention_mask,
        top_k=8,
    )

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_last_token_logit_distillation_loss_uses_last_active_token():
    teacher = torch.zeros(1, 3, 8)
    student = teacher.clone()
    attention_mask = torch.tensor([[1, 1, 0]])

    teacher[0, 1, 4] = 5.0
    student[0, 1, 4] = -5.0
    student[0, 2, 4] = 5.0

    loss = last_token_logit_distillation_loss(
        student_logits=student,
        teacher_logits=teacher,
        attention_mask=attention_mask,
        top_k=4,
    )

    assert float(loss.item()) > 0.0
