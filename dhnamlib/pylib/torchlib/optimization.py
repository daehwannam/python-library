
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    first_value=None, last_value=None, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    When `num_warmup_steps == 1`, the learning rate decreases only.
    When `num_warmup_steps == num_training_steps`, the learning rate increases only.
    When `num_warmup_steps == num_training_steps == 1`, the learning rate is always 1.

    This scheduler is modified from `transformers.optimization.get_linear_schedule_with_warmup`:
    https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup
    """

    assert num_warmup_steps >= 1
    assert num_training_steps >= num_warmup_steps

    if first_value is not None:
        assert 0 < first_value < 1
    else:
        first_value = 1 / num_warmup_steps

    if last_value is not None:
        assert 0 < last_value < 1
    else:
        last_value = 1 / ((num_training_steps - num_warmup_steps) + 1)

    def lr_lambda(current_step):
        # current_step starts from 0 by default (current_step == last_epoch + 1)

        if current_step < num_warmup_steps - 1:
            return first_value + (1 - first_value) * current_step / (num_warmup_steps - 1)
        elif current_step >= num_training_steps - 1:
            return last_value
        else:
            return last_value + (1 - last_value) * (1 - (((current_step - num_warmup_steps) + 1) /
                                                         (num_training_steps - num_warmup_steps)))

    # a scheduler calls `step` once it's created
    return LambdaLR(optimizer, lr_lambda, last_epoch)
