# voc training schedule for detrex, modified according to coco_schedule.py



from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler



# default scheduler for detr-like training pipeline.
def default_voc_scheduler(epochs=50, decay_epochs=40, warmup_epochs=0, batchsize=16):
    """
    Returns the config for a default multi-step LR scheduler for voc datasets.
    Learning rate is decayed once at the end of training.

    For default VOC training pipeline, we use a combination of trainval2007 and trainval2012 (16551 images in total).


    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.
        batchsize (int): 

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    iters_per_batch = int(16651 / batchsize)

    total_iters = epochs * iters_per_batch
    decay_iters = decay_epochs * iters_per_batch
    warmup_iters = warmup_epochs * iters_per_batch
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[decay_iters, total_iters],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_iters / total_iters,
        warmup_method="linear",
        warmup_factor=0.001,
    )





# default scheduler for detr
lr_multiplier_50ep = default_voc_scheduler(50, 40, 0)
lr_multiplier_36ep = default_voc_scheduler(36, 30, 0)
lr_multiplier_24ep = default_voc_scheduler(24, 20, 0)
lr_multiplier_12ep = default_voc_scheduler(12, 11, 0)

# warmup scheduler for detr
lr_multiplier_50ep_warmup = default_voc_scheduler(50, 40, 1e-3)
lr_multiplier_12ep_warmup = default_voc_scheduler(12, 11, 1e-3)
