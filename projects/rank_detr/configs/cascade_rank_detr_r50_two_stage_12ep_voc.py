from detrex.config import get_config
from .models.rank_detr_r50 import model
from configs.common.voc_schedule import default_voc_scheduler
import os

CODE_VERSION  = os.environ.get("code_version")
if CODE_VERSION is None:
    raise ValueError("code version must be specified!")

# ========================================
# basic setting
# ========================================
batch_size = 4
total_imgs = 16651
num_epochs = 12
assert num_epochs == 12

setting_code = f"bs{batch_size}_epoch{num_epochs}"

# ========================================
# dataloader config
# ========================================
dataloader = get_config("common/data/voc_detr.py").dataloader
dataloader.train.total_batch_size = batch_size
dataloader.train.num_workers = 8


dataset_code = "voc"


# ========================================
# model config
# ========================================
# for VOC dataset
model.num_classes = 20
model.criterion.num_classes = 20
model.transformer.encoder.num_layers=3 
model.transformer.decoder.num_layers=3 
model.num_queries_one2one = 300
model.num_queries_one2many = 1500 
model.transformer.two_stage_num_proposals = model.num_queries_one2one + model.num_queries_one2many

# two-stage scheme
model.with_box_refine = True
model.as_two_stage = True
model.rank_adaptive_classhead = True
model.transformer.decoder.query_rank_layer = True
model.criterion.GIoU_aware_class_loss = True
model.criterion.matcher.iou_order_alpha = 4.0
model.criterion.matcher.matcher_change_iter = 67500

model.transformer.topk_ratio=0.2



model_code = f"CascadeRankDETR_twostage_one2one{model.num_queries_one2one}one2many{model.num_queries_one2many}_enc{model.transformer.encoder.num_layers}_dec{model.transformer.decoder.num_layers}_topkratio{model.transformer.topk_ratio}"

# ========================================
# optimizer config
# ========================================
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = default_voc_scheduler(12, 11, 0, batch_size)
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1
optim_code = f"lr{optimizer.lr}"



# ========================================
# training config modification
# ========================================
train = get_config("common/train.py").train
# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"

# max training iterations
train.max_iter = int( num_epochs * total_imgs / batch_size)

# run evaluation every 5000 iters
train.eval_period = 5000

# log training infomation every 20 iters
train.log_period = 20

# save checkpoint every 5000 iters
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# ========================================
# logging
# ========================================
# wandb log
train.wandb.enabled = True
train.wandb.params.name = "-".join([CODE_VERSION, model_code, dataset_code, setting_code, optim_code, ])
train.wandb.params.project = "rank_detr" 
train.output_dir = "./output/" + "${train.wandb.params.name}"
# dump the testing results into output_dir for visualization
# NOTE that VOC standard evaluator don't need output_dir
dataloader.evaluator.output_dir = train.output_dir



