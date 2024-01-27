from detrex.config import get_config
from .models.anchor_detr_r50 import model

dataloader = get_config("common/data/voc_cocostyle.py").dataloader # change to voc dataset
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
train.output_dir = "./output/shallow_anchor_detr_r50_50ep"



# log training infomation every 20 iters
train.log_period = 20



# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16



#NOTE======================== shallow encoder decoder layers =======================================
#model.transformer.num_encoder_layers=2 
#model.transformer.num_decoder_layers=2 
#model.transformer.num_query_position=50
model.transformer.num_classes=20
model.criterion.num_classes=20



#NOTE======================== modifications to train on voc =======================================
# max training iterations
train.max_iter = 32000 # 5011 * 50 / 16=15659.375
# run evaluation every 5000 iters
train.eval_period = 1000
# save checkpoint every 5000 iters
train.checkpointer.period = 2000

train.wandb.enabled = True
train.wandb.params.name = "anchor detr"

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 8
#NOTE======================== modifications to evaluate on voc =======================================