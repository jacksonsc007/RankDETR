from .rank_detr_r50_50ep import train, dataloader, optimizer, model
from detrex.config import get_config
from configs.common.voc_schedule import default_voc_scheduler

dataloader = get_config("common/data/voc_detr.py").dataloader
batch_size=4
dataloader.train.total_batch_size = batch_size

lr_multiplier = default_voc_scheduler(12, 11, 0, batch_size)


# modify model config
model.with_box_refine = True
model.as_two_stage = True

model.rank_adaptive_classhead = True
model.transformer.decoder.query_rank_layer = True
model.criterion.GIoU_aware_class_loss = True
model.criterion.matcher.iou_order_alpha = 4.0
model.criterion.matcher.matcher_change_iter = 67500

model.num_classes = 20
model.criterion.num_classes = 20
model.transformer.encoder.num_layers=2 
model.transformer.decoder.num_layers=2 

model.num_queries_one2one = 300
model.num_queries_one2many = 1500
model.transformer.two_stage_num_proposals = model.num_queries_one2one + model.num_queries_one2many

# use single scale
model.backbone.out_features=["res3",
                             "res4"
                            ]
from detectron2.layers import ShapeSpec
model.neck.input_shapes={ "res3": ShapeSpec(channels=512), 
                          "res4":ShapeSpec(channels=1024)
                        }
model.neck.in_features=["res3", 
                        "res4"
                        ]
model.neck.num_outs=2
model.transformer.num_feature_levels=2


#NOTE======================== modifications to train on voc =======================================
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/rank_detr_r50_two_stage_12ep"
total_imgs = 16651
num_epochs = 12
train.max_iter = int( num_epochs * total_imgs / batch_size)
# run evaluation every 5000 iters
train.eval_period = 5000
# save checkpoint every 5000 iters
train.checkpointer.period = 5000

# wandb log
train.wandb.enabled = True
train.wandb.params.name = "rank_detr_voc_baseline_res3_res4"
train.output_dir = "./output/" + "${train.wandb.params.name}"