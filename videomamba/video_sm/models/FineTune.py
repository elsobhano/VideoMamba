from timm.models import create_model
import pytorch_lightning as pl

class FineTuneModel(pl.LightningModule):
    def __init__(self, 
                args,
                lr=3e-4, 
                encoder_ckpt=None,
                eval_freq=10,
                ):
        super().__init__()
        self.save_hyperparameters()

        ################Set the Mamba Model####################
        model = create_model(
            args.model,
            img_size=args.input_size,
            pretrained=False if args.finetune else True,
            num_classes=args.nb_classes,
            fc_drop_rate=args.fc_drop_rate,
            drop_path_rate=args.drop_path,
            kernel_size=args.tubelet_size,
            num_frames=args.num_frames,
            use_checkpoint=args.use_checkpoint,
            checkpoint_num=args.checkpoint_num,
        )
        self.lr = None
        
        self.criterion = None
        ######################Prompts#######################
        
    def forward(self, **kwargs):
        pass
        
    
    def on_train_epoch_start(self):
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        loss = None
        return loss
    def validation_step(self, batch, batch_idx):
        logits = None
        loss = None
        return {"logits": logits, "loss": loss}

    def on_validation_epoch_end(self):
        pass
        
    
    def _log_to_wandb(self, targets, hypotheses, hypotheses_teacher, split: str, epoch: int):
        pass
    
    def test_step(self, batch, batch_idx):
        pass
    
    def on_test_epoch_end(self):
        pass
    
    def teacher_forcing_generate(self, logits):
        pass
    
    def generate(self, list_of_frames):
        pass
    
    def calc_loss(self, logits, y):
        pass
    
    def add_weight_decay(self, weight_decay, skip_list=()):
        """Custom method to create parameter groups with/without weight decay."""
        pass
    def configure_optimizers(self):
        pass
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        pass