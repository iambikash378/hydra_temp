import pytorch_lightning as pl
import torch

class BaseModule(pl.LightningModule):
    def __init__(
            self,
            net: nn.Module,
            optimizer : torch.optim.Optimizer,
            scheduler : torch.optim.lr_scheduler.LRScheduler = None,
            loss_fn,
            threshold: float = 0.5,
            multi_class : bool = False ,
            log_outputs : bool = True
    ):
        super().__init__()

        self.net = net 
        self.loss_fn = loss_fn
        self.accuracy = pl.metrics.Accuracy()

        self.save_hyperparameters(logger = False, ignore = ["net"])

    
    def forward(self, **kwargs):
        return self.net(**kwargs)
    
    def on_train_start(self):
        pass
    

    def training_step(self, batch, batch_idx : int):
        images, labels = batch
        outputs = self.net(images)
        _ , preds = torch.max(outputs, 1)
        loss = self.loss_fn(outputs, labels)

        self.log("Train Loss ", loss, on_step = True, on_epoch = True, prog_bar = True)
        self.log("Train Accuracy ", self.accuracy(preds, labels) , on_step = True, on_epoch = True, prog_bar = True)
        
        return loss
    
    def on_train_batch_end(self):
        pass
    
    def on_train_epoch_end(self):
        self.accuracy.reset()


    def validation_step(self, batch, batch_idx: int):
        images, labels = batch
        outputs = self.net(images)
        _ , preds = torch.max(outputs, 1)
        loss = self.loss_fn(outputs, labels)

        self.log("Train Loss ", loss, on_step = True, on_epoch = True, prog_bar = True)
        self.log("Train Accuracy ", self.accuracy(preds, labels) , on_step = True, on_epoch = True, prog_bar = True)


    def on_validation_batch_end(self):
        self.accuracy.reset()

    def on_validation_epoch_end(self):
        pass

    def test_step():
        pass

    def configure_optimizers(self):
