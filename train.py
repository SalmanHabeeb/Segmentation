from model import UNet
from utils import get_dataset, get_dataloaders, ImageTransform, MaskTransform, DiceScore, get_pretrained_backbone
from trainer import Trainer

if __name__ == "__main__":

    img_tfs = ImageTransform((64, 64))
    mask_tfs = MaskTransform((64, 64), normalize=True, convert2gray=True)
    TRAIN_PATHS = {
                      "IMAGE_PATH" : "./riwa_v2/images",
                      "MASK_PATH"  : "./riwa_v2/masks",
                  }
    VAL_PATHS = {
                    "IMAGE_PATH" : "./riwa_v2/validation/images",
                    "MASK_PATH"  : "./riwa_v2/validation/masks",
                }
    dataset = get_dataset(TRAIN_PATHS, VAL_PATHS, img_tfs, mask_tfs)
    dataloaders = get_dataloaders(dataset)

    back_bone = get_pretrained_backbone()
    model = UNet(
                in_channels=3,
                out_channels=1,
                image_shape=(64, 64),
                n_layers=5,
                back_bone=back_bone
            )

    loss_func = nn.MSELoss()
    metric = DiceScore()

    optimizer = torch.optim.Adam
    learning_rate=1e-3

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    reduce_lr_factor = 0.1
    patience = 5

    trainer = Trainer(
                  model=model,
                  dataloaders=dataloaders,
                  loss_func=loss_func,
                  metric=metric,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  learning_rate=learning_rate,
                  reduce_lr_factor = reduce_lr_factor,
                  patience=patience,
              )
    trainer.train(n_epochs=40)
    trainer.plot_loss()
    trainer.save("RiverSegModelWeights.pt")
