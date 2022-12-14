import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from scripts.dataset import ProstateDataModule


class PixelsPlotter(pl.Callback):
    def __init__(self, cfg: DictConfig, dataset: ProstateDataModule):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch != 0 and trainer.current_epoch %2 == 0:
            self._plot_pixels(self.dataset.train, pl_module, self.cfg.logs.train_pixels, self.dataset.features, "train", trainer.current_epoch)
        return super().on_train_epoch_end(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch != 0 and trainer.current_epoch %2 == 0:
            self._plot_pixels(self.dataset.val, pl_module, self.cfg.logs.val_pixels, self.dataset.features, "val", trainer.current_epoch)
        return super().on_validation_epoch_end(trainer, pl_module)
    
    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._plot_pixels(self.dataset.test, pl_module, self.cfg.logs.test_pixels, self.dataset.features, "test", trainer.current_epoch)
        return super().on_test_epoch_end(trainer, pl_module)

    def _plot_pixels(self, data: np.ndarray, model: pl.LightningModule, pixels_to_log: list, features, split: str, epoch: int):

        pixels = torch.from_numpy(data[pixels_to_log, :]).to(model.device)
        pixels_hat = model(pixels)
        
         #  .to("cpu").detach().numpy()

        fig, axs = plt.subplots(2, len(pixels_to_log)//2,
                                    figsize=(7.5 * len(pixels_to_log)//2, 5 * 2))
        fig.subplots_adjust(top=0.8)
        plt.suptitle("{}\ncomparison plot".format(split))

        for ax, pixel_index, pixel, pixel_hat in zip(axs.flat, pixels_to_log, pixels, pixels_hat):
            ax.set_title("pixel {}\nMSE: {:.5f}".format(pixel_index, F.mse_loss(pixel, pixel_hat)))
            print("pixel {}\nMSE: {:.5f}".format(pixel_index, F.mse_loss(pixel, pixel_hat)))
            ax.plot(features, pixel.to("cpu").detach(), label="original", alpha=0.7)
            ax.plot(features, pixel_hat.to("cpu").detach(), label="reconstructed", alpha=0.7)
        
        axs[0, 0].legend()
        for ax in axs[-1, :].flat:
            ax.set_xlabel('mz')
        for ax in axs[:, 0].flat:
            ax.set_ylabel('i')

        plt.savefig("assets/weights/checkpoints/imgs/pixels_comparison_plot_epoch_{}.png".format(epoch))
        plt.close()
