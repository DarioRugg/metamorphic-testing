import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from dataset import ProstateDataModule


class PixelPlotter(pl.Callback):

    def __init__(self, cfg: DictConfig, dataset: ProstateDataModule):
        self.train_pixels = self._get_pixels(dataset.train, cfg.logs.train_pixels)
        self.val_pixels = self._get_pixels(dataset.val, cfg.logs.val_pixels)
        self.test_pixels = self._get_pixels(dataset.test, cfg.logs.test_pixels)

    def _get_pixels(data: np.ndrray, pixels: list):
        return data[pixels, :]

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        train_pixels_hat = trainer.model(self.train_pixels)

        # tensorflow code to work on 
        fig, axs = plt.subplots(1, len(self.pixels),
                                    figsize=(5 * len(self.pixels), 5.5))
            fig.subplots_adjust(top=0.8)
            plt.suptitle("comparison plot\nepoch: {} loss: {}".format(epoch, logs["loss"]))

            for i, pixel_id in enumerate(self.pixels):
                pixel = np.expand_dims(self.dataset["dataset"][pixel_id, :], 0)
                pixel_recon = self.model.predict(pixel)
                axs[i].set_title("pixel {} ({}:{})\nMSE: {:.5f}".format(pixel_id,
                                                                        self.dataset["x"][pixel_id],
                                                                        self.dataset["y"][pixel_id],
                                                                        mean_squared_error(pixel, pixel_recon)))
                axs[i].plot(self.features, pixel[0], label="original", alpha=0.7)
                axs[i].plot(self.features, pixel_recon[0], label="reconstructed", alpha=0.7)

                axs[i].set_xlabel('mz')

            axs[0].set_ylabel('i')
            axs[-1].legend()

            plt.savefig(self.plots_path / "pixels_comparison_plot_epoch_{}.png".format(epoch))
            plt.close()

    def plot_logs(self):
        plt.title("loss")
        plt.plot(range(len(self.loss)), self.loss)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig("./assets/img/logs/loss")
        plt.close()

        plt.title("kl")
        plt.plot(range(len(self.kl)), self.kl)
        plt.ylabel('kl')
        plt.xlabel('epoch')
        plt.savefig("./assets/img/logs/kl")
        plt.close()

        plt.title("mse")
        plt.plot(range(len(self.mse)), self.mse)
        plt.ylabel('mse')
        plt.xlabel('epoch')
        plt.savefig("./assets/img/logs/mse")
        plt.close()


class CustomLogger(pl.Callback):

    def __init__(self):
        self.loss = []
        self.mse = []
        self.kl = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.loss.append(trainer.logged_metrics['elbo'].numpy())
        self.mse.append(trainer.logged_metrics['recon_loss'].numpy())
        self.kl.append(trainer.logged_metrics['kl'].numpy())

    def plot_logs(self):
        plt.title("loss")
        plt.plot(range(len(self.loss)), self.loss)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig("./assets/img/logs/loss")
        plt.close()

        plt.title("kl")
        plt.plot(range(len(self.kl)), self.kl)
        plt.ylabel('kl')
        plt.xlabel('epoch')
        plt.savefig("./assets/img/logs/kl")
        plt.close()

        plt.title("mse")
        plt.plot(range(len(self.mse)), self.mse)
        plt.ylabel('mse')
        plt.xlabel('epoch')
        plt.savefig("./assets/img/logs/mse")
        plt.close()
