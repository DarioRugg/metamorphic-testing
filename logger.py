from black import out
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from zmq import device
from dataset import ProstateDataModule


# class PixelPlotter(pl.Callback):

#     def __init__(self, cfg: DictConfig, dataset: ProstateDataModule):
#         self.train_pixels = self._get_pixels(dataset.train, cfg.logs.train_pixels)
#         self.val_pixels = self._get_pixels(dataset.val, cfg.logs.val_pixels)
#         self.test_pixels = self._get_pixels(dataset.test, cfg.logs.test_pixels)

#     def _get_pixels(data: np.ndrray, pixels: list):
#         return data[pixels, :]

#     def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         train_pixels_hat = trainer.model(self.train_pixels)
#         num_pixels = train_pixels_hat.shape[0]

#         # tensorflow code to work on 
#         fig, axs = plt.subplots(1, num_pixels,
#                                     figsize=(5 * num_pixels, 5.5))
#         fig.subplots_adjust(top=0.8)
#         plt.suptitle("comparison plot\nepoch: {} loss: {}".format(trainer.epoch, logs["loss"]))

#         for i, pixel_id in enumerate(self.pixels):
#             pixel = np.expand_dims(self.dataset["dataset"][pixel_id, :], 0)
#             pixel_recon = self.model.predict(pixel)
#             axs[i].set_title("pixel {} ({}:{})\nMSE: {:.5f}".format(pixel_id,
#                                                                     self.dataset["x"][pixel_id],
#                                                                     self.dataset["y"][pixel_id],
#                                                                     mean_squared_error(pixel, pixel_recon)))
#             axs[i].plot(self.features, pixel[0], label="original", alpha=0.7)
#             axs[i].plot(self.features, pixel_recon[0], label="reconstructed", alpha=0.7)

#             axs[i].set_xlabel('mz')

#         axs[0].set_ylabel('i')
#         axs[-1].legend()

#         plt.savefig(self.plots_path / "pixels_comparison_plot_epoch_{}.png".format(epoch))
#         plt.close()

#     def plot_logs(self):
#         plt.title("loss")
#         plt.plot(range(len(self.loss)), self.loss)
#         plt.ylabel('loss')
#         plt.xlabel('epoch')
#         plt.savefig("./assets/img/logs/loss")
#         plt.close()

#         plt.title("kl")
#         plt.plot(range(len(self.kl)), self.kl)
#         plt.ylabel('kl')
#         plt.xlabel('epoch')
#         plt.savefig("./assets/img/logs/kl")
#         plt.close()

#         plt.title("mse")
#         plt.plot(range(len(self.mse)), self.mse)
#         plt.ylabel('mse')
#         plt.xlabel('epoch')
#         plt.savefig("./assets/img/logs/mse")
#         plt.close()


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


class ModelWithLoggingFunctions(pl.LightningModule):
    def __init__(self, cfg: DictConfig, dataset: ProstateDataModule):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset

    def training_epoch_end(self,outputs):
        if self.current_epoch != 0 and self.current_epoch%4 == 0:
            self._plot_pixels(self.dataset.train, self.cfg.logs.train_pixels, self.dataset.features, "train")
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self,outputs):
        if self.current_epoch != 0 and self.current_epoch%4 == 0:
            self._plot_pixels(self.dataset.val, self.cfg.logs.val_pixels, self.dataset.features, "val")
        return super().validation_epoch_end(outputs)
    
    def test_epoch_end(self, outputs):
        self._plot_pixels(self.dataset.test, self.cfg.logs.test_pixels, self.dataset.features, "test")
        return super().test_epoch_end(outputs)

    def _plot_pixels(self, data: np.ndarray, pixels_to_log: list, features, split: str):
        pixels = data[pixels_to_log, :]

        pixels_hat = self(torch.from_numpy(pixels).to(f'cuda:0')).to("cpu").detach().numpy()

        fig, axs = plt.subplots(2, len(pixels_to_log)//2,
                                    figsize=(7.5 * len(pixels_to_log)//2, 5 * 2))
        fig.subplots_adjust(top=0.8)
        plt.suptitle("{}\ncomparison plot".format(split))

        for ax, pixel_index, pixel, pixel_hat in zip(axs.flat, pixels_to_log, pixels, pixels_hat):
            ax.set_title("pixel {}\nMSE: {:.5f}".format(pixel_index, F.mse_loss(torch.from_numpy(pixel), torch.from_numpy(pixel_hat))))
            ax.plot(features, pixel, label="original", alpha=0.7)
            ax.plot(features, pixel_hat, label="reconstructed", alpha=0.7)
        
        axs[0, 0].legend()
        for ax in axs[-1, :].flat:
            ax.set_xlabel('mz')
        for ax in axs[:, 0].flat:
            ax.set_ylabel('i')

        #plt.savefig(self.plots_path / "pixels_comparison_plot_epoch_{}.png".format(epoch))
        plt.show()
        plt.close()


class ModelWithLoggingFunctionsOld(pl.LightningModule):
    def __init__(self, cfg: DictConfig, dataset: ProstateDataModule):
        super().__init__()
        print(dataset.train)
        self.train_pixels_to_log = self._get_pixels(dataset.train, cfg.logs.train_pixels)
        self.val_pixels_to_log = self._get_pixels(dataset.val, cfg.logs.val_pixels)
        self.test_pixels_to_log = self._get_pixels(dataset.test, cfg.logs.test_pixels)

        self.features_to_log = dataset.features

    def _get_pixels(data: np.ndarray, pixels: list):
        return data[pixels, :]

    def training_epoch_end(self,outputs):
        self._plot_pixels(self.train_pixels_to_log, self(self.train_pixels_to_log), outputs)
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self,outputs):
        self._plot_pixels(self.val_pixels_to_log, self(self.val_pixels_to_log), outputs)
        return super().validation_epoch_end(outputs)
    
    def test_epoch_end(self, outputs):
        self._plot_pixels(self.test_pixels_to_log, self(self.test_pixels_to_log), outputs)
        return super().test_epoch_end(outputs)

    def _plot_pixels(self, pixels, pixels_hat, outputs):
        num_pixels = pixels_hat.shape[0]

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # tensorflow code to work on 
        fig, axs = plt.subplots(1, num_pixels,
                                    figsize=(5 * num_pixels, 5.5))
        fig.subplots_adjust(top=0.8)
        plt.suptitle("comparison plot\nloss: {}".format(avg_loss))

        for i, pixel, pixel_hat in enumerate(zip(pixels, pixels_hat)):
            axs[i].set_title("pixel {}\nMSE: {:.5f}".format(i, F.mse_loss(pixel, pixel_hat)))
            axs[i].plot(self.features_to_log, pixel, label="original", alpha=0.7)
            axs[i].plot(self.features_to_log, pixel_hat, label="reconstructed", alpha=0.7)

            axs[i].set_xlabel('mz')

        axs[0].set_ylabel('i')
        axs[-1].legend()

        self.logger.experiment.add_image("reconstruction plot", )

        #plt.savefig(self.plots_path / "pixels_comparison_plot_epoch_{}.png".format(epoch))
        #plt.close()
        plt.show()