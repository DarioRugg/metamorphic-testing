import pytorch_lightning as pl
from matplotlib import pyplot as plt


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
