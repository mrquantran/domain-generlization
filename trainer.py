from torch import optim, nn
import torch
import pytorch_lightning as pl
import models
import numpy as np
from models.config_cls import Config
from models.vae import VAE
import os, copy
from sklearn.cluster import KMeans
import sklearn
from fid_score import calculate_fid_given_data
from utils import make_kl_df, data_to_device, check_input_unpacked, change_latent_dim, value_is_changing, \
    make_joint_samples, get_last_n_vals, trenddetector, calculate_acc
from visualization import plot_kls_df
from visualization import t_sne

class ALDVAE(pl.LightningModule):
    """
    VAE trainer. Configures, trains and tests the model.

    :param feature_dims: dictionary with feature dimensions of training data
    :type feature_dims: dict
    :param cfg: instance of Config class
    :type cfg: Config
    """

    def __init__(self, cfg, feature_dims: dict):
        super().__init__()
        self.config = self.check_config(cfg)
        self.model = None
        self.adaptive = self.config.adaptive
        self.optimizer = None
        self.objective = None
        self.datamodule = None
        self.mod_names = self.get_mod_names()
        self.feature_dims = feature_dims
        self.latents = cfg.n_latents
        self.get_model()
        self.example_input_array = None
        self.clustering = [[],[], []]
        self.recon_losses = [[]]
        self.fid_scores = [[],[]]
        self.latent_n = self.config.initial_latent_n
        self.patience = self.config.initial_patience
        self.compression_state = "decreasing"
        self.last_latent_decrease = 0
        self.fid_stats = None
        self.detection_window = 20

    def check_config(self, cfg):
        if not isinstance(cfg, models.config_cls.Config):
            cfg = Config(cfg)
        return cfg

    def get_mod_names(self):
        mod_names = {}
        for i, m in enumerate(self.config.mods):
            mod_names["mod_{}".format(i + 1)] = m["mod_type"]
        return mod_names

    @property
    def datamod(self):
        try:
            return self.trainer.datamodule
        except:
            return self.datamodule

    def configure_optimizers(self):
        """
        Sets up the optimizer specified in the config
        """
        if self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=float(self.config.lr), amsgrad=True)
        else:
            raise NotImplementedError
        return self.optimizer

    def get_model(self):
        """
        Sets up the model according to the config file
        """
        if self.config.pre_trained:
            self.model = self.load_from_checkpoint(self.config.pre_trained, cfg=self.config)
            return self.model
        vaes = {}
        for i, m in enumerate(self.config.mods):
            vaes["mod_{}".format(i + 1)] = VAE(m["encoder"], m["decoder"], self.feature_dims[m["mod_type"]],
                                               self.config.n_latents, m["recon_loss"], m["private_latents"], m["growtype"],
                                               obj_fn=self.config.obj,
                                               beta=self.config.beta, id_name="mod_{}".format(i + 1))
        self.model = vaes["mod_1"]
        self.save_hyperparameters()
        return self.model

    def training_step(self, train_batch, batch_idx):
        """
        Iterates over the train loader
        """
        loss_d = self.model.objective(train_batch)
        for key in loss_d.keys():
            if key != "reconstruction_loss":
                self.log("train_{}".format(key), loss_d[key].sum().item(), batch_size=self.config.batch_size)
            else:
                if len(self.config.mods) > 1:
                    for i, p_l in enumerate(loss_d[key]):
                        self.log("Mod_{}_TrainLoss".format(i), p_l.sum().item(), batch_size=self.config.batch_size)
                else:
                    self.log("TrainLoss", loss_d[key].sum().item(), batch_size=self.config.batch_size)
        return loss_d["loss"]

    def validation_step(self, val_batch, batch_idx):
        """
        Iterates over the val loader
        """
        loss_d = self.model.objective(val_batch)
        for key in loss_d.keys():
            if key != "reconstruction_loss":
                self.log("val_{}".format(key), loss_d[key].sum().item(), batch_size=self.config.batch_size)
            else:
                if len(self.config.mods) > 1:
                    for i, p_l in enumerate(loss_d[key]):
                        self.log("Mod_{}_ValLoss".format(i), p_l.sum().item(), batch_size=self.config.batch_size)
                else:
                    self.log("ValLoss", loss_d[key].sum().item(), batch_size=self.config.batch_size)
                    if not isinstance(self.recon_losses[-1], list):
                        self.recon_losses.append([])
                    self.recon_losses[-1].append(loss_d[key].mean())

        return loss_d["loss"]

    def test_step(self, test_batch, batch_idx):
        """
        Iterates over the test loader
        """
        loss_d = self.model.objective(test_batch)
        for key in loss_d.keys():
            if key != "reconstruction_loss":
                self.log("test_{}".format(key), loss_d[key].sum().item(), batch_size=self.config.batch_size)
            else:
                if len(self.config.mods) > 1:
                    for i, p_l in enumerate(loss_d[key]):
                        self.log("Mod_{}_TestLoss".format(i), p_l.sum().item(), batch_size=self.config.batch_size)
                else:
                    self.log("TestLoss", loss_d[key].sum().item(), batch_size=self.config.batch_size)
        return loss_d["loss"]

    def validation_epoch_end(self, outputs):
        """
        Save visualizations

        :param outputs: Loss that comes from validation_step
        :type outputs: torch.tensor
        """
        cluster_eval = self.analyse_data(fn_list=["clusters"])
        if cluster_eval and cluster_eval["silhouette"] is not None:
            self.log("Silhouette score", cluster_eval["silhouette"])
            self.log("ACC", cluster_eval["acc"])
        if (self.trainer.current_epoch) % self.config.viz_freq == 0:
            savepath = os.path.join(self.config.mPath, "visuals/epoch_{}/".format(self.trainer.current_epoch))
            os.makedirs(savepath, exist_ok=True)
            self.analyse_data(savedir=savepath)
            self.save_reconstructions(savedir=savepath)
            if self.config.dataset_name != "sprites":
                self.save_joint_samples(savedir=savepath, traversals=True)
                self.save_joint_samples(savedir=savepath, num_samples=100, traversals=False)
        ns = 100
        recons = self.save_joint_samples(num_samples=ns, savedir=None, traversals=False)["mod_1_raw"]
        samples, _ = self.datamod.get_num_samples(ns, split="train")
        recons2 = self.save_reconstructions(data=samples, savedir=None)
        if self.config.dataset_name == "sprites":  # reshape the animations for FID calculation
            samples_ = samples["mod_1"]["data"].reshape(ns, 3,64,8*64).float().clone().detach().cpu()
            recons = recons.clone().detach().reshape(*samples["mod_1"]["data"].reshape(ns, 3,64,8*64).shape).float().cpu()
            recons2 = recons2.clone().detach().reshape(*samples["mod_1"]["data"].reshape(ns, 3,64,8*64).shape).float().cpu()
        else:
            samples_ = samples["mod_1"]["data"].float().clone().detach().cpu()
            recons = recons.clone().detach().reshape(*samples["mod_1"]["data"].shape).float().cpu()
            recons2 = recons2.clone().detach().reshape(*samples["mod_1"]["data"].shape).float().cpu()
        if self.fid_stats ==  None:  # calculate the dataset statistics only once to save time
            FID, stats = calculate_fid_given_data([samples_, recons])
            self.fid_stats = stats
        else:
            FID, _ = calculate_fid_given_data([recons], stats=self.fid_stats)
        FID_recon,_ = calculate_fid_given_data([recons2], stats=self.fid_stats)
        self.fid_scores[0].append(FID)
        self.fid_scores[1].append(FID_recon)
        self.log("FID", FID)
        self.log("FID Recon", FID_recon)
        self.recon_losses[-1] = float(torch.stack(self.recon_losses[-1]).mean().cpu().detach())
        cluster_eval = self.analyse_data(fn_list=["clusters"], num_samples=1000)
        for ind, name in enumerate(["silhouette", "acc"]):
            if cluster_eval[name] is not None:
                self.clustering[ind].append(cluster_eval[name])
        all_data = {"Silhouette score": self.clustering[0],
                    "FID": self.fid_scores[0],
                    "FID Recon": self.fid_scores[1],
                    "Recon loss": self.recon_losses,
                    }
        if self.adaptive == 1 and self.compression_state != "stop" and self.trainer.current_epoch > 0:
            if (self.trainer.current_epoch - self.last_latent_decrease) % self.patience == 0:
                self.decider(all_data)
        self.log_progress()
        self.recon_losses.append([])

    def changes_detection(self, data):
        prob = trenddetector(data)
        return prob

    def get_weights(self, all_data, window):
        probs = {}
        for metric in all_data.keys():
                probs[metric] = self.changes_detection(all_data[metric][-window:-1])
        return list(probs.values())

    def decider(self, all_data):
        """Make a decision on what to do next based on the slope of Silhouette, Recon loss and FID"""
        if len(self.clustering[0]) >= self.detection_window:
            slope1 = value_is_changing(self.clustering[0], window=10)
            if slope1 == True and self.latent_n > 2:
                self.latent_n = 1
                self.patience = 5
            if self.latent_n == 1:
                weights = self.get_weights(all_data, self.detection_window)
                if weights[0] > 0 and weights[1] > 0 and weights[2] > 0 and weights[3] < 0:
                    self.compression_state = "stop"
        self.apply_change_latents()

    def log_progress(self):
        self.log("latent_dim", float(self.latents))
        self.log("State", {"growing":1.0, "stop":0.0, "decreasing":-1.0}[self.compression_state])
        self.log("Mean validation recon loss", self.recon_losses[-1])

    def apply_change_latents(self):
        """Change the latent dimensionality based on the current compression state (decreasing/stop/growing)"""
        if self.compression_state == "decreasing":
            if self.latents > self.latent_n:
                self.latents -= 1*self.latent_n
        elif self.compression_state == "growing":
            self.latents += 1 * self.latent_n
        if self.compression_state in ["decreasing", "growing"]:
            self.last_latent_decrease = self.trainer.current_epoch
            self.model = change_latent_dim(self.model, self.latents)


    def test_epoch_end(self, outputs):
        savepath = os.path.join(self.config.mPath, "visuals/epoch_{}_test/".format(self.trainer.current_epoch))
        os.makedirs(savepath, exist_ok=True)
        self.analyse_data(savedir=savepath, split="test")
        self.save_reconstructions(savedir=savepath, split="test")
        self.save_joint_samples(savedir=savepath, traversals=True)
        self.save_joint_samples(num_samples=100, savedir=savepath, traversals=False)
        if self.datamod.datasets[0].eval_statistics_fn() is not None:
            self.datamod.datasets[0].eval_statistics_fn()(self)

    def save_reconstructions(self, data=None, num_samples=24, savedir=None, split="val"):
        """
        Reconstructs data and saves output, also iterates over missing modalities on the input to cross-generate

        :param num_samples: number of samples to take from the dataloader for reconstruction
        :type num_samples: int
        :param savedir: where to save the reconstructions
        :type savedir: str
        :param split: val/test, whether to take samples from test or validation dataloader
        :type split: str
        """

        def save(output, mods, name=None):
            for k in data.keys():
                if mods[k]["data"] is None:
                    mods.pop(k)
            for key in output.mods.keys():
                recon_list = [x.loc for x in output.mods[key].decoder_dist] if isinstance(output.mods[key].decoder_dist,
                                                                                          list) \
                    else output.mods[key].decoder_dist.loc
                data_class = self.datamod.datasets[int(key.split("_")[-1]) - 1]
                if name is not None:
                    p = os.path.join(savedir, "recon_{}_to_{}.png".format(name, data_class.mod_type))
                    data_class.save_recons(mods, recon_list, p, self.mod_names)
            return recon_list

        if data is None:
            data, labels = self.datamod.get_num_samples(num_samples, split=split)
        data_i = check_input_unpacked(data_to_device(data, self.device))
        if savedir is not None:
            save(self.model.forward(data_i), data, "all")
            for m in range(len(data.keys())):
                mods = copy.deepcopy(data)
                for d in mods.keys():
                    mods[d]["data"], mods[d]["masks"] = None, None
                mods["mod_{}".format(m + 1)] = data["mod_{}".format(m + 1)]
                mod_name = self.config.mods[m]["mod_type"]
                output = self.model.forward(check_input_unpacked(mods))
                save(output, copy.deepcopy(mods), mod_name)
        else:
            return save(self.model.forward(data_i), data)

    def save_joint_samples(self, num_samples=16, savedir=None, traversals=False):
        """
        Generate joint samples from random vectors and save them

        :param num_samples: number of samples to generate
        :type num_samples: int
        :param savedir: where to save the reconstructions
        :type savedir: str
        :param traversals: whether to make traversals for each dimension (True) or randomly sample latents (False)
        :type traversals: bool
        """
        recons = {}
        traversal_ranges = [(-6,6), (-4,4), (-2,2), (-1,1)]
        for rng in traversal_ranges:
            if len(self.config.mods) > 1:
                for i, vae in enumerate(self.model.vaes):
                    recons["mod_{}".format(i+1)], recons["mod_{}_raw".format(i+1)] = make_joint_samples(self.model, i, self.datamod, self.latents, traversals,
                                                                      savedir, num_samples, trav_range=rng, current_vae=vae)
            else:
                recons["mod_1"], recons["mod_1_raw"] = make_joint_samples(self.model, 0, self.datamod, self.latents, traversals,
                                                     savedir, num_samples, trav_range=rng)
        return recons

    def analyse_data(self, data=None, labels=None, num_samples=500, path_name="",
                     savedir=None, split="val", fn_list=["kld", "tsne", "clusters"]):
        """
        Encodes data and plots T-SNE. If no data is passed, a dataloader (based on split="val"/"test") will be used.

        :param data: test data
        :type data: torch.tensor
        :param labels: labels for the data for labelled T-SNE (optional) - list of strings
        :type labels: list
        :param num_samples: number of samples to use for visualization
        :type num_samples: int
        :param path_name: label under which to save the visualizations
        :type path_name: str
        :param savedir: where to save the reconstructions
        :type savedir: str
        :param split: val/test, whether to take samples from test or validation dataloader
        :type split: str
        """
        if not data:
            data, labels = self.datamod.get_num_samples(num_samples, split=split)
        output_dic = self.eval_forward(data)
        pz = self.model.pz(*[x for x in self.model.pz_params])
        zss_sampled = [pz.sample(torch.Size([1, num_samples])).view(-1, pz.batch_shape[-1]),
                       *[zs["latents"].view(-1, zs["latents"].size(-1)) for zs in output_dic["latent_samples"]]]
        if path_name == "" and not self.config.eval_only:
            path_name = "_e_{}".format(self.trainer.current_epoch)
        if "kld" in fn_list:
            kl_df = make_kl_df(output_dic["encoder_dist"], pz)
            plot_kls_df(kl_df, os.path.join(savedir, 'kl_distance{}.png'.format(path_name)))
        if hasattr(labels[0], "__len__") and len(labels[0]) > 1 and any([isinstance(labels[0], list), type(labels[0]).__module__ == "numpy"])\
                and not isinstance(labels[0], str):
            cluster_scores = {"silhouette": []}
            for i, _ in enumerate(labels[0]):
                label = [x[i] for x in labels]
                path_name += "_feature{}".format(i)
                cs = self.analyse_clusters(zss_sampled, savedir, path_name, list(label), fn_list)
                cluster_scores["silhouette"].append(cs["silhouette"])
            cluster_scores["silhouette"] = np.mean(np.asarray(cluster_scores["silhouette"]))
        else:
            cluster_scores = self.analyse_clusters(zss_sampled, savedir, path_name, labels, fn_list)
        return cluster_scores

    def analyse_clusters(self, zss_sampled, savedir, path_name, labels, fn_list):
        cluster_scores = {"silhouette": None, "acc":None}
        if "tsne" in fn_list:
            t_sne([x for x in zss_sampled[1:]], os.path.join(savedir, 't_sne{}.png'.format(path_name)), labels,
                  self.mod_names)
        if "clusters" in fn_list and labels is not None:
            d = [x for x in zss_sampled[1:]]
            km = KMeans(n_clusters=self.config.exp_classes, random_state=42)
            km.fit_predict(np.concatenate([x.detach().cpu().numpy() for x in d]))
            acc = calculate_acc(np.asarray(labels), np.asarray(km.labels_))
            cluster_scores["acc"] = acc
            cluster_scores["silhouette"] = sklearn.metrics.silhouette_score(np.concatenate([x.detach().cpu().numpy() for x in d]), labels, metric='euclidean')
        return cluster_scores

    def eval_forward(self, data):
        data_i = check_input_unpacked(data_to_device(data, self.device))
        output = self.model.forward(data_i)
        output_dic = output.unpack_values()
        return output_dic
