#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import time

import mlflow
import mlflow.pytorch
import pickle
import torch
import numpy as np
import qqdm

import deploy.deployer_odom


class Trainer(deploy.deployer_odom.Deployer):

    def __init__(self, config):
        super(Trainer, self).__init__(config=config)
        self.training_bool = True

        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()},
                                           {'params': self.lossTransformation.parameters(),'lr': 0.0001}],
                                          lr=self.config["learning_rate"])

        # Load checkpoint
        if self.config["checkpoint"]:
            checkpoint = torch.load(self.config["checkpoint"], map_location=self.device)
            ## Model weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print('\033[92m' + "Model weights loaded from " + self.config["checkpoint"] + "\033[0;0m")
            ## Optimizer parameters
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print('\033[92m' + "Optimizer parameters loaded from " + self.config["checkpoint"] + "\033[0;0m")
            ## Directly train unsupervised in that case, since model is pretrained
            self.config["unsupervised_at_start"] = True

        if self.config["inference_only"]:
            print(
                "Config error: Inference only does not make sense during training. Changing to inference_only=False.")
            self.config["inference_only"] = False

    def train_epoch(self, epoch, dataloader):

        epoch_losses = {
            "loss_epoch": 0.0,
            "st_epoch": 0.0,
            "sq_epoch": 0.0,
            "loss_t_epoch": 0.0,
            "loss_q_epoch": 0.0,
            "seg_epoch": 0.0,
        }
        counter = 0

        qqdm_dataloader = qqdm.qqdm(dataloader, desc=qqdm.format_str('blue', 'Epoch ' + str(epoch)))
        # qqdm_dataloader = dataloader
        for preprocessed_dicts in qqdm_dataloader:
            # Load corresponnding preprocessed kd_tree
            for preprocessed_dict in preprocessed_dicts:
                # Move data to devices:
                for key in preprocessed_dict:
                    if hasattr(preprocessed_dict[key], 'to'):
                        preprocessed_dict[key] = preprocessed_dict[key].to(self.device)

            self.optimizer.zero_grad()

            epoch_losses, _ = (
                self.step(
                    preprocessed_dicts=preprocessed_dicts,
                    epoch_losses=epoch_losses,
                    log_images_bool=counter == self.steps_per_epoch - 1 or counter == 0))

            # Plotting and logging --> only first one in batch
            preprocessed_data = preprocessed_dicts[0]

            qqdm_dataloader.set_infos({'loss': f'{float(epoch_losses["loss_epoch"] / (counter + 1)):.6f}'})

            counter += 1

        return epoch_losses

    def train(self):

        dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 collate_fn=Trainer.list_collate,
                                                 num_workers=self.config["num_dataloader_workers"],
                                                 pin_memory=True if self.config[
                                                                        "device"] == torch.device("cuda") else False)
        
        # Check whether experiment already exists
        client = mlflow.tracking.MlflowClient()
        experiment_list = client.list_experiments()
        id = None
        for experiment in experiment_list:
            if experiment.name == self.config["experiment"]:
                id = experiment.experiment_id

        if id is None:
            print("Creating new MLFlow experiment: " + self.config["experiment"])
            id = mlflow.create_experiment(self.config["experiment"])
        else:
            print("MLFlow experiment " + self.config["experiment"] + " already exists. Starting a new run within it.")
        print("----------------------------------")

        with mlflow.start_run(experiment_id=id, run_name="Training: " + self.config["training_run_name"]):
            self.log_config()
            for epoch in range(self.config["max_epoch"]):
                # Train for 1 epoch
                epoch_losses = self.train_epoch(epoch=epoch, dataloader=dataloader)

                # Compute metrics
                epoch_losses["loss_epoch"] /= self.steps_per_epoch
                epoch_losses["st_epoch"] /= self.steps_per_epoch
                epoch_losses["sq_epoch"] /= self.steps_per_epoch
                epoch_losses["loss_t_epoch"] /= self.steps_per_epoch
                epoch_losses["loss_q_epoch"] /= self.steps_per_epoch
                epoch_losses["seg_epoch"] /= self.steps_per_epoch

                # Print update
                print("--------------------------")
                print("Epoch Summary: " + format(epoch, '05d') + ", loss: " + str(
                    epoch_losses["loss_epoch"])+ ", loss_q: " + str(
                    epoch_losses["loss_q_epoch"])+ ", loss_t: " + str(
                    epoch_losses["loss_t_epoch"])+ ", loss_seg: " + str(
                    epoch_losses["seg_epoch"]))

                # Logging
                print("Logging metrics and artifacts...")
                # Log metrics
                mlflow.log_metric("loss", float(epoch_losses["loss_epoch"]), step=epoch)
                mlflow.log_metric("st", float(epoch_losses["st_epoch"]), step=epoch)
                mlflow.log_metric("sq", float(epoch_losses["sq_epoch"]), step=epoch)
                mlflow.log_metric("loss_t", float(epoch_losses["loss_t_epoch"]), step=epoch)
                mlflow.log_metric("loss_q", float(epoch_losses["loss_q_epoch"]), step=epoch)
                mlflow.log_metric("seg", float(epoch_losses["seg_epoch"]), step=epoch)

                # Save latest checkpoint, and create checkpoint backup all 5 epochs
                ## Every epoch --> will always be overwritten by latest version
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': float(epoch_losses["loss_epoch"]),
                    'parameters': self.config
                }, "/tmp/" + self.config["training_run_name"] + "_latest_checkpoint.pth")
                mlflow.log_artifact("/tmp/" + self.config["training_run_name"] + "_latest_checkpoint.pth")
                ## All 5 epochs --> will be logged permanently in MLFlow
                if not epoch % 5:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': float(epoch_losses["loss_epoch"]),
                        'parameters': self.config
                    }, "/tmp/" + self.config["training_run_name"] + "_checkpoint_epoch_" + str(epoch) + ".pth")
                    mlflow.log_artifact(
                        "/tmp/" + self.config["training_run_name"] + "_checkpoint_epoch_" + str(epoch) + ".pth")

                # Save latest pickled full model
                if not self.config["use_jit"]:
                    mlflow.pytorch.log_model(self.model, "latest_model_pickled")

                # if self.config["visualize_images"] and not self.config["po2po_alone"]:
                #     self.log_image(epoch=epoch, string="_image")

                print("...done.")
