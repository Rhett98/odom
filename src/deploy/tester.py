#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import mlflow
import numpy as np
import pickle
import torch

import deploy.deployer


class Tester(deploy.deployer.Deployer):

    def __init__(self, config):
        super(Tester, self).__init__(config=config)
        self.training = False

        # Load checkpoint
        if self.config["checkpoint"]:
            checkpoint = torch.load(self.config["checkpoint"], map_location=self.device)
            ## Model weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print('\033[92m' + "Model weights loaded from " + self.config["checkpoint"] + "\033[0;0m")
        else:
            raise Exception("No checkpoint specified.")

        print("Batch size set to 1 for the testing.")
        self.batch_size = 1

        # Split up dataset by sequences, since maps are created sequence-wise
        self.computed_transformations_datasets = []
        for dataset in self.config["datasets"]:
            computed_transformations_sequences = [[] for element in self.config[dataset]["data_identifiers"]]
            self.computed_transformations_datasets.append(computed_transformations_sequences)

    def test_dataset(self, dataloader):

        epoch_losses = {
            "loss_epoch": 0.0,
            "st_epoch": 0.0,
            "sq_epoch": 0.0,
            "loss_t_epoch": 0.0,
            "loss_q_epoch": 0.0,
        }
        index_of_dataset = 0
        index_of_sequence = 0
        dataset = self.config["datasets"][0]

        for index, preprocessed_dicts in enumerate(dataloader):
            for preprocessed_dict in preprocessed_dicts:

                # Move data to device
                for key in preprocessed_dict:
                    if hasattr(preprocessed_dict[key], 'to'):
                        preprocessed_dict[key] = preprocessed_dict[key].to(self.device)

            if not self.config["inference_only"]:
                epoch_losses, computed_transformation = self.step(
                    preprocessed_dicts=preprocessed_dicts,
                    epoch_losses=epoch_losses,
                    log_images_bool=not index % 10)
            else:
                computed_transformation = self.step(
                    preprocessed_dicts=preprocessed_dicts,
                    epoch_losses=epoch_losses,
                    log_images_bool=not index % 10)

            for preprocessed_dict in preprocessed_dicts:
                # Case: we reached the next sequence or the next dataset --> log
                if preprocessed_dict["index_sequence"] != index_of_sequence or preprocessed_dict[
                    "index_dataset"] != index_of_dataset:
                    self.log_map(index_of_dataset=index_of_dataset,
                                 index_of_sequence=index_of_sequence,
                                 dataset=dataset,
                                 data_identifier=self.config[dataset]["data_identifiers"][index_of_sequence])
                    index_of_sequence = preprocessed_dict["index_sequence"]
                    index_of_dataset = preprocessed_dict["index_dataset"]
                    dataset = preprocessed_dict["dataset"]

                self.computed_transformations_datasets[preprocessed_dict["index_dataset"]][
                    preprocessed_dict["index_sequence"]].append(
                    computed_transformation.detach().cpu().numpy())

            if not index % 10:
                if not self.config["inference_only"]:
                    print("Index: " + str(index) + " / " + str(len(dataloader)) + ", loss: " + str(
                        epoch_losses["loss_epoch"] / (index + 1)))
                    if self.config["visualize_images"]:
                        self.log_image(epoch=index, string="_" + dataset)
                else:
                    print("Index: " + str(index) + " / " + str(len(dataloader)) + ", dataset: " +
                          dataset + ", sequence: " + str(index_of_sequence))
        self.log_map(index_of_dataset=index_of_dataset,
                     index_of_sequence=index_of_sequence,
                     dataset=dataset,
                     data_identifier=self.config[dataset]["data_identifiers"][
                         index_of_sequence])
        return epoch_losses

    def test(self):

        dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 collate_fn=Tester.list_collate,
                                                 num_workers=self.config["num_dataloader_workers"],
                                                 pin_memory=True if self.config[
                                                                        "device"] == torch.device(
                                                     "cuda") else False)

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

        with mlflow.start_run(experiment_id=id, run_name="Test " + self.config["run_name"]):
            self.log_config()
            epoch_losses = self.test_dataset(dataloader=dataloader)

            dataset_index = 0

            if not self.config["inference_only"]:
                epoch_losses["loss_epoch"] /= self.steps_per_epoch
                epoch_losses["st_epoch"] /= self.steps_per_epoch
                epoch_losses["sq_epoch"] /= self.steps_per_epoch
                epoch_losses["loss_t_epoch"] /= self.steps_per_epoch
                epoch_losses["loss_q_epoch"] /= self.steps_per_epoch
                print("Epoch Summary: " + format(dataset_index, '05d') + ", loss: " + str(
                    epoch_losses["loss_epoch"])+ ", loss_q: " + str(
                    epoch_losses["loss_q_epoch"])+ ", loss_t: " + str(
                    epoch_losses["loss_t_epoch"]))

                mlflow.log_metric("loss", float(epoch_losses["loss_epoch"]), step=dataset_index)
                mlflow.log_metric("st", float(epoch_losses["st_epoch"]), step=dataset_index)
                mlflow.log_metric("sq", float(epoch_losses["sq_epoch"]), step=dataset_index)
                mlflow.log_metric("loss_t", float(epoch_losses["loss_t_epoch"]), step=dataset_index)
                mlflow.log_metric("loss_q", float(epoch_losses["loss_q_epoch"]), step=dataset_index)
