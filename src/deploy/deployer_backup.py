#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import copy
import math

import torch
import mlflow
import numpy as np

import utility.plotting
import utility.poses
import utility.projection
import data.dataset
# import models.model
import models.model_modified
import losses.icp_losses
import models.pwclo

# from utility.geometry import euler_to_quaternion, quaternion_to_euler


class Deployer(object):

    def __init__(self, config):
        torch.cuda.empty_cache()

        # Parameters and data
        self.config = config
        self.device = config["device"]
        self.batch_size = config["batch_size"]
        self.dataset = data.dataset.PreprocessedPointCloudDataset(config=config)
        self.steps_per_epoch = int(len(self.dataset) / self.batch_size)
        if self.config["mode"] == "testing":
            self.ground_truth_dataset = data.dataset.PoseDataset(config=self.config)

        # Projections and model
        self.img_projection = utility.projection.ImageProjectionLayer(config=config)
        if self.config["use_jit"]:
            datasets = self.config["datasets"]
            ## Need to provide example tensor for torch jit
            example_tensor = torch.zeros((1, 4,
                                          self.config[datasets[0]]["vertical_cells"],
                                          self.config[datasets[0]]["horizontal_cells"]),
                                         device=self.device)
            del datasets
            self.model = torch.jit.trace(
                models.model.OdometryModel(config=self.config).to(self.device),
                example_inputs=(example_tensor, example_tensor))
        else:
            # self.model = models.model.OdometryModel(config=self.config).to(self.device)
            self.model = models.model_modified.OdometryModel(config=self.config).to(self.device)
            # self.model = models.pwclo.OdometryModel().to(self.device)

        # Geometry handler
        self.geometry_handler = models.model_parts.GeometryHandler(config=config)

        # Loss and optimizer
        # self.lossTransformation = torch.nn.MSELoss()
        self.lossTransformation = losses.icp_losses.supervisedLosses()
        self.lossPointCloud = losses.icp_losses.ICPLosses(config=self.config)
        # self.lossBCE = torch.nn.BCELoss()
        self.training_bool = False

        # Permanent variables for internal use
        self.log_img_1 = []
        self.log_img_2 = []
        self.log_img_2_transformed = []
        self.log_pointwise_loss = []
        self.log_normals_target = []
        self.log_normals_transformed_source = []

    @staticmethod
    def list_collate(batch_dicts):
        data_dicts = [batch_dict for batch_dict in batch_dicts]
        return data_dicts

    def create_images(self, preprocessed_data, losses, plotting):
        ## Create image of target normals
        image_1_at_normals, _, _, _, _ = self.img_projection(input=torch.cat((
            preprocessed_data["scan_1"],
            preprocessed_data["normal_list_1"]), dim=1), dataset=preprocessed_data["dataset"])

        ## Create image for points where normals exist (in source and target)
        image_2_transformed_and_normals_and_pointwise_loss, _, _, _, _ = \
            self.img_projection(input=torch.cat((plotting["scan_2_transformed"],
                                                 plotting["normals_2_transformed"],
                                                 losses["loss_po2pl_pointwise"]), dim=1),
                                dataset=preprocessed_data["dataset"])

        self.log_pointwise_loss = image_2_transformed_and_normals_and_pointwise_loss[:, 6:9]
        self.log_normals_target = image_1_at_normals[:, 3:6]
        self.log_normals_transformed_source = image_2_transformed_and_normals_and_pointwise_loss[
                                              :, 3:6]

    def log_image(self, epoch, string):
        utility.plotting.plot_lidar_image(
            input=[self.log_img_1, self.log_img_2, self.log_img_2_transformed,
                   self.log_pointwise_loss, self.log_normals_target,
                   self.log_normals_transformed_source],
            label="target",
            iteration=(epoch + 1) * self.steps_per_epoch if self.training_bool else epoch,
            path="/tmp/" + self.config["run_name"] + "_" + format(epoch, '05d') + string + ".png",
            training=self.training_bool)
        mlflow.log_artifact("/tmp/" + self.config["run_name"] + "_" + format(epoch, '05d') + string + ".png")

    def log_map(self, index_of_dataset, index_of_sequence, dataset, data_identifier):
        gt_translations = self.ground_truth_dataset.return_translations(
            index_of_dataset=index_of_dataset, index_of_sequence=index_of_sequence)
        gt_poses = self.ground_truth_dataset.return_poses(
            index_of_dataset=index_of_dataset, index_of_sequence=index_of_sequence)

        # Extract transformations and absolute poses
        computed_transformations = self.computed_transformations_datasets[index_of_dataset][
            index_of_sequence]
        computed_poses = utility.poses.compute_poses(
            computed_transformations=computed_transformations)

        # Log things to mlflow artifacts
        utility.poses.write_poses_to_text_file(
            file_name="/tmp/" + self.config["run_name"] + "_poses_text_file_" + dataset + "_" + format(data_identifier,
                                                                                                       '02d') + ".txt",
            poses=computed_poses)
        mlflow.log_artifact(
            "/tmp/" + self.config["run_name"] + "_poses_text_file_" + dataset + "_" + format(data_identifier,
                                                                                             '02d') + ".txt")
        np.save(
            "/tmp/" + self.config["run_name"] + "_transformations_" + dataset + "_" + format(data_identifier,
                                                                                             '02d') + ".npy",
            computed_transformations)
        np.save(
            "/tmp/" + self.config["run_name"] + "_poses_" + dataset + "_" + format(data_identifier, '02d') + ".npy",
            computed_poses)
        mlflow.log_artifact(
            "/tmp/" + self.config["run_name"] + "_transformations_" + dataset + "_" + format(data_identifier,
                                                                                             '02d') + ".npy")
        mlflow.log_artifact(
            "/tmp/" + self.config["run_name"] + "_poses_" + dataset + "_" + format(data_identifier, '02d') + ".npy")
        utility.plotting.plot_map(computed_poses=computed_poses,
                                  path_y="/tmp/" + self.config["run_name"] + "_map_" + dataset + "_" + format(
                                      data_identifier, '02d') + "_y.png",
                                  path_2d="/tmp/" + self.config["run_name"] + "_map_" + dataset + "_" + format(
                                      data_identifier, '02d') + "_2d.png",
                                  path_3d="/tmp/" + self.config["run_name"] + "_map_" + dataset + "_" + format(
                                      data_identifier, '02d') + "_3d.png",
                                  groundtruth=gt_translations,
                                  dataset=dataset)
        mlflow.log_artifact(
            "/tmp/" + self.config["run_name"] + "_map_" + dataset + "_" + format(data_identifier, '02d') + "_y.png")
        mlflow.log_artifact(
            "/tmp/" + self.config["run_name"] + "_map_" + dataset + "_" + format(data_identifier, '02d') + "_2d.png")
        mlflow.log_artifact(
            "/tmp/" + self.config["run_name"] + "_map_" + dataset + "_" + format(data_identifier, '02d') + "_3d.png")
        if gt_poses is not None:
            utility.plotting.plot_translation_and_rotation(
                computed_transformations=np.asarray(computed_transformations),
                path="/tmp/" + self.config["run_name"] + "_plot_trans_rot_" + dataset + "_" + format(data_identifier,
                                                                                                     '02d') + ".pdf",
                groundtruth=gt_poses,
                dataset=dataset)
            mlflow.log_artifact(
                "/tmp/" + self.config["run_name"] + "_plot_trans_rot_" + dataset + "_" + format(data_identifier,
                                                                                                '02d') + ".pdf")

    def log_config(self):
        for dict_entry in self.config:
            mlflow.log_param(dict_entry, self.config[dict_entry])

    def transform_image_to_point_cloud(self, transformation_matrix, image):
        # print("image shape:", image.shape)
        point_cloud_transformed = torch.matmul(transformation_matrix[:, :3, :3],
                                               image[:, :3, :, :].view(-1, 3, image.shape[2] *
                                                                       image.shape[3]))
        index_array_not_zero = (point_cloud_transformed[:, 0] != torch.zeros(1).to(
            self.device)) | (point_cloud_transformed[:, 1] != torch.zeros(1).to(self.device)) | (
                                       point_cloud_transformed[:, 2] != torch.zeros(1).to(
                                   self.device))

        for batch_index in range(len(point_cloud_transformed)):
            point_cloud_transformed_batch = point_cloud_transformed[batch_index, :,
                                            index_array_not_zero[batch_index]] + \
                                            transformation_matrix[batch_index, :3, 3].view(1, 3, -1)
        point_cloud_transformed = point_cloud_transformed_batch

        return point_cloud_transformed

    def rotate_point_cloud_transformation_matrix(self, transformation_matrix, point_cloud):
        return transformation_matrix[:, :3, :3].matmul(point_cloud[:, :3, :])

    def transform_point_cloud_transformation_matrix(self, transformation_matrix, point_cloud):
        transformed_point_cloud = self.rotate_point_cloud_transformation_matrix(
            transformation_matrix=transformation_matrix,
            point_cloud=point_cloud)
        transformed_point_cloud += transformation_matrix[:, :3, 3].view(-1, 3, 1)
        return transformed_point_cloud

    def rotate_point_cloud_euler_vector(self, euler, point_cloud):
        translation = torch.zeros(3, device=self.device)
        euler = euler.to(self.device)
        transformation_matrix = self.geometry_handler.get_transformation_matrix_angle_axis(
            translation=translation,
            euler=euler, device=self.device)
        return self.transform_point_cloud_transformation_matrix(
            transformation_matrix=transformation_matrix,
            point_cloud=point_cloud)

    def augment_input(self, preprocessed_data):
        # Random rotation
        if self.config["random_point_cloud_rotations"]:
            raise Exception("Needs to be verified for larger batches")
            if self.config["random_rotations_only_yaw"]:
                direction = torch.zeros((1, 3), device=self.device)
                direction[0, 2] = 1
            else:
                direction = (torch.rand((1, 3), device=self.device))
            direction = direction / torch.norm(direction)
            magnitude = (torch.rand(1, device=self.device) - 0.5) * (
                    self.config["magnitude_random_rot"] / 180.0 * torch.Tensor([math.pi]).to(
                self.device))
            euler = direction * magnitude
            preprocessed_data["scan_2"] = self.rotate_point_cloud_euler_vector(
                point_cloud=preprocessed_data["scan_2"], euler=euler)
            preprocessed_data["normal_list_2"] = self.rotate_point_cloud_euler_vector(
                point_cloud=preprocessed_data["normal_list_2"], euler=euler)

        return preprocessed_data

    def normalize_input(self, preprocessed_data):
        ranges_1 = torch.norm(preprocessed_data["scan_1"], dim=1)
        ranges_2 = torch.norm(preprocessed_data["scan_2"], dim=1)
        means_1 = torch.mean(ranges_1, dim=1, keepdim=True)
        means_2 = torch.mean(ranges_2, dim=1, keepdim=True)

        # Normalization mean is mean of both means (i.e. independent of number of points of each scan)
        means_1_2 = torch.cat((means_1, means_2), dim=1)
        normalization_mean = torch.mean(means_1_2, dim=1)
        preprocessed_data["scan_1"] /= normalization_mean
        preprocessed_data["scan_2"] /= normalization_mean
        preprocessed_data["scaling_factor"] = normalization_mean

        return preprocessed_data, normalization_mean

    def step(self, preprocessed_dicts, epoch_losses=None, log_images_bool=False):

        # Use every batchindex separately
        images_model_1 = torch.zeros(self.batch_size, 4,
                                     self.config[preprocessed_dicts[0]["dataset"]]["vertical_cells"],
                                     self.config[preprocessed_dicts[0]["dataset"]]["horizontal_cells"],
                                     device=self.device)
        images_model_2 = torch.zeros_like(images_model_1)
        for index, preprocessed_dict in enumerate(preprocessed_dicts):
            if self.training_bool:
                preprocessed_dict = self.augment_input(preprocessed_data=preprocessed_dict)
            if self.config["normalization_scaling"]:
                preprocessed_data, scaling_factor = self.normalize_input(preprocessed_data=preprocessed_dict)

            # Training / Testing
            image_1, _, _, point_cloud_indices_1, _ = self.img_projection(
                input=preprocessed_dict["scan_1"], dataset=preprocessed_dict["dataset"])
            image_2, _, _, point_cloud_indices_2, image_to_pc_indices_2 = self.img_projection(
                input=preprocessed_dict["scan_2"], dataset=preprocessed_dict["dataset"])

            ## Only keep points that were projected to image
            preprocessed_dict["scan_1"] = preprocessed_dict["scan_1"][:, :, point_cloud_indices_1]
            preprocessed_dict["scan_2"] = preprocessed_dict["scan_2"][:, :, point_cloud_indices_2]
            
            target_transformation = torch.tensor(preprocessed_dict["pose"]).unsqueeze(0).cuda().float()   
            target_euler = self.geometry_handler.get_euler_angles_from_matrix(target_transformation, device=self.device)
            target_trans = target_transformation[0,:3,3]

            image_model_1 = image_1[0]
            image_model_2 = image_2[0]
            
            # Write projected image to batch
            images_model_1[index] = image_model_1
            images_model_2[index] = image_model_2
            images_to_pcs_indices_2 = [image_to_pc_indices_2]
        self.log_img_1 = image_1[:, :3]
        self.log_img_2 = image_2[:, :3]

        # Feed into model as batch
        (translations, rotation_representation) = self.model(images_model_1,
                                                             images_model_2)
        
        computed_transformations = self.geometry_handler.get_transformation_matrix_quaternion(
            translation=translations, quaternion=rotation_representation, device=self.device)
        
        computed_euler = self.geometry_handler.get_euler_angles_from_matrix(computed_transformations, device=self.device)
        computed_trans = computed_transformations[0,:3,3]

        # Following part only done when loss needs to be computed
        if not self.config["inference_only"]:
            ## Losses
            loss_transformation = self.lossTransformation(target_euler, computed_euler, target_trans, computed_trans)

            loss_transformation /= self.batch_size
            loss = loss_transformation  # Overwrite loss for identity fitting
            # for name, parms in self.model.named_parameters():	
            #         print('-->name:', name)
            #         print('-->para:', parms)
            #         print('-->grad_requirs:',parms.requires_grad)
            #         # print('-->grad_value:',parms.grad)
            #         print("===")

            if self.training_bool:
                loss.backward()
                self.optimizer.step()
                # for name, parms in self.model.named_parameters():	
                #     print('-->name:', name)
                #     # print('-->para:', parms)
                #     print('-->grad_requirs:',parms.requires_grad)
                #     print('-->grad_value:',parms.grad)
                #     print("===")
                # print(self.optimizer)

            if self.config["normalization_scaling"]:
                for index, preprocessed_dict in enumerate(preprocessed_dicts):
                    computed_transformations[index, :3, 3] *= preprocessed_dict["scaling_factor"]

            epoch_losses["loss_epoch"] += loss.detach().cpu().numpy()

            return epoch_losses, computed_transformations
        else:
            if self.config["normalization_scaling"]:
                for index, preprocessed_dict in enumerate(preprocessed_dicts):
                    computed_transformations[index, :3, 3] *= preprocessed_dict["scaling_factor"]

            return computed_transformations


