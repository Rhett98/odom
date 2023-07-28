import math
import torch
import mlflow
import numpy as np

import utility.plotting
import utility.poses
import utility.projection
import data.dataset_label
import models.refinement_model
import losses.motion_losses
import models.pwclo

class Deployer(object):

    def __init__(self, config):
        torch.cuda.empty_cache()

        # Parameters and data
        self.config = config
        self.device = config["device"]
        self.batch_size = config["batch_size"]
        self.dataset = data.dataset_label.PreprocessedPointCloudDataset(config=config)
        self.steps_per_epoch = int(len(self.dataset) / self.batch_size)
        if self.config["mode"] == "testing":
            self.ground_truth_dataset = data.dataset_label.PoseDataset(config=self.config)

        # Projections and model
        self.img_projection = utility.projection.ImageProjection(config=config)
        self.model = models.refinement_model.OdometryModel(config=self.config).to(self.device)

        # Geometry handler
        self.geometry_handler = models.model_parts.GeometryHandler(config=config)

        # Loss and optimizer
        self.lossTransformation = losses.motion_losses.UncertaintyLoss().to(self.device)

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
        return torch.matmul(transformation_matrix[:, :3, :3], point_cloud[:, :3, :])

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
            point_cloud=point_cloud.unsqueeze(0))
        
    def rotate_transformation_matrix_vector(self, euler, origin_matrix):
        translation = torch.zeros(3, device=self.device)
        euler = euler.to(self.device)
        transformation_matrix = self.geometry_handler.get_transformation_matrix_angle_axis(
            translation=translation,
            euler=euler, device=self.device)
        return torch.matmul(origin_matrix, transformation_matrix.squeeze(0).double())

    def augment_input(self, scan, pose):
        # Random rotation
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
        preprocess_scan = self.rotate_point_cloud_euler_vector(
            point_cloud=scan, euler=euler)
        preprocess_pose = self.rotate_transformation_matrix_vector(
            origin_matrix=pose, euler=euler)
        return preprocess_scan, preprocess_pose

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
        batch_scan_1 = torch.zeros(self.batch_size, 3, self.config["kitti"]["max_points"], device=self.device)
        batch_scan_2 = torch.zeros_like(batch_scan_1)
        batch_target_q = torch.zeros(self.batch_size, 4, device=self.device)
        batch_target_t = torch.zeros(self.batch_size, 3, device=self.device)
        # Use every batchindex separately
        for index, preprocessed_dict in enumerate(preprocessed_dicts):
            # preprocess data
            scan_1 = preprocessed_dict["scan_1"]
            scan_2 = preprocessed_dict["scan_2"].to(self.device)

            padded_scan_1 = torch.zeros((3, self.config["kitti"]["max_points"]), dtype=torch.float, device=self.device)
            padded_scan_2 = torch.zeros((3, self.config["kitti"]["max_points"]), dtype=torch.float, device=self.device)

            # 将原始点云复制到新的张量中，实现填充
            padded_scan_1[:, :scan_1.shape[1]] = scan_1[:3, :]
            padded_scan_2[:, :scan_2.shape[1]] = scan_2[:3, :]
            
            # if self.config["normalization_scaling"]:
            #     preprocessed_data, scaling_factor = self.normalize_input(preprocessed_data=preprocessed_dict)
            
            # augment rotation
            if self.training_bool and self.config["random_point_cloud_rotations"]:
                preprocessed_scan_2,  target_transformation = self.augment_input(padded_scan_2, 
                                                    torch.tensor(preprocessed_dict["pose"]).to(self.device))
                batch_scan_1[index] = padded_scan_1
                batch_scan_2[index] = preprocessed_scan_2
                
            else:
                batch_scan_1[index] = padded_scan_1
                batch_scan_2[index] = padded_scan_2
                target_transformation = torch.tensor(preprocessed_dict["pose"]).to(self.device)
            
            # preprocess label
            target_q = self.geometry_handler.get_quaternion_from_transformation_matrix(target_transformation.unsqueeze(0))
            target_t = target_transformation[:3, 3]
            batch_target_q[index] = target_q
            batch_target_t[index] = target_t
            
        # Feed into model as batch
        # (translations, rotation_representation) = self.model(images_model_1, images_model_2)
        det_q, det_t = self.model(batch_scan_1, batch_scan_2)
        
        computed_transformations = self.geometry_handler.get_transformation_matrix_quaternion(
            translation=det_t[-1], quaternion=det_q[-1], device=self.device)

        # Following part only done when loss needs to be computed
        if not self.config["inference_only"]:
            ## Losses
            loss_transformation = self.lossTransformation(batch_target_q, det_q, batch_target_t, det_t)
            loss = loss_transformation["loss"]/self.batch_size  # Overwrite loss for identity fitting

            if self.training_bool:
                loss.backward()
                self.optimizer.step()

            if self.config["normalization_scaling"]:
                for index, preprocessed_dict in enumerate(preprocessed_dicts):
                    computed_transformations[index, :3, 3] *= preprocessed_dict["scaling_factor"]

            epoch_losses["loss_epoch"] += loss.detach().cpu().numpy()
            epoch_losses["st_epoch"] += loss_transformation["st"].detach().cpu().numpy()
            epoch_losses["sq_epoch"] += loss_transformation["sq"].detach().cpu().numpy()
            epoch_losses["loss_t_epoch"] += loss_transformation["loss_t"].detach().cpu().numpy()
            epoch_losses["loss_q_epoch"] += loss_transformation["loss_q"].detach().cpu().numpy()

            return epoch_losses, computed_transformations
        else:
            if self.config["normalization_scaling"]:
                for index, preprocessed_dict in enumerate(preprocessed_dicts):
                    computed_transformations[index, :3, 3] *= preprocessed_dict["scaling_factor"]

            return computed_transformations
        
if __name__ == '__main__':      
    import torch

    # 假设您有两个点云张量，分别是 points1 和 points2

    # points1 是形状为 (1, N1, 3) 的点云张量
    points1 = torch.tensor([[[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]]])

    # points2 是形状为 (1, N2, 3) 的点云张量
    points2 = torch.tensor([[[7.0, 8.0, 9.0],
                            [10.0, 11.0, 12.0],
                            [13.0, 14.0, 15.0]]])

    # 使用 torch.cat() 在第一个维度上进行拼接，合并两个点云
    merged_points = torch.cat((points1, points2), dim=0)

    # 现在，merged_points 的形状为 (2, N, 3)
    # 其中 2 表示有两个点云，N = N1 + N2 表示合并后的总点数
    # 3 表示每个点云中点的维度

    print(merged_points)


