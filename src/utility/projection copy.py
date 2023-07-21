#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import numpy as np
import torch
import numba


class ImageProjectionLayer(torch.nn.Module):

    def __init__(self, config):
        super(ImageProjectionLayer, self).__init__()
        self.device = config["device"]
        self.config = config
        self.horizontal_field_of_view = config["horizontal_field_of_view"]
        self.proj_W = self.config["kitti"]["vertical_cells"]
        self.proj_H = self.config["kitti"]["horizontal_cells"]
        self.proj_fov_down = self.horizontal_field_of_view[0]
        self.proj_fov_up = self.horizontal_field_of_view[1]
        # The following will be set while doing projections (different sensors etc.):
        # self.height_pixel, self.vertical_field_of_view

    def compute_2D_coordinates(self, point_cloud, width_pixel, height_pixel,
                               vertical_field_of_view):
        u = ((torch.atan2(point_cloud[:, 1, :], point_cloud[:, 0, :]) -
              self.horizontal_field_of_view[0]) / (
                     self.horizontal_field_of_view[1] - self.horizontal_field_of_view[0]) * (
                     width_pixel - 1))
        v = ((torch.atan2(point_cloud[:, 2, :], torch.norm(point_cloud[:, :2, :], dim=1)) -
              vertical_field_of_view[0]) / (
                     vertical_field_of_view[1] - vertical_field_of_view[0]) * (
                     height_pixel - 1))
        return u, v

    # Keeps closest point, because u and v are computed previously based on range-sorted point cloud
    @staticmethod
    @numba.njit
    def remove_duplicate_indices(u, v, occupancy_grid, unique_bool, image_to_pointcloud_indices):
        for index in range(len(u)):
            if not occupancy_grid[v[index], u[index]]:
                occupancy_grid[v[index], u[index]] = True
                unique_bool[index] = True
                image_to_pointcloud_indices[0, index, 0] = v[index]
                image_to_pointcloud_indices[0, index, 1] = u[index]
        return unique_bool, image_to_pointcloud_indices

    # input is unordered point cloud scan, e.g. shape [2000,4], with [.,0], [.,1], [.,2], [.,3] being x, y, z, i values
    # Gets projected to an image
    # returned point cloud only contains unique points per pixel-discretization, i.e. the closest one
    def project_to_img(self, point_cloud, dataset):
        # Get sensor specific parameters
        width_pixel = self.config[dataset]["horizontal_cells"]
        height_pixel = self.config[dataset]["vertical_cells"]
        vertical_field_of_view = self.config[dataset]["vertical_field_of_view"]

        # Add range to point cloud
        point_cloud_with_range = torch.zeros(
            (point_cloud.shape[0], point_cloud.shape[1] + 1, point_cloud.shape[2]),
            device=self.device, requires_grad=False)
        point_cloud_with_range[:, :point_cloud.shape[1], :] = point_cloud
        distance = torch.norm(point_cloud_with_range[:, :3, :], dim=1)
        point_cloud_with_range[:, -1, :] = distance.detach()
        del point_cloud  # safety such that only correctly sorted one is used
        # Only keep closest points
        sort_indices = torch.argsort(
            point_cloud_with_range[:, point_cloud_with_range.shape[1] - 1, :], dim=1)

        # for batch_idx in range(len(point_cloud_with_range)):
        point_cloud_with_range = point_cloud_with_range[:, :, sort_indices[0]]

        u, v = self.compute_2D_coordinates(point_cloud=point_cloud_with_range,
                                           width_pixel=width_pixel,
                                           height_pixel=height_pixel,
                                           vertical_field_of_view=vertical_field_of_view)

        inside_fov_bool = (torch.round(u) <= width_pixel - 1) & (torch.round(u) >= 0) & (
                torch.round(v) <= height_pixel - 1) & (torch.round(v) >= 0)
        u_filtered = torch.round(u[inside_fov_bool])
        v_filtered = torch.round(v[inside_fov_bool])
        point_cloud_with_range = point_cloud_with_range[:, :, inside_fov_bool[0]]

        occupancy_grid = np.zeros((height_pixel, width_pixel), dtype=bool)
        # Find pixel to point cloud mapping (for masking in loss later on)
        image_to_pointcloud_indices = np.zeros((1, len(u_filtered), 2), dtype=int)
        unique_bool = np.zeros((len(u_filtered)), dtype=bool)
        unique_bool, image_to_pointcloud_indices = ImageProjectionLayer.remove_duplicate_indices(
            u=(u_filtered.long().to(torch.device("cpu")).numpy()),
            v=(v_filtered.long().to(torch.device("cpu")).numpy()),
            occupancy_grid=occupancy_grid,
            unique_bool=unique_bool,
            image_to_pointcloud_indices=image_to_pointcloud_indices)
        unique_bool = torch.from_numpy(unique_bool).to(self.device)
        image_to_pointcloud_indices = torch.from_numpy(image_to_pointcloud_indices).to(self.device)

        u_filtered = u_filtered[unique_bool]
        v_filtered = v_filtered[unique_bool]
        point_cloud_with_range = point_cloud_with_range[:, :, unique_bool]
        image_to_pointcloud_indices = image_to_pointcloud_indices[:, unique_bool]

        image_representation = torch.zeros(
            (point_cloud_with_range.shape[0], point_cloud_with_range.shape[1], height_pixel,
             width_pixel), device=self.device, requires_grad=False)

        image_representation[:, :, v_filtered.long(), u_filtered.long()] = \
            point_cloud_with_range.to(self.device)

        return image_representation, u, v, sort_indices[inside_fov_bool][
            unique_bool], image_to_pointcloud_indices

    def forward(self, input, dataset):
        return self.project_to_img(point_cloud=input, dataset=dataset)
    
    
class ImageProjection():
    def __init__(self, config):
        super(ImageProjection, self).__init__()
        self.device = config["device"]
        self.config = config
        self.horizontal_field_of_view = config["horizontal_field_of_view"]
        self.proj_W = self.config["kitti"]["vertical_cells"]
        self.proj_H = self.config["kitti"]["horizontal_cells"]
        self.proj_fov_down = self.horizontal_field_of_view[0]
        self.proj_fov_up = self.horizontal_field_of_view[1]
        
    def do_spherical_projection(self, in_scan, in_label=None):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # conver torch.Tensor to numpy.ndarray
        in_scan = in_scan.permute(1,0).numpy()
        self.points = in_scan[:, :3]
        self.remissions = in_scan[:, 3]
        
        """ Reset scan members. """
        # self.points = np.zeros((0, 3), dtype=np.float32)  # [m, 3]: x, y, z
        # self.remissions = np.zeros((0, 1), dtype=np.float32)  # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), 0, dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), 0, dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), 0, dtype=np.float32)

        if in_label is not None:
            self.proj_label = np.full((self.proj_H, self.proj_W), 0, dtype=np.int32)
        
        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), 0, dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y

        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)
        # delete error point
        valid_indice = depth > 0.0
        depth = depth[valid_indice]
        
        # get scan components
        scan_x = self.points[:, 0][valid_indice]
        scan_y = self.points[:, 1][valid_indice]
        scan_z = self.points[:, 2][valid_indice]
        
        if in_label is not None:
            self.label = in_label[valid_indice]
        
        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)

        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        if in_label is not None:
            label = self.label[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        
        if in_label is not None:
            self.proj_label[proj_y, proj_x] = label
            
            return torch.from_numpy(self.proj_range).clone().unsqueeze(0), \
                torch.from_numpy(self.proj_xyz).clone().permute(2, 0, 1), \
                torch.from_numpy(self.proj_remission).clone().unsqueeze(0), \
                torch.from_numpy(self.proj_label).clone().unsqueeze(0)
        else:
            return torch.from_numpy(self.proj_range).clone().unsqueeze(0), \
                    torch.from_numpy(self.proj_xyz).clone().permute(2, 0, 1),\
                    torch.from_numpy(self.proj_remission).clone().unsqueeze(0)
