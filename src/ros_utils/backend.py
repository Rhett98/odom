from typing import List
from threading import Lock

import matplotlib.pyplot as plt
import numpy as np
import math

import gtsam
import gtsam.utils.plot as gtsam_plot

def create_poses() -> List[gtsam.Pose3]:
     """Creates ground truth poses of the robot."""
     P0 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
     P1 = np.array([[0, -1, 0, 15],
                    [1, 0, 0, 15],
                    [0, 0, 1, 20],
                    [0, 0, 0, 1]])
     P2 = np.array([[np.cos(np.pi/4), 0, np.sin(np.pi/4), 30],
                    [0, 1, 0, 30],
                    [-np.sin(np.pi/4), 0, np.cos(np.pi/4), 30],
                    [0, 0, 0, 1]])
     P3 = np.array([[0, 1, 0, 30],
                    [0, 0, -1, 0],
                    [-1, 0, 0, -15],
                    [0, 0, 0, 1]])
     P4 = np.array([[-1, 0, 0, 0],
                    [0, -1, 0, -10],
                    [0, 0, 1, -10],
                    [0, 0, 0, 1]])
     P5 = P0[:]
 
     return [gtsam.Pose3(P0), gtsam.Pose3(P1), gtsam.Pose3(P2),
             gtsam.Pose3(P3), gtsam.Pose3(P4), gtsam.Pose3(P5)]

def gtsam_pose_to_np(gtsam_pose):
    position = np.array([
        gtsam_pose.x(),
        gtsam_pose.y(),
        gtsam_pose.z()])
    quat = gtsam_pose.rotation().quaternion()
    orientation = np.array([quat[0], quat[1], quat[2], quat[3]]) # wxyz
    return position, orientation

def np_to_gtsam_pose(position, orientation):
    return gtsam.Pose3(
        gtsam.Rot3.Quaternion(
            orientation[0],
            orientation[1],
            orientation[2],
            orientation[3]),
        gtsam.Point3(
            position[0],
            position[1],
            position[2])
    )

class FactorGraph(object):
    """
        A simple factor graph.
    """
    def __init__(self):
        # Declare the 3D translational standard deviations of the prior factor's Gaussian model, in meters.
        prior_xyz_sigma = 0.3
        # Declare the 3D rotational standard deviations of the prior factor's Gaussian model, in degrees.
        prior_rpy_sigma = 5
        priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_rpy_sigma*np.pi/180,
                                                                    prior_rpy_sigma*np.pi/180,
                                                                    prior_rpy_sigma*np.pi/180,
                                                                    prior_xyz_sigma,
                                                                    prior_xyz_sigma,
                                                                    prior_xyz_sigma]))
           
        self.lock = Lock()

        # All variables required for a persistent state of the graph or rebuilding it from memory.
        self.initial_poses = []

        # The working memory version of graph slam using gtsam.
        self.graph_backend = gtsam.NonlinearFactorGraph()
        self.graph_estimates = gtsam.Values()
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.0001)
        parameters.setRelinearizeSkip(1)
        self.isam = gtsam.ISAM2(parameters)

        # Initialize the graph with the origin pose.
        priorPose = gtsam.Pose3(gtsam.Rot3.Quaternion(0,0,0,0), gtsam.Point3(0,0,0))  # prior at origin
        self.graph_estimates.insert(1, priorPose)
        self.initial_poses.append(priorPose)
        self.graph_backend.push_back(gtsam.PriorFactorPose3(1, priorPose, priorNoise))

    def save(self, filenname):
        np.savez(filenname, initial_poses=self.initial_poses)

    def get_current_estimate(self):
        """
            Returns the current estimate of the trajectory without reoptimizing it.
        """
        return self.initial_poses

    def get_current_index(self):
        """
            Gets the current last index of the main datastream.
        """
        return len(self.initial_poses)
    
    def optimize(self, steps=1):
        """
            Optimize the graph.

            Returns the trajectory of the main_poses after optimization.
        """
        self.lock.acquire()
        print("Optimizing: " + str(self.get_current_index()))
        result = []
        print("self.graph_estimates:", self.graph_estimates)
        # Update the graph
        self.isam.update(self.graph_backend, self.graph_estimates)
        for i in range(steps):
            self.isam.update()

        # Extract the estimates from the graph back into the initial_estimates.
        current_estimate = self.isam.calculateEstimate()
        for i in current_estimate.keys():
            pose = current_estimate.atPose3(i)
            result.append(pose)

        self.initial_poses = result

        # Reset everything for the next run
        self.graph_backend = gtsam.NonlinearFactorGraph()
        self.graph_estimates.clear()
        
        self.lock.release()
        return result

    def append_relative_pose(self, pose, covariance):
        """
            Adds a relative pose in the graph to the main datastream.
            This increments the index and those poses are used for initial estimates.

            The pose is expected as an array T 4x4.

            You must provide a valid covariance matrix for the pose.
        """
        self.lock.acquire()

        # Prepare helper variables
        idx = self.get_current_index() + 1

        # Insert pose into initial_estimates
        self.initial_poses.append(pose)
        self.graph_estimates.insert(idx, pose)

        # Insert relative edge into graph.
        self.graph_backend.push_back(gtsam.BetweenFactorPose3(idx-1, idx, pose, covariance))

        self.lock.release()



if __name__ == '__main__':
    # Declare the 3D translational standard deviations of the odometry factor's Gaussian model, in meters.
    odometry_xyz_sigma = 0.2

    # Declare the 3D rotational standard deviations of the odometry factor's Gaussian model, in degrees.
    odometry_rpy_sigma = 5
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([odometry_rpy_sigma*np.pi/180,
                                                                odometry_rpy_sigma*np.pi/180,
                                                                odometry_rpy_sigma*np.pi/180,
                                                                odometry_xyz_sigma,
                                                                odometry_xyz_sigma,
                                                                odometry_xyz_sigma]))
  
    true_poses = create_poses()
 
    # Create the ground truth odometry transformations, xyz translations, and roll-pitch-yaw rotations
    # between each robot pose in the trajectory.
    odometry_tf = [true_poses[i-1].transformPoseTo(true_poses[i]) for i in range(1, len(true_poses))]
    odometry_xyz = [(odometry_tf[i].x(), odometry_tf[i].y(), odometry_tf[i].z()) for i in range(len(odometry_tf))]
    odometry_rpy = [odometry_tf[i].rotation().rpy() for i in range(len(odometry_tf))]

    # Corrupt xyz translations and roll-pitch-yaw rotations with gaussian noise to create noisy odometry measurements.
    noisy_measurements = [np.random.multivariate_normal(np.hstack((odometry_rpy[i],odometry_xyz[i])), \
                                                        ODOMETRY_NOISE.covariance()) for i in range(len(odometry_tf))]
    backend = FactorGraph()
  
    for i in range(0, len(true_poses)-1):
        noisy_odometry = noisy_measurements[i]
        noisy_tf = gtsam.Pose3(gtsam.Rot3.RzRyRx(noisy_odometry[:3]), noisy_odometry[3:6].reshape(-1,1))
        print("input:", gtsam_pose_to_np(noisy_tf))
        backend.append_relative_pose(true_poses[i], ODOMETRY_NOISE)
        result = backend.optimize()
        print("result: ", gtsam_pose_to_np(result[-1]))
        print("********************")
    
