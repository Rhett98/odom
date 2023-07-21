from __future__ import division
import os
import sys
path = os.getcwd()
sys.path.append(path)
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.model_parts
import models.resnet_modified
import utility.projection
from models.BaseBlocks import ResBlock, UpBlock,ResContextBlock

class SalsaNext(nn.Module):
    def __init__(self, nclasses=3, input_scan=2):
        super(SalsaNext, self).__init__()
        self.nclasses = nclasses
        self.input_size = 5 * input_scan

        print("Depth of backbone input = ", self.input_size)
        ###
        
        self.downCntx = ResContextBlock(self.input_size, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False)
         
        self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))

    def forward(self, x):
        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)

        up4e = self.upBlock1(down5c,down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)
        logits = self.logits(up1e)

        logits = logits
        logits = F.softmax(logits, dim=1)
        return logits

class OdometryModel(torch.nn.Module):
    def __init__(self, config):
        super(OdometryModel, self).__init__()

        self.device = config["device"]
        self.config = config
        self.batch_size = config["batch_size"]
        self.pre_feature_extraction = config["pre_feature_extraction"]
        in_channels = 8
        num_feature_extraction_layers = 5
        self.img_projection = utility.projection.ImageProjection(config=config)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 0))

        print("Used activation function is: " + self.config["activation_fct"] + ".")
        if self.config["activation_fct"] != "relu" and self.config["activation_fct"] != "tanh":
            raise Exception('The specified activation function must be either "relu" or "tanh".')

        # Here are trainable parameters
        if config["pre_feature_extraction"]:
            layers = []
            for layer_index in range(num_feature_extraction_layers):
                input_channels = (
                    int(in_channels / 2) if layer_index == 0 else (layer_index) * in_channels)
                layers.append(models.model_parts.CircularPad(padding=(1, 1, 0, 0)))
                layers.append(torch.nn.Conv2d(in_channels=input_channels,
                                              out_channels=(layer_index + 1) * in_channels,
                                              kernel_size=3, padding=(1, 0), bias=False))
                if self.config["activation_fct"] == "relu":
                    layers.append(torch.nn.ReLU(inplace=True))
                else:
                    layers.append(torch.nn.Tanh())
            self.feature_extractor = torch.nn.Sequential(*layers)
            print("Number of trainable parameters in our feature extractor: " + \
                  f'{sum(p.numel() for p in self.feature_extractor.parameters()):,}')
        print('Resnet in_channels:',in_channels if not config[
                "pre_feature_extraction"] else 2 * num_feature_extraction_layers * in_channels)
        
        self.resnet1 = models.resnet_modified.ResNetModified(
            in_channels=in_channels if not config[
                "pre_feature_extraction"] else 2 * num_feature_extraction_layers * in_channels,
            num_outputs=config["resnet_outputs"],
            use_dropout=self.config["use_dropout"],
            layers=self.config["layers"],
            factor_fewer_resnet_channels=self.config["factor_fewer_resnet_channels"],
            activation_fct=self.config["activation_fct"],
            output_layer=1)

        self.resnet2 = models.resnet_modified.ResNetModified(
            in_channels=in_channels if not config[
                "pre_feature_extraction"] else 2 * num_feature_extraction_layers * in_channels,
            num_outputs=config["resnet_outputs"],
            use_dropout=self.config["use_dropout"],
            layers=self.config["layers"],
            factor_fewer_resnet_channels=self.config["factor_fewer_resnet_channels"],
            activation_fct=self.config["activation_fct"],
            output_layer=2)

        self.resnet3 = models.resnet_modified.ResNetModified(
            in_channels=in_channels if not config[
                "pre_feature_extraction"] else 2 * num_feature_extraction_layers * in_channels,
            num_outputs=config["resnet_outputs"],
            use_dropout=self.config["use_dropout"],
            layers=self.config["layers"],
            factor_fewer_resnet_channels=self.config["factor_fewer_resnet_channels"],
            activation_fct=self.config["activation_fct"],
            output_layer=3)

        self.resnet4 = models.resnet_modified.ResNetModified(
            in_channels=in_channels if not config[
                "pre_feature_extraction"] else 2 * num_feature_extraction_layers * in_channels,
            num_outputs=config["resnet_outputs"],
            use_dropout=self.config["use_dropout"],
            layers=self.config["layers"],
            factor_fewer_resnet_channels=self.config["factor_fewer_resnet_channels"],
            activation_fct=self.config["activation_fct"],
            output_layer=4)

        rot_out_features = 4
        self.fully_connected_rotation1 = torch.nn.Sequential(
            torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(in_features=32, out_features=rot_out_features))
        self.fully_connected_translation1 = torch.nn.Sequential(
            torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(in_features=32, out_features=3))
        
        self.fully_connected_rotation2 = torch.nn.Sequential(
            torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(in_features=128, out_features=64),
            torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(in_features=64, out_features=rot_out_features))
        self.fully_connected_translation2 = torch.nn.Sequential(
            torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(in_features=128, out_features=64),
            torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(in_features=64, out_features=3))
        
        self.fully_connected_rotation3 = torch.nn.Sequential(
            torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(in_features=256, out_features=100),
            torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(in_features=100, out_features=rot_out_features))
        self.fully_connected_translation3 = torch.nn.Sequential(
            torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(in_features=256, out_features=100),
            torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(in_features=100, out_features=3))
        
        self.fully_connected_rotation4 = torch.nn.Sequential(
            torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(in_features=config["resnet_outputs"], out_features=100),
            torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(in_features=100, out_features=rot_out_features))
        self.fully_connected_translation4 = torch.nn.Sequential(
            torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(in_features=config["resnet_outputs"], out_features=100),
            torch.nn.ReLU() if self.config["activation_fct"] == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(in_features=100, out_features=3))

        # geometry_handler does not contain any trainable parameters
        self.geometry_handler = models.model_parts.GeometryHandler(config=config)
    
    def proj_image(self, input_scan, preprocessed_dicts):
        """project pointcloud to image coordinate
        input_scan: (b, 4, n) or (b, 3, n) without remission
        """
        # Use every batchindex separately
        images_model = torch.zeros(self.batch_size, 4,
                                     self.config[preprocessed_dicts[0]["dataset"]]["vertical_cells"],
                                     self.config[preprocessed_dicts[0]["dataset"]]["horizontal_cells"],
                                     device=self.device)
        proj_xyz_model = torch.Tensor().to(self.device)
        for index in range(0, self.batch_size):
            # preprocessed_dict = self.augment_input(preprocessed_data=preprocessed_dict)
            # Training / Testing
            image_range, image_xyz, image_remission = self.img_projection.do_spherical_projection(
                input_scan[index])
            # image_model = torch.cat([image_range, image_xyz, image_remission])
            image_model = torch.cat([image_range, image_xyz])
            # Write projected image to batch
            images_model[index] = image_model
            proj_xyz_model = image_xyz.view(1, 3, -1).to(self.device)
        return images_model, proj_xyz_model
    
    def warp_pointcloud(self, pc_xyz, translation, rotation):
        inv_q = models.model_parts.inv_q(rotation)
        # change (B, 3, N) to (B, N, 4)
        pc_xyz = pc_xyz.transpose(1,2).clone()
        pc_xyz = torch.cat([torch.zeros(pc_xyz.size(0), pc_xyz.size(1), 1, device=pc_xyz.device), pc_xyz], dim=2)
        # multi : q*p*q- 
        warp_pc = models.model_parts.mul_q_point(rotation, pc_xyz)
        warp_pc = models.model_parts.mul_point_q(warp_pc, inv_q)[..., 1:] + translation
        return warp_pc.transpose(1,2)
    
    def warp_det_result(self, q0, t0, q1, t1):
        inv_q1 = models.model_parts.inv_q(q1)
        warp_q = models.model_parts.mul_q_point(q1, q0.unsqueeze(1))
        # change (B, 3) to (B, 1, 4)
        t0 = t0.unsqueeze(1).clone()
        t0 = torch.cat([torch.zeros(t0.size(0), t0.size(1), 1, device=t0.device), t0], dim=2)
        warp_t = models.model_parts.mul_q_point(q1, t0)
        warp_t = models.model_parts.mul_point_q(warp_t, inv_q1)[..., 1:] + t1
        return warp_q.squeeze(1), warp_t.squeeze(1)
    
    def forward_features(self, image_1, image_2):
        if self.pre_feature_extraction:
            x1 = self.feature_extractor(image_1)
            x2 = self.feature_extractor(image_2)
            x = torch.cat((x1, x2), dim=1)
        else:
            x = torch.cat((image_1, image_2), dim=1)
        return x
    
    def forward(self, preprocessed_dicts):
        # xyzr_1 = preprocessed_dicts[0]["scan_1"]
        # xyzr_2= preprocessed_dicts[0]["scan_2"]
        # print(preprocessed_dicts[0]["scan_1"].shape)
        xyz_1 = preprocessed_dicts[0]["scan_1"][:3, :].unsqueeze(0)
        xyz_2 = preprocessed_dicts[0]["scan_2"][:3, :].unsqueeze(0)
        
        image_1, proj_point_1 = self.proj_image(xyz_1, preprocessed_dicts)
        image_2, _ = self.proj_image(xyz_2, preprocessed_dicts)
        
        ### l1-layer ###
        features_l1 = self.resnet1(self.forward_features(image_1=image_1, image_2=image_2))
        l1_q = self.fully_connected_rotation1(features_l1)
        l1_t = self.fully_connected_translation1(features_l1)
        l1_q = l1_q / torch.norm(l1_q)
        
        ### l2-layer ###
        warp_xyz_1_l1 = self.warp_pointcloud(proj_point_1, l1_t, l1_q)
        image_1_l2, _= self.proj_image(warp_xyz_1_l1, preprocessed_dicts)
        
        features_l2 = self.resnet2(self.forward_features(image_1=image_1_l2, image_2=image_2))
        l2_q_det = self.fully_connected_rotation2(features_l2)
        l2_t_det = self.fully_connected_translation2(features_l2)

        l2_q_det = l2_q_det / torch.norm(l2_q_det)
        l2_q, l2_t = self.warp_det_result(l1_q, l1_t, l2_q_det, l2_t_det)
        
        ### l3-layer ###
        warp_xyz_1_l2 = self.warp_pointcloud(proj_point_1, l2_t, l2_q)
        image_1_l3, _ = self.proj_image(warp_xyz_1_l2, preprocessed_dicts)
        
        features_l3 = self.resnet3(self.forward_features(image_1=image_1_l3, image_2=image_2))
        l3_q_det = self.fully_connected_rotation3(features_l3)
        l3_t_det = self.fully_connected_translation3(features_l3)

        l3_q_det = l3_q_det / torch.norm(l3_q_det)
        l3_q, l3_t = self.warp_det_result(l2_q, l2_t, l3_q_det, l3_t_det)
        
        ### l4-layer ###
        warp_xyz_1_l3 = self.warp_pointcloud(proj_point_1, l3_t, l3_q)
        image_1_l4, _ = self.proj_image(warp_xyz_1_l3, preprocessed_dicts)
        
        features_l4 = self.resnet4(self.forward_features(image_1=image_1_l4, image_2=image_2))
        l4_q_det = self.fully_connected_rotation4(features_l4)
        l4_t_det = self.fully_connected_translation4(features_l4)
        l4_q_det = l4_q_det / torch.norm(l4_q_det)
        l4_q, l4_t = self.warp_det_result(l3_q, l3_t, l4_q_det, l4_t_det)

        return (l1_q, l2_q, l3_q, l4_q),(l1_t, l2_t, l3_t, l4_t)

if __name__ == '__main__':
    from thop import profile
    
    import yaml
    config = yaml.safe_load(open('/home/yu/Resp/delora/config/config_datasets.yaml', 'r'))
    a = yaml.safe_load(open('/home/yu/Resp/delora/config/deployment_options.yaml', 'r'))
    config.update(a)
    b = yaml.safe_load(open('/home/yu/Resp/delora/config/hyperparameters.yaml', 'r'))
    config.update(b)
    # model = OdometryModel(config)
    # # fn = model.warp_pointcloud(torch.randn(1, 1000, 3 ),torch.randn(1, 3), torch.randn(1, 4))
    # # print(fn.shape)
    # # dummy_input = torch.randn(1, 4, 64, 720),torch.randn(1, 4, 64, 720)
    # # flops, params = profile(model, (dummy_input))
    # # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    
    # points = torch.tensor([[[1.0], [2.0],[3.0]]])  
    # quaternions = torch.tensor([[0.0, 0.707, 0.707, 0.0]])  
    # translations = torch.tensor([[1.0, 2.0, 3.0]])  
    # translations2 = torch.tensor([[0, 0, 0]]) 
    # transformed_points = model.warp_pointcloud(points, translations2, quaternions)
    # print("变换前的点:\n", points)
    # print("变换后的点:\n", transformed_points)
    
    # import numpy as np
    # import quaternion
    # q1 = np.quaternion(0,1,2,3 )
    # q2 = np.quaternion(0.0, 0.707, 0.707, 0.0)
    # q2_ = np.quaternion(0.0, -0.707, -0.707, 0.0)
    # q = q2*q1
    # print(q)
    # print(q*q2_)
    
    model = SalsaNext(3,2)
    input = torch.randn(1,10,64,2048)
    out = model(input)
    print(out)


    