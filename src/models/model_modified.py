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
        in_channels = 10
        num_feature_extraction_layers = 5
        self.img_projection = utility.projection.ImageProjection(config=config)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 0))
        self.moving_seg_network = SalsaNext(nclasses=3, input_scan=1)
        
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
        input_scan: (b, 4, n) or (b, 3, n) 
        """
        # Use every batchindex separately
        B, _, _ = input_scan.shape
        images_model = torch.zeros(B, 5,
                                     self.config[preprocessed_dicts[0]["dataset"]]["vertical_cells"],
                                     self.config[preprocessed_dicts[0]["dataset"]]["horizontal_cells"],
                                     device=self.device)
        proj_xyz_model = torch.Tensor().to(self.device)
        for index in range(0, B):
            # preprocessed_dict = self.augment_input(preprocessed_data=preprocessed_dict)
            # Training / Testing
            image_range, image_xyz, image_remission = self.img_projection.do_spherical_projection(
                input_scan[index])
            # image_model = torch.cat([image_range, image_xyz, image_remission])
            image_model = torch.cat([image_range, image_xyz, image_remission])
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
        image_2, proj_point_2 = self.proj_image(xyz_2, preprocessed_dicts)
        
        # delete moving point
        moving_label = self.moving_seg_network(torch.cat([image_1, image_2]))
        moving_label_argmax = moving_label.argmax(dim=1)
        static_pc_1 = del_moving_point(proj_point_1, moving_label_argmax[0].unsqueeze(0), 0.5)
        static_pc_2 = del_moving_point(proj_point_2, moving_label_argmax[1].unsqueeze(0), 0.5)
        static_image_1, static_proj_point_1 = self.proj_image(static_pc_1, preprocessed_dicts)
        static_image_2, _ = self.proj_image(static_pc_2, preprocessed_dicts)

        
        ### l1-layer ###
        features_l1 = self.resnet1(self.forward_features(image_1=static_image_1, image_2=static_image_2))
        l1_q = self.fully_connected_rotation1(features_l1)
        l1_t = self.fully_connected_translation1(features_l1)
        l1_q = l1_q / torch.norm(l1_q)
        
        ### l2-layer ###
        warp_xyz_1_l1 = self.warp_pointcloud(static_proj_point_1, l1_t, l1_q)
        image_1_l2, _= self.proj_image(warp_xyz_1_l1, preprocessed_dicts)
        
        features_l2 = self.resnet2(self.forward_features(image_1=image_1_l2, image_2=image_2))
        l2_q_det = self.fully_connected_rotation2(features_l2)
        l2_t_det = self.fully_connected_translation2(features_l2)

        l2_q_det = l2_q_det / torch.norm(l2_q_det)
        l2_q, l2_t = self.warp_det_result(l1_q, l1_t, l2_q_det, l2_t_det)
        
        ### l3-layer ###
        warp_xyz_1_l2 = self.warp_pointcloud(static_proj_point_1, l2_t, l2_q)
        image_1_l3, _ = self.proj_image(warp_xyz_1_l2, preprocessed_dicts)
        
        features_l3 = self.resnet3(self.forward_features(image_1=image_1_l3, image_2=image_2))
        l3_q_det = self.fully_connected_rotation3(features_l3)
        l3_t_det = self.fully_connected_translation3(features_l3)

        l3_q_det = l3_q_det / torch.norm(l3_q_det)
        l3_q, l3_t = self.warp_det_result(l2_q, l2_t, l3_q_det, l3_t_det)
        
        ### l4-layer ###
        warp_xyz_1_l3 = self.warp_pointcloud(static_proj_point_1, l3_t, l3_q)
        image_1_l4, _ = self.proj_image(warp_xyz_1_l3, preprocessed_dicts)
        
        features_l4 = self.resnet4(self.forward_features(image_1=image_1_l4, image_2=image_2))
        l4_q_det = self.fully_connected_rotation4(features_l4)
        l4_t_det = self.fully_connected_translation4(features_l4)
        l4_q_det = l4_q_det / torch.norm(l4_q_det)
        l4_q, l4_t = self.warp_det_result(l3_q, l3_t, l4_q_det, l4_t_det)

        return (l1_q, l2_q, l3_q, l4_q),(l1_t, l2_t, l3_t, l4_t), moving_label

def del_moving_point(batch_pc, batch_label, rate=0.5):
    """
    pc(B, 3, N) : 原始点云
    label_img(B, W, H) : 原始点云投影得到的img, 经过动态分割网络得到的label, unlabel:0, static:1, moving:2
    rate(float) : 移除动态点云的比率
    return: static_pc(B, 3, N) : 输入到odometry网络的静态点云, 动态点填充为0
    """
    B, W, H = batch_label.shape
    batch_static_pc = torch.zeros(B, 3, W*H)
    for index in range(0, B):
        pc = batch_pc[index]
        label_pc = batch_label[index].view(W*H, 1)
        # 找到标签中值为2的像素，表示动态点云
        moving_pixels = torch.nonzero(label_pc == 2, as_tuple=False)

        # 计算要移除的动态点云数量
        num_points = moving_pixels.size(0)
        num_points_to_remove = int(rate * num_points)

        # 随机选择要保留的动态点云索引
        shuffled_indices = torch.randperm(num_points)
        moving_pixels_to_remove = moving_pixels[shuffled_indices[:num_points_to_remove]]

        # 创建一个和原始点云相同的零张量
        static_pc = torch.zeros_like(pc)

        # 从原始点云中挑选出静态点云，将动态点云部分置0
        static_pc[:, :] = pc[:, :]
        static_pc[:, moving_pixels_to_remove[:, 1]] = 0

        batch_static_pc[index] = static_pc
        
    return batch_static_pc

def list_collate(batch_dicts):
        data_dicts = [batch_dict for batch_dict in batch_dicts]
        return data_dicts

if __name__ == '__main__':
    from thop import profile
    import losses.motion_losses
    import losses.segment_losses
    import yaml
    config = yaml.safe_load(open('/home/yu/Resp/odom/config/config_datasets_test.yaml', 'r'))
    a = yaml.safe_load(open('/home/yu/Resp/odom/config/deployment_options.yaml', 'r'))
    config.update(a)
    f = open('config/semantic-kitti-mos.yaml')
    sematic_options = yaml.load(f, Loader=yaml.FullLoader)
    config.update(sematic_options)
    b = yaml.safe_load(open('/home/yu/Resp/odom/config/hyperparameters.yaml', 'r'))
    config.update(b)
    dataset = "kitti"
    config[dataset]["data_identifiers"] = config[dataset]["training_identifiers"]
    # model = OdometryModel(config)
    # # fn = model.warp_pointcloud(torch.randn(1, 1000, 3 ),torch.randn(1, 3), torch.randn(1, 4))
    # # print(fn.shape)
    # # dummy_input = torch.randn(1, 4, 64, 720),torch.randn(1, 4, 64, 720)
    # # flops, params = profile(model, (dummy_input))
    # # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    
    
    # model = SalsaNext(3,2)
    # input = torch.randn(2, 10, 64, 2048)
    # pc = torch.randn(2, 64*2048, 3)
    # out = model(input)
    # print(out.shape)
    # print(out.argmax(dim=1).shape)
    # print("____________")
    # print(pc)
    # print(del_moving_point(pc, out.argmax(dim=1)).shape)
    
    model = OdometryModel(config)
    import data.dataset_label
    import models.model_parts
    geometry_handler = models.model_parts.GeometryHandler(config=config)
    DATASET = data.dataset_label.PreprocessedPointCloudDataset(config)
    lossTransformation = losses.motion_losses.UncertaintyLoss()
    lossSegmentation = losses.segment_losses.SegmentLoss(config=config)
    dataloader = torch.utils.data.DataLoader(dataset=DATASET,
                                                 batch_size=1,
                                                 shuffle=True,
                                                 collate_fn=list_collate,
                                                 num_workers=0,
                                                 pin_memory=False)
    for preprocessed_dicts in dataloader:
        det_q, det_t, predict = model(preprocessed_dicts)
        target_transformation = torch.tensor(preprocessed_dicts[0]["pose"]).unsqueeze(0).float()   
        target_q = geometry_handler.get_quaternion_from_transformation_matrix(target_transformation)
        target_t = target_transformation[:, :3, 3]
        loss_trans = lossTransformation(target_q, det_q, target_t, det_t)
        loss_seg = lossSegmentation(torch.cat([preprocessed_dicts[0]["proj_label_1"],preprocessed_dicts[0]["proj_label_2"]]), predict)
        print(loss_seg)
        break

    # # 2D loss example (used, for example, with image inputs)
    # N, C = 5, 4
    # loss = nn.NLLLoss()
    # # input is of size N x C x height x width
    # data = torch.randn(N, 16, 10, 10)
    # conv = nn.Conv2d(16, C, (3, 3))
    # m = nn.LogSoftmax(dim=1)
    # # each element in target has to have 0 <= value < C
    # target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
    # predict = m(conv(data))
    # output = loss(predict, target)
    # print(predict.shape, target.shape)
    # print(output)

    
    

    