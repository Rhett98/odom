#!/usr/bin/env python3

import torch

class supervisedLosses(torch.nn.Module):
    def __init__(self):
        super(supervisedLosses, self).__init__()
        self.loss_q = torch.nn.MSELoss()
        self.loss_t = torch.nn.L1Loss()

    def forward(self, target_q, det_q, target_t, det_t):
        loss_q, loss_t = 0, 0
        weights = [0.2, 0.4, 0.8, 1.6]
        for i in range(0, len(det_q)):    
            loss_q += weights[i] * self.loss_q(target_q, det_q[i])
            loss_t += weights[i] * self.loss_t(target_t, det_t[i])

        losses = {
            "loss": 100*loss_q + loss_t,
            "st": loss_q,
            "sq": loss_q,
            "loss_q": loss_q,
            "loss_t": loss_t,
        }
        return losses
    
class GeometricLoss(torch.nn.Module):
    """ Geometric loss function from PoseNet paper """
    def __init__(self, st=0.0, sq=-2.5):
        super(GeometricLoss, self).__init__()
        self.st = torch.nn.Parameter(torch.tensor(st, requires_grad=True))
        self.sq = torch.nn.Parameter(torch.tensor(sq, requires_grad=True))
        self.loss_q = torch.nn.MSELoss()
        self.loss_t = torch.nn.L1Loss()
        
    def forward(self, target_q, det_q, target_t, det_t):         
        loss_q, loss_t = 0, 0
        weights = [0.2, 0.4, 0.8, 1.6]
        for i in range(0, len(det_q)):    
            loss_q += weights[i] * self.loss_q(target_q, det_q[i])
            loss_t += weights[i] * self.loss_t(target_t, det_t[i])
        loss = torch.exp(-self.st)*loss_t + self.st \
               + torch.exp(-self.sq)*loss_q + self.sq   
        
        losses = {
            "loss": loss,
            "st": self.st,
            "sq": self.sq,
            "loss_q": loss_q,
            "loss_t": loss_t,
        }
        # print("value: ", self.st, self.sq)
        return losses

class UncertaintyLoss(torch.nn.Module):
    def __init__(self, v_num=7):
        super(UncertaintyLoss, self).__init__()
        sigma = torch.randn(v_num)
        self.sigma = torch.nn.Parameter(sigma)
        self.v_num = v_num
        self.loss_q = torch.nn.MSELoss()
        self.loss_t = torch.nn.L1Loss()
        
    def forward(self, target_q, det_q, target_t, det_t): 
        loss_q0, loss_q1, loss_q2, loss_q3 = 0, 0, 0, 0
        loss_t0, loss_t1, loss_t2 = 0, 0, 0
        weights = [0.2, 0.4, 0.8, 1.6]
 
        for i in range(0, len(det_q)):    
            loss_q0 += weights[i] * self.loss_q(target_q[:,0], det_q[i][:,0])
            loss_q1 += weights[i] * self.loss_q(target_q[:,1], det_q[i][:,1])
            loss_q2 += weights[i] * self.loss_q(target_q[:,2], det_q[i][:,2])
            loss_q3 += weights[i] * self.loss_q(target_q[:,3], det_q[i][:,3])
            loss_t0 += weights[i] * self.loss_t(target_t[:,0], det_t[i][:,0])  
            loss_t1 += weights[i] * self.loss_t(target_t[:,1], det_t[i][:,1])
            loss_t2 += weights[i] * self.loss_t(target_t[:,2], det_t[i][:,2])
        print(loss_q0, loss_q1, loss_q2, loss_q3)
        print(loss_t0, loss_t1, loss_t2)
        loss = torch.exp(-self.sigma[0])*loss_q0 + self.sigma[0] +\
                    torch.exp(-self.sigma[1])*loss_q1 + self.sigma[1] +\
                    torch.exp(-self.sigma[2])*loss_q2 + self.sigma[2] +\
                    torch.exp(-self.sigma[3])*loss_q3 + self.sigma[3] +\
                    torch.exp(-self.sigma[4])*loss_t0 + self.sigma[4] +\
                    torch.exp(-self.sigma[5])*loss_t1 + self.sigma[5] +\
                    torch.exp(-self.sigma[6])*loss_t2 + self.sigma[6] 
        losses = {
            "loss": loss,
            "st": self.sigma[0],
            "sq": self.sigma[4],
            "loss_q": loss_q0 + loss_q1 + loss_q2 + loss_q3,
            "loss_t": loss_t0 + loss_t1 + loss_t2,
        }
        return losses
    
# class UncertaintyLoss(torch.nn.Module):
#     def __init__(self, v_num=7):
#         super(UncertaintyLoss, self).__init__()
#         sigma = torch.randn(v_num)
#         self.sigma = torch.nn.Parameter(sigma)
#         self.v_num = v_num
#         self.loss_q = torch.nn.MSELoss()
#         self.loss_t = torch.nn.L1Loss()
        
#     def forward(self, target_q, det_q, target_t, det_t): 
#         # print(det_q.shape, det_t.shape)
#         loss_q0, loss_q1, loss_q2, loss_q3 = 0, 0, 0, 0
#         loss_t0, loss_t1, loss_t2 = 0, 0, 0
#         loss_q0 = self.loss_q(target_q[:,0], det_q[:,0])
#         loss_q1 = self.loss_q(target_q[:,1], det_q[:,1])
#         loss_q2 = self.loss_q(target_q[:,2], det_q[:,2])
#         loss_q3 = self.loss_q(target_q[:,3], det_q[:,3])
#         loss_t0 = self.loss_t(target_t[:,0], det_t[:,0])  
#         loss_t1 = self.loss_t(target_t[:,1], det_t[:,1])
#         loss_t2 = self.loss_t(target_t[:,2], det_t[:,2])
        
#         loss = torch.exp(-self.sigma[0])*loss_q0 + self.sigma[0] +\
#             torch.exp(-self.sigma[1])*loss_q1 + self.sigma[1] +\
#             torch.exp(-self.sigma[2])*loss_q2 + self.sigma[2] +\
#             torch.exp(-self.sigma[3])*loss_q3 + self.sigma[3] +\
#             torch.exp(-self.sigma[4])*loss_t0 + self.sigma[4] +\
#             torch.exp(-self.sigma[5])*loss_t1 + self.sigma[5] +\
#             torch.exp(-self.sigma[6])*loss_t2 + self.sigma[6] 
#         # loss = loss_q0 / (2 * self.sigma[0] ** 2) + loss_q1 / (2 * self.sigma[1] ** 2) +\
#         #        loss_q2 / (2 * self.sigma[2] ** 2) + loss_q3 / (2 * self.sigma[3] ** 2) +\
#         #        loss_t0 / (2 * self.sigma[4] ** 2) + loss_t1 / (2 * self.sigma[5] ** 2) +\
#         #        loss_t2 / (2 * self.sigma[6] ** 2) 
#         # loss += torch.log(self.sigma.pow(2).prod())
#         # print(loss_q0 , loss_q1 , loss_q2 , loss_q3, loss_t0 , loss_t1 , loss_t2)
#         losses = {
#             "loss": loss,
#             "st": self.sigma[0],
#             "sq": self.sigma[1],
#             "loss_q": loss_q0 + loss_q1 + loss_q2 + loss_q3,
#             "loss_t": loss_t0 + loss_t1 + loss_t2,
#         }
#         return losses