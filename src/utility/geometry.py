#!/usr/bin/env python3
# Parts of the code taken from pytorch3d (https://pytorch3d.readthedocs.io/)
import torch


def _angle_from_tan(
        axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2


def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def euler_to_quaternion(euler_angles):
    """
    将欧拉角转换为四元数表示
    
    参数:
        euler_angles: 欧拉角表示的旋转，[roll, pitch, yaw]
        
    返回值:
        四元数表示的旋转，[x, y, z, w]
    """
    roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return torch.tensor([qx, qy, qz, qw]).unsqueeze(0)

def quaternion_to_euler(quaternion):
    """
    将四元数转换为欧拉角表示
    
    参数:
        quaternion: 四元数表示的旋转，[x, y, z, w]
        
    返回值:
        欧拉角表示的旋转，[roll, pitch, yaw]
    """
    qx, qy, qz, qw= quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    
    # 归一化四元数
    norm = torch.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm

    # 计算欧拉角
    roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = torch.asin(2 * (qw * qy - qx * qz))
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    return torch.tensor([roll, pitch, yaw])

def quatt2T(t, q):
    t0, t1, t2 = t[:, 0], t[:, 1], t[:, 2]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Nq = w * w + x * x + y * y + z * z
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    
    c1 = torch.tensor([1.0], device=q.device)
    add = torch.tensor([0.0, 0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], -1)
    T = torch.cat([(c1 - (yY + zZ)).unsqueeze(1),
                   (xY - wZ).unsqueeze(1),
                   (xZ + wY).unsqueeze(1),
                   t0.unsqueeze(1)], dim=1)

    T = torch.cat([T,
                   (xY + wZ).unsqueeze(1),
                   (c1 - (xX + zZ)).unsqueeze(1),
                   (yZ - wX).unsqueeze(1),
                   t1.unsqueeze(1)], dim=1)

    T = torch.cat([T,
                   (xZ - wY).unsqueeze(1),
                   (yZ + wX).unsqueeze(1),
                   (c1 - (xX + yY)).unsqueeze(1),
                   t2.unsqueeze(1)], dim=1)

    T = torch.cat([T.view(T.shape[0],3,4), add.unsqueeze(1)], dim=1)

    return T

def T2quat_tran(R):
    # 计算四元数的分量
    w = torch.sqrt(1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) / 2
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)

    # 归一化四元数
    quaternion = torch.stack([w, x, y, z], dim=1)
    quaternion = quaternion / quaternion.norm(dim=1, keepdim=True)
    tran = R[:, :3, 3]
    return quaternion, tran

def transformPC(PC, matrix):
    """use matrix to transform current pc

    Args:
        PC (tensor): (B, 3, H, W)
        matrix (tensor): (B, 4, 4)
    """
    B, C, H, W = PC.shape
    PC = PC.view(B, -1, C)
    padding_column = torch.ones(B, H*W, 1, dtype=PC.dtype, device=PC.device)
    PC = torch.cat((PC, padding_column), dim=2).permute(0, 2, 1)
    output = torch.bmm(matrix, PC)
    return output[:, :3, :].view(B, C, H, W)