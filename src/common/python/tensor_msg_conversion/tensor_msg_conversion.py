from interactive_yolo_interfaces.msg import Float32Tensor, BoolTensor, Bbox
import numpy as np
import torch
from typing import List
import lz4.frame


def torchTensorToFloat32Tensor(tensor:torch.Tensor)->Float32Tensor:
    """
    Convert a torch tensor to a Float32Tensor message.
    """
    return ndarrayToFloat32Tensor(tensor.to(dtype=torch.float32, device="cpu").numpy())

def float32TensorToTorchTensor(float32_tensor:Float32Tensor)->torch.Tensor:
    """
    Convert a Float32Tensor message to a torch tensor.
    """
    if len(float32_tensor.tensor_data) == 0:
        return None
    
    return torch.from_numpy(float32TensorToNdarray(float32_tensor).copy()).to(dtype=torch.float32, device="cpu")

def ndarrayToFloat32Tensor(array:np.ndarray)->Float32Tensor:
    """
    Convert a torch tensor to a Float32Tensor message.
    """

    data = array.flatten().astype(np.float32).tobytes()

    float32_tensor = Float32Tensor()

    float32_tensor.shape = list(array.shape)
    float32_tensor.tensor_data = data.hex()

    return float32_tensor

def float32TensorToNdarray(float32_tensor:Float32Tensor)->np.ndarray:
    """
    Convert a Float32Tensor message to a torch tensor.
    """
    if len(float32_tensor.tensor_data) == 0:
        return None
    
    data = bytes.fromhex(float32_tensor.tensor_data)

    return np.frombuffer(data, dtype=np.float32).reshape(float32_tensor.shape)

def ndArrayToBoolTensor(array:np.ndarray)->BoolTensor:
    """
    Convert a numpy array to a BoolTensor message.
    """

    data = np.packbits(array.flatten().astype(np.bool_), bitorder="big").astype(np.uint8).tobytes()
    compressed_data = lz4.frame.compress(data, compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC)

    bool_tensor = BoolTensor()

    bool_tensor.shape = list(array.shape)
    bool_tensor.tensor_data = compressed_data.hex()

    return bool_tensor

def boolTensorToNdArray(bool_tensor:BoolTensor)->np.ndarray:
    """
    Convert a BoolTensor message to a numpy array.
    """
    if len(bool_tensor.tensor_data) == 0:
        return None
    
    unpack_length = 1
    for dim in bool_tensor.shape:
        unpack_length *= dim

    compressed_data = bytes.fromhex(bool_tensor.tensor_data)
    data = lz4.frame.decompress(compressed_data)

    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big", count=unpack_length).reshape(bool_tensor.shape).astype(np.bool_)

def torchTensorToBoolTensor(bool_tensor:torch.Tensor)->BoolTensor:
    """
    Convert a torch tensor to a BoolTensor message.
    """
    return ndArrayToBoolTensor(bool_tensor.to(dtype=torch.bool, device="cpu").numpy())

def boolTensorToTorchTensor(bool_tensor:BoolTensor)->torch.Tensor:
    """
    Convert a BoolTensor message to a torch tensor.
    """
    if len(bool_tensor.tensor_data) == 0:
        return None
    
    return torch.from_numpy(boolTensorToNdArray(bool_tensor).copy()).to(dtype=torch.bool, device="cpu")
