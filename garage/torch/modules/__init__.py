"""PyTorch Modules."""
# yapf: disable
# isort:skip_file
from garage.torch.modules.cnn_module import CNNModule
from garage.torch.modules.gaussian_mlp_module import (
    GaussianMLPIndependentStdModule)  # noqa: E501
from garage.torch.modules.gaussian_mlp_module import (
    GaussianMLPTwoHeadedModule)  # noqa: E501
from garage.torch.modules.gaussian_mlp_module import GaussianMLPModule
from garage.torch.modules.mlp_module import MLPModule
from garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule
# DiscreteCNNModule must go after MLPModule
from garage.torch.modules.discrete_cnn_module import DiscreteCNNModule
from garage.torch.modules.multi_headed_lstm_module import MultiHeadedLSTMModule
from garage.torch.modules.lstm_module import LSTMModule
# yapf: enable

__all__ = [
    'CNNModule',
    'DiscreteCNNModule',
    'MLPModule',
    'MultiHeadedMLPModule',
    'GaussianMLPModule',
    'GaussianMLPIndependentStdModule',
    'GaussianMLPTwoHeadedModule',
    'MultiHeadedLSTMModule',
    'LSTMModule'
]
