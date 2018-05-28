from torch import nn as nn

from zerogercrnn.experiments.token_level.common import TokenMain
from zerogercrnn.experiments.token_level.base.model import TokenBaseModel
from zerogercrnn.lib.core import BaseModule
from zerogercrnn.lib.metrics import Metrics, MaxPredictionAccuracyMetrics


class TokenBaseMain(TokenMain):

    def __init__(self, args):
        super().__init__(args)

    def create_model(self, args) -> BaseModule:
        return TokenBaseModel(
            num_tokens=args.tokens_num,
            embedding_dim=args.token_embedding_dim,
            hidden_size=args.hidden_size
        )

    def create_criterion(self, args) -> nn.Module:
        return nn.CrossEntropyLoss()

    def create_train_metrics(self, args) -> Metrics:
        return MaxPredictionAccuracyMetrics()

    def create_eval_metrics(self, args) -> Metrics:
        return MaxPredictionAccuracyMetrics()
