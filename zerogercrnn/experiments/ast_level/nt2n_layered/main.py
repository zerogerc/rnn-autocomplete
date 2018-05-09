import torch.nn as nn

from zerogercrnn.experiments.ast_level.nt2n_te.main import NT2NPretrainedTerminalsMain
from zerogercrnn.experiments.ast_level.nt2n_seq.model import NT2NLayerModel
from zerogercrnn.lib.metrics import MaxPredictionAccuracyMetrics


class NT2NSequentialMain(NT2NPretrainedTerminalsMain):

    def create_model(self, args):
        return NT2NLayerModel(
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminal_embeddings=self.terminal_embeddings,
            hidden_dim=args.hidden_size,
            prediction_dim=args.non_terminals_num,
            layered_hidden_size=args.layered_hidden_size,
            dropout=args.dropout
        )

    def create_criterion(self, args):
        return nn.CrossEntropyLoss()

    def create_metrics(self, args):
        return MaxPredictionAccuracyMetrics()
