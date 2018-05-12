from zerogercrnn.experiments.ast_level.common import ASTMain, TerminalMetrics, TerminalsCrossEntropyLoss
from zerogercrnn.experiments.ast_level.ntn2t_base.model import NTN2TBaseModel
from zerogercrnn.lib.metrics import MaxPredictionAccuracyMetrics


class NTN2TBaseMain(ASTMain):

    def create_model(self, args):
        return NTN2TBaseModel(
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminals_num=args.terminals_num,
            terminal_embedding_dim=args.terminal_embedding_dim,
            hidden_dim=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

    def create_criterion(self, args):
        return TerminalsCrossEntropyLoss()

    def create_metrics(self, args):
        return TerminalMetrics(base=MaxPredictionAccuracyMetrics())
