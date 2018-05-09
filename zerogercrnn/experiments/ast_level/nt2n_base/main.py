from zerogercrnn.experiments.ast_level.common import ASTMain, NonTerminalMetrics, NonTerminalsCrossEntropyLoss
from zerogercrnn.experiments.ast_level.nt2n_base.model import NT2NBaseModel
from zerogercrnn.lib.metrics import MaxPredictionAccuracyMetrics


class NT2NBaseMain(ASTMain):
    def create_model(self, args):
        return NT2NBaseModel(
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminals_num=args.terminals_num,
            terminal_embedding_dim=args.terminal_embedding_dim,
            hidden_dim=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

    def create_criterion(self, args):
        return NonTerminalsCrossEntropyLoss()

    def create_metrics(self, args):
        return NonTerminalMetrics(base=MaxPredictionAccuracyMetrics())
