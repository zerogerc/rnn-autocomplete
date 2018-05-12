from zerogercrnn.experiments.ast_level.common import ASTMain
from zerogercrnn.experiments.ast_level.common import NonTerminalMetrics, NonTerminalsCrossEntropyLoss
from zerogercrnn.experiments.ast_level.nt2n_layered_attention.model import NT2NLayeredAttentionModel
from zerogercrnn.lib.metrics import MaxPredictionAccuracyMetrics, SequentialMetrics, MaxPredictionWrapper, ResultsSaver
from zerogercrnn.experiments.ast_level.metrics import NonTerminalsMetricsWrapper, SingleNonTerminalAccuracyMetrics


class NT2NLayeredAttentionMain(ASTMain):

    def create_model(self, args):
        return NT2NLayeredAttentionModel(
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminals_num=args.terminals_num,
            terminal_embedding_dim=args.terminal_embedding_dim,
            layered_hidden_size=args.layered_hidden_size,
            hidden_dim=args.hidden_size,
            dropout=args.dropout
        )

    def create_criterion(self, args):
        return NonTerminalsCrossEntropyLoss()

    def create_metrics(self, args):
        return SequentialMetrics([
            NonTerminalMetrics(base=MaxPredictionAccuracyMetrics()),
            SingleNonTerminalAccuracyMetrics(
                non_terminals_number=args.non_terminals_num,
                non_terminals_file=args.non_terminals_file
            ),
            NonTerminalsMetricsWrapper(MaxPredictionWrapper(ResultsSaver(dir_to_save=args.eval_results_directory)))
        ])
