from zerogercrnn.experiments.ast_level.common import ASTMain, NonTerminalMetrics, NonTerminalsCrossEntropyLoss
from zerogercrnn.experiments.ast_level.metrics import NonTerminalsMetricsWrapper, SingleNonTerminalAccuracyMetrics
from zerogercrnn.experiments.ast_level.nt2n_base.model import NT2NBaseModel
from zerogercrnn.lib.metrics import SequentialMetrics, MaxPredictionAccuracyMetrics, ResultsSaver, MaxPredictionWrapper, \
    MetricsCombination


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
        return MetricsCombination(
            train_metrics=NonTerminalMetrics(base=MaxPredictionAccuracyMetrics()),
            eval_metrics=SequentialMetrics([
                NonTerminalMetrics(base=MaxPredictionAccuracyMetrics()),
                SingleNonTerminalAccuracyMetrics(
                    non_terminals_file=args.non_terminals_file,
                    results_dir=args.eval_results_directory
                ),
                NonTerminalsMetricsWrapper(MaxPredictionWrapper(ResultsSaver(dir_to_save=args.eval_results_directory)))
            ])
        )
