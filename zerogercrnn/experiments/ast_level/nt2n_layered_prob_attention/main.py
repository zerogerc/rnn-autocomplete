from zerogercrnn.experiments.ast_level.common import ASTMain, NonTerminalMetrics, NonTerminalsCrossEntropyLoss
from zerogercrnn.experiments.ast_level.metrics import SingleNonTerminalAccuracyMetrics, NonTerminalsMetricsWrapper
from zerogercrnn.experiments.ast_level.nt2n_layered_prob_attention.model import NT2NLayeredProbabilisticAttentionModel
from zerogercrnn.lib.metrics import MaxPredictionAccuracyMetrics, MetricsCombination, SequentialMetrics, \
    MaxPredictionWrapper, ResultsSaver


class NT2NLayeredProbabilisticAttentionMain(ASTMain):

    def create_model(self, args):
        return NT2NLayeredProbabilisticAttentionModel(
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            layered_probabilities_file=args.nodes_depths_stat_file,
            terminals_num=args.terminals_num,
            terminal_embedding_dim=args.terminal_embedding_dim,
            layered_hidden_size=args.layered_hidden_size,
            hidden_dim=args.hidden_size,
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
