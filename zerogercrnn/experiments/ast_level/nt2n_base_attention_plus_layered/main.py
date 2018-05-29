from zerogercrnn.experiments.ast_level.common import ASTMain, NonTerminalMetrics, NonTerminalsCrossEntropyLoss
from zerogercrnn.experiments.ast_level.metrics import NonTerminalsMetricsWrapper, SingleNonTerminalAccuracyMetrics
from zerogercrnn.experiments.ast_level.nt2n_base_attention_plus_layered.model import NT2NBaseAttentionPlusLayeredModel
from zerogercrnn.lib.metrics import SequentialMetrics, MaxPredictionAccuracyMetrics, ResultsSaver, MaxPredictionWrapper, TopKWrapper, FeaturesMeanVarianceMetrics
from zerogercrnn.lib.utils import register_input_hook


class NT2NBaseAttentionPlusLayeredMain(ASTMain):
    def create_model(self, args):
        return NT2NBaseAttentionPlusLayeredModel(
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminals_num=args.terminals_num,
            terminal_embedding_dim=args.terminal_embedding_dim,
            hidden_dim=args.hidden_size,
            layered_hidden_size=args.layered_hidden_size,
            dropout=args.dropout
        )

    def create_criterion(self, args):
        return NonTerminalsCrossEntropyLoss()

    def create_train_metrics(self, args):
        return NonTerminalMetrics(base=MaxPredictionAccuracyMetrics())

    def create_eval_metrics(self, args):
        return SequentialMetrics([
            NonTerminalMetrics(base=MaxPredictionAccuracyMetrics()),
            SingleNonTerminalAccuracyMetrics(
                non_terminals_file=args.non_terminals_file,
                results_dir=args.eval_results_directory
            ),
            NonTerminalsMetricsWrapper(TopKWrapper(base=ResultsSaver(dir_to_save=args.eval_results_directory)))
        ])

    def register_eval_hooks(self):
        return []


def add_eval_hooks(model: NT2NBaseAttentionPlusLayeredModel):
    metrics = FeaturesMeanVarianceMetrics()
    register_input_hook(model.h2o, metrics)

    return [metrics]
