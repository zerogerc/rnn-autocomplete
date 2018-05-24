from zerogercrnn.experiments.ast_level.common import ASTMain, NonTerminalMetrics, NonTerminalsCrossEntropyLoss
from zerogercrnn.experiments.ast_level.metrics import NonTerminalsMetricsWrapper, SingleNonTerminalAccuracyMetrics
from zerogercrnn.experiments.ast_level.nt2n_base_attention_norm.model import NT2NBaseAttentionLayerNormalizedModel
from zerogercrnn.lib.metrics import SequentialMetrics, MaxPredictionAccuracyMetrics, ResultsSaver, MaxPredictionWrapper, \
    FeaturesMeanVarianceMetrics


class NT2NBaseAttentionNormalizedMain(ASTMain):
    def create_model(self, args):
        return NT2NBaseAttentionLayerNormalizedModel(
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

    def create_train_metrics(self, args):
        return NonTerminalMetrics(base=MaxPredictionAccuracyMetrics())

    def create_eval_metrics(self, args):
        return SequentialMetrics([
            NonTerminalMetrics(base=MaxPredictionAccuracyMetrics()),
            SingleNonTerminalAccuracyMetrics(
                non_terminals_file=args.non_terminals_file,
                results_dir=args.eval_results_directory
            ),
            NonTerminalsMetricsWrapper(MaxPredictionWrapper(ResultsSaver(dir_to_save=args.eval_results_directory)))
        ])

    def register_eval_hooks(self):
        return add_eval_hooks(self.model)


def register_forward_hook(module, metrics, picker):
    module.register_forward_hook(lambda _, m_input, m_output: metrics.report(picker(m_input, m_output)))


def register_output_hook(module, metrics, picker=None):
    if picker is None:
        picker = lambda m_output: m_output
    register_forward_hook(module, metrics, lambda m_input, m_output: picker(m_output))


def register_input_hook(module, metrics, picker=None):
    if picker is None:
        picker = lambda m_input: m_input[0]
    register_forward_hook(module, metrics, lambda m_input, m_output: picker(m_input))


def add_eval_hooks(model: NT2NBaseAttentionLayerNormalizedModel):
    before_norm_metrics = FeaturesMeanVarianceMetrics(directory='eval/temp/nt2n_base_attention_norm_before')
    after_norm_metrics = FeaturesMeanVarianceMetrics(directory='eval/temp/nt2n_base_attention_norm_after')

    register_input_hook(model.recurrent_norm, before_norm_metrics)
    register_output_hook(model.recurrent_norm, after_norm_metrics)

    return before_norm_metrics, after_norm_metrics
