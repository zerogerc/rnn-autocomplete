from zerogercrnn.experiments.ast_level.common import ASTMain
from zerogercrnn.experiments.ast_level.common import NonTerminalMetrics, NonTerminalsCrossEntropyLoss
from zerogercrnn.experiments.ast_level.metrics import NonTerminalsMetricsWrapper, SingleNonTerminalAccuracyMetrics
from zerogercrnn.experiments.ast_level.nt2n_layered_single.model import NT2NSingleLSTMLayeredAttentionModel
from zerogercrnn.lib.metrics import MaxPredictionAccuracyMetrics, SequentialMetrics, MaxPredictionWrapper, \
    ResultsSaver, TensorVisualizer3DMetrics, FeaturesMeanVarianceMetrics


class NT2NSingleLSTMLayeredAttentionMain(ASTMain):

    def create_model(self, args):
        return NT2NSingleLSTMLayeredAttentionModel(
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminals_num=args.terminals_num,
            terminal_embedding_dim=args.terminal_embedding_dim,
            hidden_dim=args.hidden_size,
            node_depths_embedding_dim=args.node_depths_embedding_dim,
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


def add_eval_hooks(model: NT2NSingleLSTMLayeredAttentionModel):
    before_output_metrics = TensorVisualizer3DMetrics(file='eval/temp/output_sum_before_matrix')
    after_output_metrics = TensorVisualizer3DMetrics(file='eval/temp/output_sum_after_matrix')

    register_input_hook(model.h_norm, before_output_metrics)
    register_output_hook(model.h_norm, after_output_metrics)

    concatenated_input_metrics = FeaturesMeanVarianceMetrics(dim=0)
    register_input_hook(model.recurrent_core, concatenated_input_metrics)

    concatenated_hidden_metrics = FeaturesMeanVarianceMetrics(dim=0, directory='eval/temp/concat_hidden')
    register_input_hook(
        model.h_norm,
        concatenated_hidden_metrics,
        picker=lambda m_input: m_input[0].view(-1, m_input[0].size()[-1])
    )

    return before_output_metrics, after_output_metrics, concatenated_input_metrics, concatenated_hidden_metrics
