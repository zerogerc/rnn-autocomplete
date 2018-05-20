from zerogercrnn.experiments.ast_level.common import ASTMain
from zerogercrnn.experiments.ast_level.common import NonTerminalMetrics, NonTerminalsCrossEntropyLoss
from zerogercrnn.experiments.ast_level.metrics import NonTerminalsMetricsWrapper, SingleNonTerminalAccuracyMetrics
from zerogercrnn.experiments.ast_level.nt2n_layered_attention_norm.model import NT2NLayeredAttentionNormalizedModel
from zerogercrnn.lib.metrics import MaxPredictionAccuracyMetrics, SequentialMetrics, MaxPredictionWrapper, \
    ResultsSaver, TensorVisualizer2DMetrics, TensorVisualizer3DMetrics


class NT2NLayeredAttentionNormalizedMain(ASTMain):

    def create_model(self, args):
        return NT2NLayeredAttentionNormalizedModel(
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminals_num=args.terminals_num,
            terminal_embedding_dim=args.terminal_embedding_dim,
            layered_hidden_size=args.layered_hidden_size,
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


def add_eval_hooks(model: NT2NLayeredAttentionNormalizedModel):
    # before_output_metrics = TensorVisualizer2DMetrics(file='eval/temp/output_sum_before_matrix')
    # after_output_metrics = TensorVisualizer2DMetrics(file='eval/temp/output_sum_after_matrix')

    before_output_metrics = TensorVisualizer3DMetrics(file='eval/temp/output_sum_before_matrix')
    after_output_metrics = TensorVisualizer3DMetrics(file='eval/temp/output_sum_after_matrix')

    attention = TensorVisualizer2DMetrics(file='eval/temp/attention_matrix')

    def attention_hook(module, m_input, m_output):
        attention.report(m_output.squeeze(2))

    def layered_normalization_hook(module, m_input, m_output):
        assert m_input[0].size() == m_output.size()
        before_output_metrics.report(m_input[0])
        after_output_metrics.report(m_output)

    # def output_normalization_hook(module, m_input, m_output):
    #     assert m_input[0].size() == m_output.size()
    #     before_output_metrics.report(m_input[0])
    #     after_output_metrics.report(m_output)

    model.h_norm.register_forward_hook(layered_normalization_hook)
    model.attn.register_forward_hook(attention_hook)
    # model.layered_recurrent.norm.register_forward_hook(layered_normalization_hook)

    return before_output_metrics, after_output_metrics, attention
