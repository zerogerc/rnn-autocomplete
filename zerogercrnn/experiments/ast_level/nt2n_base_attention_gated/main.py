import torch
import torch.nn as nn
import torch.nn.functional as F
from zerogercrnn.experiments.ast_level.data import ASTTarget

from zerogercrnn.experiments.ast_level.common import ASTMain, NonTerminalMetrics, NonTerminalsCrossEntropyLoss
from zerogercrnn.experiments.ast_level.metrics import NonTerminalsMetricsWrapper, SingleNonTerminalAccuracyMetrics
from zerogercrnn.experiments.ast_level.nt2n_base_attention_gated.model import NT2NBaseAttentionGatedBufferModel
from zerogercrnn.lib.metrics import SequentialMetrics, MaxPredictionAccuracyMetrics, ResultsSaver, MaxPredictionWrapper


class GatedLoss(NonTerminalsCrossEntropyLoss):
    def __init__(self, model: NT2NBaseAttentionGatedBufferModel):
        super().__init__()
        self.model = model
        self.it = 0

    @staticmethod
    def l1(p):
        return F.l1_loss(p, target=torch.zeros_like(p), size_average=False)

    def forward(self, prediction: torch.Tensor, target: ASTTarget):
        base_loss = super().forward(prediction, target)

        # l1_crit = torch.nn.L1Loss(size_average=False)

        # reg_loss = 0
        # reg_loss += GatedLoss.l1(self.model.gated_attention.w_cntx.affine.weight)
        # reg_loss += GatedLoss.l1(self.model.gated_attention.w_h.affine.weight)

        l2_loss = self.model.gated_attention.w_cntx.affine.weight.norm(2)
        l2_loss = l2_loss + self.model.gated_attention.w_h.affine.weight.norm(2)

        if self.it % 1000 == 0:
            print(self.model.gated_attention.w_cntx.affine.weight.norm(2))
            print(self.model.gated_attention.w_h.affine.weight.norm(2))

        self.it += 1

        return base_loss + 0.001 * l2_loss


class NT2NBaseAttentionGatedBufferMain(ASTMain):
    def create_model(self, args):
        return NT2NBaseAttentionGatedBufferModel(
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminals_num=args.terminals_num,
            terminal_embedding_dim=args.terminal_embedding_dim,
            hidden_dim=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

    def create_criterion(self, args):
        return GatedLoss(self.model)

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
