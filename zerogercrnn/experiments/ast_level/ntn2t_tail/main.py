from zerogercrnn.experiments.ast_level.common import ASTMain, TerminalMetrics, TerminalsCrossEntropyLoss
from zerogercrnn.experiments.ast_level.common import create_terminal_embeddings
from zerogercrnn.experiments.ast_level.ntn2t_tail.model import NTN2TTailAttentionModel
from zerogercrnn.lib.metrics import MaxPredictionAccuracyMetrics


class NTN2TTailAttentionMain(ASTMain):

    def create_terminal_embeddings(self, args):
        return create_terminal_embeddings(args)

    def create_model(self, args):
        return NTN2TTailAttentionModel(
            seq_len=args.seq_len,
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminal_embeddings=self.terminal_embeddings,
            hidden_dim=args.hidden_size,
            dropout=args.dropout
        )

    def create_criterion(self, args):
        return TerminalsCrossEntropyLoss()

    def create_train_metrics(self, args):
        return TerminalMetrics(base=MaxPredictionAccuracyMetrics())