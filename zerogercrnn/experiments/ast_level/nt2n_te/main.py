from zerogercrnn.experiments.ast_level.common import ASTMain, NonTerminalsCrossEntropyLoss, NonTerminalMetrics, \
    create_terminal_embeddings
from zerogercrnn.experiments.ast_level.nt2n_te.model import NT2NPretrainedTerminalsModel
from zerogercrnn.lib.metrics import MaxPredictionAccuracyMetrics


class NT2NPretrainedTerminalsMain(ASTMain):
    """NT2N Model where embeddings for terminals is pretrained."""

    def create_terminal_embeddings(self, args):
        return create_terminal_embeddings(args)

    def create_model(self, args):
        return NT2NPretrainedTerminalsModel(
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminal_embeddings=self.terminal_embeddings,
            hidden_dim=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

    def create_criterion(self, args):
        return NonTerminalsCrossEntropyLoss()

    def create_train_metrics(self, args):
        return NonTerminalMetrics(base=MaxPredictionAccuracyMetrics())
