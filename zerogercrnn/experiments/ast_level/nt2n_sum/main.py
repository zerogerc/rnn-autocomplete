from zerogercrnn.experiments.ast_level.nt2n_te.main import NT2NPretrainedTerminalsMain
from zerogercrnn.experiments.ast_level.nt2n_sum.model import NT2NSumAttentionModel


class NT2NSumAttentionMain(NT2NPretrainedTerminalsMain):
    def create_model(self, args):
        return NT2NSumAttentionModel(
            context_len=50,  # last 50 for context
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminal_embeddings=self.terminal_embeddings,
            hidden_dim=args.hidden_size,
            prediction_dim=args.non_terminals_num,
            dropout=args.dropout
        )
