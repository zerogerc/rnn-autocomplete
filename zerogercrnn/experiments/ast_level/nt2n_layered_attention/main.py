from zerogercrnn.experiments.ast_level.nt2n_layered_attention.model import NT2NLayeredAttentionModel
from zerogercrnn.experiments.ast_level.nt2n_te.main import NT2NPretrainedTerminalsMain


class NT2NLayeredAttentionMain(NT2NPretrainedTerminalsMain):

    def create_model(self, args):
        return NT2NLayeredAttentionModel(
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminal_embeddings=self.terminal_embeddings,
            layered_hidden_size=args.layered_hidden_size,
            hidden_dim=args.hidden_size,
            prediction_dim=args.non_terminals_num,
            dropout=args.dropout
        )
