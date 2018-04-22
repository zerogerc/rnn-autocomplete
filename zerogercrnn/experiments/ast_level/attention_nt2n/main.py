from zerogercrnn.experiments.ast_level.nt2n.main import NT2NMain
from zerogercrnn.experiments.ast_level.attention_nt2n.model import NT2NAttentionModel


class NT2NAttentionMain(NT2NMain):
    def create_model(self, args):
        return NT2NAttentionModel(
            seq_len=args.seq_len,
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminal_embeddings=self.terminal_embeddings,
            hidden_dim=args.hidden_size,
            prediction_dim=args.non_terminals_num,
            dropout=args.dropout
        )
