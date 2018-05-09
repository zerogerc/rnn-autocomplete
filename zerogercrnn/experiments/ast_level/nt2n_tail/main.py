from zerogercrnn.experiments.ast_level.nt2n_te.main import NT2NPretrainedTerminalsMain
from zerogercrnn.experiments.ast_level.nt2n_tail.model import NT2NTailAttentionModel


class NT2NTailAttentionMain(NT2NPretrainedTerminalsMain):
    def create_model(self, args):
        return NT2NTailAttentionModel(
            seq_len=args.seq_len,
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminal_embeddings=self.terminal_embeddings,
            hidden_dim=args.hidden_size,
            prediction_dim=args.non_terminals_num,
            dropout=args.dropout
        )
