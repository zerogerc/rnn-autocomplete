from zerogercrnn.experiments.ast_level.nt2n_te.main import NT2NPretrainedTerminalsMain
from zerogercrnn.experiments.ast_level.ntn2t_tail.model import NTN2TTailAttentionModel


class NTN2TTailAttentionMain(NT2NPretrainedTerminalsMain):
    def create_model(self, args):
        return NTN2TTailAttentionModel(
            seq_len=args.seq_len,
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminal_embeddings=self.terminal_embeddings,
            hidden_dim=args.hidden_size,
            dropout=args.dropout
        )
