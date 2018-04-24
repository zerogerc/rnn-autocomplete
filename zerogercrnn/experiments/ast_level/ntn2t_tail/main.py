from zerogercrnn.experiments.ast_level.nt2n.main import NT2NMain
from zerogercrnn.experiments.ast_level.ntn2t_tail.model import NTN2TTailAttentionModel
from zerogercrnn.experiments.ast_level.ntn2t.main import NTN2TMain


class NT2NTailAttentionMain(NTN2TMain):
    def create_model(self, args):
        return NTN2TTailAttentionModel(
            seq_len=args.seq_len,
            non_terminals_num=args.non_terminals_num,
            non_terminal_embedding_dim=args.non_terminal_embedding_dim,
            terminal_embeddings=self.terminal_embeddings,
            hidden_dim=args.hidden_size,
            dropout=args.dropout
        )
