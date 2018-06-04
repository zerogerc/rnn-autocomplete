def add_general_arguments(parser):
    parser.add_argument('--title', type=str, help='Title for this run. Used in tensorboard and in saving of models.')
    parser.add_argument('--train_file', type=str, help='File with training data')
    parser.add_argument('--eval_file', type=str, help='File with eval data')
    parser.add_argument('--data_limit', type=int, help='How much lines of data to process (only for fast checking)')
    parser.add_argument('--model_save_dir', type=str, help='Where to save trained models')
    parser.add_argument('--saved_model', type=str, help='File with trained model if not fresh train')
    parser.add_argument('--log', action='store_true', help='Log performance?')


def add_batching_data_args(parser):
    parser.add_argument('--seq_len', type=int, help='Recurrent layer time unrolling')
    parser.add_argument('--batch_size', type=int, help='Size of batch')


def add_optimization_args(parser):
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of epochs to run model')
    parser.add_argument('--decay_after_epoch', type=int, help='Multiply lr by decay_multiplier each epoch')
    parser.add_argument('--decay_multiplier', type=float, help='Multiply lr by this number after decay_after_epoch')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for l2 regularization')


def add_recurrent_core_args(parser):
    parser.add_argument('--hidden_size', type=int, help='Hidden size of recurrent part of model')
    parser.add_argument('--num_layers', type=int, help='Number of recurrent layers')
    parser.add_argument('--dropout', type=float, help='Dropout to apply to recurrent layer')
    # Layered LSTM args, ignored if not layered
    parser.add_argument('--layered_hidden_size', type=int, help='Size of hidden state in layered lstm')
    parser.add_argument('--num_tree_layers', type=int, help='Number of layers to distribute hidden size')


def add_non_terminal_args(parser):
    parser.add_argument('--non_terminals_num', type=int, help='Number of different non-terminals')
    parser.add_argument('--non_terminal_embedding_dim', type=int, help='Dimension of non-terminal embeddings')
    parser.add_argument('--non_terminals_file', type=str, help='Json file with all non-terminals')
    parser.add_argument('--non_terminal_embeddings_file', type=str, help='File with pretrained non-terminal embeddings')


def add_terminal_args(parser):
    parser.add_argument('--terminals_num', type=int, help='Number of different terminals')
    parser.add_argument('--terminal_embedding_dim', type=int, help='Dimension of terminal embeddings')
    parser.add_argument('--terminals_file', type=str, help='Json file with all terminals')
    parser.add_argument('--terminal_embeddings_file', type=str, help='File with pretrained terminal embeddings')


def add_tokens_args(parser):
    parser.add_argument('--tokens_num', type=int, help='Number of different tokens in train file')
    parser.add_argument('--token_embedding_dim', type=int, help='Size of continuous token representation')