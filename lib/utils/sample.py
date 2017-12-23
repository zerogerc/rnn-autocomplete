import torch


def convert_output_to_input_default(input_tensor, output_tensor):
    """
    Convert output_tensor of RNN network to next input for sequence generation
    :param input_tensor: tensort the network was run on to generate output
    :param output_tensor: network(input_tensor)
    :return: next valid input for prediction
    """
    return torch.cat((input_tensor[1:], output_tensor[-1:]))


def extract_next_prediction_default(input_tensor, output_tensor):
    """
        Extract last prediction from output_tensor of RNN network.
        :param input_tensor: tensor the network was run on to generate output
        :param output_tensor: network(input_tensor)
        :return: next valid input for prediction
        """
    return output_tensor[-1:]


class RNNSampler:
    def __init__(
            self,
            network,
            input_generator=convert_output_to_input_default,
            sample_extractor=extract_next_prediction_default
    ):
        """
        Create RNNSampler.
        :param network: network to sample.
        :param input_generator: converter from input, output to next input
        :param sample_extractor: converter from input, output to sample
        """
        self.network = network

    def sample(self, input_tensor, iters):
        """
        Sample sequence using input_tensor as start.

        :param input_tensor: tensor of size [seq_len, 1, input_len]
        :param iters: number of times to sample
        :return: generated tensor of size [iters, input_len]
        """
        seq_len, _, input_len = input_tensor.size()

        sequence = []
        current_input = input_tensor.contiguous()
        for it in range(iters - seq_len):
            output_tensor = self.network.predict(current_input)

            sequence.append(extract_next_prediction_default(current_input, output_tensor))
            current_input = convert_output_to_input_default(current_input, output_tensor)

        return torch.cat(sequence, dim=0)
