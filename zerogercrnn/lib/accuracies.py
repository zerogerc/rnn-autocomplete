import torch


def indexed_topk_hits(prediction, target, index):
    """
    :param prediction: tensor of size [N, K], where K is the number of top predictions
    :param target: tensor of size [N]
    :param index: tensor of size [I] with indexes to count accuracy on
    :return: array with T topk hits, total_entries
    """
    selected_prediction = torch.index_select(prediction, 0, index)
    selected_target = torch.index_select(target, 0, index)

    if selected_prediction.size()[0] == 0:
        return torch.zeros((prediction.size()[-1]), dtype=torch.int64), 0
    return topk_hits(selected_prediction, selected_target)


def topk_hits(prediction, target):
    """
    :param prediction: tensor of size [N, K], where K is the number of top predictions
    :param target: tensor of size [N]
    :return: array with T topk hits, total_entries
    """
    n = prediction.size()[0]
    k = prediction.size()[1]

    hits = torch.zeros(k, dtype=torch.int64)
    correct = prediction.eq(target.unsqueeze(1).expand_as(prediction))
    for tk in range(k):
        cur_hits = correct[:, :tk + 1]
        hits[tk] += cur_hits.sum()

    return hits, n
