class LastKBuffer:
    def __init__(self, window_len, buffer_size):
        assert window_len <= buffer_size
