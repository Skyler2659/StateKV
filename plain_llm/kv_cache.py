class PlainKVCache:
    def __init__(self):
        pass

    def __call__(self, past_key_values):
        return past_key_values

    def evict_for_space(self, past_key_values, num_coming):
        return past_key_values

    def evict_range(self, past_key_values, start, end):
        return past_key_values
