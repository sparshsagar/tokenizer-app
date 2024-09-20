class WordPieceTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}

    def tokenize(self, sentence):
        tokens = sentence.split()
        encoded_tokens = []
        for token in tokens:
            sub_tokens = list(token)
            for i, sub_token in enumerate(sub_tokens):
                if i == 0:
                    if sub_token not in self.vocab:
                        self.vocab[sub_token] = len(self.vocab)
                    encoded_tokens.append(sub_token)
                else:
                    prefixed_token = "##" + sub_token
                    if prefixed_token not in self.vocab:
                        self.vocab[prefixed_token] = len(self.vocab)
                    encoded_tokens.append(prefixed_token)
        return encoded_tokens

    def get_pair_scores(self, tokens):
        pair_freq = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in pair_freq:
                pair_freq[pair] += 1
            else:
                pair_freq[pair] = 1

        pair_scores = {}
        for pair, freq in pair_freq.items():
            pair_scores[pair] = freq / (self.vocab[pair[0]] + self.vocab[pair[1]])
        return pair_scores

    def merge_tokens(self, tokens_list, pair, new_token):
        new_tokens_list = []
        i = 0
        while i < len(tokens_list):
            if i < len(tokens_list) - 1 and tokens_list[i] == pair[0] and tokens_list[i + 1] == pair[1]:
                new_tokens_list.append(new_token)
                i += 2  # skip the original tokens
            else:
                new_tokens_list.append(tokens_list[i])
                i += 1
        return new_tokens_list

    def train(self, sentence):
        tokens_list = self.tokenize(sentence)
        num_merges = self.vocab_size - len(set(tokens_list))
        for i in range(num_merges):
            pair_scores = self.get_pair_scores(tokens_list)
            if not pair_scores:
                break
            best_pair = max(pair_scores, key=pair_scores.get)
            new_token = best_pair[0] + best_pair[1].replace("##", "")
            if "##" in best_pair[1] and not best_pair[0].startswith("##"):
                new_token = best_pair[0] + best_pair[1].replace("##", "")
                new_token = "##" + new_token
            elif best_pair[0].startswith("##"):
                new_token = best_pair[0] + best_pair[1].replace("##", "")
            else:
                new_token = best_pair[0] + best_pair[1]
            print(f"Merging {best_pair} into new token '{new_token}'")
            tokens_list = self.merge_tokens(tokens_list, best_pair, new_token)
            self.vocab[new_token] = len(self.vocab)
            self.merges[best_pair] = new_token

    def encode(self, sentence):
        tokens = self.tokenize(sentence)
        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            merge_candidates = [pair for pair in pairs if pair in self.merges]
            if not merge_candidates:
                break
            for pair in merge_candidates:
                new_token = self.merges[pair]
                tokens = self.merge_tokens(tokens, pair, new_token)
        return tokens

    # New decode method
    def decode(self, tokens):
        decoded_sentence = ""
        for token in tokens:
            if token.startswith("##"):
                decoded_sentence += token[2:]  # Remove "##" and append to the last word
            else:
                if decoded_sentence:
                    decoded_sentence += " "  # Add space before new word if not first
                decoded_sentence += token
        return decoded_sentence
