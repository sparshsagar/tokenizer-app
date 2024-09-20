import streamlit as st
import regex as re

class GPT4Tokenizer:
    def __init__(self):
        self.pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.vocab_size = 400
        self.merges = {}
        self.vocab = {}
    
    # Find consecutive pairs   
    def get_stats(self, token_ids, stats):
        for pair in zip(token_ids, token_ids[1:]):
            stats[pair] = stats.get(pair, 0) + 1
        return stats
    
    # Merge token ids
    def merge(self, token_ids, pair, new_index):
        _token_ids = []
        i = 0
        while i < len(token_ids):
            if (i < len(token_ids)-1) and (token_ids[i]==pair[0]) and (token_ids[i+1]==pair[1]):
                _token_ids.append(new_index)
                i += 2

            else:
                _token_ids.append(token_ids[i])
                i += 1
        return _token_ids
    
    def train(self, text, verbose=False):
        assert self.vocab_size >= 256
        num_merges = self.vocab_size - 256
        
        text_chunks = re.findall(self.pattern, text)
        token_ids = [list(chunk.encode('utf-8')) for chunk in text_chunks]
        
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        
        for i in range(num_merges):
            stats = {}
            for chunk_token in token_ids:
                self.get_stats(chunk_token, stats)
            
            # If no pairs are found, stop the training process
            if not stats:
                if verbose:
                    print(f"No more pairs to merge at step {i}. Stopping early.")
                break
            
            # Get the most frequent pair
            top_pair = max(stats, key=stats.get)
            index = 256 + i
            if verbose:
                print(f"merged : {top_pair} -> {index}")
                
            token_ids = [self.merge(chunk_token, top_pair, index) for chunk_token in token_ids]   
                
            self.vocab[index] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            self.merges[top_pair] = index

    # encode chunk
    def encode_chunks(self, chunk_bytes):
        chunk_token_ids = list(chunk_bytes)
        while len(chunk_token_ids) >=2: 
            stats = {}
            self.get_stats(chunk_token_ids, stats)
            pair = min(stats, key= lambda x: self.merges.get(x, float("inf")))
            if pair not in self.merges:
                break
            index = self.merges[pair]
            chunk_token_ids = self.merge(chunk_token_ids, pair, index)
        return chunk_token_ids
    
    # encode full text
    def encode(self, text):
        text_chunks = re.findall(self.pattern, text)
        token_ids = []
        
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_tokens_ids = self.encode_chunks(chunk_bytes)
            token_ids.extend(chunk_tokens_ids)
        return token_ids
                
    # decoding
    def decode(self, token_ids):
        print("vocab:",self.vocab)
        chunk_bytes = []
        for token in token_ids:
            if token in self.vocab:
                chunk_bytes.append(self.vocab[token])
            else:
                raise ValueError(f"Invalid token id: {token}")
         
        
        b_tokens_ids = b"".join(chunk_bytes)
        text = b_tokens_ids.decode('utf-8', errors= "replace")
        return text

