from collections import defaultdict

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


import itertools

import tqdm
import numpy as np
class BPETokenizer():
    def __init__(self, vocab_size):
        self.word_freq = defaultdict(lambda : 0)
        self.vocab = []
        self.vocab_freqs = defaultdict(lambda : 0)
        self.merges = {}
        
        self.smallest_vocab = []
        
        self.vocab_size = vocab_size
        self.special_symbol = "Ġ"
    
    def pre_tokenize_str(self, text):
        text_tokenized_with_spaces = [[[' '] + nltk.word_tokenize(w)] if idx != 0 else [nltk.word_tokenize(w)]  for idx, w in enumerate(text.split(' '))]
        text_tokenized_with_spaces = list(itertools.chain(*list(itertools.chain(*text_tokenized_with_spaces))))
        
        for i in range(len(text_tokenized_with_spaces)):
            if text_tokenized_with_spaces[i] == ' ':
                text_tokenized_with_spaces[i] = self.special_symbol
                
        tokenized_text = []
        i = 0
         
        while i < len(text_tokenized_with_spaces):
            if i < len(text_tokenized_with_spaces) - 1:
                if text_tokenized_with_spaces[i] == self.special_symbol and text_tokenized_with_spaces[i + 1] != self.special_symbol:
                    tokenized_text.append(self.special_symbol + text_tokenized_with_spaces[i + 1])
                    i += 2
                else:
                    tokenized_text.append(text_tokenized_with_spaces[i])
                    i += 1
            else:
                tokenized_text.append(text_tokenized_with_spaces[i])
                i += 1
                
        return tokenized_text        
            
    def compute_pair_freqs(self, splits):
        pair_freqs = defaultdict(lambda : 0)
        
        for word, freq in self.word_freq.items():
            split = splits[word]
            
            if len(split) == 1:
                continue
            
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                
                pair_freqs[pair] += freq
                
        return pair_freqs
    
    def merge_pair(self, a, b, splits):
        for word in self.word_freq:
            split = splits[word]
            
            if len(split) == 1:
                continue
            
            i = 0
            
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1
                    
            splits[word] = split
            
        return splits
    
    def train_tokenizer(self, corpus):
        for text in tqdm.tqdm(corpus):
            words = self.pre_tokenize_str(text)
            
            for word in words:
                self.word_freq[word] += 1
            
        alphabet = set()
        
        for word in self.word_freq.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.add(letter)
                self.vocab_freqs[letter] += 1
        
        alphabet = list(alphabet)
        alphabet.sort()    
        self.vocab = alphabet#["<|endoftext|>"] + alphabet
        self.smallest_vocab = self.vocab.copy()
        
        splits = {word : [c for c in word] for word in self.word_freq.keys()}
        
        prev_vocab_len = len(self.vocab)
        
        pbar = tqdm.tqdm(total=self.vocab_size)
        pbar.update(prev_vocab_len)
        
        while len(self.vocab) < self.vocab_size:
            pbar.update(len(self.vocab) - prev_vocab_len)
            prev_vocab_len = len(self.vocab)
        
            pair_freqs = self.compute_pair_freqs(splits)
        
        
            max_freq = None
            best_pair = ''
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
        
            self.merges[best_pair] = ''.join(best_pair)
            self.vocab_freqs[''.join(best_pair)] = max_freq
        
            splits = self.merge_pair(*best_pair, splits)
            
            self.vocab.append(best_pair[0] + best_pair[1])
            
    
    def tokenize(self, text):
        pre_tokenized_text = self.pre_tokenize_str(text)
        
        splits = [[l for l in word] for word in pre_tokenized_text]
                
        for pair, merge in self.merges.items():
            i = 0
            
            for idx, split in enumerate(splits):
                i = 0
                
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split
                
        return sum(splits, [])

class UnigramTokenizer():
    def __init__(self, vocab_size, 
                 initial_vocab_multiplier,
                 shrink_multiplier=0.1,
                 sub_em_steps=2):
        self.vocab_size = vocab_size
        self.initial_vocab_multiplier = initial_vocab_multiplier
        self.shrink_multplier = shrink_multiplier
        self.sub_em_steps = sub_em_steps
        
        self.initial_tokenizer = BPETokenizer(int(self.vocab_size*self.initial_vocab_multiplier))
        
        self.cur_vocab_subword_freqs = None
        self.cur_vocab_subword_logprob = None
        
        self.alphabet = None
    
    def pre_tokinze_str(self, text):
        return self.initial_tokenizer.pre_tokenize_str(text)
    
    def get_initial_word_freq(self):
        return self.initial_tokenizer.word_freq
    
    def train_initial_tokenizer(self, corpus):
        self.initial_tokenizer.train_tokenizer(corpus)
        self.alphabet = set(self.initial_tokenizer.smallest_vocab)
        
        self.cur_vocab_subword_freqs = self.initial_tokenizer.vocab_freqs
        tot_cnt = sum(list(self.cur_vocab_subword_freqs.values()))
        self.cur_vocab_subword_logprob = {k : np.log(v / tot_cnt) for k, v in self.cur_vocab_subword_freqs.items()}

    def get_initial_subword_logprob(self):
        vocab_freqs = self.initial_tokenizer.vocab_freqs
        tot_cnt = sum(list(vocab_freqs.values()))
        subword_logp = {k : np.log(v / tot_cnt) for k, v in vocab_freqs.items()}
        return subword_logp
          
    @staticmethod
    def viterbi_forward(word, subword_logp):
        best_subw_slices = [None]*(len(word) + 1)
        neg_loglik = np.zeros(len(word) + 1)
        
        for eow in range(1, len(word) + 1):
            neg_loglik[eow] = np.inf
            
            for bow in range(eow):
                subw = word[bow:eow]
                
                if subw in subword_logp:
                    logp = subword_logp[subw]
                    
                    s = neg_loglik[bow] - logp
                    if s < neg_loglik[eow]:
                        neg_loglik[eow] = s
                        best_subw_slices[eow] = (bow, eow)
        return neg_loglik, best_subw_slices
    
    @staticmethod
    def viterbi_backward(word, subw_slices, neg_loglik):
        subwords = []
        subwords_slices = []
        
        next_slices = subw_slices[-1]
        
        while next_slices is not None:
            subw = word[next_slices[0]:next_slices[1]]
            subwords.append(subw)
            subwords_slices.append((next_slices[0],next_slices[1]))
            next_slices = subw_slices[next_slices[0]]
        subwords.reverse()
    
        return subwords, subwords_slices, neg_loglik[-1]
    
    @staticmethod
    def get_viterbi_path(word, subword_logp):
        neg_loglik, best_subw_slices = UnigramTokenizer.viterbi_forward(word, subword_logp)
        subwords, subwords_slices, vit_path_loss = UnigramTokenizer.viterbi_backward(word, best_subw_slices, neg_loglik)
        
        return subwords, subwords_slices, vit_path_loss
    
    
    def run_e_step(self, estimated_logprob):
        initial_word_freq = self.get_initial_word_freq()
        
        viterbi_subword_freq = defaultdict(lambda : 0)
        vit_path_loss_full = 0
        
        for word in initial_word_freq:
            word_freq = initial_word_freq[word]
            
            subwords_v, _, vit_path_loss = UnigramTokenizer.get_viterbi_path(word, estimated_logprob)
            vit_path_loss_full += vit_path_loss*word_freq
            for subword_v in subwords_v:
                viterbi_subword_freq[subword_v] += word_freq
        
        return  viterbi_subword_freq, vit_path_loss_full
    
    def run_m_step(self, viterbi_subword_freq):
        
        tot_cnt = sum(list(viterbi_subword_freq.values()))
        viterbi_logprob = {k : np.log(v / tot_cnt) for k, v in viterbi_subword_freq.items()}
        
        return viterbi_logprob
    
    
    def delta_loss(self, token, estimated_word_freqs, estimated_logprob):
        if token not in estimated_word_freqs:
            return None, np.inf
        
        if token in self.alphabet:
            return None, -np.inf
        
        if len(token) == 1:
            return None, -np.inf 
        
        most_probable_split = None
        most_probable_split_score = None
        
        token_logprob = estimated_logprob[token]
        estimated_logprob[token] = -np.inf
        
        most_probable_split, _, most_probable_split_score = UnigramTokenizer.get_viterbi_path(token, estimated_logprob)
        most_probable_split_score *= -1
        
        estimated_logprob[token] = token_logprob
                    
        if most_probable_split_score is None:
            return None, -np.inf
        
        return most_probable_split, \
               most_probable_split_score*estimated_word_freqs[token] - estimated_logprob[token]*estimated_word_freqs[token]
               
    def rebuid_vocab(self, tokens):
        new_subword_freqs = {}
        
        for token in tokens:
            new_subword_freqs[token] = self.cur_vocab_subword_freqs[token]
        self.cur_vocab_subword_freqs = new_subword_freqs
            
        tot_cnt = sum(list(self.cur_vocab_subword_freqs.values()))
        self.cur_vocab_subword_logprob = {k : np.log(v / tot_cnt) for k, v in self.cur_vocab_subword_freqs.items()}
            
               
    def train_tokenizer(self, corpus):
        self.train_initial_tokenizer(corpus)
        
        while len(self.cur_vocab_subword_freqs.keys()) > self.vocab_size:
            
            viterbi_word_freq = self.cur_vocab_subword_freqs
            viterbi_logprob = self.cur_vocab_subword_logprob
            
            for i in range(self.sub_em_steps):
                viterbi_word_freq, _ = self.run_e_step(viterbi_logprob)
                viterbi_logprob = self.run_m_step(viterbi_word_freq)
            viterbi_losses = []

            for token in self.cur_vocab_subword_freqs:  
                _, delta = self.delta_loss(token, viterbi_word_freq, viterbi_logprob)
                viterbi_losses.append((token, delta))
                        
            viterbi_losses = sorted(viterbi_losses, key=lambda x: x[1])
            
            viterbi_losses = viterbi_losses[:max(int(len(viterbi_losses)*(1. - self.shrink_multplier)), self.vocab_size)]
            tokens = list(map(lambda x: x[0], viterbi_losses))
            tokens = set(tokens).union(set(self.alphabet))
            tokens = list(tokens)
            
            self.rebuid_vocab(tokens)
            
            if len(viterbi_losses) == self.vocab_size:
                break
        
    def tokenize(self, text):
        words = self.pre_tokinze_str(text)
        tokens = []
        
        for word in words:
            cur_token, _, _ = self.get_viterbi_path(word, self.cur_vocab_subword_logprob)
            tokens.extend(cur_token)
        
        return tokens