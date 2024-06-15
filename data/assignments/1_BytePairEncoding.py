# %% [markdown]
# # Byte Pair Encoding
# 
# We want to implement BPE.

# %% [markdown]
# ## Byte Pair Encoding A) [10 points]
# 
# First we want to do pre-tokenization using white spaces.
# 
# Please complete the function `pretokenize` below. This takes a list of sentences or documents and returns a list of tokenized sentences or documents. Look at the example in the docstring for more information.

# %%
from typing import List

def pretokenize(sentences: List[str]) -> List[List[str]]:
    """
    Tokenizes a list of sentences into a list of lists of tokens.

    Args:
        sentences (List[str]): List of sentences to be tokenized.

    Returns:
        List[List[str]]: List of lists of tokens, where each inner list represents
                         the tokens of a single sentence.
    Example:
        >>> sentences = ["Hello world", "This is a test"]
        >>> pretokenize(sentences)
        [['Hello', 'world'], ['This', 'is', 'a', 'test']]
    """
    # YOUR CODE HERE
    raise NotImplementedError()
    
example_sentences = [
    "This is an  example sentence",
    "Another sentence",
    "The final sentence"
]

tokenized = pretokenize(example_sentences)
tokenized

# %%


# %% [markdown]
# ## Byte Pair Encoding B) [10 points]
# 
# For BPE we first need an initial vocabulary. The input is a pretokenized list of sentences / documents.
# 
# The output should be a set of characters present in this list.

# %%
from typing import List, Set

def build_initial_vocabulary(corpus: List[List[str]]) -> Set[str]:
    """
    Build the initial vocabulary from a corpus of tokenized sentences.

    Args:
        corpus (List[List[str]]): A list of tokenized sentences, where each sentence
            is represented as a list of strings (tokens).

    Returns:
        Set[str]: A set containing all unique tokens in the corpus.

    Example:
        >>> corpus = [['hello', 'world'], ['This', 'is', 'a', 'test']]
        >>> build_initial_vocabulary(corpus)
        {'T', 'a', 'd', 'e', 'h', 'i', 'l', 'o', 'r', 's', 't', 'w'}
    """
    # YOUR CODE HERE
    raise NotImplementedError()
    
build_initial_vocabulary(pretokenize(["hello world", "This is a test"]))

# %%


# %% [markdown]
# ## Byte Pair Encoding C) [10 points]
# 
# 
# Now we want to build our dictionary for the split tokens. Complete the function `get_splits` below. Look at the example in the docstring!
# 
# Make sure to add the end of word symbol (`</w>`) to each token.

# %%
from collections import Counter
from typing import Dict, Tuple

def get_splits(corpus: List[List[str]]) -> Dict[Tuple[str], int]:
    """
    Get subword splits of tokens in a corpus.

    Args:
        corpus (List[List[str]]): A list of sentences where each sentence is represented
            as a list of tokens.

    Returns:
        Dict[Tuple[str], int]: A dictionary where keys are tuples representing subword splits
            and values are the counts of occurrences of those splits in the corpus.

    Example:
        >>> corpus = [['apple', 'banana', 'apple'], ['apple']]
        >>> get_splits(corpus)
        {('a', 'p', 'p', 'l', 'e', '</w>'): 3, ('b', 'a', 'n', 'a', 'n', 'a', '</w>'): 1}
    """
    # YOUR CODE HERE
    raise NotImplementedError()
    
get_splits(pretokenize(["apple banana apple", "apple"])) 

# %%


# %% [markdown]
# ## Byte Pair Encoding D) [10 points]
# 
# In the next step we want to find the most common pair from a splits dictionary.
# 
# Complete the function `find_most_frequent_pair` which returns the most frequent pair alongside its count (e.g. `(('a', 'n'), 2)`)

# %%
def find_most_frequent_pair(splits: Dict[Tuple[str], int]) -> Tuple[Tuple[str, str], int]:
    """
    Find the most frequent pair of characters from a dictionary of split words along with its count.

    Args:
        splits (Dict[Tuple[str], int]): A dictionary where keys are tuples of split words and values are their counts.

    Returns:
        Tuple[Tuple[str, str], int]: A tuple containing the most frequent pair of characters and its count.

    Example:
        >>> splits = {('a', 'p', 'p', 'l', 'e', '</w>'): 3,
                      ('b', 'a', 'n', 'a', 'n', 'a', '</w>'): 1}
        >>> find_most_frequent_pair(splits)
        (('a', 'n'), 2)
    """
    # YOUR CODE HERE
    raise NotImplementedError()
    
find_most_frequent_pair(get_splits(pretokenize(["apple banana apple", "apple"])))

# %%


# %% [markdown]
# ## Byte Pair Encoding E) [15 points]
# 
# Now write a function that takes a pair and the splits and merges all occurences of the pair in the splits.

# %%
def merge_split(split: Tuple[str], pair: Tuple[str, str]):
    """
    Merge a split tuple if it contains the given pair.

    Args:
        split (Tuple[str]): The split tuple to merge.
        pair (Tuple[str, str]): The pair to merge.

    Returns:
        Tuple[str]: The merged split tuple.
        
    Example:
        >>> merge_split(split=('a', 'b', 'c', 'b', 'c'), pair=('b', 'c'))
        ('a', 'bc', 'bc')
    """
    # YOUR CODE HERE
    raise NotImplementedError()
    
def merge_splits(splits: Dict[Tuple[str], int], pair: Tuple[str, str]):
    """
    Merge all split tuples in a dictionary that contain the given pair.

    Args:
        splits (Dict[Tuple[str], int]): A dictionary of split tuples and their counts.
        pair (Tuple[str, str]): The pair to merge.

    Returns:
        Dict[Tuple[str], int]: A dictionary with merged split tuples and their counts.
        
    Example:
        >>> merge_splits({('a', 'p', 'p', 'l', 'e', '</w>'): 3,
                          ('b', 'a', 'n', 'a', 'n', 'a', '</w>'): 1}, 
                          ('a', 'n'))
        {('a', 'p', 'p', 'l', 'e', '</w>'): 3,
         ('b', 'an', 'an', 'a', '</w>'): 1}
    """
    # YOUR CODE HERE
    raise NotImplementedError()
    
splits = get_splits(pretokenize(["apple banana apple", "apple"]))

most_frequent_pair, count = find_most_frequent_pair(splits)

merge_splits(splits, most_frequent_pair)

# %%


# %% [markdown]
# ## Byte Pair Encoding E) [40 points]
# 
# Now let us put this all together into a single class. Complete the methods `train`, `encode` and `decode`.
# 
# - `train` will learn the vocabulary and a list of merged pairs to use for encoding / tokenizing.
# - `encode` will tokenize a list of strings using the merge rules by applying them in order
# - `decode` will take a BPE encoded list of lists and merge subwords
# 
# Look at the examples in the docstrings for more information.

# %%
class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer.
    
    This tokenizer learns a vocabulary and encodes/decodes text using the Byte-Pair Encoding algorithm.
    """
    
    def __init__(self):
        """
        Initialize the BPETokenizer.
        """
        self.vocab: set = set()
        self.end_of_word: str = "</w>"
        self.merge_pairs: List[Tuple[str, str]] = []
        
    def train(self, corpus: List[str], max_vocab_size: int) -> None:
        """
        Train the tokenizer on a given corpus.
        First pretokenizes the corpus using whitespace
        Then uses BPE to update the vocabulary and learn the merge pairs

        Args:
            corpus (List[str]): The corpus of text for training.
            max_vocab_size (int): The maximum size of the vocabulary.

        Returns:
            None
            
        Example:
        >>> corpus = [
            "lowest lower newer newest",
            "low lower new"
        ]
        >>> tokenizer.train(corpus, max_vocab_size=20)
        """
        # YOUR CODE HERE
        raise NotImplementedError()
        
    def encode(self, corpus: List[str]) -> List[List[str]]:
        """
        Encode / Tokenize a corpus of text using the learned vocabulary and merge pairs.

        Args:
            corpus (List[str]): The corpus of text to be encoded.

        Returns:
            List[List[str]]: The encoded corpus.
        
        Example:
        >>> corpus = [
            "lowest lower newer newest",
            "low lower new"
        ]
        >>> tokenizer.train(corpus, max_vocab_size=20)
        >>> tokenizer.encode(corpus)
        [['lowest</w>', 'lower</w>', 'newer</w>', 'newe', 'st</w>'],
         ['lo', 'w</w>', 'lower</w>', 'ne', 'w</w>']]
        
        """
        # YOUR CODE HERE
        raise NotImplementedError()
        
    def decode(self, tokenized: List[List[str]]) -> List[List[str]]:
        """
        Decode a corpus of tokenized text.

        Args:
            tokenized (List[List[str]]): The tokenized text to be decoded.

        Returns:
            List[List[str]]: The decoded text.
            
        Example:
        >>> corpus = [
            "lowest lower newer newest",
            "low lower new"
        ]
        >>> tokenizer.train(corpus, max_vocab_size=20)
        >>> tokenizer.decode([['lowest</w>', 'lower</w>', 'newer</w>', 'newe', 'st</w>'],
                              ['lo', 'w</w>', 'lower</w>', 'ne', 'w</w>']])
        [['lowest', 'lower', 'newer', 'newest'], ['low', 'lower', 'new']]                              
        """
        # YOUR CODE HERE
        raise NotImplementedError()
        
corpus = [
    "lowest lower newer newest",
    "low lower new"
]
tokenizer = BPETokenizer()
tokenizer.train(corpus, 20)
tokenizer.decode(tokenizer.encode(corpus))

# %%


# %% [markdown]
# ## Byte Pair Encoding F) [5 points]
# 
# Use your BPE tokenizer on the movie script of spider. Then encode a random sentence using the tokenizer. Finally decode the sentence again.
# 
# Training might take ~3 minutes.

# %%
with open("/srv/shares/NLP/datasets/yelp/reviews_sents.txt", "r") as f:
    sentences = f.read().split("\n")
    
# YOUR CODE HERE
raise NotImplementedError()


