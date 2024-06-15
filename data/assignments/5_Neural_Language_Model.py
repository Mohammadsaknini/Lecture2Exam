# %% [markdown]
# # Neural Language Model
# 
# In this task we want to implement the neural language model given in chapter 7 of the book (p. 16) using PyTorch.
# 
# <img src="Neural_Language_Model_files/img/model.png">
# 
# © Tim Metzler, Hochschule Bonn-Rhein-Sieg

# %% [markdown]
# ## Helper Functions
# 
# You are given some helper functions to make this assignment easier.
# 
# First a tokenizer that turns strings into lists of token ids.

# %%
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from typing import List

def tokenizer_from_strings(strings: List[str], vocab_size: int = None) -> Tokenizer:
    """
    Create and train a WordLevel Tokenizer for tokenizing text from the given strings.

    Args:
        strings (List[str]): A list of strings containing the text data for training.
        vocab_size (int, optional): The maximum vocabulary size to limit the number of tokens
                                    in the tokenizer. If None, the vocabulary size is determined automatically.
                                    Defaults to None.

    Returns:
        Tokenizer: A trained WordLevel Tokenizer capable of tokenizing text data.    
    """    
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

    # We can also pass a vocab_size to the trainer to only keep the most frequent words
    # Special tokens are tokens that we want to use but are not part of the text we train on
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "<s>", "</s>"], 
    )
    if vocab_size is not None:
        trainer.vocab_size = vocab_size
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(strings, trainer=trainer)
    return tokenizer


# %%
# Example texts for our tokenizer. This can be a list of one our more documents
my_text = [
    "<s> I like NLP. </s>",
    "It is very interesting",
    "But it is also hard"
]

tokenizer = tokenizer_from_strings(my_text)

# Let us look at our trained vocabulary
print("The vocabulary")
print(tokenizer.get_vocab())

# Size of the vocabulary
print("\nThe size of the vocabulary")
print(tokenizer.get_vocab_size())

# Now lets turn a sentence into a list of token indices
encoded_input = tokenizer.encode("NLP is hard")

# This is how it splits it into tokens
print("\nThe tokens from our input string")
print(encoded_input.tokens)
# This is how we get the ids
print("\nThe ids for the tokens")
print(encoded_input.ids)

# Let us look what happens if we put in unknown words
encoded_input = tokenizer.encode("NLP is a tough subject")
print("\nThe tokens from our input string. Notice how everything unknown is represented as [UNK]")
print(encoded_input.tokens)
print("\nThe ids for the tokens")
print(encoded_input.ids)

# Finally we can also turn back ids into strings
tokenizer.decode([6, 13, 8])

# %% [markdown]
# ## Neural Language Model A)
# ### One Hot Encoder
# 
# First we create a one-hot encoder that produces tensors. We will use the built-in function `torch.nn.functional.one_hot` to create our embeddings.
# 
# First look at the example in the following cell.
# 
# Then complete the class `OneHotEncoder` below. You need to implement the method `encode`, which encodes a single index into a one-hot embedding and the method `encode_sequence`, which will take a list of indices and should return a list of one-hot embeddings.

# %%
from torch import tensor
from torch.nn.functional import one_hot

index = 5
vocab_size = 20

my_one_hot_embedding = one_hot(
    tensor(index),
    num_classes=vocab_size
).float()

print(my_one_hot_embedding)          # The embedding
print(my_one_hot_embedding.shape)    # The size of the embedding
print(my_one_hot_embedding.argmax()) # The index of the 1

# %%
from torch.nn.functional import one_hot
from torch import tensor, float32
from typing import List

class OneHotEncoder:
    
    def __init__(self, vocab_size: int):
        """
        OneHotEncoder class for converting token IDs to one-hot encoded tensors.

        Args:
            vocab_size (int): The size of the vocabulary, i.e., the number of unique tokens.
        """
        self.vocab_size = vocab_size
        
    def encode(self, token_id: int) -> tensor:
        """
        Encode a single token ID as a one-hot encoded tensor.

        Args:
            token_id (int): The token ID to be encoded.

        Returns:
            tensor: The one-hot encoded tensor representing the input token ID.
        """
        # YOUR CODE HERE
        raise NotImplementedError()
    
    def encode_sequence(self, token_ids: List[int]) -> List[tensor]:
        """
        Encode a sequence of token IDs as a list of one-hot encoded tensors.

        Args:
            token_ids (List[int]): A list of token IDs to be encoded.

        Returns:
            List[tensor]: A list of one-hot encoded tensors representing the input token IDs.
        """
        # YOUR CODE HERE
        raise NotImplementedError()

# %%
# This is for you to test if your implementation is working

vocabulary_size = 50

encoder = OneHotEncoder(vocabulary_size)

# Test with a single index
embedding = encoder.encode(5)

assert embedding.shape[0] == vocabulary_size, "All embeddings should have the size of the vocabulary"
assert embedding.argmax() == 5, "The single one should be at index 5"

# Test with a list of indices
indices = [5, 3, 7]
embeddings = encoder.encode_sequence(indices)

assert len(embeddings) == 3, "We put in three indices so we want three embeddings"
assert isinstance(embeddings, list), "We want to return a list"

for idx, embedding in zip(indices, embeddings):
    assert embedding.shape[0] == vocabulary_size, "All embeddings should have the size of the vocabulary"
    assert embedding.argmax() == idx, "The single one should be at index 5" 



# %% [markdown]
# ## Neural Language Model B)
# ### The model
# 
# Complete the model class below.
# 
# **Hint: To get from three tensors of size $d$ to a tensor of size $3d$ we need to concatenate. See the next cell for an example**

# %%
import torch

# Create our encoder and encode three indices
encoder = OneHotEncoder(12)
indices = [5, 3, 7]
embedding1, embedding2, embedding3 = encoder.encode_sequence(indices)

# Concatenate them into a single embedding of size 3*vocab_size
concatenated_embeddings = torch.concatenate((embedding1, embedding2, embedding3))

print(concatenated_embeddings.shape) # This is three times our vocabulary size
print(concatenated_embeddings)

# %%
import torch.nn as nn
from torch import TensorType

class NeuralLanguageModel(nn.Module):
    """
    A neural language model that predicts a word from two input words
    """
    
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int):
        """
        Initializes the NeuralLanguageModel.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_size (int): The size of word embeddings.
            hidden_size (int): The size of the hidden layer
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        
        # Create your layers here. All layers are linear
        # We want an embedding layer
        # Then a hidden layer
        # Then an output layer
        # Then we define our activation functions and the softmax
        # YOUR CODE HERE
        raise NotImplementedError()
        
    def forward(
        self, 
        word1: TensorType, 
        word2: TensorType, 
        word3: TensorType, 
        inference: bool=False
    ) -> TensorType:
        """
        Forward pass of the neural language model.

        Args:
            word1 (torch.TensorType): Tensor representing the first word (one-hot).
            word2 (torch.TensorType): Tensor representing the second word (one-hot).
            word3 (torch.TensorType): Tensor representing the third word (one-hot).
            inference (bool, optional): Flag representing if we are doing inference or not.
                                        This is needed since during training PyTorch does not work well with
                                        the softmax. 

        Returns:
            torch.TensorType: Output tensor representing the probability distribution over the vocabulary.
        """
        # This will be our output
        y = None
        # YOUR CODE HERE
        raise NotImplementedError()
        
        # The loss we will use later does not play well with softmax. 
        # So we only apply it for inferencing
        if inference:
            y = self.softmax(y)        
        return y

# %%
# This is to test your implementation

# First create the model
vocab_size = 50
model = NeuralLanguageModel(
    vocab_size=vocab_size,
    embedding_size=16,
    hidden_size=10
)

# Next create some inputs for our model
encoder = OneHotEncoder(vocab_size)

word1, word2, word3 = encoder.encode_sequence([3, 7, 2])

# Now we feed it to our model
output = model(word1, word2, word3)

print(output)
assert output.shape[0] == vocab_size, "Our output should have the size of the vocabulary"

# Next we do the same for inference (we check if the softmax is applied then)
output = model(word1, word2, word3, inference=True)
print(output)

assert abs(output.sum().item() - 1) < 10e-6, "The outputs should sum up to 1"


# %% [markdown]
# ## The Dataset
# 
# The dataset class was already implemented for you. Look at it and understand what it does.

# %%
from torch.utils.data import Dataset, DataLoader
from typing import List, Any, Tuple

def sliding_window(sequence: List[Any], window_size: int) -> List[Any]:
    """
    Generate a sliding window over a sequence (list).

    Args:
        sequence (list): The input sequence.
        window_size (int): The size of the sliding window.

    Yields:
        list: A window of elements from the input sequence.
    """
    for i in range(len(sequence) - window_size + 1):
        yield sequence[i:i + window_size]
        

class NGramTextDataset(Dataset):
    """
    A PyTorch Dataset class for generating trigram-based text datasets.

    Args:
        sentences (List[str]): A list of input sentences.
        vocab_size (int, optional): The size of the vocabulary to use for tokenization.
                                    Defaults to None.

    Methods:
        __len__(self) -> int: Returns the total number of examples in the dataset.

        __getitem__(self, index: int) -> Tuple[List[int], int]:
            Returns a tuple containing the input trigram and its corresponding label for the specified index.

    Example:
        sentences = ["This is a sample sentence.", "Another example sentence."]
        dataset = TrigramTextDataset(sentences, vocab_size=10000)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    """
    
    def __init__(self, sentences: List[str], vocab_size: int=None):
        """
        Initializes the TrigramTextDataset with input sentences and vocabulary size.

        Args:
            sentences (List[str]): A list of input sentences.
            vocab_size (int, optional): The size of the vocabulary to use for tokenization.
                                        Defaults to None.
        """
        # First augment the sentences with a start and end symbol
        # We add three of each since we look at four grams
        sentences = [
            '<s> <s> <s>' + sentence + ' </s> </s> </s>'
            for sentence in sentences
        ]
        # Next we train our tokenizer
        self.tokenizer = tokenizer_from_strings(sentences, vocab_size)
        self.encoder = OneHotEncoder(self.tokenizer.get_vocab_size())
        
        # Prepare our examples
        self.inputs = []
        self.labels = []
        
        for sentence in sentences:
            # Go over each trigram of the encoded sentence
            for trigram in sliding_window(self.tokenizer.encode(sentence).ids, 4):
                # Take the first two tokens as input
                self.inputs.append(trigram[:-1])
                # Take the last token as the label
                self.labels.append(trigram[-1])
                
    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset.

        Returns:
            int: The number of examples in the dataset.
        """
        return len(self.labels)
    
    def __getitem__(self, index: int) -> Tuple[List[torch.tensor], torch.tensor]:
        """
        Returns a tuple containing the input trigram and its corresponding label for the specified index.

        Args:
            index (int): The index of the example to retrieve.

        Returns:
            Tuple[List[int], int]: A tuple containing the input trigram (a list of integers) and
            its corresponding label (an integer).
        """
        return self.encoder.encode_sequence(self.inputs[index]), self.encoder.encode(self.labels[index])

# %%
# Open our training data
# This file has one sentence per line
with open('/srv/shares/NLP/datasets/marvel/spider_man_homecoming.txt', 'r') as f:
    text = f.read()
    
# Create the dataset
dataset = NGramTextDataset(text.split("\n"))

# Look at one example
inputs, label = dataset[4]

print(inputs)
print(label)

# We can also turn this back into strings using the tokenizer
word1, word2, word3 = inputs

# Turn the one-hot vectors back to indices
indices = [
    word1.argmax().item(),
    word2.argmax().item(),
    word3.argmax().item()
]

label_index = label.argmax().item()

# Turn these indices back to strings
dataset.tokenizer.decode(indices), dataset.tokenizer.decode([label_index])

# %% [markdown]
# ## Neural Language Model C)
# ### The training loop and optimizer
# 
# Please implement the training loop below. 
# This method receives a model, an optimizer, a loss function and a dataloader.

# %%
from typing import List
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
import numpy as np

def train_one_epoch(
    model: nn.Module, 
    optimizer: Optimizer, 
    loss_fn: _Loss, 
    dataloader: DataLoader) -> float:
    """
    Trains a neural network model for one epoch using the specified data.

    Parameters:
        model (nn.Module): The neural network model to be trained.
        optimizer (Optimizer): The optimizer used for updating model weights.
        loss_fn (_Loss): The loss function used to compute the training loss.
        dataloader (DataLoader): The data loader providing batches of training data.

    Returns:
        float: The mean training loss for the entire epoch.
    """
    # YOUR CODE HERE
    raise NotImplementedError()

# %% [markdown]
# ## Neural Language Model D)
# ### Creating the optimizer, loss, model and dataloader
# 
# Use a batch size of 256 for your data loader.
# 
# Initialize your model with an embedding size of 8 and a hidden size of 12.
# 
# Use AdamW as your optimizer with a learning rate of 0.01.
# 
# Use the CrossEntropyLoss as the loss function

# %%

model = None
dataloader = None
loss_fn = None
optimizer = None
# YOUR CODE HERE
raise NotImplementedError()

# %% [markdown]
# ## Neural Language Model E)
# ### Train the model
# 
# Train the model for at least 10 epochs (this should take about 3 minutes).
# 
# **Hint: If you want a progress bar for your loops you can use the following code:**
# 
# 

# %%
from tqdm.notebook import tqdm_notebook
# YOUR CODE HERE
raise NotImplementedError()

# %% [markdown]
# ## Neural Language Model F)
# ### Plot the losses
# 
# Create a plot with the batch losses.
# Your x axis is the epoch, your y axis is the loss.
# 
# Don't forget labels, a title and a grid

# %%
import matplotlib.pyplot as plt
# YOUR CODE HERE
raise NotImplementedError()

# %%
# Here we can do inference
sentence = "<s> <s> I'm a little station on the ground"

tokenized = dataset.tokenizer.encode(sentence)
token_ids = tokenized.ids

onehots = dataset.encoder.encode_sequence(token_ids)[-3:]

predicted_index = model(*onehots, inference=True).argmax().item()

dataset.tokenizer.decode([predicted_index])

def generate_random_sentence(model, dataset, input_sequence):
    sentence_end = False
    while not sentence_end:
        tokenized = dataset.tokenizer.encode(input_sequence)
        token_ids = tokenized.ids

        onehots = dataset.encoder.encode_sequence(token_ids)[-3:]
        
        probabilities = model(*onehots, inference=True).detach().numpy()
        
        
        
        next_index = np.random.choice(model.vocab_size, p=probabilities)
        
        # We specified <s> and </s> as special tokens. To have them be part of the output 
        # we need to set the flag skip_special_tokens=False
        next_word = dataset.tokenizer.decode([next_index], skip_special_tokens=False)
        sentence_end = next_word == "</s>"
        input_sequence += f" {next_word}"
    return input_sequence

generate_random_sentence(model, dataset, "<s> <s> <s> No")


