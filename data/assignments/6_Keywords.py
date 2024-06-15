# %% [markdown]
# 
# # Introduction to spaCy
# 
# SpaCy is a tool that does tokenization, parsing, tagging and named entity regocnition (among other things).
# 
# When we parse a document via spaCy, we get an object that holds sentences and tokens, as well as their POS tags, dependency relations and so on.
# 
# Look at the next cell for an example.
# 
# © Tim Metzler, Hochschule Bonn-Rhein-Sieg

# %%
import spacy

# Load the English language model
nlp = spacy.load('/srv/shares/NLP/spacy/en_core_web_sm')

# Our sample input
text = 'SpaCy is capable of    tagging, parsing and annotating text. It recognizes sentences and stop words.'

# Parse the sample input
doc = nlp(text)

# For every sentence
for sent in doc.sents:
    # For every token
    for token in sent:
        # Print the token itself, the pos tag, 
        # dependency tag and whether spacy thinks this is a stop word
        print(token, token.pos_, token.dep_, token.is_stop)
        
print('-'*30)
print('The nouns and proper nouns in this text are:')
# Print only the nouns:
for token in doc:
    if token.pos_ in ['NOUN', 'PROPN']:
        print(token)

# %% [markdown]
# ## SpaCy A) [5 points]
# ### Splitting text into sentences
# 
# You are given the text in the next cell.
# 
# ```
# text = '''
# This is a sentence. 
# Mr. A. said this was another! 
# But is this a sentence? 
# The abbreviation Merch. means merchant(s).
# At certain univ. in the U.S. and U.K. they study NLP.
# '''
# ```
# 
# Use spaCy to split this into sentences. Store the resulting sentences (each as a **single** string) in the list ```sentences```. Make sure to convert the tokens to strings (e.g. via str(token)).

# %%
import spacy
nlp = spacy.load('/srv/shares/NLP/spacy/en_core_web_sm')

text = '''
This is a sentence. Mr. A. said this was another! 
But is this a sentence? The abbreviation Merch. means merchant(s).
At certain Univ. in the U.S. and U.K. they study NLP.
'''
sentences = []

# YOUR CODE HERE
raise NotImplementedError()

for sentence in sentences:
    print(sentence)
    print('.')
    assert type(sentence) == str, 'You need to convert this to a single string!'

# %%
# This is a test cell, please ignore it!

# %% [markdown]
# ## SpaCy B) [5 points]
# 
# ### Cluster the text by POS tag
# 
# Next we want to cluster the text by the corresponding part-of-speech (POS) tags. 
# 
# The result should be a dictionary ```pos_tags``` where the keys are the POS tags and the values are lists of words with those POS tags. Make sure your words are converted to **strings**.
# 
# *Example:*
# 
# ```
# pos_tags['VERB'] # Output: ['said', 'means', 'study']
# pos_tags['ADJ']  # Output: ['certain']
# ...
# ```

# %%
import spacy
nlp = spacy.load('/srv/shares/NLP/spacy/en_core_web_sm')

text = '''
This is a sentence. Mr. A. said this was another! 
But is this a sentence? The abbreviation Merch. means merchant(s).
At certain Univ. in the U.S. and U.K. they study NLP.
'''

pos_tags = dict()

# YOUR CODE HERE
raise NotImplementedError()

for key in pos_tags:
    print('The words with the POS tag {} are {}.'.format(key, pos_tags[key]))
    for token in pos_tags[key]:
        assert type(token) == str, 'Each token should be a string'

# %%
# This is a test cell, please ignore it!

# %% [markdown]
# # SpaCy C) [5 points]
# 
# ### Stop word removal
# 
# Stop words are words that appear often in a language and don't hold much meaning for a NLP task. Examples are the words ```a, to, the, this, has, ...```. This depends on the task and domain you are working on.
# 
# SpaCy has its own internal list of stop words. Use spaCy to remove all stop words from the given text. Store your result as a **single string** in the variable ```stopwords_removed```.

# %%
import spacy
nlp = spacy.load('/srv/shares/NLP/spacy/en_core_web_sm')

text = '''
This is a sentence. Mr. A. said this was another! 
But is this a sentence? The abbreviation Merch. means merchant(s).
At certain Univ. in the U.S. and U.K. they study NLP.
'''

stopwords_removed = ''

# YOUR CODE HERE
raise NotImplementedError()

print(stopwords_removed)
assert type(stopwords_removed) == str, 'Your answer should be a single string!'

# %%
# This is a test cell, please ignore it!

# %% [markdown]
# # SpaCy D) [2 points]
# 
# ### Dependency Tree
# 
# We now want to use spaCy to visualize the dependency tree of a certain sentence. Look at the Jupyter Example on the [spaCy website](https://spacy.io/usage/visualizers/). Render the tree.

# %%
import spacy
from spacy import displacy

nlp = spacy.load('/srv/shares/NLP/spacy/en_core_web_sm')

text = 'Dependency Parsing is helpful for many tasks.'

# YOUR CODE HERE
raise NotImplementedError()

# %% [markdown]
# # SpaCy E) [5 points]
# 
# ### Dependency Parsing
# 
# Use spaCy to extract all subjects and objects from the text. We define a subject as any word that has ```subj``` in its dependency tag (e.g. ```nsubj```, ```nsubjpass```, ...). Similarly we define an object as any token that has ```obj``` in its dependency tag (e.g. ```dobj```, ```pobj```, etc.).
# 
# For each sentence extract the subject, root node ```ROOT``` of the tree and object and store them as a single string in a list. Name this list ```subj_obj```.
# 
# *Example:*
# 
# ```
# text = 'Learning multiple ways of representing text is cool. We can access parts of the sentence with dependency tags.'
# 
# subj_obj = ['Learning ways text is', 'We access parts sentence tags']

# %%
import spacy
nlp = spacy.load('/srv/shares/NLP/spacy/en_core_web_sm')

text = '''
This is a sentence. Mr. A. said this was another! 
But is this a sentence? The abbreviation Merch. means merchant(s).
At certain Univ. in the U.S. and U.K. they study NLP.
'''

subj_obj = []
# YOUR CODE HERE
raise NotImplementedError()

for cleaned_sent in subj_obj:
    print(cleaned_sent)
    assert type(cleaned_sent) == str, 'Each cleaned sentence should be a string!'

# %%
# This is a test cell, please ignore it!

# %% [markdown]
# # Keyword Extraction
# 
# In this assignment we want to write a keyword extractor. There are several methods of which we want to explore a few.
# 
# We want to extract keywords from our Yelp reviews.
# 
# ##  POS tag based extraction
# 
# When we look at keywords we realize that they are often combinations of nouns and adjectives. The idea is to find all sequences of nouns and adjectives in a corpus and count them. The $n$ most frequent ones are then our keywords.
# 
# A keyword (or keyphrase) by this definition is any combination of nouns (NOUN) and adjectives (ADJ) that ends in a noun. We also count proper nouns (PROPN) as nouns.
# 
# © Tim Metzler, Hochschule Bonn-Rhein-Sieg

# %% [markdown]
# ## POS tag based extraction A) [35 points]
# 
# ### POSKeywordExtractor
# 
# Please complete the function ```keywords``` in the class ```POSKeywordExtractor```.
# 
# You are given the file ```wiki_nlp.txt```, which has the raw text from all top-level Wikipedia pages under the category ```Natural language processing```. Use this for extracting your keywords.
# 
# *Example:*
# 
# Let us look at the definition of an index term or keyword from Wikipedia. Here I highlighted all combinations of nouns and adjectives that end in a noun. All the highlighted words are potential keywords.
# 
# An **index term**, **subject term**, **subject heading**, or **descriptor**, in **information retrieval**, is a **term** that captures the **essence** of the **topic** of a **document**. **Index terms** make up a **controlled vocabulary** for **use** in **bibliographic records**.
# 
# *Rules:*
# 
# - A keyphrase is a sequence of nouns, adjectives and proper nouns ending in a noun or proper noun.
# - Keywords / Keyphrases **can not go over sentence boundaries**.
# - We always take the longest sequence of nouns, adjectives and proper nouns
#   - Consider the sentence ```She studies natural language processing.```. The only extracted keyphrase here will be ```('natural', 'language', 'processing')```.
# - Consider the sentence ```neural networks massively increased the performance.```:
#   - Here our keyphrase would be ```neural networks```, not ```neural networks massively```.
#   - Our keyphrases are always the longest sequence of nouns and adjectives ending in a noun

# %%
%%time
from typing import List, Tuple, Iterable
from collections import Counter
import spacy
from spacy.tokens import Token
import pickle


class POSKeywordExtractor:
    
    def __init__(self):
        # Set up SpaCy in a more efficient way by disabling what we do not need
        # This is the dependency parser (parser) and the named entity recognizer (ner)
        self.nlp = spacy.load(
            '/srv/shares/NLP/spacy/en_core_web_sm', 
            disable=['ner', 'parser']
        )
        # Add the sentencizer to quickly split our text into sentences
        self.nlp.add_pipe('sentencizer')
        # Increase the maximum length of text SpaCy can parse in one go
        self.nlp.max_length = 1500000
        
    def validate_keyphrase(self, candidate: Iterable[Token]) -> Iterable[Token]:
        '''
        Takes in a list of tokens which are all proper nouns, nouns or adjectives
        and returns the longest sequence that ends in a proper noun or noun
        
        Args:
            candidate         -- List of spacy tokens
        Returns:
            longest_keyphrase -- The longest sequence that ends in a noun
                                 or proper noun
                                 
        Example:
            candidate = [neural, networks, massively]
            longest_keyphrase = [neural, networks]
        '''
        # YOUR CODE HERE
        raise NotImplementedError()
        
    def keywords(self, text: str, n_keywords: int, min_words: int) -> List[Tuple[Tuple[str], int]]:
        '''
        Extract the top n most frequent keywords from the text.
        Keywords are sequences of adjectives and nouns that end in a noun
        
        Arguments:
            text       -- the raw text from which to extract keywords
            n_keywords -- the number of keywords to return
            min_words  -- the number of words a potential keyphrase has to include
                          if this is set to 2, then only keyphrases consisting of 2+ words are counted
        Returns:
            keywords   -- List of keywords and their count, sorted by the count
        '''
        doc = self.nlp(text)
        keywords = []
        # YOUR CODE HERE
        raise NotImplementedError()
        return keywords

    
with open('/srv/shares/NLP/datasets/wiki/wiki_nlp.txt', 'r') as corpus_file:
    text = corpus_file.read()
    
keywords = POSKeywordExtractor().keywords(text.lower(), n_keywords=15, min_words=1)

'''
Expected output:
The keyword ('words',) appears 353 times.
The keyword ('text',) appears 342 times.
The keyword ('example',) appears 263 times.
The keyword ('word',) appears 231 times.
The keyword ('natural', 'language', 'processing') appears 184 times.
...
'''
for keyword in keywords:
    print('The keyword {} appears {} times.'.format(*keyword))

# %%
# This is a test cell, please ignore it!

# %% [markdown]
# ### POS tag based extraction B) [4 points]
# 
# Rerun the keyword extrator with a minimum word count of ```min_words=2``` and a keyword count of ```n_keywords=15```.
# 
# Store this in the variable ```keywords_2```. Print the result.
# 
# Make sure to convert the input text to lower case!

# %%
keywords_2 = []

# YOUR CODE HERE
raise NotImplementedError()

# %%
# This is a test cell, please ignore it!

# %% [markdown]
# 
# # Stop word based keyword extraction
# 
# One approach to extract keywords is by splitting the text at the stop words. Then we count these potential keywords and output the top $n$ keywords. Make sure to only include words proper words. Here we define proper words as those words that match the regular expression ```r'\b(\w{2,})\b'``` (words that consist of at least 2 alphanumerical characters, including hyphens). 
# 
# © Tim Metzler, Hochschule Bonn-Rhein-Sieg

# %% [markdown]
# ## Stop word based keyword extraction A) [35 points]
# 
# Complete the function ```keywords``` in the class ```StopWordKeywordExtractor```.

# %%
%%time
from typing import List, Tuple
from collections import Counter
import re
import spacy

class StopWordKeywordExtractor:
    
    def __init__(self):
        # Set up SpaCy in a more efficient way by disabling what we do not need
        # This is the dependency parser (parser) and the named entity recognizer (ner)
        self.nlp = spacy.load('/srv/shares/NLP/spacy/en_core_web_sm', disable=['ner', 'parser'])
        # Add the sentencizer to quickly split our text into sentences
        self.nlp.add_pipe('sentencizer')
        # Increase the maximum length of text SpaCy can parse in one go
        self.nlp.max_length = 1500000
        
    def is_proper_word(self, token:str) -> bool:
        '''
        Checks if the word is a proper word by our definition
        
        Arguments:
            token     -- The token as a string
        Return:
            is_proper -- True / False
        '''
        match = re.search(r'\b(\w{2,})\b', token)
        return match and token == match[0] 
    
    def keywords(self, text: str, n_keywords: int, min_words: int) -> List[Tuple[Tuple[str], int]]:
        '''
        Extract the top n most frequent keywords from the text.
        Keywords are sequences of adjectives and nouns that end in a noun
        
        Arguments:
            text       -- the raw text from which to extract keywords
            n_keywords -- the number of keywords to return
            min_words  -- the number of words a potential keyphrase has to include
                          if this is set to 2, then only keyphrases consisting of 2+ words are counted
        Returns:
            keywords   -- List of keywords and their count, sorted by the count
                          Example: [(('potato'), 12), (('potato', 'harvesting'), 9), ...]
        '''
        doc = self.nlp(text)
        keywords = []
        # YOUR CODE HERE
        raise NotImplementedError()
        return keywords
        
with open('/srv/shares/NLP/datasets/wiki/wiki_nlp.txt', 'r') as corpus_file:
    text = corpus_file.read()
    
keywords = StopWordKeywordExtractor().keywords(text.lower(), n_keywords=15, min_words=1)

'''
Expected output:
The keyword ('words',) appears 273 times.
The keyword ('text',) appears 263 times.
The keyword ('example',) appears 257 times.
The keyword ('word',) appears 201 times.
The keyword ('references',) appears 184 times.
The keyword ('natural', 'language', 'processing') appears 165 times.
...
'''
for keyword in keywords:
    print('The keyword {} appears {} times.'.format(*keyword))

# %%
# This is a test cell, please ignore it!

# %% [markdown]
# ## Stop word based keyword extraction B) [4 points]
# 
# Rerun the keyword extrator with a minimum word count of ```min_words=2``` and a keyword count of ```n_keywords=15```.
# 
# Store this in the variable ```keywords_2```. Print the result.
# 
# Make sure to convert the input text to lower case!

# %%
keywords_2 = []

# YOUR CODE HERE
raise NotImplementedError()

# %%
# This is a test cell, please ignore it!


