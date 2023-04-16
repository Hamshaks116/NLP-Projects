from sklearn.base import BaseEstimator, TransformerMixin
from bs4 import BeautifulSoup
import numpy as np
import spacy
import re
from spacy.matcher import Matcher
from spacy.tokens import Token
from pathlib import Path
from nltk.stem.porter import PorterStemmer

class SpacyPreprocessor(BaseEstimator, TransformerMixin):
    '''
    model : spacy model to be used. For example fpr english language we can specify: en_core_web_sm

    '''
    np.random.seed(0)
    def __init__(self, model, batch_size = 64, lammetize=True, lower=True, remove_stop=True, 
                 remove_punct=True, remove_email=True, remove_url=True, remove_num=False, stemming = False,
                 add_user_mention_prefix=True, remove_hashtag_prefix=False):
        self.model = model
        self.batch_size = batch_size
        self.remove_stop = remove_stop
        self.remove_punct = remove_punct
        self.remove_num = remove_num
        self.remove_url = remove_url
        self.remove_email = remove_email
        self.lammetize = lammetize
        self.lower = lower
        self.stemming = stemming
        self.add_user_mention_prefix = add_user_mention_prefix
        self.remove_hashtag_prefix = remove_hashtag_prefix

 # helpfer functions for basic cleaning 

    def basic_clean(self, text):
        
        '''
        This fuction removes HTML tags from text
        '''
        if (bool(BeautifulSoup(text, "html.parser").find())==True):         
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text()
        else:
            pass
        return re.sub(r'[\n\r]',' ', text) 

    # helper function for pre-processing with spacy and Porter Stemmer
    
    def spacy_preprocessor(self,texts):

        final_result = []
        nlp = spacy.load(self.model)
        if self.lammetize:   
          disabled = nlp.select_pipes(disable= [ 'parser', 'ner'])
        else:
          disabled = nlp.select_pipes(disable= ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
        
        ## Add @ as a prefix so that we can separate the word from @ 
        prefixes = list(nlp.Defaults.prefixes)

        if self.add_user_mention_prefix:
            prefixes += ['@']

        ## Remove # as a prefix so that we can keep hashtags and words together
        if self.remove_hashtag_prefix:
            prefixes.remove(r'#')

        prefix_regex = spacy.util.compile_prefix_regex(prefixes)
        nlp.tokenizer.prefix_search = prefix_regex.search

        matcher = Matcher(nlp.vocab)
        if self.remove_stop:
            matcher.add("stop_words", [[{"is_stop" : True}]])
        if self.remove_punct:
            matcher.add("punctuation",[ [{"is_punct": True}]])
        if self.remove_num:
            matcher.add("numbers", [[{"like_num": True}]])
        if self.remove_url:
            matcher.add("urls", [[{"like_url": True}]])
        if self.remove_email:
            matcher.add("emails", [[{"like_email": True}]])
            
        Token.set_extension('is_remove', default=False, force=True)

        cleaned_text = []
        for doc in nlp.pipe(texts,batch_size=self.batch_size ):
            matches = matcher(doc)
            for _, start, end in matches:
                for token in doc[start:end]:
                    token._.is_remove =True
                    
            if self.lammetize:                     
                text = ' '.join(token.lemma_ for token in doc if (token._.is_remove==False))
            elif self.stemming:
                text = ' '.join(PorterStemmer().stem(token.text) for token in doc if (token._.is_remove==False))
            else:
                text = ' '.join(token.text for token in doc if (token._.is_remove==False))
                                   
            if self.lower:
                text=text.lower()
            cleaned_text.append(text)
        return cleaned_text

    def fit(self, X,y=None):
        return self

    def transform(self, X, y=None):
        try:
            if str(type(X)) not in ["<class 'list'>","<class 'numpy.ndarray'>"]:
                raise Exception('Expected list or numpy array got {}'.format(type(X)))
            x_clean = [self.basic_clean(text) for text in X]
            x_clean_final = self.spacy_preprocessor(x_clean)
            return x_clean_final
        except Exception as error:
            print('An exception occured: ' + repr(error))
