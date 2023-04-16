from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import spacy
import re
import sys
import custom_preprocessor as cp
from pathlib import Path


class ManualFeatures(TransformerMixin, BaseEstimator):
    
    def __init__(self, spacy_model, pos_features = True, ner_features = True, count_features = True):
        
        self.spacy_model = spacy_model
        self.pos_features = pos_features
        self.ner_features = ner_features
        self.count_features = count_features    
    
        
    # Define some helper functions
    def get_pos_features(self, cleaned_text):
        nlp = spacy.load(self.spacy_model)
        noun_count = []
        aux_count = []
        verb_count = []
        adj_count =[]
        disabled = nlp.select_pipes(disable= ['lemmatizer', 'ner'])
        for doc in nlp.pipe(cleaned_text, batch_size=1000, n_process=-1):
            nouns = [token.text for token in doc if (token.pos_ in ["NOUN","PROPN"])] 
            auxs =  [token.text for token in doc if (token.pos_ in ["AUX"])] 
            verbs =  [token.text for token in doc if (token.pos_ in ["VERB"])] 
            adjectives =  [token.text for token in doc if (token.pos_ in ["ADJ"])]        

            noun_count.append(int(len(nouns)))
            aux_count.append(int(len(auxs)))
            verb_count.append(int(len(verbs)))
            adj_count.append(int(len(adjectives)))
        return np.transpose(np.vstack((noun_count, aux_count, verb_count, adj_count)))
            
        
    def get_ner_features(self, cleaned_text):
        nlp = spacy.load(self.spacy_model)
        count_ner  = []
        disabled = nlp.select_pipes(disable= ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
        for doc in nlp.pipe(cleaned_text, batch_size=1000, n_process=-1):
            ners = [ent.label_ for ent in doc.ents] 
            count_ner.append(len(ners))
        return np.array(count_ner).reshape(-1,1)   
   
    def get_count_features(self, cleaned_text):
        list_count_words =[]
        list_count_characters =[]
        list_count_characters_no_space =[]
        #list_avg_word_length=[]
        list_count_digits=[]
        list_count_numbers=[]
        for sent in cleaned_text:
            words = re.sub(r'\d+\s','',sent)
            numbers = re.findall(r'\d+', sent)
            print(sent)
            #print(numbers)

            count_word = len(words.split())
            count_char = len(words)
            count_char_no_space = len(''.join(words.split()))
           
            count_numbers = len(numbers)
            count_digits = len(''.join(numbers))

            list_count_words.append(count_word)
            list_count_characters.append(count_char)
            list_count_characters_no_space.append(count_char_no_space)
            list_count_digits.append(count_digits)
            list_count_numbers.append(count_numbers)  
            
        count_features = np.vstack((list_count_words, list_count_characters,
                                  list_count_characters_no_space,
                                  list_count_digits,list_count_numbers ))
        return np.transpose(count_features)
        
 
         
    def fit(self, X, y = None):
        return self
    
    def transform(self, X,y=None):
        try:
            if str(type(X)) not in ["<class 'list'>","<class 'numpy.ndarray'>"]:
                raise Exception('Expected list or numpy array got {}'.format(type(X)))

            
            preprocessor1 = cp.SpacyPreprocessor(model = 'en_core_web_sm', lammetize=False, lower = False, 
                                   remove_stop=False )
            preprocessor2 = cp.SpacyPreprocessor(model = 'en_core_web_sm', lammetize=False, lower = False, 
                                   remove_stop=False, remove_punct= False )
            
            feature_names =[]
            if (self.pos_features or self.ner_features):
                cleaned_x_count_ner_pos = preprocessor2.fit_transform(X)
            
            if self.count_features:
                cleaned_x_count_features = preprocessor1.fit_transform(X)
                count_features = self.get_count_features(cleaned_x_count_features)
                feature_names.extend(['count_words', 'count_characters',
                                  'count_characters_no_space', 
                                  'count_digits','count_numbers'])
				
            else:
                count_features = np.empty(shape = (0, 0))
                
            if self.pos_features: 
                pos_features = self.get_pos_features(cleaned_x_count_ner_pos)
                feature_names.extend(['noun_count', 'aux_count', 'verb_count', 'adj_count'])
            else:
                 pos_features = np.empty(shape = (0, 0))
                
            if self.ner_features: 
                ner_features =self.get_ner_features(cleaned_x_count_ner_pos)
                feature_names.extend(['ner'])
            else:
                 ner_features = np.empty(shape = (0, 0))
                
            return np.hstack((count_features, ner_features, pos_features)), feature_names
            

        except Exception as error:
            print('An exception occured: ' + repr(error))
    