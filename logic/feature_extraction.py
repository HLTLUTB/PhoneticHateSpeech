import sys
import epitran
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from logic.text_analysis import TextAnalysis
from logic.utils import Utils
from root import DIR_EMBEDDING


class FeatureExtraction(BaseEstimator, TransformerMixin):

    def __init__(self, lang='es', text_analysis=None):
        try:
            if text_analysis is None:
                self.ta = TextAnalysis(lang=lang)
            else:
                self.ta = text_analysis
            file_syllable_embedding_en = DIR_EMBEDDING + 'syllable_embedding_en.model'
            file_syllable_embedding_es = DIR_EMBEDDING + 'syllable_embedding_es.model'
            file_phoneme_embedding_en = DIR_EMBEDDING + 'phoneme_embedding_en.model'
            file_phoneme_embedding_es = DIR_EMBEDDING + 'phoneme_embedding_es.model'
            print('Loading Lexicons and Embedding.....')
            if lang == 'es':
                epi = epitran.Epitran('spa-Latn')
                syllable_embedding = Word2Vec.load(file_syllable_embedding_es)
                phoneme_embedding = Word2Vec.load(file_phoneme_embedding_es)
            else:
                epi = epitran.Epitran('eng-Latn')
                syllable_embedding = Word2Vec.load(file_syllable_embedding_en)
                phoneme_embedding = Word2Vec.load(file_phoneme_embedding_en)

            self.epi = epi
            self.syllable_embedding = syllable_embedding
            self.phoneme_embedding = phoneme_embedding
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error FeatureExtraction: {0}'.format(e))

    def fit(self, x, y=None):
        return self

    def transform(self, list_messages):
        try:
            result = self.get_features(list_messages)
            return result
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error transform: {0}'.format(e))

    def get_features(self, messages, model_type='111', binary_vad='0000'):
        try:
            # S:Syllable, F: Frequency Phoneme, S: One/All Phoneme

            syllable_features = self.get_feature_syllable(messages)
            phoneme_frequency = self.get_frequency_phoneme(messages)
            one_syllable = self.get_feature_phoneme(messages)
            all_syllable = self.get_feature_phoneme(messages, syllable=True)

            result = np.zeros((len(messages), 0), dtype="float32")
            if int(model_type[0]) == 1:
                result = np.append(result, syllable_features, axis=1)
            elif int(model_type[1]) == 1:
                result = np.append(result, phoneme_frequency, axis=1)
            elif int(model_type[2]) == 1:
                result = np.append(result, one_syllable, axis=1)
            elif int(model_type[3]) == 1:
                result = np.append(result, all_syllable, axis=1)

            return result
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_features: {0}'.format(e))
            return None

    def get_feature_syllable(self, messages):
        try:
            messages_phonetic = []
            model = self.syllable_embedding
            num_features = model.vector_size
            index2phoneme_set = set(model.wv.index2word)
            num_phonemes = 1
            feature_vec = []
            list_syllable = [token['syllables'] for token in self.ta.tagger(messages) if token['syllables'] is not None]
            for syllable in list_syllable:
                for s in syllable:
                    syllable_phonetic = self.epi.transliterate(s, normpunc=True)
                    messages_phonetic.append(syllable_phonetic)
                    if syllable_phonetic in index2phoneme_set:
                        vec = model.wv[syllable_phonetic]
                        feature_vec.append(vec)
                        num_phonemes += 1
            feature_vec = np.array(feature_vec, dtype="float32")
            feature_vec = np.sum(feature_vec, axis=0)
            feature_vec = np.divide(feature_vec, num_phonemes)
            print('Phonetic text: {0}'.format(messages_phonetic))
            print('Embedding: {0}'.format(feature_vec))
            return feature_vec
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_feature_syllable: {0}'.format(e))
            return None

    def get_frequency_phoneme(self, messages):
        try:
            messages_phonetic = []
            model = self.phoneme_embedding
            index2phoneme = list(model.wv.index2word)
            num_features = len(index2phoneme)
            feature_vec = np.zeros(num_features, dtype="float32")
            list_syllable = [token['syllables'] for token in self.ta.tagger(messages) if token['syllables'] is not None]
            for syllable in list_syllable:
                for s in syllable:
                    syllable_phonetic = self.epi.transliterate(s, normpunc=True)
                    messages_phonetic.append(syllable_phonetic)
                    if syllable_phonetic in index2phoneme:
                        index = index2phoneme.index(syllable_phonetic)
                        value = feature_vec[index]
                        feature_vec[index] = value + 1
            print('Phonetic text: {0}'.format(messages_phonetic))
            print('Frequency: {0}'.format(feature_vec))
            return feature_vec
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_frequency_phoneme: {0}'.format(e))
            return None

    def get_feature_phoneme(self, messages, one=False):
        try:
            messages_phonetic = None
            model = self.phoneme_embedding
            num_features = model.vector_size
            index2phoneme_set = set(model.wv.index2word)
            size = 1
            feature_vec = []
            list_syllable = [token['syllables'] for token in self.ta.tagger(messages) if token['syllables'] is not None]
            if one:
                try:
                    first_syllable = str(list_syllable[0][0])
                    first_syllable = first_syllable[0] if (first_syllable is not None) and (len(first_syllable) > 0) else ''
                    syllable_phonetic = self.epi.transliterate(first_syllable)
                    messages_phonetic = first_syllable
                    if syllable_phonetic in index2phoneme_set:
                        vec = model.wv[syllable_phonetic]
                        feature_vec.append(vec)
                    else:
                        feature_vec.append(np.zeros(num_features, dtype="float32"))
                except Exception as e_epi:
                    print('Error transliterate: {0}'.format(e_epi))
                    pass
            else:
                list_phoneme = self.epi.trans_list(messages)
                messages_phonetic = list_phoneme
                size = len(list_phoneme)
                for phoneme in list_phoneme:
                    if phoneme in index2phoneme_set:
                        vec = model.wv[phoneme]
                        feature_vec.append(vec)
                    else:
                        feature_vec.append(np.zeros(num_features, dtype="float32"))

            feature_vec = np.array(feature_vec, dtype="float32")
            feature_vec = np.sum(feature_vec, axis=0)
            feature_vec = np.divide(feature_vec, size)
            print('Phonetic text: {0}'.format(messages_phonetic))
            print('Embedding: {0}'.format(feature_vec))
            return feature_vec
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_feature_phoneme: {0}'.format(e))
            return None
