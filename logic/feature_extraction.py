import re
import sys
import epitran
from nltk import TweetTokenizer
from scipy.stats import kurtosis, skew
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin

from logic.linguistic_senticnet import LinguisticSenticNet
from logic.text_analysis import TextAnalysis
from logic.utils import Utils
from logic.lexical_features import lexical_es, lexical_en
from root import DIR_EMBEDDING


class FeatureExtraction(BaseEstimator, TransformerMixin):

    def __init__(self, lang='es', text_analysis=None):
        try:
            ta = None
            if text_analysis is None:
                ta = TextAnalysis(lang=lang)
            else:
                ta = text_analysis
            self.ta = ta
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
            self.lexical = lexical_es if lang == 'es' else lexical_en
            self.lsn = LinguisticSenticNet(text_analysis=self.ta)
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

    def get_features(self, messages: str, type_features: list = [1, 1, 1, 1]):
        try:
            # L: Lexical, S:Syllable, F: Frequency Phoneme, P: All Phoneme
            syllable_features = list(abs(self.get_feature_syllable(messages))) if type_features[0] else []
            phoneme_frequency = list(abs(self.get_frequency_phoneme(messages))) if type_features[1] else []
            all_phoneme = list(abs(self.get_feature_phoneme(messages))) if type_features[2] else []
            lexical_features = list(abs(self.get_features_lexical(messages))) if type_features[3] else []
            features = lexical_features + syllable_features + all_phoneme + phoneme_frequency
            result = np.array(features, dtype=np.float32)
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
            # print('Phonetic text: {0}'.format(messages_phonetic))
            # print('Embedding: {0}'.format(feature_vec))
            return feature_vec
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_feature_syllable: {0}'.format(e))
            return None

    def get_frequency_phoneme(self, messages):
        try:
            messages_phonetic = []
            total_freq = 1
            model = self.syllable_embedding
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
                        total_freq += feature_vec[index]
            # print('Phonetic text: {0}'.format(messages_phonetic))
            # print('Frequency: {0}'.format(feature_vec))
            return feature_vec / total_freq
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
            # print('Phonetic text: {0}'.format(messages_phonetic))
            # print('Embedding: {0}'.format(feature_vec))
            return feature_vec
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_feature_phoneme: {0}'.format(e))
            return None

    def get_features_lexical(self, message):
        result = None
        try:
            lexical = self.lexical
            text_tokenizer = TweetTokenizer()
            tags = ('mention', 'url', 'hashtag', 'emoji', 'rt')
            vector = dict()
            vector['plarity'] = float(self.lsn.polarity_text(text=message)['polarity_value'])
            tokens_text = text_tokenizer.tokenize(message)
            if len(tokens_text) > 0:
                vector['weighted_position'], vector['weighted_normalized'] = self.weighted_position(tokens_text)

                vector['label_mention'] = float(sum(1 for word in tokens_text if word == 'mention'))
                vector['label_url'] = float(sum(1 for word in tokens_text if word == 'url'))
                vector['label_hashtag'] = float(sum(1 for word in tokens_text if word == 'hashtag'))
                vector['label_emoji'] = float(sum(1 for word in tokens_text if word == 'emoji'))
                vector['label_retweets'] = float(sum(1 for word in tokens_text if word == 'rt'))

                vector['lexical_diversity'] = self.lexical_diversity(message)

                label_word = vector['label_mention'] + vector['label_url'] + vector['label_hashtag']
                label_word = label_word + vector['label_emoji'] + vector['label_retweets']
                vector['label_word'] = float(len(tokens_text) - label_word)

                vector['first_person_singular'] = float(
                    sum(1 for word in tokens_text if word in lexical['first_person_singular']))
                vector['second_person_singular'] = float(
                    sum(1 for word in tokens_text if word in lexical['second_person_singular']))
                vector['third_person_singular'] = float(
                    sum(1 for word in tokens_text if word in lexical['third_person_singular']))
                vector['first_person_plurar'] = float(
                    sum(1 for word in tokens_text if word in lexical['first_person_plurar']))
                vector['second_person_plurar'] = float(
                    sum(1 for word in tokens_text if word in lexical['second_person_plurar']))
                vector['third_person_plurar'] = float(
                    sum(1 for word in tokens_text if word in lexical['third_person_plurar']))

                vector['avg_word'] = np.nanmean([len(word) for word in tokens_text if word not in tags])
                vector['avg_word'] = vector['avg_word'] if not np.isnan(vector['avg_word']) else 0.0
                vector['avg_word'] = round(vector['avg_word'], 4)

                vector['kur_word'] = kurtosis([len(word) for word in tokens_text if word not in tags])
                vector['kur_word'] = vector['kur_word'] if not np.isnan(vector['kur_word']) else 0.0
                vector['kur_word'] = round(vector['kur_word'], 4)

                vector['skew_word'] = skew(np.array([len(word) for word in tokens_text if word not in tags]))
                vector['skew_word'] = vector['skew_word'] if not np.isnan(vector['skew_word']) else 0.0
                vector['skew_word'] = round(vector['skew_word'], 4)

                # adverbios
                vector['adverb_neg'] = sum(1 for word in tokens_text if word in lexical['adverb_neg'])
                vector['adverb_neg'] = float(vector['adverb_neg'])

                vector['adverb_time'] = sum(1 for word in tokens_text if word in lexical['adverb_time'])
                vector['adverb_time'] = float(vector['adverb_time'])

                vector['adverb_place'] = sum(1 for word in tokens_text if word in lexical['adverb_place'])
                vector['adverb_place'] = float(vector['adverb_place'])

                vector['adverb_mode'] = sum(1 for word in tokens_text if word in lexical['adverb_mode'])
                vector['adverb_mode'] = float(vector['adverb_mode'])

                vector['adverb_cant'] = sum(1 for word in tokens_text if word in lexical['adverb_cant'])
                vector['adverb_cant'] = float(vector['adverb_cant'])

                vector['adverb_all'] = float(vector['adverb_neg'] + vector['adverb_time'] + vector['adverb_place'])
                vector['adverb_all'] = float(vector['adverb_all'] + vector['adverb_mode'] + vector['adverb_cant'])

                vector['adjetives_neg'] = sum(1 for word in tokens_text if word in lexical['adjetives_neg'])
                vector['adjetives_neg'] = float(vector['adjetives_neg'])

                vector['adjetives_pos'] = sum(1 for word in tokens_text if word in lexical['adjetives_pos'])
                vector['adjetives_pos'] = float(vector['adjetives_pos'])

                vector['who_general'] = sum(1 for word in tokens_text if word in lexical['who_general'])
                vector['who_general'] = float(vector['who_general'])

                vector['who_male'] = sum(1 for word in tokens_text if word in lexical['who_male'])
                vector['who_male'] = float(vector['who_male'])

                vector['who_female'] = sum(1 for word in tokens_text if word in lexical['who_female'])
                vector['who_female'] = float(vector['who_female'])

                vector['hate'] = sum(1 for word in tokens_text if word in lexical['hate'])
                vector['hate'] = float(vector['hate'])

                vector['noun'] = self.pos_frequency(message)['NOUN'] * 0.8
                vector['verb'] = self.pos_frequency(message)['VERB'] * 0.5
                vector['adj'] = self.pos_frequency(message)['ADJ'] * 0.4
                vector['pos_others'] = self.pos_frequency(message)['ANOTHER'] * 0.1

                result = np.array(list(vector.values()))
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_lexical_features: {0}'.format(e))
        return result

    @staticmethod
    def lexical_diversity(text):
        result = None
        try:
            text_out = re.sub(r"[\U00010000-\U0010ffff]", '', text)
            text_out = re.sub(
                r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+'
                r'|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
                '', text_out)
            text_out = text_out.lower()
            result = round((len(set(text_out)) / len(text_out)), 4)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error lexical_diversity: {0}'.format(e))
        return result

    @staticmethod
    def weighted_position(tokens_text):
        result = None
        try:
            size = len(tokens_text)
            weighted_words = 0.0
            weighted_normalized = 0.0
            for w in tokens_text:
                weighted_words += 1 / (1 + tokens_text.index(w))
                weighted_normalized += (1 + tokens_text.index(w)) / size
            result = (weighted_words, weighted_normalized)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error weighted_position: {0}'.format(e))
        return result

    def pos_frequency(self, text):
        dict_token = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ANOTHER': 0}
        try:
            doc = self.ta.tagger(text)
            for token in doc:
                if token['pos'] == 'NOUN':
                    value = dict_token['NOUN']
                    dict_token['NOUN'] = value + 1
                elif token['pos'] == 'VERB':
                    value = dict_token['VERB']
                    dict_token['VERB'] = value + 1
                elif token['pos'] == 'ADJ':
                    value = dict_token['ADJ']
                    dict_token['ADJ'] = value + 1
                else:
                    value = dict_token['ANOTHER']
                    dict_token['ANOTHER'] = value + 1
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error pos_frequency: {0}'.format(e))
        return dict_token
