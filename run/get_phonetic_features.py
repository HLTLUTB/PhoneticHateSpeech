from logic.text_analysis import TextAnalysis
from logic.feature_extraction import FeatureExtraction

lang = 'es'
setting = {'sep': ';', 'url': True, 'mention': True, 'emoji': False,
           'hashtag': True, 'lemmatizer': False, 'stopwords': True}

text = 'Vine a ver si se habían muerto en el apocalipsis y aún sigo leyendo sus tuits bien de la chingada'

ta = TextAnalysis(lang=lang)
fe = FeatureExtraction(lang=lang, text_analysis=ta)

text_clean = ta.clean_text(text)

print('Original text'.format(text_clean))

print('Get Syllabification Features')
text_syllable = fe.get_feature_syllable(text_clean)
print(text_syllable)

print('Get One Phoneme Features')
fe.get_feature_phoneme(text_clean, one=True)

print('Get All Phoneme Features')
fe.get_feature_phoneme(text_clean, one=False)

print('Get frequency Phoneme Features')
fe.get_frequency_phoneme(text_clean)
