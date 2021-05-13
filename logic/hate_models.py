import os
import pickle
from sklearn import preprocessing
from tqdm import tqdm
from logic.data_transformation import DataTransformation
from logic.feature_extraction import FeatureExtraction
from logic.text_analysis import TextAnalysis
from root import DIR_OUTPUT, DIR_MODELS


class HateModels(object):

    def __init__(self, lang: str = 'es', name_model: str = None,
                 dataset: str = 'pan21-author-profiling-test-without-gold'):
        self.lang = lang
        self.ta = TextAnalysis(lang=lang)
        self.features = FeatureExtraction(lang=lang, text_analysis=self.ta)
        self.test = DataTransformation(dataset=dataset, lang=lang, type_data='test').get_data()
        file_model = '{0}{1}.pkl'.format(DIR_MODELS, name_model)
        self.clf = pickle.load(open(file_model, 'rb'))

    def run(self, type_features: list = [1, 1, 1, 1]):
        try:
            out = []
            print('Predicting users ...')
            count_one = 0
            count_zero = 0
            for user, cont in tqdm(self.test.items()):
                x_test = self.ta.clean_text(cont, stopwords=False)
                x_test = [list(self.features.get_features(x_test, type_features))]
                # x_test = preprocessing.normalize(x_test, norm='l2')
                predict = int(self.clf.predict(x_test)[0])
                if predict == 1:
                    count_one += 1
                else:
                    count_zero += 1
                out.append({'id': user, 'lang': self.lang, 'type': predict})
            print('Statistical result:\n# Ones: {0}\n# Zeros: {1}'.format(count_one, count_zero))
            # make files
            print('Generating files ...')
            path = '{0}{1}{2}'.format(DIR_OUTPUT, self.lang, os.sep)
            for row in tqdm(out):
                path_file = '{0}{1}{2}'.format(path, row['id'], '.xml')
                file = open(path_file, "w")
                text = '<author id="{0}" lang="{1}" type="{2}"/>'.format(row['id'], row['lang'], row['type'])
                file.writelines(text)
                file.close()
        except Exception as e:
            print('Error baseline: {0}'.format(e))

    def testing_model(self, cont: str = '', type_features: list = [1, 1, 1, 1]):
        x_test = self.ta.clean_text(cont, stopwords=False)
        x_test = [list(self.features.get_features(x_test, type_features))]
        # x_test = preprocessing.normalize(x_test, norm='l2')
        predict = int(self.clf.predict(x_test)[0])
        print('Predict: {0}'.format(predict))


if __name__ == "__main__":
    tm = HateModels(lang='es', name_model='hate_randomforest_es',
                    dataset='pan21-author-profiling-test-without-gold')
    cont = tm.ta.transformer_file(file='9151a34f406a463711a1f5e61e80a219.xml')
    tm.testing_model(cont=cont['9151a34f406a463711a1f5e61e80a219'], type_features=[1, 1, 1, 1])