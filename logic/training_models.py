import datetime
import pickle
import time
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from logic.data_transformation import DataTransformation
from logic.classifiers import Classifiers
from logic.feature_extraction import FeatureExtraction
from logic.text_analysis import TextAnalysis
from root import DIR_MODELS


class TrainModels(object):

    def __init__(self, lang: str = 'es', iteration: int = 10, fold: int = 10,
                 dataset: str = 'pan21-author-profiling-training-2021-03-14'):
        self.lang = lang
        self.iteration = iteration
        self.fold = fold
        self.classifiers = Classifiers.dict_classifiers
        self.ta = TextAnalysis(lang=lang)
        self.features = FeatureExtraction(lang=lang, text_analysis=self.ta)
        self.data = DataTransformation(dataset=dataset, lang=lang).get_data()

    def run(self, type_features: list = [1, 1, 1, 1]):
        try:
            date_file = datetime.datetime.now().strftime("%Y-%m-%d")
            print('***Clean data training')
            x = [self.ta.clean_text(row['content'], stopwords=False) for row in tqdm(self.data)]
            y = np.array([row['value'] for row in self.data], dtype=np.int)

            print('***Get training features')
            x = [self.features.get_features(msg, type_features) for msg in tqdm(x)]

            cv = StratifiedShuffleSplit(n_splits=self.fold, test_size=0.30, random_state=42)

            best = 0.0
            best_clf = None
            name_best = None
            for clf_name, clf_ in self.classifiers.items():
                classifier_name = clf_name
                clf = clf_
                start_time = time.time()
                print('**Training {0} ...'.format(classifier_name))
                scores_acc = []
                scores_recall = []
                scores_f1 = []
                for i in range(1, self.iteration + 1):
                    clf.fit(x, y)
                    accuracy = cross_val_score(clf, x, y, cv=cv)
                    scores_acc.append(accuracy)
                    recall = cross_val_score(clf, x, y, cv=cv, scoring='recall')
                    scores_recall.append(recall)
                    f1 = cross_val_score(clf, x, y, cv=cv, scoring='f1')
                    scores_f1.append(f1)

                # Calculated Time processing
                t_sec = round(time.time() - start_time)
                (t_min, t_sec) = divmod(t_sec, 60)
                (t_hour, t_min) = divmod(t_min, 60)
                time_processing = '{} hour:{} min:{} sec'.format(t_hour, t_min, t_sec)

                mean_score_acc = np.mean(scores_acc)
                std_score_acc = np.std(scores_acc)
                mean_score_recall = np.mean(scores_recall)
                std_score_recall = np.std(scores_recall)
                mean_score_f1 = np.mean(scores_f1)
                std_score_f1 = np.std(scores_f1)
                # Calculated statistical
                print('-' * 40)
                print("Results for {} classifier".format(classifier_name))
                print("Mean Accuracy: %0.3f (+/- %0.3f)" % (mean_score_acc, std_score_acc))
                print("Mean Recall: %0.3f (+/- %0.3f)" % (mean_score_recall, std_score_recall))
                print("Mean F1: %0.3f (+/- %0.3f)" % (mean_score_f1, std_score_f1))
                print("Time processing: {0}".format(time_processing))
                print('-' * 40)
                if mean_score_acc > best:
                    best = mean_score_acc
                    best_clf = clf
                    name_best = classifier_name
            file_model = '{0}hate_model_{1}.pkl'.format(DIR_MODELS, self.lang)
            with open(file_model, 'wb') as file:
                pickle.dump(best_clf, file)
                print('Best classifier is {0} with Accuracy: {1}'.format(name_best, best))
        except Exception as e:
            print('Error baseline: {0}'.format(e))


if __name__ == "__main__":
    tm = TrainModels(lang='es', iteration=10, fold=10,
                     dataset='pan21-author-profiling-training-2021-03-14')
    tm.run(type_features=[1, 1, 1, 1])