import multiprocessing
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class Classifiers(object):

    cores = multiprocessing.cpu_count() - 1
    dict_classifiers = dict()
    # dict_classifiers['SVM'] = SVC(kernel='linear', C=0.5, probability=True)
    dict_classifiers['LogisticRegression'] = LogisticRegression(C=10, n_jobs=cores)
    dict_classifiers['RandomForest'] = RandomForestClassifier(n_jobs=cores)
    dict_classifiers['DecisionTree'] = DecisionTreeClassifier()
    dict_classifiers['Bagging'] = BaggingClassifier(n_estimators=20, random_state=42)
    dict_classifiers['GradientBoosting'] = GradientBoostingClassifier(n_estimators=20, random_state=7)
    dict_classifiers['AdaBoost'] = AdaBoostClassifier(n_estimators=20, random_state=7)