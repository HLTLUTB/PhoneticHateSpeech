from logic.training_models import TrainModels

# L: Lexical, S:Syllable, F: Frequency Phoneme, P: All Phoneme
tm = TrainModels(lang='en', iteration=10, fold=10,
                 dataset='pan21-author-profiling-training-2021-03-14')
tm.run(type_features=[1, 1, 1, 1])