import os
ROOT = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = "{0}{1}data".format(ROOT, os.sep)
DIR_EMBEDDING = "{0}{1}embedding{1}".format(DIR_DATA, os.sep)
DIR_TRAIN = "{0}{1}train{1}{2}{1}".format(DIR_DATA, os.sep, 'pan21-author-profiling-training-2021-03-14')
DIR_TEST = "{0}{1}test{1}{2}{1}".format(DIR_DATA, os.sep, 'pan21-author-profiling-test-without-gold')
DIR_OUTPUT = "{0}{1}output{1}".format(DIR_DATA, os.sep)
DIR_MODELS = "{0}{1}models{1}".format(DIR_DATA, os.sep)
DIR_LEXICON = "{0}{1}lexicon{1}".format(DIR_DATA, os.sep)
DATA_BABEL = 'data.babel.data_'
