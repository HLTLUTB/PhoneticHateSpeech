from logic.hate_models import HateModels

tm = HateModels(lang='es', name_model='hate_model_es',
                dataset='pan21-author-profiling-test-without-gold')
tm.run(type_features=[1, 1, 1, 1])