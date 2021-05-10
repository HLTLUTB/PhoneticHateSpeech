# Phonetic Detection for Hate Speech Spreaders on Twitter

The main objective of the software is to determine whether its author spreads hate speech on Twitter.

## Preinstall Models
- pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
- pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.3.1/en_core_web_md-2.3.1.tar.gz
- pip install https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-2.3.1/es_core_news_sm-2.3.1.tar.gz
- pip install https://github.com/explosion/spacy-models/releases/download/es_core_news_md-2.3.1/es_core_news_md-2.3.1.tar.gz

## Train models
- ipython run/training_models_es.py
- ipython run/training_models_en.py

## Test models

- ipython run/hate_model_es.py
- ipython run/hate_model_en.py

## Results
- Accuracy Spanish model 0.73 
- Accuracy English model 0.60
- Features 
  - Lexical
  - Phoneme syllables embedding
  - Phoneme frequency
  - Phoneme embedding
  
## Team

- Edwin Puertas, PhDc | <epuerta@utb.edu.co>