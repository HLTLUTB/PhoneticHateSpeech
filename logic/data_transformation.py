import os
from os import listdir
from os.path import isfile
from os import scandir
import xml.etree.ElementTree as ET
from sys import path
from root import DIR_INPUT


class DataTransformation(object):

    def __init__(self, dataset: str = 'pan21-author-profiling-training-2021-03-14',
                 lang: str = 'es', type_data: str = 'train'):
        self.dataset = dataset
        self.type_data = type_data
        self.path_dir = '{0}{1}{2}{1}{3}{1}'.format(DIR_INPUT, os.sep, dataset, lang)

    def get_data(self):
        out_put = []
        # Get all the names of the files in the path
        user_tweets = {}
        truth_file = ''
        # r=root, d=directories, f = files
        for r, d, f in os.walk(self.path_dir):
            for file in f:
                if file.endswith(".xml"):
                    tree = ET.parse('{0}{1}'.format(self.path_dir, file))
                    root = tree.getroot()
                    list_content = [i.text for i in root.iter('document')]
                    content = '\n'.join(list_content)
                    user_tweets[file.replace('.xml', '')] = content
                elif file.endswith(".txt"):
                    truth_file = file

        if self.type_data == 'train':
            with open(self.path_dir + truth_file, 'r+', encoding="utf-8") as file:
                for line in file:
                    entry = line.split(':::')
                    user = entry[0]
                    value = int(entry[1])
                    if user in user_tweets:
                        if value == 1:
                            out_put.append({'user': user, 'content': user_tweets[user], 'value': 1})
                        else:
                            out_put.append({'user': user, 'content': user_tweets[user], 'value': 0})

            return out_put
        else:
            return user_tweets


if __name__ == '__main__':
    dt = DataTransformation(dataset='pan21-author-profiling-training-2021-03-14',
                            lang='es', type_data='train')
    print(dt.get_data())