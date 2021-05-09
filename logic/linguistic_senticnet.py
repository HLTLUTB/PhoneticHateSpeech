import importlib
from logic.text_analysis import TextAnalysis
from logic.triggers import rules
from root import DATA_BABEL


class LinguisticSenticNet:
    """
    :Date: 2018-05-12
    :Version: 1.0
    :Author: Edwin Puertas - Pontificia Universidad Javeriana, Bogotá
    :Copyright: To be defined
    Simple API to use SenticNet 5.

    This class offers dependency analysis task to be performed.

    """
    def __init__(self, lang='es', text_analysis=None):
        try:
            data_module = importlib.import_module(DATA_BABEL + lang)
            self.data = data_module.senticnet
            self.triggers = rules
            if text_analysis is None:
                self.ta = TextAnalysis(lang=lang)
            else:
                self.ta = text_analysis
        except Exception as e:
            print('Error __init__: {0}'.format(e))

    def message_concept(self, text):
        """
        Return all the information about a text: semantics,
        sentics and polarity.
        """
        result = None
        try:
            list_words = []
            left_conjunct, right_conjunct = self.discourse_structures(text)
            if left_conjunct is not None or right_conjunct is not None:
                trace = []
                coordinated = self.coordinated(left_conjunct, right_conjunct)
                left = [i for i in coordinated['left']['trace']]
                right = [i for i in coordinated['right']['trace']]
                trace.extend(left)
                trace.extend(right)
                polarity_value = 0.0
                result = {'msg': text, 'polarity_label': coordinated['polarity_label'], 'polarity_value': polarity_value, 'trace': trace}
            else:
                result = self.polarity_text(text)
            list_tmp = result['trace']

            for item in list_tmp:
                if item['semantics'] is not None:
                    list_words.extend(item['semantics'])

            result['words'] = [str(i).replace('_', ' ') for i in list_words]
        except Exception as e:
            print('Error message_concept: {0}'.format(e))
        return result

    def concept(self, concept):
        """
        Return all the information about a concept: semantics,
        sentics and polarity.
        """
        result = {}
        try:
            result['polarity_value'] = self.polarity_value(concept)
            result['polarity_intense'] = self.polarity_intense(concept)
            result['moodtags'] = self.moodtags(concept)
            result['sentics'] = self.sentics(concept)
            result['semantics'] = self.semantics(concept)
        except Exception as e:
            print('Error concept: {0}'.format(e))
        return result

    def semantics(self, concept):
        """
        Return the semantics associated with a concept.
        """
        val = None
        try:
            if concept.find(' ') > 0:
                concept = concept.replace(" ", "_")

            if concept in self.data:
                val = self.data[concept][8:]
        except Exception as e:
            print('Error semantics: {0}'.format(e))
        return val

    def sentics(self, concept):
        """
        Return sentics of a concept.
        """
        sentics = None
        try:
            if concept.find(' ') > 0:
                concept = concept.replace(" ", "_")

            if concept in self.data:
                concept_info = self.data[concept]
                sentics = {"pleasantness": concept_info[0],
                           "attention": concept_info[1],
                           "sensitivity": concept_info[2],
                           "aptitude": concept_info[3]}
        except Exception as e:
            print('Error sentics: {0}'.format(e))
        return sentics

    def polarity_value(self, concept):
        """
        Return the polarity value of a concept.
        """
        val = 0.0
        try:
            if concept.find(' ') > 0:
                concept = concept.replace(" ", "_")

            if concept in self.data:
                val = list(self.data[concept])
                val = val[6]
        except Exception as e:
            print('Error polarity_value: {0}'.format(e))
        return val

    def pleasantness_value(self, concept):
        """
        Return the polarity value of a concept.
        """
        val = 0.0
        try:
            if concept.find(' ') > 0:
                concept = concept.replace(" ", "_")

            if concept in self.data:
                val = self.data[concept][0]
        except Exception as e:
            print('Error pleasantness_value: {0}'.format(e))
        return val

    def attention_value(self, concept):
        """
        Return the polarity value of a concept.
        """
        val = 0.0
        try:
            if concept.find(' ') > 0:
                concept = concept.replace(" ", "_")

            if concept in self.data:
                val = self.data[concept][1]
        except Exception as e:
            print('Error attention_value: {0}'.format(e))
        return val

    def sensitivity_value(self, concept):
        """
        Return the polarity value of a concept.
        """
        val = 0.0
        try:
            if concept.find(' ') > 0:
                concept = concept.replace(" ", "_")

            if concept in self.data:
                val = self.data[concept][2]
        except Exception as e:
            print('Error sensitivity_value: {0}'.format(e))
        return val

    def aptitude_value(self, concept):
        """
        Return the polarity value of a concept.
        """
        val = 0.0
        try:
            if concept.find(' ') > 0:
                concept = concept.replace(" ", "_")

            if concept in self.data:
                val = self.data[concept][3]
        except Exception as e:
            print('Error aptitude_value: {0}'.format(e))
        return val

    def polarity_intense(self, concept):
        """
        Return the polarity intense of a concept.
        """
        val = None
        try:
            if concept.find(' ') > 0:
                concept = concept.replace(" ", "_")

            if concept in self.data:
                val = self.data[concept][7]
        except Exception as e:
            print('Error polarity_intense: {0}'.format(e))
        return val

    def moodtags(self, concept):
        """
        Return the moodtags of a concept.
        """
        val = None
        try:
            if concept.find(' ') > 0:
                concept = concept.replace(" ", "_")

            if concept in self.data:
                val = self.data[concept][4:6]
        except Exception as e:
            print('Error moodtags: {0}'.format(e))
        return val

    def polarity_text(self, text):
        result = None
        try:
            status_msg = 'NEUTRAL'
            polarity_NOUN = 0.0
            polarity_ADV = 0.0
            polarity_VERB = 0.0
            polarity = 0.0
            trace = []
            chunks = []
            count_chunks = 1
            dict_chunks = self.ta.syntax_patterns(text)
            for type_chunk, list_chunk in dict_chunks.items():
                if len(list_chunk) > 0:
                    for chunk in list_chunk:
                        polarity_value = 0.0
                        if self.sentics(chunk) is None:
                            chunk = self.ta.clean_text(chunk)
                            if self.sentics(chunk) is not None and chunk not in chunks:
                                chunks.append(chunk)
                                dict_trace = {'text': chunk}
                                dict_trace.update(self.concept(chunk))
                                trace.append(dict_trace)
                                polarity_value = float(self.polarity_value(chunk)) 
                                count_chunks += 1
                        else:
                            if chunk not in chunks:
                                chunks.append(chunk)
                                dict_trace = {'text': chunk}
                                dict_trace.update(self.concept(chunk))
                                trace.append(dict_trace)
                                polarity_value = float(self.polarity_value(chunk))
                                count_chunks += 1
                                
                        if type_chunk == 'NOUN':
                            polarity_NOUN += polarity_value
                        elif type_chunk == 'VERB':
                            polarity_VERB += (1 * polarity_value)
                        elif type_chunk == 'ADV':
                            if self.polarity_inversion(chunk):
                                polarity_ADV += (-2 * polarity_value)
                            else:
                                polarity_ADV += (2 * polarity_value)

            polarity = (polarity_NOUN + polarity_VERB + polarity_ADV)
            polarity = round((polarity / count_chunks ), 3)
            if polarity > 0.10:
                status_msg = 'POSITIVE'
            elif polarity < -0.10:
                status_msg = 'NEGATIVE'
            else:
                status_msg = 'NEUTRAL'
            result = {'msg': text, 'polarity_label': status_msg, 'polarity_value': polarity, 'trace': trace}
        except Exception as e:
            print('Error polarity_text: {0}'.format(e))
        return result

    def polarity_inversion(self, chunk):
        result = False
        try:
            for row in self.triggers['negative']:
                value = chunk.find(row)
                if value > -1:
                    result = True
        except Exception as e:
            print('Error polarity_inversion: {0}'.format(e))
        return result

    def discourse_structures(self, text):
        resp = None
        try:
            text_len = len(text)
            left_conjunct = None
            right_conjunct = None
            for trig in self.triggers['discourse']:
                value = text.find(trig)
                size_word = len(trig)
                if value > -1:
                    if (value >=0) and (value <= size_word):
                        comma = text.find(',')
                        if comma > -1:
                            left_conjunct = text[0:comma]
                            right_conjunct = text[comma: text_len]
                            break
                    else:
                        left_conjunct = text[0:value]
                        right_conjunct = text[value: text_len]
                        break
            resp = left_conjunct, right_conjunct
        except Exception as e:
            print('Error discourse_structures: {0}'.format(e))
        return resp

    def coordinated(self, left_conjunct, right_conjunct):
        try:
            polarity_label = 'NEUTRAL'
            polarity_value = 0.0
            left = 0.0
            right = 0.0
            left_result = None
            right_result = None
            if left_conjunct is not None:
                left_result = self.polarity_text(left_conjunct)
                left = left_result['polarity_value']
            if right_conjunct is not None:
                right_result = self.polarity_text(right_conjunct)
                right = right_result['polarity_value']
            if (left > 0.0) and (right < 0.0):
                polarity_label = 'POSITIVE'
                polarity_value = left
            elif (left < 0.0 or left > 0.0 or left is None) and right > 0.0:
                polarity_label = 'POSITIVE'
                polarity_value = right
            elif left > 0.0 and right is None:
                polarity_label = 'NEGATIVE'
                polarity_value = -1.0
            elif left < 0.0 and right is None:
                polarity_label = 'POSITIVE'
                polarity_value = 1.0
            elif (left < 0.0 or left is None) and right < 0.0:
                polarity_label = 'NEGATIVE'
                polarity_value = right
            return {'polarity_label': polarity_label, 'polarity_value': polarity_value,
                    'left': left_result, 'right': right_result}
        except Exception as e:
            print('Error coordinated {0}'.format(e))
            return None


if __name__ == "__main__":
    text_ipsos = 'PORQUE NUNCA NOS HAN ATENDIDO DE LA MEJOR MANERA YA QUE SIEMPRE PONEN PEROS PARA CIERTAS COSAS EN ' \
                 'CUANTO A QUE NUNCA HABLAN CON LA VERDAD PARA LOS CRÉDITOS YA QUE PASAMOS 5 SOLICITUDES Y SIEMPRE ' \
                 'SALEN CON UN CUENTO DIFERENTE QUE LAS BOTARON QUE NO LA MANDARON, QUE NO ESTÁ EL QUE AUTORIZA, ' \
                 'SIEMPRE TIENEN QUE RECIBIR EL PEDIDO CUANDO ELLOS LO DICEN NO CUANDO UNO PUEDE SI NO QUE NO VUELVEN ' \
                 'A ATENDER, AMENAZAN A UNO DE QUITARLES EL SERVICIO.'

    lsn = LinguisticSenticNet()
    result = lsn.polarity_text(text=text_ipsos)
    print(result)