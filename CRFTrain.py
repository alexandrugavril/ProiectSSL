import nltk
import os
import untangle
import pickle
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

class CRFTrain:
    def __init__(self, *args, **kwargs):
        if 'pickle' in kwargs:
            pickleFile = kwargs.get('pickle', None)
            print("Reading from pickle:" + str(pickleFile))
            self.full_set = self.load_tags(pickleFile)
        elif 'folder' in kwargs:
            folderFile = kwargs.get('folder', None)
            if(folderFile[-1] != '\\'):
                folderFile = folderFile + '\\'
            print("Reading from folder:" + str(folderFile))
            annotations = self.get_all_annotations_in_folder(folderFile)
            self.full_set = []
            for ann in annotations:
                self.full_set.append(self.get_pos_tagging(ann))
        else:
            raise Exception("Not a valid file!")

    def word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][2]

        features = {
            'pos' : postag,
            'word' : word,
            'postag': postag[:2],
            'BOS' : False,
            'EOS' : False
        }
        if i > 0:
            features.update(
                    {
                        'word-1' : sent[i-1][0],
                        #'type-1' : sent[i-1][1],
                        'pos-1' : sent[i-1][2],

                    }
            )
        else:
            features['BOS'] = True

        if i > 1:
            features.update(
                    {
                        'word-2' : sent[i-2][0],
                        #'type-2' : sent[i-2][1],
                        'pos-2' : sent[i-2][2]
                    }
            )
        if i < len(sent)-1:
            features.update(
                    {
                        'word+1' : sent[i+1][0],
                        #'type+1' : sent[i+1][1],
                        'pos+1' : sent[i+1][2]

                    }
            )
        else:
            features['EOS'] = True
        if i < len(sent)-2:
            features.update(
                    {
                        'word+2' : sent[i+2][0],
                        #'type+2' : sent[i+2][1],
                        'pos+2' : sent[i+2][2],

                    }
            )
        return features

    def test_set_test(self):
        if(self.trained):
            labels = list(self.crf.classes_)
            labels.remove("N")
            print("Starting Testing on " + str(len(self.x_test)) + " sentences...")
            y_pred = self.crf.predict(self.x_test)
            val = metrics.flat_f1_score(self.y_test, y_pred,
                              average='weighted', labels=labels)
            return val
        else:
            print("CRF was not trained!")
            return -1

    def get_annotations_for_sentence(self,sentence):
        tokens = nltk.word_tokenize(sentence)
        pos_tagged_tokens = nltk.pos_tag(tokens)

        final_tokens = []
        for (word, pos_tag) in pos_tagged_tokens:
            final_tokens.append((word, "N", pos_tag))
        features = self.sent2features(final_tokens)

        ann_sent = zip(tokens, self.crf.predict_single(features))
        print(ann_sent)
        return ann_sent

    def train(self, test_size=0.2, max_iterations=100, fold5valid=False ):
        full_set_labels = []
        for sent in self.full_set:
            set_lab = []
            for word in sent:
                    set_lab.append(word[1])
            full_set_labels.append(set_lab)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.full_set,
                                                                                full_set_labels,
                                                                                test_size=test_size,
                                                                                random_state=0)
        self.x_train = [self.sent2features(s) for s in self.x_train]
        self.x_test = [self.sent2features(s) for s in self.x_test]

        print("Starting Training on " + str(len(self.x_train)) + " sentences...")
        batch_size = len(self.x_train) / 5
        scores = []
        if fold5valid:
            for i in range(5):
                self.crf = sklearn_crfsuite.CRF(
                    algorithm='lbfgs',
                    c1=0.1,
                    c2=0.1,
                    max_iterations=max_iterations,
                    all_possible_transitions=True
                    )

                indices = range(i*batch_size, (i+1)*batch_size)
                train_batch = [i for j, i in enumerate(self.x_train) if j not in indices]
                test_batch = [i for j, i in enumerate(self.x_train) if j in indices]

                train_labels = [i for j, i in enumerate(self.y_train) if j not in indices]
                test_labels = [i for j, i in enumerate(self.y_train) if j in indices]

                self.crf.fit(train_batch, train_labels)
                labels = list(self.crf.classes_)
                labels.remove("N")
                y_pred = self.crf.predict(test_batch)
                val = metrics.flat_f1_score(test_labels, y_pred,
                                  average='weighted', labels=labels)
                scores.append(val)

            import numpy
            scores = numpy.array(scores)
            print("5 Fold scores:" + str(scores))
            f1score = scores.mean(), scores.std() * 2
            print("F1 Score: %0.2f (+/- %0.2f)" % (f1score))

            #self.crf.fit(self.x_train, self.y_train)
            self.trained = True
            print("Finished training...")
            return f1score

        else:
            import scipy
            from sklearn.metrics import make_scorer
            from sklearn.grid_search import RandomizedSearchCV

            self.crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                all_possible_transitions=True
            )
            params_space = {
                'c1': scipy.stats.expon(scale=0.5),
                'c2': scipy.stats.expon(scale=0.05),
                'max_iterations': range(20,100),
            }
            self.crf.fit(self.x_train, self.y_train)
            labels = list(self.crf.classes_)
            labels.remove('N')
            # use the same metric for evaluation
            f1_scorer = make_scorer(metrics.flat_f1_score,
                                    average='weighted', labels=labels)

            # search
            rs = RandomizedSearchCV(self.crf, params_space,
                                    cv=5,
                                    verbose=1,
                                    n_jobs=-1,
                                    n_iter=100,
                                    scoring=f1_scorer)
            rs.fit(self.x_train, self.y_train)
            # crf = rs.best_estimator_
            print('best params:', rs.best_params_)
            print('best CV score:', rs.best_score_)
            print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
            self.trained = True
            return rs

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        print(sent[0])
        return [label for (token, postag, label) in sent]

    def sent2tokens(self, sent):
        return [token for token, postag, label in sent]

    def get_tags_from_gate_xml(self, gate_file):
        doc = untangle.parse(gate_file)
        sentence = doc.GateDocument.TextWithNodes.cdata
        annotations = doc.GateDocument.AnnotationSet[0].Annotation
        annotations = sorted(annotations, key=lambda x: int(x['StartNode']))

        result = []
        for an in annotations:
           result.append((sentence[int(an['StartNode']):int(an['EndNode'])], an['Type']))

        return result

    def get_all_annotations_in_folder(self, folder):
        annotations = []
        for file in os.listdir(folder):
            if 'DS_Store' in file:
                continue
            try:
                annotations.append(self.get_tags_from_gate_xml('%s%s' % (folder, file)))
            except Exception as e:
                print(file + " cannot be parsed!")
        return annotations

    def get_pos_tagging(self, tokens):
        words = [i for (i,_) in tokens]
        tags = [j for (_,j) in tokens]
        poss = [j for (_,j) in nltk.pos_tag(words)]
        return zip(words, tags, poss)

    def save_tags(self):
        pickle.dump(self.full_set, open( "save.p", "wb" ))

    def load_tags(self, file):
        return pickle.load(open(file, "rb" ))
