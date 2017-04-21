import nltk
import os
import sys
import untangle
import argparse
import pickle
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

class CRFTrain:
    def __init__(self, *args, **kwargs):
        if 'pickle' in kwargs:
            pickleFile = kwargs.get('pickle', None)
            print("Reading from pickle:" + str(pickleFile))
            self.full_set = self.load_tags(pickleFile)
        elif 'folder' in kwargs:
            folderFile = kwargs.get('folder', None)
            print("Reading from pickle:" + str(folderFile))
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

    def train(self, test_size=0.2):
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

        self.crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
        )
        self.crf.fit(self.x_train, self.y_train)
        self.trained = True
        print("Finished training...")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reads annotations and automatically tags them.')
    parser.add_argument('--folder', type=str, nargs='?', help="Annotations folder path")
    parser.add_argument('--pickle', type=str, nargs='?', help="Load saved pickle")
    args = parser.parse_args()

    if args.pickle is not None:
        crftrain = CRFTrain(pickle = args.pickle)
    elif args.folder is not None:
        crftrain = CRFTrain(folder = args.folder)
        crftrain.save_tags()
    else:
        parser.print_help()
        sys.exit(-1)

    crftrain.train(test_size=0.2)
    print("F1 Score: " + str(crftrain.test_set_test()))

    crftrain.get_annotations_for_sentence("When was Superman born")
    crftrain.get_annotations_for_sentence("When was the Battle of Gettysburg held")
    crftrain.get_annotations_for_sentence("When was the lord of the rings written")


