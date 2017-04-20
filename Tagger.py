import nltk
import os
import sys
import untangle
import argparse
import pickle
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

def word2features(sent, i):
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
                    'type-1' : sent[i-1][1],
                    'pos-1' : sent[i-1][2],

                }
        )
    else:
        features['BOS'] = True

    if i > 1:
        features.update(
                {
                    'word-2' : sent[i-2][0],
                    'type-2' : sent[i-2][1],
                    'pos-2' : sent[i-2][2]
                }
        )
    if i < len(sent)-1:
        features.update(
                {
                    'word+1' : sent[i+1][0],
                    'type+1' : sent[i+1][1],
                    'pos+1' : sent[i+1][2]

                }
        )
    else:
        features['EOS'] = True
    if i < len(sent)-2:
        features.update(
                {
                    'word+2' : sent[i+2][0],
                    'type+2' : sent[i+2][1],
                    'pos+2' : sent[i+2][2],

                }
        )
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    print(sent[0])
    return [label for (token, postag, label) in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def get_tags_from_gate_xml(gate_file):
    doc = untangle.parse(gate_file)
    sentence = doc.GateDocument.TextWithNodes.cdata
    annotations = doc.GateDocument.AnnotationSet[0].Annotation
    annotations = sorted(annotations, key=lambda x: int(x['StartNode']))

    result = []
    for an in annotations:
       result.append((sentence[int(an['StartNode']):int(an['EndNode'])], an['Type']))

    return result

def get_all_annotations_in_folder(folder):
    annotations = []
    for file in os.listdir(folder):
        if 'DS_Store' in file:
            continue
        try:
            annotations.append(get_tags_from_gate_xml('%s%s' % (folder, file)))
        except Exception as e:
            print(file + " cannot be parsed!")
    return annotations

def get_pos_tagging(tokens):
    words = [i for (i,_) in tokens]
    tags = [j for (_,j) in tokens]
    poss = [j for (_,j) in nltk.pos_tag(words)]
    return zip(words, tags, poss)

def save_tags(features):
    pickle.dump( features, open( "save.p", "wb" ))

def load_tags(file):
    return pickle.load(open( file, "rb" ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reads annotations and automatically tags them.')
    parser.add_argument('--folder', type=str, nargs='?', help="Annotations folder path")
    parser.add_argument('--pickle', type=str, nargs='?', help="Load saved pickle")
    args = parser.parse_args()

    if args.pickle is not None:
        print("Reading from pickle:" + args.pickle)
        full_set = load_tags(args.pickle)
    elif args.folder is not None:
        folderPath = args.folder
        print("Reading from folder:" + folderPath)
        if folderPath[-1] != '\\':
            folderPath = folderPath + '\\'
        annotations = get_all_annotations_in_folder(folderPath)
        full_set = []
        for ann in annotations:
            full_set.append(get_pos_tagging(ann))
        save_tags(full_set)

    else:
        parser.print_help()
        sys.exit(-1)

    full_set_labels = []
    for sent in full_set:
            set_lab = []
            for word in sent:
                set_lab.append(word[1])
            full_set_labels.append(set_lab)
    x_train, x_test, y_train, y_test = train_test_split(full_set, full_set_labels, test_size=0.2, random_state=0)
    x_train = [sent2features(s) for s in x_train]
    x_test = [sent2features(s) for s in x_test]

    print("Starting Training on " + str(len(x_train)) + " sentences...")

    crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
    )
    crf.fit(x_train, y_train)
    labels = list(crf.classes_)
    labels.remove("N")

    print("Starting Testing on " + str(len(x_test)) + " sentences...")
    y_pred = crf.predict(x_test)
    val = metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)
    print("F1 Score:" + str(val))