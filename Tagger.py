import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from CRFTrain import CRFTrain

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
    f1scores = []
    f1score = crftrain.train(test_size=0.2, max_iterations=100)
    #plt.plot(range(20,200), f1scores)
    #plt.savefig("F1Scores20.200Type.png")

    #print("F1 Score: " + str(crftrain.test_set_test()))d

    crftrain.get_annotations_for_sentence("When was Superman born")
    crftrain.get_annotations_for_sentence("When was the Battle of Gettysburg held")
    crftrain.get_annotations_for_sentence("When was the lord of the rings written")


