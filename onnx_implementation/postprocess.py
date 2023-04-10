
import numpy
# label_map = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '', 'del', ' ']

def postprocess(out, labels, outputNames):
    # 0-25 for letters A-Z, 26-27 for empty character ('') and del
    output_probabilities = numpy.zeros(shape=28)
    for letter_idx, prob in out[outputNames[1]][0]:
        output_probabilities[letter_idx] = prob

    return output_probabilities

