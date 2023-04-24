
import numpy
# label_map = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '', 'del', ' ']

# returns numpy array of probabilities for each output
def postprocess(out, labels, outputNames):
    # 0-25 for letters A-Z, 26-27 for empty character ('') and del
    # output_probabilities = numpy.zeros(shape=(len(out[outputNames[1]]), 28))
    output_probabilities = numpy.zeros(shape=(len(out[1]), 28))

    # for output_idx, output in enumerate(out[outputNames[1]]):
    for output_idx, output in enumerate(out[1]):
        for letter_idx, prob in output.items():
            output_probabilities[output_idx][letter_idx] = prob

    return output_probabilities

