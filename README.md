# QMIND x Distributive x JKUAT
## Project Bravo - Live ASL Recognition

### Meet The Team
- Matthew Li
- Sammi Wang
    - 4th year Cognitive Science student
- Liam Salass
- Noah Waisbrod
- Hendrix Gryspeerdt
    - First year engineering student

### Required Libraries

- openCV
- MediaPipe
- numpy
- pandas
- process_image
- scipy
- matplotlib
- seaborn
- TensorFlow

### The Dataset

### The Models

#### ASL_model1
- image input: numpy.ndarray with shape (28, 28)
- returns: numpy array with shape (, 25)
- trains with labels as integers 0, 1, ..., 24 to indicate which letter is to be selected

#### asl_model2
- image input: numpy.ndarray with shape (28, 28)
- returns: numpy.ndarray with shape (, 24)
- training input: numpy.ndarray with shape (n, 24) where n is the number of training cases and each label is an array of shape (, 24). 
- label: the value at the index corresponding to the letter the image represents is 1, values at all other indexes are 0

### The Results

### Next Steps
