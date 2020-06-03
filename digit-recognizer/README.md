# Digit-Recognizer
In this competitions I had apply mulitple method/technique DNN, CNN, pretrained CNN.

## Data Exploration 
Number of data in each label

## Preprocessing
### Reshape
Given the data in 1D, to fit into a CNN input data need to be in 3D.
imagedata = imagedata.values.reshape(-1,28,28)
imagedata = np.expand_dims(imagedata, axis=-1)

## Feature engineering
### Normalize
Each image pixel is a range of values from 0-255. To normalize dividing by 255, keep the data in the range of [0-1].
imagedata = imagedata/255

### Data augmentation
rotation_range, zoom_range, width_shift_range, height_shift_range
*Note fliping is not done, cause if we will to vertical_flip 6 become 9.

### Additional
Inorder to train using the pretrained CNN (eg. VGG, Inception). The input data must be atlest of (32,32) shape with 3 channel.


