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
Inorder to train using the pretrained CNN (eg. VGG, Inception). The input data must be minimum size of (32,32) shape with 3 channel.

## DNN Architecture
With hyperparameter tuning (numofnode, multiple hidden layer, dropout) done:

Single hidden layer with no dropout performed the best,
**INPUT -> HiddenLayer -> OUTPUT**
529 node: Epochs=30, Train accuracy=1.0, Validation accuracy=0.9828673601150513

Summition result,
0.97871

Others result,
529 node +Dropout: Epochs=30, Train accuracy=0.9986307621002197, Validation accuracy=0.9808447360992432
529-396 node: Epochs=30, Train accuracy=0.998928427696228, Validation accuracy=0.97846519947052

## CNN Architecture
**INPUT -> [[CONV]x2 -> [CONV + strides]]x2 -> [Dropout] -> [Flatten] -> [FC] ->  [Dropout] -> Output**

Pooling layer is replace with [CONV + strides]. 

Refrence:
Pooling layer vs CONV with stride Layer
https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling

