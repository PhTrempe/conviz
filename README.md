<div align="center">
    <img src="http://gdurl.com/NKrM" width="512">
</div>
<br>

| Better Code Hub compliance |
|:--------------------------:|
| [![Better Code Hub compliance](https://bettercodehub.com/edge/badge/PhTrempe/conviz)](https://bettercodehub.com/) |

## Description

Conviz is a convolutional neural network layer visualization library developed 
in Python and used with Keras.

## How to Install

```
pip install conviz
```

## A Small Example to Get You Started

```python
from conviz.models import cifar10
from conviz.utils.image_util import ImageUtil
from conviz.visualizer import Visualizer


# Loads a model trained on the CIFAR10 dataset
model = cifar10.load()

# Creates and binds a visualizer to the model
visualizer = Visualizer(model)

# Gets the layer of the model to visualize
layer = model.get_layer("conv1")

# Generates the visualization for the selected layer as a 4 by 4 grid of filters
img = visualizer.visualize(layer, (4, 4), ga_rate=0.1, num_steps=1000)

# Saves the generated visualization image to a file
ImageUtil.save_image(img, "./conv1.png")
```
