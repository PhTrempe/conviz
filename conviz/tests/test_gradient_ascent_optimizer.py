import inspect
import os
from unittest import TestCase

from keras.models import load_model

from conviz.src.gradient_ascent_optimizer import GradientAscentOptimizer


class TestGradientAscentOptimizer(TestCase):
    def setUp(self):
        self.dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        self.cifar10_model_path = os.path.join(self.dir, "data", "cifar10.h5")
        self.cifar10_model = load_model(self.cifar10_model_path)

    def test_optimize(self):
        # Create a gradient ascent optimizer
        gao = GradientAscentOptimizer(
            x=self.cifar10_model.input,
            ga_rate=0.5,
            num_steps=5
        )

        # Find the "conv1" layer of the model
        conv_layer_name = "conv1"
        conv_layer = next(layer for layer in self.cifar10_model.layers
                          if layer.name == conv_layer_name)

        num_conv_filters = conv_layer.output_shape[3]
        expected_out_img_shape = (*conv_layer.input_shape[1:3], 3)

        # Run gradient ascent optimization for all filters of layer "conv1"
        for conv_filter_idx in range(num_conv_filters):
            # Perform gradient ascent for current convolutional filter
            conv_filter_img, conv_filter_score = gao.optimize(
                conv_layer=conv_layer,
                conv_filter_idx=conv_filter_idx
            )

            # Verify that the score value is valid
            self.assertTrue(conv_filter_score >= 0)

            # Verify that the generated filter image has the correct shape
            self.assertTupleEqual(
                conv_filter_img.shape,
                expected_out_img_shape,
                "Generated filter image has incorrect shape."
            )
