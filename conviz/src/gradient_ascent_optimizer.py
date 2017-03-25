import numpy
from keras import backend as b


class GradientAscentOptimizer(object):
    def __init__(self, x, ga_rate, num_steps):
        self.x = x
        self.ga_rate = ga_rate
        self.num_steps = num_steps
        self.epsilon = numpy.finfo(float).eps

    def optimize(self, conv_layer, conv_filter_idx):
        objective = self._build_objective(conv_layer, conv_filter_idx)
        gradients = self._build_gradients(objective)
        ga_step = self._build_gradient_ascent_step_op(objective, gradients)
        img_dim = (*conv_layer.input_shape[1:3],)
        img_data = self._build_initial_image(img_dim)
        img_data, score = self._run_optimization(img_data, ga_step)
        conv_filter_img = self._optimization_result_to_image(img_data)
        return conv_filter_img, score

    @staticmethod
    def _build_objective(conv_layer, conv_filter_idx):
        return b.mean(conv_layer.output[:, :, :, conv_filter_idx])

    @staticmethod
    def _build_initial_image(img_dim):
        return numpy.random.random((1, *img_dim, 3)) - 0.5

    def _build_gradients(self, objective):
        return self._l2_normalize(b.gradients(objective, self.x))

    def _l2_normalize(self, x):
        return x / (b.sqrt(b.mean(b.square(x))) + self.epsilon)

    def _build_gradient_ascent_step_op(self, objective, gradients):
        return b.function([self.x], [objective, gradients])

    def _run_optimization(self, img_data, ga_step):
        score = -numpy.inf
        for step in range(self.num_steps):
            score, grads_value = ga_step([img_data])
            img_data += grads_value[0] * self.ga_rate
        return img_data, score

    def _optimization_result_to_image(self, a):
        a = numpy.array(a[0])
        a_0_centered = a - a.mean()
        a_std_01 = a_0_centered / (a_0_centered.std() + self.epsilon) * 0.1
        a_clip_0_1 = numpy.clip(a_std_01 + 0.5, 0.0, 1.0)
        return (a_clip_0_1 * 255).astype("uint8")
