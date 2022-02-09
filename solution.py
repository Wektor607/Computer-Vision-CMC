from operator import ilshift

from numpy.core.fromnumeric import mean
from torch._C import dtype
from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr # это числе есть размер шага спуска

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.lr * parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter
            :return: the updater function for that parameter
        """
        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            updater.inertia = self.momentum * updater.inertia + self.lr * parameter_grad
            return parameter - updater.inertia
            # your code here /\
        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size, ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return inputs.clip(0)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size, ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        x = self.forward_inputs
        return grad_outputs * (x >= 0)
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        # your code here \/
        h, w = inputs.shape
        if(np.size(input) == 0):
            max = 0
        else:
            max = np.max(inputs, axis = 1)
        tmp = np.exp(inputs - np.transpose(np.resize(max, (w, h))))
        sum_tmp = np.transpose(np.resize(np.sum(tmp, axis = 1), (w, h)))
        return tmp / sum_tmp
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        # your code here \/
        y = self.forward_outputs
        n =  grad_outputs.shape[0]
        d =  grad_outputs.shape[1]
        new_grad = np.empty(y.shape)
        for i in range(n):
            y_t = y[i].reshape(-1, 1)
            new_grad[i] = grad_outputs[i] * y[i] - np.matmul(y_t, np.matmul(np.transpose(y_t), grad_outputs[i]))
        return new_grad 
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, c)), output values

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/
        return(np.matmul(inputs, self.weights) + self.biases)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/
        n = grad_outputs.shape[0]
        inputs = self.forward_inputs

        self.biases_grad = np.mean(grad_outputs, axis = 0)
        self.weights_grad = np.matmul(np.transpose(inputs), grad_outputs) / n

        return np.matmul(grad_outputs, np.transpose(self.weights))
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n,)), loss scalars for batch

                n - batch size
                d - number of units
        """
        # your code here \/
        return (-1) * np.sum(y_gt * np.log(y_pred), axis = 1)
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n, d)), gradient loss to y_pred

                n - batch size
                d - number of units
        """
        # your code here \/
        res = np.zeros((y_gt.shape[0], y_gt.shape[1]))
        for i in range(y_gt.shape[0]):
            for j in range(y_gt.shape[1]):
                if(eps > y_pred[i][j]):
                    res[i][j] = (-1) * (y_gt[i][j] / eps)
                else:
                    res[i][j] = (-1) * (y_gt[i][j] / y_pred[i][j])
        return res
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    loss = CategoricalCrossentropy()
    optimizer = SGDMomentum(lr=0.002, momentum= 0.9)
    model = Model(loss, optimizer)

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(units=8, input_shape=(784, )))
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=32, epochs=5, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model

from itertools import product
# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # your code here \/
    n = inputs.shape[0]
    d_inp = inputs.shape[1]
    input_height = inputs.shape[2]
    input_width = inputs.shape[3]

    c = kernels.shape[0]
    d_ker = kernels.shape[1]
    ker_height = kernels.shape[2]
    ker_width = kernels.shape[3]
    kernels = kernels[:, :, ::-1, ::-1].transpose(2, 3, 1, 0)

    out_height = input_height - ker_height + 1 + 2 * padding
    out_width = input_width - ker_width + 1 + 2 * padding
    output = np.empty((n, c, out_height, out_width)).transpose(0, 2, 3, 1)

    imagePadded = np.zeros((n, d_inp, input_height + 2 * padding, input_width + 2 * padding))
    if (padding == 0):
        imagePadded = inputs
    else:
        imagePadded[:, :, int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = inputs
    
    imagePadded = imagePadded.transpose(0, 2, 3, 1)
    
    for n, i in product(range(output.shape[0]), range(output.shape[1])):
        for j, cout in product(range(output.shape[2]), range(output.shape[3])):
            output[n, i, j, cout] = np.sum(imagePadded[n, i:i+kernels.shape[0], j:j+kernels.shape[1], :] * kernels[:,:,:,cout])
    return output.transpose(0, 3, 1, 2)
    # your code here /\


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, c, h, w)), output values

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        batch_size = inputs.shape[0]
        self.conv2d = convolve(inputs, self.kernels, (self.kernels.shape[2] - 1) // 2)
        #self.biases = np.stack([np.stack([np.stack([self.biases for _ in range(self.conv2d.shape[0])], axis = 0) for _ in range(self.conv2d.shape[2])], axis = 2) for _ in range(self.conv2d.shape[3])], axis = 3)
        biases = np.stack([self.biases for _ in range(self.conv2d.shape[0])], axis = 0)
        biases = np.stack([biases for _ in range(self.conv2d.shape[2])], axis = 2)
        biases = np.stack([biases for _ in range(self.conv2d.shape[3])], axis = 3)
        return self.conv2d + biases
        # your code here /\
    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        input_shape = self.input_shape
        batch_size = grad_outputs.shape[0]
        x = self.forward_inputs[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)
        kernels = self.kernels[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)
        self.biases_grad = np.sum(grad_outputs, axis=(0, 2, 3)).reshape(self.biases_grad.shape) / batch_size
        self.kernels_grad = convolve(x, grad_outputs.transpose(1, 0, 2, 3), (self.kernels.shape[2] - 1) // 2).transpose(1, 0, 2, 3) / batch_size
        return convolve(grad_outputs, kernels, (self.kernels.shape[2] - 1) // 2)
        # your code here /\

# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, ih, iw)), input values

            :return: np.array((n, d, oh, ow)), output values

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        n, d, i_h, i_w = inputs.shape
        o_h = i_h // self.pool_size 
        o_w = i_w // self.pool_size

        res = np.lib.stride_tricks.as_strided(inputs, shape=(n, d, o_h, o_w, self.pool_size, self.pool_size), strides=(inputs.strides[0], inputs.strides[1], self.pool_size * inputs.strides[2], self.pool_size * inputs.strides[3], inputs.strides[2], inputs.strides[3]))

        res = np.reshape(res, (n, d, o_h, o_w, self.pool_size ** 2))
        self.forward_idxs = np.zeros_like(res)
        if self.pool_mode == 'max':
            pool = np.max(res, axis = 4)
            arg_max = np.expand_dims(np.argmax(res, axis = -1), axis = -1)
            np.put_along_axis(self.forward_idxs, arg_max, 1, axis=-1)
            self.forward_idxs = np.reshape(np.transpose(np.reshape(self.forward_idxs, (n, d, o_h, o_w, self.pool_size, self.pool_size)),(0, 1, 2, 4, 3, 5)), (inputs.shape))
        else:
            pool = np.mean(res, axis = 4)

        return pool
        # your code here /\
        
    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

            :return: np.array((n, d, ih, iw)), dLoss/dInputs

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        batch_size = grad_outputs.shape[0]
        # your code here \/
        if self.pool_mode == 'max':
            grad_outputs = np.repeat(np.repeat(grad_outputs, self.pool_size, axis = 2), self.pool_size, axis=3)
            grad_outputs = grad_outputs * self.forward_idxs
        else:
            grad_outputs = np.repeat(np.repeat(grad_outputs, self.pool_size, axis = 2), self.pool_size, axis=3) / (self.pool_size ** 2)

        return grad_outputs
        # your code here /\

# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        d = inputs.shape[1]
        self.gamma = np.reshape(self.gamma, (1, d, 1, 1))
        self.beta = np.reshape(self.beta, (1, d, 1, 1))
        self.mean = None
        self.var = None
        if self.is_training:
            self.mean = inputs.mean(axis = (0, 2, 3))
            self.var  = inputs.var(axis = (0, 2, 3))
            self.mean = np.reshape(self.mean, (1, d, 1, 1))
            self.var = np.reshape(self.var, (1, d, 1, 1))

            self.forward_inverse_std = 1 / np.sqrt(self.var + eps)
            self.forward_centered_inputs = inputs - self.mean
            self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std

            output = self.gamma * self.forward_normalized_inputs + self.beta
            self.running_mean = self.momentum * self.running_mean.reshape(1, d, 1, 1) + (1 - self.momentum) * self.mean
            self.running_var  = self.momentum * self.running_var.reshape(1, d, 1, 1) + (1 - self.momentum) * self.var

        else:
            mean = self.running_mean.reshape(1, d, 1, 1)
            var = self.running_var.reshape(1, d, 1, 1)
            # mean = np.reshape(mean, (1, d, 1, 1))
            # var = np.reshape(var, (1, d, 1, 1))
            output = self.gamma * ((inputs - mean) / (np.sqrt(var + eps))) + self.beta

        return output
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        batch_size = grad_outputs.shape[0]
        
        x_norm = self.forward_normalized_inputs
        
        #5
        self.gamma_grad = np.sum(grad_outputs * x_norm, axis=(0,2,3)).reshape(self.gamma_grad.shape) / batch_size
        
        #6
        self.beta_grad = np.sum(grad_outputs, axis=(0,2,3)).reshape(self.beta_grad.shape) / batch_size
        
        #1
        dx_norm = grad_outputs * self.gamma.reshape(1, grad_outputs.shape[1], 1, 1).astype("float64")
        
        #2
        grad_var = np.sum(dx_norm * x_norm, axis=(0,2,3))
        
        #3
        
        grad_mean = np.sum(dx_norm, axis=(0,2,3))
        
        grad_mean = grad_mean.reshape(1, grad_outputs.shape[1], 1, 1)
        grad_var = grad_var.reshape(1, grad_outputs.shape[1], 1, 1)
        
        #4
        grad_inputs = (dx_norm - (grad_var * x_norm + grad_mean) / (batch_size * grad_outputs.shape[2] * grad_outputs.shape[3])) * self.forward_inverse_std

        return grad_inputs
        # your code here /\
    


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (np.prod(self.input_shape),)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, (d * h * w))), output values

                n - batch size
                d - number of input channels
                (h, w) - image shape
        """
        # your code here \/
        self.prev_shape = inputs.shape
        return inputs.reshape((inputs.shape[0], -1))
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of units
                (h, w) - input image shape
        """
        # your code here \/
        batch_size = grad_outputs.shape[0]
        return grad_outputs.reshape(self.prev_shape)
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        # your code here \/
        if self.is_training:
            self.forward_mask = np.random.uniform(size = inputs.shape) > self.p
            outputs = inputs * self.forward_mask
        else:
            outputs = inputs
        return outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        # your code here \/
        return grad_outputs * self.forward_mask
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGDMomentum(0.01, 0.9))
    model.add(Conv2D(output_channels = 16, kernel_size= 3, input_shape=(3, 32, 32)))
    model.add(ReLU())
    model.add(Pooling2D())

    model.add(Conv2D(128, 3))
    model.add(ReLU())
    model.add(Pooling2D())

    model.add(Conv2D(256, 3))
    model.add(ReLU())
    model.add(Pooling2D())

    model.add(Flatten())

    model.add(Dense(10))
    model.add(ReLU())
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size = 32, epochs = 5, x_valid = x_valid, y_valid = y_valid)

    # your code here /\
    return model

# ============================================================================
