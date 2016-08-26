import chainer
from chainer import functions as F
from chainer import links as L
from net import ConvLSTM


class NatureDQNHead(chainer.ChainList):
    """DQN's head (Nature version)"""

    def __init__(self, n_input_channels=4, n_output_channels=512,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        layers = [
            L.Convolution2D(n_input_channels, 32, 8, stride=4, bias=bias),
            L.Convolution2D(32, 64, 4, stride=2, bias=bias),
            L.Convolution2D(64, 64, 3, stride=1, bias=bias),
            L.Linear(3136, n_output_channels, bias=bias),
        ]

        super(NatureDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.activation(layer(h))
        return h


class NIPSDQNHead(chainer.ChainList):
    """DQN's head (NIPS workshop version)"""

    def __init__(self, n_input_channels=4, n_output_channels=256,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        layers = [
            L.Convolution2D(n_input_channels, 16, 8, stride=4, bias=bias),
            L.Convolution2D(16, 32, 4, stride=2, bias=bias),
            L.Linear(2592, n_output_channels, bias=bias),
        ]

        super(NIPSDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.activation(layer(h))
        return h


class PredNet(chainer.Chain):
    def __init__(self):
        self.layers = 1
        self.width = 84
        self.height = 84
        self.n_output_channels = 256
        self.sizes = [(1, 3, self.width, self.height)]

        super(PredNet, self).__init__(
            conv_lstm_r0=ConvLSTM(self.width, self.height, (3*2,), 3),
            conv_p0=L.Convolution2D(3, 3, 3, pad=1),
            conv_q1=L.Convolution2D(3, 3, 8, stride=4),
            linear_q2=L.Linear(1200, 256)
        )
        self.reset_state()

    def __call__(self, s):
        # if P is blank, zelo-fill P.
        for nth in range(self.layers):
            if getattr(self, 'P' + str(nth)) is None:
                setattr(self, 'P' + str(nth), chainer.Variable(
                    self.xp.zeros(self.sizes[nth], dtype=s.data.dtype),
                    volatile='auto'))

        E0 = F.concat((F.relu(s - self.P0), F.relu(self.P0 - s)))
        R0 = self.conv_lstm_r0((E0,))
        self.P0 = F.clipped_relu(self.conv_p0(R0), 1.0)
        q = F.relu(self.conv_q1(R0))
        q = F.relu(self.linear_q2(q))

        return q

    def to_cpu(self):
        super(PredNet, self).to_cpu()
        for nth in range(self.layers):
            if getattr(self, 'P' + str(nth)) is not None:
                getattr(self, 'P' + str(nth)).to_cpu()

    def to_gpu(self, device=None):
        super(PredNet, self).to_gpu(device)
        for nth in range(self.layers):
            if getattr(self, 'P' + str(nth)) is not None:
                getattr(self, 'P' + str(nth)).to_gpu(device)

    def reset_state(self):
        for nth in range(self.layers):
            setattr(self, 'P' + str(nth), None)
        self.conv_lstm_r0.reset_state()
