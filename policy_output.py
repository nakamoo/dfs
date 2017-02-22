import chainer
from chainer import functions as F
from cached_property import cached_property
import numpy as np


class PolicyOutput(object):
    """Struct that holds policy output and subproducts."""
    pass


def _sample_discrete_actions(batch_probs):
    """Sample a batch of actions from a batch of action probabilities.

    Args:
      batch_probs (ndarray): batch of action probabilities BxA
    Returns:
      List consisting of sampled actions
    """
    action_indices = []

    # Subtract a tiny value from probabilities in order to avoid
    # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
    batch_probs = batch_probs - np.finfo(np.float32).epsneg

    for i in range(batch_probs.shape[0]):
        histogram = np.random.multinomial(1, batch_probs[i])
        action_indices.append(int(np.nonzero(histogram)[0]))
    return action_indices


class SoftmaxPolicyOutput(PolicyOutput):
    def __init__(self, logits):
        self.logits = logits

    @cached_property
    def most_probable_actions(self):
        return np.argmax(self.probs.data, axis=1)

    @cached_property
    def probs(self):
        return F.softmax(self.logits)

    @cached_property
    def log_probs(self):
        return F.log_softmax(self.logits)

    @cached_property
    def action_indices(self):
        return _sample_discrete_actions(self.probs.data)

    @cached_property
    def sampled_actions_log_probs(self):
        return F.select_item(
            self.log_probs,
            chainer.Variable(np.asarray(self.action_indices, dtype=np.int32)))

    @cached_property
    def entropy(self):
        return - F.sum(self.probs * self.log_probs, axis=1)


class GaussianPolicyOutput(PolicyOutput):
    def __init__(self, mu, action_indices):
        self.mu = mu
        self.action_indices = action_indices

    @cached_property
    def probs(self):
        return self.logits_mu

    @cached_property
    def frameskip(self):
        fs_list = []
        for i in range(self.mu.data.shape[0]):
            mu_i = self.mu.data[i][self.action_indices[i]]
            fs = float(np.random.normal(mu_i, 2))
            fs_list.append(int(np.ceil(max(fs, 0.5))))
        return fs_list

    @cached_property
    def sample_output(self):
        return F.select_item(
            self.mu,
            chainer.Variable(np.asarray(self.action_indices, dtype=np.int32)))


