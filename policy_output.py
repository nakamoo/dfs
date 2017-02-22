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
    def __init__(self, logits_mu, logits_var):
        self.logits_mu = logits_mu
        self.logits_var = logits_var

        # print("self.logits_mu.data: ", self.logits_mu.data)

    @cached_property
    def probs(self):
        return self.logits_mu

    def dynamic_frame_skip(self, action_indices):
        # the function has same name as for SoftmaxPolicyOutput so that the function
        # can be called from a3c.py without changes
        # however, the function samples from gaussian distributions

        fs_list = []
        mu, sigma2 = self.logits_mu.data, self.logits_var.data

        # print("mu.data: ", mu.data)
        # print("sigma2.data: ", sigma2.data)
        for i in range(mu.shape[0]):
            mu_i = mu[i][action_indices[i]]
            fs = float(np.random.normal(mu_i, sigma2))
            fs_list.append(int(np.ceil(max(fs, 0.5))))
        return fs_list

    def sampled_actions_log_probs(self, action_indices):
        # sample action
        fs = self.dynamic_frame_skip(action_indices)
        fs = chainer.Variable(np.array(fs).astype(np.float32))
        mu_selected = F.select_item(self.logits_mu,
                    chainer.Variable(np.asarray(action_indices, dtype=np.int32)))

        # compute neg. log likelihood
        # print("chainer.Variable(action).dtype: ", chainer.Variable(fs).dtype)
        # print("mu.dtype: ", mu_selected.dtype)
        # print("F.log(sigma2).dtype: ", F.log(self.logits_var).dtype)

        return - F.gaussian_nll(fs, mu_selected, F.log(self.logits_var[0]))

    @cached_property
    def entropy(self):
        return - F.sum(0.5 * (F.log(2 * np.pi * self.logits_var) + 1))
