from .function import Function

_NOT_PROVIDED = object()


class StochasticFunction(Function):

    def __init__(self):
        self.reward = _NOT_PROVIDED

    def _do_backward(self, grad_output, retain_variables):
        if self.reward is _NOT_PROVIDED:
            raise RuntimeError("differentiating stochastic functions requires "
                               "providing a reward")
        result = super(StochasticFunction, self)._do_backward((self.reward,), retain_variables)
        if not retain_variables:
            self.reward = None
        return result

    def _do_forward(self, *inputs):
        result = super(StochasticFunction, self)._do_forward(*inputs)
        # save output type and size, to check the type of reward
        assert isinstance(output, Variable)
        self.output_info = (type(output.data), output.size())
        return result

    def _reinforce(self, reward):
        if type(reward) != self.output_info.type or reward.size() != self.output_info.size:
            raise ValueError("mismatch between reward and output type or size: "
                             "got {} of size {}, but expected {} of size {}".format(
                                type(reward).__name__, 'x'.join(map(str, reward.size())),
                                self.output_info.type, 'x'.join(map(str, self.output_info.size))))
        self.reward = reward
