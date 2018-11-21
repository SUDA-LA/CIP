import abc


class Module(abc.ABC):
    def __init__(self):
        self.input = None
        self.output = None
        pass

    @abc.abstractmethod
    def forward(self, *input):
        pass

    @abc.abstractmethod
    def backward(self, *input):
        pass
