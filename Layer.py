from Neuron import Neuron

class Layer:
    def __init__(self, dim, num):
        # dim is the dimension of each neuron in the layer
        # num is the number of neurons in the layer
        self.neurons = [Neuron(dim) for _ in range(num)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        '''
        output = []
        for neuron in self.neurons:
            for p in neuron.parameters:
                output.append(p)
        '''

        return [p for neuron in self.neurons for p in neuron.parameters()] # same thing
