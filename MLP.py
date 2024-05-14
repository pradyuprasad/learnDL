from Layer import Layer

class MLP:
    def __init__(self, dim_inputs, layers_list):
        # dim_inputs is the dimension of the inputs
        # layers_list is a list showing how many neurons in each layer
        # note that in every layer, the number of neurons is the dimension of the output when you call
        # so basically, if there is a layer of 3 neurons and then a layer with 4 neurons, then the second layer must accept inputs of size 3


        # for example dim_inputs is 3
        # and layers_list is [4, 3, 1]
        # which means the first layer must have size 3 with 4 neurons
        # second must have size 4 with 3 neurons and so on
        sz = [dim_inputs] + layers_list
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(layers_list))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            # basically keep multiplying the matrix with each layer's coefficients
        
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
