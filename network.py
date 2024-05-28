
class Network:
    def __init__(self) -> None:
        self.layers = []
        self.edges = []
    
    def num_layers(self) -> int:
        return len(self.layers)
    
    def make_layers(self, num:int) -> None:
        for i in range(num):
            self.layers.append(Layer())

    def connect_specific_layers(self, origin:'Layer', destination:'Layer') -> None:
        for i in range(len(origin.vertices)):
            for j in range(len(destination.vertices)):
                self.edges.append(Edge(origin.vertices[i], destination.vertices[j]))
        
        for i in range(len(destination.vertices)):
            self.edges.append(Edge(origin.bias,destination.vertices[i]))
    
    # Add edges between all vertices
    def connect_layers(self):
        for i in range(len(self.layers)-1):
            origin = self.layers[i]
            destination = self.layers[i+1]
            self.connect_specific_layers(origin,destination)

    def forward_pass(self) -> None:
        for layer in self.layers:
            layer.compute()
    
    def mean_sqare_error(self, true_vals:list) -> int:
        val = 0
        lf = self.layers[-1]
        for i in range(len(lf)):
            val += (true_vals[i] - lf[i])**2
        return val/(len(lf))

class Layer:
    def __init__(self, activation_fcn='squared') -> None:
        self.vertices = []
        self.bias = Vertex()
        self.activation_fcn = activation_fcn

    # Returns number of vertices + 1 for the bias term
    def num_nodes(self) -> int:
        return len(self.vertices) + 1
    
    def fill_layer(self, num:int) -> None:
        for i in range(num):
            self.vertices.append(Vertex(activation_fcn=self.activation_fcn))
    
    def compute(self) -> None:
        for i in range(len(self.vertices)):
            self.vertices[i].compute_value()
    
class Vertex:
    def __init__(self, value=1, activation_fcn='squared') -> None:
        self.activation_fcn = activation_fcn
        self.value = value
        self.gradient = 0
        self.outs = []
        self.ins = [] 
    
    def squared(self, val:int) -> int:
        return val * val
    
    # Activation function is just z^2 for now
    def activation(self, val:int) -> int:
        if self.activation_fcn == 'squared':
            return self.squared(val)
        else:
            print("ERROR: using 'Squared'")
            return self.squared(val)
    
    def compute_value(self) -> None:
        if len(self.ins) == 0:
            return
        else:
            val = 0
            for i in range(len(self.ins)):
                val += (self.ins[i])[0].value * (self.ins[i])[1].weight
            self.value = self.activation(val)

class Edge:
    def __init__(self, origin:Vertex, destination:Vertex, weight=1) -> None:
        self.weight = weight
        self.gradient = 0
        self.origin = origin
        self.destination = destination
        self.update_vertices()

    def update_vertices(self):
        self.origin.outs.append((self.destination,self))
        self.destination.ins.append((self.origin,self))
