import numpy as np

class Network:
    def __init__(self) -> None:
        self.layers = []
        self.has_error = False
    
    def num_layers(self) -> int:
        return len(self.layers)
    
    def make_layers(self, num:int) -> None:
        for i in range(num):
            self.layers.append(Layer())

    def forward_pass(self) -> None:
        for layer in self.layers:
            layer.compute()
    
    def connect_specific_layers(self, origin:'Layer', destination:'Layer'):
        for i in range(len(origin.vertices)):
            for j in range(len(destination.vertices)):
                e = Edge()
                a = None
                if origin.vertices[i][1] == None:
                    a = origin.vertices[i][0]
                else:
                    a = origin.vertices[i][1]
                a.outs.append(e)
                e.ins.append(a)
                e.outs.append(destination.vertices[j][0])
                destination.vertices[j][0].ins.append(e)
        
        for i in range(len(destination.vertices)):
            e = Edge()
            origin.bias.outs.append(e)
            e.ins.append(origin.bias)
            e.outs.append(destination.vertices[i][0])
            destination.vertices[i][0].ins.append(e)
    
    def connect_layers(self):
        for i in range(len(self.layers)-1):
            origin = self.layers[i]
            destination = self.layers[i+1]
            self.connect_specific_layers(origin,destination)
    
    def add_mserror_vertex(self):
        self.has_error = True
        le = Layer()
        mserror = MSError()
        le.vertices.append((mserror,None))
        L = self.layers[-1]
        for i in range(len(L.vertices)):
            L.vertices[i][1].outs.append(mserror)
            mserror.ins.append(L.vertices[i][1])
        self.layers.append(le)
    
    def set_true_out(self, true_vals:list):
        L = self.layers[-1]
        L.vertices[0][0].set_true_vals(true_vals)
        

class Layer:
    def __init__(self, activation_fcn='none') -> None:
        self.vertices = []
        self.bias = Bias()
        self.activation_fcn = activation_fcn
    
    def __repr__(self) -> str:
        return "vertices: " + str(self.vertices) + " bias: " + str(self.bias) + " actv: " + self.activation_fcn
    
    def fill_layer(self, num:int, kind:str) -> None:
        if kind.upper() == "INPUT":
            for i in range(num):
                self.vertices.append((Input(),None))
        if "CONG" in kind.upper():
            for i in range(num):
                activ = None
                if self.activation_fcn.upper() == "NONE":
                    activ = None
                elif self.activation_fcn.upper() == "SQUARED":
                    activ = Squared()
                elif self.activation_fcn.upper() == "SIGMOID":
                    activ = Sigmoid()
                new = Congregate()
                new.outs.append(activ)
                activ.ins.append(new)
                self.vertices.append((new,activ))
    
    def compute(self) -> None:
        for i in range(len(self.vertices)):
            self.vertices[i][0].compute_value()
            if self.vertices[i][1] != None:
                self.vertices[i][1].compute_value()
                for j in range(len(self.vertices[i][1].outs)):
                    self.vertices[i][1].outs[j].compute_value()
            else:
                for j in range(len(self.vertices[i][0].outs)):
                    self.vertices[i][0].outs[j].compute_value()
    
    def set_input(self, data:list):
        for i in range(len(self.vertices)):
            self.vertices[i][0].val = data[i]

class Vertex:
    count = 0

    def __init__(self, kind:int) -> None:
        self.kind = kind
        self.ins = []
        self.outs = []
        self.val = 0
        Vertex.count += 1
        self.name = f"V{Vertex.count}"
    
    def __repr__(self) -> str:
        return self.name + " outs: " +  str(self.outs) + " val: " + str(self.val)
        
    def compute_value(self):
        return

class Node(Vertex):
    def __init__(self, fcn:str) -> None:
        super().__init__(kind=0)
        self.fcn = fcn

class Bias(Node):
    def __init__(self) -> None:
        super().__init__(fcn="none")

class Input(Node):
    def __init__(self) -> None:
        super().__init__(fcn="none")

class Congregate(Node):
    def __init__(self) -> None:
        super().__init__(fcn="cong")
    
    def compute_value(self):
        val = 0
        for i in range(len(self.ins)):
            val += self.ins[i].val
        self.val = val

class Squared(Node):
    def __init__(self) -> None:
        super().__init__(fcn="squared")
    
    def compute_value(self):
        if len(self.ins) > 1:
            print("ERROR WITH NETWORK CONSTRUCTION")
            exit(1)

        self.val = self.ins[0].val * self.ins[0].val

class Sigmoid(Node):
    def __init__(self) -> None:
        super().__init__(fcn="sigmoid")
    
    def compute_value(self):
        if len(self.ins) > 0:
            print("ERROR WITH NETWORK CONSTRUCTION")
            exit(1)
        
        self.val = 1/(1 + np.exp(-1 * self.ins[0].val))

class MSError(Node):
    def __init__(self) -> None:
        super().__init__(fcn="mserror")
        self.true_vals = []
    
    def set_true_vals(self, true_vals:list):
        self.true_vals = true_vals
    
    def compute_value(self):
        val = 0
        for i in range(len(self.ins)):
            val += (self.true_vals[i]-self.ins[i].val)**2
        val *= 1/len(self.ins)
        self.val = val

class Edge(Vertex):
    count = 0

    def __init__(self, weight=1) -> None:
        super().__init__(kind=1)
        self.weight = weight
        Edge.count += 1
        self.name = f"E{Edge.count}"
    
    def compute_value(self):
        self.val = self.ins[0].val*self.weight
    
    def __repr__(self) -> str:
        # return "kind: " + str(self.kind) + " ins: " + str(self.ins) + " outs: " + str(self.outs) + " val: " + str(self.val) + " weight: " + str(self.weight)
        return self.name + " outs: " +  str(self.outs) + " weight: " + str(self.weight) + " val: " + str(self.val)
