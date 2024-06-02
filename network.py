import numpy as np

class Network:
    def __init__(self) -> None:
        self.layers = []
        self.edges = []
        self.output_vals = []

        self.learning_rate = 0.00001
        self.err = 0

        self.has_error = False
    
    # Updates the weights and resets the batch
    def update_weights(self):
        for edge in self.edges:
            avg = edge.batch[0]/edge.batch[1]
            edge.batch = [0,0]
            edge.weight -= self.learning_rate*avg
    
    # Clear gradients and add to batch for edges
    def reset_gradients(self):
        for edge in self.edges:
            edge.batch[0] += edge.gradient
            edge.batch[1] += 1
            edge.gradient = 0
        for layer in self.layers:
            for node in layer.vertices:
                node[0].gradient = 0
                if node[1]:
                    node[1].gradient = 0
            layer.bias.gradient = 0
    
    def store_output_vals(self):
        self.output_vals = []
        for elt in self.layers[-2].vertices:
            self.output_vals.append(elt[0].val)
    
    def get_output(self):
        outs = []
        for elt in self.layers[-2].vertices:
            if elt[1] != None:
                outs.append(elt[1].val)
            else:
                outs.append(elt[0].val)
        return outs

    def forward_pass(self) -> None:
        for layer in self.layers:
            layer.compute()
            if type(layer.vertices[0][0]) == MSError or \
                type(layer.vertices[0][0]) == CELoss:
                self.err = layer.vertices[0][0].val
            # If it is a softmax layer, we need to recompute the activations,
            # now that we know all the outputs.
            if layer.activation_fcn.upper() == "SOFTMAX":
                self.store_output_vals()
                for vert in layer.vertices:
                    vert[1].compute_value_special(self.output_vals)
    
    def connect_specific_layers(self, origin:'Layer', destination:'Layer'):
        for i in range(len(origin.vertices)):
            for j in range(len(destination.vertices)):
                e = Edge()
                self.edges.append(e)
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
            # Account for regression
            if L.vertices[i][1] == None:
                L.vertices[i][0].outs.append(mserror)
                mserror.ins.append(L.vertices[i][0])
            else:
                L.vertices[i][1].outs.append(mserror)
                mserror.ins.append(L.vertices[i][1])
        self.layers.append(le)
    
    def add_celoss_vertex(self):
        self.has_error = True
        le = Layer()
        celoss = CELoss()
        le.vertices.append((celoss,None))
        L = self.layers[-1]
        for i in range(len(L.vertices)):
            if L.vertices[i][1] == None:
                L.vertices[i][0].outs.append(celoss)
                celoss.ins.append(L.vertices[i][0])
            else:
                L.vertices[i][1].outs.append(celoss)
                celoss.ins.append(L.vertices[i][1])
        self.layers.append(le)
    
    def set_true_out(self, true_vals:list):
        L = self.layers[-1]
        L.vertices[0][0].set_true_vals(true_vals)

    def backward_pass(self):
        if not self.has_error:
            raise ValueError("No error vertex found")

        loss = self.layers[-1].vertices[0][0]
        stack = []
        visited = set()
        for elt in loss.ins:
            elt.gradient = loss.compute_gradient(elt)
            stack.append(elt)
        
        while stack:
            curr = stack.pop()
            visited.add(curr)

            for elt in curr.ins:
                elt.gradient += curr.gradient*curr.compute_gradient(elt)
                if elt not in visited and elt not in stack:
                    stack.append(elt)

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
                elif self.activation_fcn.upper() == "RELU":
                    activ = ReLU()
                elif self.activation_fcn.upper() == "SOFTMAX":
                    activ = Softmax()
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
        for i in range(len(self.bias.outs)):
            self.bias.outs[i].compute_value()
        
    
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
        self.gradient = 0
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
        self.val = 1

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
    
    def compute_gradient(self, inp):
        # Assuming only edges feed into congregates
        # Want the value of the variable that feeds into the weight node
        return inp.val/inp.weight

class Squared(Node):
    def __init__(self) -> None:
        super().__init__(fcn="squared")
    
    def compute_value(self):
        if len(self.ins) > 1:
            print("ERROR WITH NETWORK CONSTRUCTION")
            exit(1)

        self.val = self.ins[0].val * self.ins[0].val
    
    def compute_gradient(self, inp):
        return 2*inp.val


class Sigmoid(Node):
    def __init__(self) -> None:
        super().__init__(fcn="sigmoid")
    
    def compute_value(self):
        if len(self.ins) > 1:
            print("ERROR WITH NETWORK CONSTRUCTION")
            exit(1)
        
        self.val = 1/(1 + np.exp(-1 * self.ins[0].val))
    
    def compute_gradient(self, inp:Vertex):
        return inp.val*(1-inp.val)

class MSError(Node):
    def __init__(self) -> None:
        super().__init__(fcn="mserror")
        self.true_vals = []
        self.val_diff_mapping = {}
    
    def set_true_vals(self, true_vals:list):
        self.true_vals = true_vals
    
    def compute_value(self):
        val = 0
        for i in range(len(self.ins)):
            self.val_diff_mapping[self.ins[i]] = self.ins[i].val - self.true_vals[i]
            val += (self.true_vals[i]-self.ins[i].val)**2
        val *= 1/len(self.ins)
        self.val = val
    
    def compute_gradient(self, inp):
        return 2/len(self.ins) * (self.val_diff_mapping.get(inp))


class Edge(Vertex):
    count = 0

    def __init__(self) -> None:
        super().__init__(kind=1)
        self.weight = np.random.normal(0,np.sqrt(2)/28)
        Edge.count += 1
        self.name = f"E{Edge.count}"
        self.batch = [0,0]
    
    def compute_value(self):
        self.val = self.ins[0].val*self.weight
    
    def __repr__(self) -> str:
        return self.name + " outs: " +  str(self.outs) + " weight: " + str(self.weight) + " val: " + str(self.val)
    
    def compute_gradient(self, inp):
        return 1

class ReLU(Node):
    def __init__(self) -> None:
        super().__init__(fcn="relu")
    
    def compute_value(self):
        self.val = max(0, self.ins[0].val)
    
    def compute_gradient(self, inp: Vertex):
        return 1 if inp.val > 0 else 0

class Softmax(Node):
    def __init__(self) -> None:
        super().__init__(fcn="softmax")
        self.exp_vals = 0
        self.exp_inp = 0
    
    # Dummy computation because not all outputs will be computed when this gets called
    def compute_value(self):
        self.val = 1
    
    def compute_value_special(self,output_vals:list):
        self.exp_vals = np.exp(output_vals)
        self.exp_inp = np.exp(self.ins[0].val)
        self.val = self.exp_inp/np.sum(self.exp_vals )

    def compute_gradient(self,inp):
        # return (self.exp_inp*self.exp_vals)/((self.exp_inp+self.exp_vals)**2)
        return self.val*(1-self.val)

class CELoss(Node):
    def __init__(self) -> None:
        super().__init__(fcn="celoss")
        self.true_vals = []
        self.val_mapping = {}
    
    def set_true_vals(self, true_vals:list):
        self.true_vals = true_vals
    
    def compute_value(self):
        val = 0
        epsilon = 1e-6
        for i in range(len(self.true_vals)):
            self.val_mapping[self.ins[i]] = self.true_vals[i]
            val += self.true_vals[i] * np.log(self.ins[i].val+epsilon)
        val *= -1
        self.val = val
    
    def compute_gradient(self, inp):
        # return -1 * self.val_diff_mapping.get(inp)/inp.val
        return inp.val - self.val_mapping.get(inp)
