import numpy 


# make linear data 
true_weight = 2.0 
true_bias = -3.0 
x = numpy.linspace(-10, 10, num=100)
y = true_weight * x + true_bias

# cheated and looked at micrograd.
class Parameter:
    def __init__(self, value, prev=()): 
        print('value', value)
        self.value = value 
        self._backward = lambda: None
        self._prev = prev
        self.grad = 0.0

    def __add__(self, other): 
        if not isinstance(other, Parameter): 
            other = Parameter(other)
        ret = Parameter(self.value + other.value, (self, other))
        def backward():
            self.grad += ret.grad
            other.grad += ret.grad 
        ret._backward = backward 
        return ret 

    def __mul__(self, other):
        if not isinstance(other, Parameter): 
            other = Parameter(other)
        
        ret = Parameter(self.value * other.value, (self, other))
        def backward(): 
            self.grad += ret.grad * other.grad 
            other.grad += ret.grad * self.grad
        ret._backward = backward 
        return ret 
    
    def backward(self): 
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

a = Parameter(3.0)
b = a * 3
b.backward()