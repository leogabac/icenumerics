import math

class Vector(tuple):
    def __init__(self,Value):
        self=tuple(float(s) for s in Value)
    def add(self,other):
        return self.__class__(tuple(a+b for a,b in zip(self,other)))
    def dot(self,other):
        return sum(tuple(a*b for a,b in zip(self,other)))
    def __add__(self,other):
        return self.add(other)
    def __mul__(self,other):
        if other.__class__.__name__=="Vector":
            return self.dot(other)
        if other.__class__.__name__=="float":
            return self.__class__(tuple(other*s for s in self))
        if other.__class__.__name__=="int":
            return self*float(other)

    def round(self,i=0):
        return tuple(round(s,i) for s in self)
    
    def unit(self):
        return self*float(1/math.sqrt(self.dot(self)))
    
    def magnitude(self):
        return math.sqrt(self.dot(self))
    
