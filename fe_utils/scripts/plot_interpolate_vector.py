from fe_utils import *
from math import cos, sin, pi

se = LagrangeElement(ReferenceTriangle, 2)
ve = VectorFiniteElement(se)
m = UnitSquareMesh(10, 10)
fs = FunctionSpace(m, ve)
f = Function(fs)
f.interpolate(lambda x: (2*pi*(1 - cos(2*pi*x[0]))*sin(2*pi*x[1]),
                         -2*pi*(1 - cos(2*pi*x[1]))*sin(2*pi*x[0])))
f.plot()
