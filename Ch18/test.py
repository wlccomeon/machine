import matplotlib.pyplot as plt
from pylab import *
x=linspace(0,5,10)
y=x**2

fig=plt.figure()

axes=fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(x,y,'r')

plt.show()