import numpy as np
import matplotlib.pyplot as plt
import math as m

x=np.linspace(-100,100,100)
sd=15
me=0
pi=3.14159
y=np.exp(-0.5*((x-me)/sd)**2)/(sd*m.sqrt(2*pi))
plt.figure(figsize=(8,10))
plt.plot(x,y,label="y=2*x**2+3*x +4",color="red")
plt.grid(True)
plt.legend()
plt.show()