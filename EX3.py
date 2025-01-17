import numpy as np
import matplotlib.pyplot as plt


x=np.linspace(-100,100,100)
# print(x)
y=2*x**2+3*x +4
plt.figure(figsize=(8,10))
plt.plot(x,y,label="y=2*x**2+3*x +4",color="red")
plt.grid(True)
plt.legend()
plt.show()