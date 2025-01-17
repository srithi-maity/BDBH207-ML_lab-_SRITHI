import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-100,100,100)
y=x**2
dydx=np.gradient(y,x)
plt.figure(figsize=(8,10))
plt.plot(x,y,dydx,label="y=x**2")

# plt.plot(x,y,dydx,label="y=x**2",color="red")
# plt.plot(x,dydx,label="dydx",color="blue")
plt.grid(True)
plt.legend()
plt.show()
