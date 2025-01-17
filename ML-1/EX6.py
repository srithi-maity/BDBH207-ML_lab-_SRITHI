import numpy as np
import matplotlib.pyplot as plt
from random import randint
def sample_h(theta):
    # theta=[1,-0.2,0.35,1.1]
    x=[randint(1,100) for i in range(4)]
    print(x)
    y=randint(30,100)
    print(y)
    y_prime=[p*q for p,q in zip(theta,x)]
    print(y_prime)
    y_res=np.sum(y_prime)
    print(y_res)
    h_x=(y_res-y)**2
    return h_x
def main():
    theta=[1,-0.2,0.35,1.1]

    h_x=[]
    for n in range(2):
        h_x.append(sample_h(theta))
    result=sum(h_x)/2
    print(result)
    for i in range(4):
        final=d
        print(f"dE/dtheta[{i}] is {result}")


if __name__ == '__main__':
    main()
