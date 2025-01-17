import numpy as np

def transpose_matrix(A):
    return np.transpose(A)

def multiply_matrices(A, B):
  return np.dot(A, B)

def main():
    A=[[1,2,3],[4,5,6]]
    x=transpose_matrix(A)
    print(f"A transpose A is :{multiply_matrices(A, x)}")

if __name__ == '__main__':
    main()