import numpy as np

def gauss_elimination(a, b):
    n = len(a)
    for i in range(n):
        max_row_index = abs(a[i:, i]).argmax() + i
        if a[max_row_index, i] == 0:
            raise ValueError("Matice je singulární.")
        a[[i, max_row_index]] = a[[max_row_index, i]]
        b[[i, max_row_index]] = b[[max_row_index, i]]
        for j in range(i+1, n):
            factor = a[j, i]/a[i, i]
            a[j, :] -= factor * a[i, :]
            b[j] -= factor * b[i]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(a[i,i+1:], x[i+1:])) / a[i,i]
    return x

# Testování dat
# Zadání xyz
a = np.array([[2.0, 1.0, -1.0], [3.0, 2.0, 1.0], [2.0, -1.0, 2.0]])
b = np.array([1.0, 10.0, 6.0])

print(gauss_elimination(a,b))
