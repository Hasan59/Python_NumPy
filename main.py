import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(arr)
import numpy as np

arr = np.array((1, 2, 3, 4, 5))

print(arr)
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(arr)

print(type(arr))

import numpy as np

arr = np.array(42)

print(arr)

import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(arr)

import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr)

import numpy as np

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(arr)

a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

import numpy as np

arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr)
print('number of dimensions :', arr.ndim)

import numpy as np

arr = np.array([1, 2, 3, 4])

print(arr[0])

import numpy as np

arr = np.array([1, 2, 3, 4])

print(arr[1])

import numpy as np

arr = np.array([1, 2, 3, 4])

print(arr[2] + arr[3])
import numpy as np
arr=np.array([3,5,6,8,0,9])
print(arr[1]+arr[5])

import numpy as np

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('2nd element on 1st row: ', arr[0, 1])

import numpy as np

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('5th element on 2nd row: ', arr[1, 4])


import numpy as np

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(arr[0, 1, 2])

import numpy as np

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('Last element from 2nd dim: ', arr[1, -1])

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5])

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[4:])

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[:4])

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[-3:-1])

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5:2])

import numpy as np

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[1, 1:4])
import numpy as np

arr = np.array([10, 15, 20, 25, 30, 35, 40])

print(arr[::2])

import numpy as np

arr = np.array([1, 2, 3, 4])

print(arr.dtype)

import numpy as np

arr = np.array(['apple', 'banana', 'cherry'])

print(arr.dtype)

import numpy as np

arr = np.array([1, 2, 3, 4], dtype='S')

print(arr)
print(arr.dtype)

import numpy as np

arr = np.array([1, 2, 3, 4], dtype='i4')

print(arr)
print(arr.dtype)

import numpy as np

arr = np.array([1.1, 2.1, 3.1])

newarr = arr.astype('i')

print(newarr)
print(newarr.dtype)

import numpy as np

arr = np.array([1.1, 2.1, 3.1])

newarr = arr.astype(int)

print(newarr)
print(newarr.dtype)

import numpy as np

arr = np.array([1, 0, 3])

newarr = arr.astype(bool)

print(newarr)
print(newarr.dtype)

import numpy as np

arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42

print(arr)
print(x)

import numpy as np

arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
y=arr.copy()
arr[0] = 42

print(arr)
print(x)
print(y)

import numpy as np

arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
x[0] = 31

print(arr)
print(x)

import numpy as np

arr = np.array([1, 2, 3, 4, 5])

x = arr.copy()
y = arr.view()

print(x.base)
print(y.base)


import numpy as np

arr = np.array([[1, 2, 3, 4,5], [5, 6, 7, 8,9]])

print(arr.shape)

import numpy as np

arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr)
print('shape of array :', arr.shape)

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

print(arr.reshape(2, 4).base)

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

newarr = arr.reshape(2, 2, -1)

print(newarr)

import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

newarr = arr.reshape(-1)

print(newarr)

import numpy as np

arr = np.array([1, 2, 3])
sum=0

for x in arr:

    sum += arr[x]
    print(sum)

    import numpy as np

    arr = np.array([[1, 2, 3], [4, 5, 6]])

    for x in arr:
        for y in x:
            print(y)

            import numpy as np

            arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

            for x in arr:
                for y in x:
                    for z in y:
                        print(z)

