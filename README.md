# `better_einsum`

_`np.einsum` but better:_

- better syntax (`"C[i,k] = A[i,j] B[j,k]"` instead of `"ij, jk -> ik"`),
- names and indices can be arbitrary variable names not just single letters,
- keyword arguments (`einsum("C = A[i] B[i]", A=..., B=...)`),
- warnings on common bugs,
- an `einsum.exec` method for executing the einsum assignment in the calling scope, and
- a `base_einsum_func` keyword argument for using a different base einsum function than `np.einsum`.

`pip install better_einsum` then:

```pycon
>>> import numpy as np
>>> from better_einsum import einsum

>>> A = np.array([[1, 2], [3, 4]])
>>> B = np.array([[5, 6], [7, 8]])

>>> einsum("C[i,k] = A[i,j] * B[j,k]", A=A, B=B)  # equivalent to A.dot(B)
array([[19, 22],
       [43, 50]])

>>> einsum("C = A[i,j] * B[i,j]", A=A, B=B)  # equivalent to np.sum(A * B)
70

>>> einsum("C[...] = A[i,...] * B[i,...]", A=A, B=B)  # equivalent to np.sum(A * B, axis=0)
array([26, 44])

>>> einsum("C[i,k] = A[i,j] B[j,k]", A, B)  # * is optional; positional args are also supported
array([[19, 22],
       [43, 50]])

>>> einsum("C[i,k] = A[i,j] * B[j,k]", A, A)  # better_einsum will catch common mistakes for you
better_einsum.py: UserWarning: better_einsum: variable 'B' in calling scope points to a different object than was passed in; this usually denotes an error
array([[ 7, 10],
       [15, 22]])

>>> einsum("_[i,k] = _[i,j] * _[j,k]", A, B)  # use placeholders if you don't want to name your variables
array([[19, 22],
       [43, 50]])

>>> einsum.exec("C[i,k] = A[i,j] * B[j,k]", A=A, B=B)  # directly assigns to C
array([[19, 22],
       [43, 50]])
>>> C
array([[19, 22],
       [43, 50]])

>>> import jax.numpy as jnp
>>> from functools import partial
>>> jnp_einsum = partial(einsum, base_einsum_func=jnp.einsum)  # better_einsum for JAX
```
