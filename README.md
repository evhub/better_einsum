# `better_einsum`

_`np.einsum` but better_

```python
from better_einsum import einsum

C = einsum("C[i,k] = A[i,j] B[j,k]", A=A, B=B)
```

Supports:

- better syntax (`"C[i,k] = A[i,j] B[j,k]"` instead of `"ij, jk -> ik"`),
- keyword arguments (`einsum("C = A[i] B[i]", A=..., B=...)`),
- warnings on common bugs (e.g. if the calling scope has a different value for a variable than was passed in),
- an `einsum.exec` method for executing the einsum assignment in the calling scope, and
- an `einsum.set_base_einsum_func` method for using a different base einsum function than `np.einsum`.
