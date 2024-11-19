# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

Parallel Analytics Script:
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/carterhe/Desktop/MLE
Codes/workspace/mod3-CarterHe479/minitorch/fast_ops.py (163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/carterhe/Desktop/MLE Codes/workspace/mod3-CarterHe479/minitorch/fast_ops.py (163)
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        in_storage: Storage,                                                 |
        in_shape: Shape,                                                     |
        in_strides: Strides,                                                 |
    ) -> None:                                                               |
        # TODO: Implement for Task 3.1.                                      |
        if (                                                                 |
            len(out_strides) != len(in_strides)                              |
            or (out_strides != in_strides).any()-----------------------------| #0
            or (out_shape != in_shape).any()---------------------------------| #1
        ):                                                                   |
            for i in prange(len(out)):---------------------------------------| #5
                out_index = np.zeros(MAX_DIMS, dtype=np.int32)---------------| #2
                in_index = np.zeros(MAX_DIMS, dtype=np.int32)----------------| #3
                to_index(i, out_shape, out_index)                            |
                broadcast_index(out_index, out_shape, in_shape, in_index)    |
                out[index_to_position(out_index, out_strides)] = fn(         |
                    in_storage[index_to_position(in_index, in_strides)]      |
                )                                                            |
        else:                                                                |
            for i in prange(len(out)):---------------------------------------| #4
                out[i] = fn(in_storage[i])                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--2 has the following loops fused into it:
   +--3 (fused)
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #0, #1, #5, #2, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--5 is a parallel loop
   +--2 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--5 (parallel)
   +--2 (parallel)
   +--3 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--5 (parallel)
   +--2 (serial, fused with loop(s): 3)



Parallel region 0 (loop #5) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#5).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/carterhe/Desktop/MLE
Codes/workspace/mod3-CarterHe479/minitorch/fast_ops.py (178) is hoisted out of
the parallel loop labelled #5 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/carterhe/Desktop/MLE
Codes/workspace/mod3-CarterHe479/minitorch/fast_ops.py (179) is hoisted out of
the parallel loop labelled #5 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: in_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/carterhe/Desktop/MLE
Codes/workspace/mod3-CarterHe479/minitorch/fast_ops.py (215)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/carterhe/Desktop/MLE Codes/workspace/mod3-CarterHe479/minitorch/fast_ops.py (215)
---------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                          |
        out: Storage,                                                                  |
        out_shape: Shape,                                                              |
        out_strides: Strides,                                                          |
        a_storage: Storage,                                                            |
        a_shape: Shape,                                                                |
        a_strides: Strides,                                                            |
        b_storage: Storage,                                                            |
        b_shape: Shape,                                                                |
        b_strides: Strides,                                                            |
    ) -> None:                                                                         |
        # TODO: Implement for Task 3.1.                                                |
        if (                                                                           |
            len(out_strides) != len(a_strides)                                         |
            or len(out_strides) != len(b_strides)                                      |
            or (out_strides != a_strides).any()----------------------------------------| #6
            or (out_strides != b_strides).any()----------------------------------------| #7
            or (out_shape != a_shape).any()--------------------------------------------| #8
            or (out_shape != b_shape).any()--------------------------------------------| #9
        ):                                                                             |
            for i in prange(len(out)):-------------------------------------------------| #14
                out_index = np.zeros(MAX_DIMS, dtype=np.int32)-------------------------| #10
                a_index = np.zeros(MAX_DIMS, dtype=np.int32)---------------------------| #11
                b_index = np.zeros(MAX_DIMS, dtype=np.int32)---------------------------| #12
                to_index(i, out_shape, out_index)                                      |
                broadcast_index(out_index, out_shape, a_shape, a_index)                |
                broadcast_index(out_index, out_shape, b_shape, b_index)                |
                a_data = a_storage[index_to_position(a_index, a_strides)]              |
                b_data = b_storage[index_to_position(b_index, b_strides)]              |
                out[index_to_position(out_index, out_strides)] = fn(a_data, b_data)    |
        else:                                                                          |
            for i in prange(len(out)):-------------------------------------------------| #13
                out[i] = fn(a_storage[i], b_storage[i])                                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--10 has the following loops fused into it:
   +--11 (fused)
   +--12 (fused)
Following the attempted fusion of parallel for-loops there are 7 parallel for-
loop(s) (originating from loops labelled: #6, #7, #8, #9, #14, #10, #13).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--14 is a parallel loop
   +--10 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--14 (parallel)
   +--10 (parallel)
   +--11 (parallel)
   +--12 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--14 (parallel)
   +--10 (serial, fused with loop(s): 11, 12)



Parallel region 0 (loop #14) had 2 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#14).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/carterhe/Desktop/MLE
Codes/workspace/mod3-CarterHe479/minitorch/fast_ops.py (236) is hoisted out of
the parallel loop labelled #14 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/carterhe/Desktop/MLE
Codes/workspace/mod3-CarterHe479/minitorch/fast_ops.py (237) is hoisted out of
the parallel loop labelled #14 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: a_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/carterhe/Desktop/MLE
Codes/workspace/mod3-CarterHe479/minitorch/fast_ops.py (238) is hoisted out of
the parallel loop labelled #14 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: b_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/carterhe/Desktop/MLE
Codes/workspace/mod3-CarterHe479/minitorch/fast_ops.py (273)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/carterhe/Desktop/MLE Codes/workspace/mod3-CarterHe479/minitorch/fast_ops.py (273)
--------------------------------------------------------------|loop #ID
    def _reduce(                                              |
        out: Storage,                                         |
        out_shape: Shape,                                     |
        out_strides: Strides,                                 |
        a_storage: Storage,                                   |
        a_shape: Shape,                                       |
        a_strides: Strides,                                   |
        reduce_dim: int,                                      |
    ) -> None:                                                |
        # TODO: Implement for Task 3.1.                       |
        for i in prange(len(out)):----------------------------| #16
            out_index = np.zeros(MAX_DIMS, dtype=np.int32)----| #15
            dim = a_shape[reduce_dim]                         |
            to_index(i, out_shape, out_index)                 |
            o = index_to_position(out_index, out_strides)     |
            accum = out[o]                                    |
            j = index_to_position(out_index, a_strides)       |
            s_tem = a_strides[reduce_dim]                     |
            for _ in range(dim):                              |
                accum = fn(accum, a_storage[j])               |
                j += s_tem                                    |
            out[o] = accum                                    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #16, #15).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--16 is a parallel loop
   +--15 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--16 (parallel)
   +--15 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--16 (parallel)
   +--15 (serial)



Parallel region 0 (loop #16) had 0 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#16).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/carterhe/Desktop/MLE
Codes/workspace/mod3-CarterHe479/minitorch/fast_ops.py (284) is hoisted out of
the parallel loop labelled #16 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/carterhe/Desktop/MLE
Codes/workspace/mod3-CarterHe479/minitorch/fast_ops.py (299)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/carterhe/Desktop/MLE Codes/workspace/mod3-CarterHe479/minitorch/fast_ops.py (299)
-----------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                             |
    out: Storage,                                                                        |
    out_shape: Shape,                                                                    |
    out_strides: Strides,                                                                |
    a_storage: Storage,                                                                  |
    a_shape: Shape,                                                                      |
    a_strides: Strides,                                                                  |
    b_storage: Storage,                                                                  |
    b_shape: Shape,                                                                      |
    b_strides: Strides,                                                                  |
) -> None:                                                                               |
    """NUMBA tensor matrix multiply function.                                            |
                                                                                         |
    Should work for any tensor shapes that broadcast as long as                          |
                                                                                         |
    ```                                                                                  |
    assert a_shape[-1] == b_shape[-2]                                                    |
    ```                                                                                  |
                                                                                         |
    Optimizations:                                                                       |
                                                                                         |
    * Outer loop in parallel                                                             |
    * No index buffers or function calls                                                 |
    * Inner loop should have no global writes, 1 multiply.                               |
                                                                                         |
                                                                                         |
    Args:                                                                                |
    ----                                                                                 |
        out (Storage): storage for `out` tensor                                          |
        out_shape (Shape): shape for `out` tensor                                        |
        out_strides (Strides): strides for `out` tensor                                  |
        a_storage (Storage): storage for `a` tensor                                      |
        a_shape (Shape): shape for `a` tensor                                            |
        a_strides (Strides): strides for `a` tensor                                      |
        b_storage (Storage): storage for `b` tensor                                      |
        b_shape (Shape): shape for `b` tensor                                            |
        b_strides (Strides): strides for `b` tensor                                      |
                                                                                         |
    Returns:                                                                             |
    -------                                                                              |
        None : Fills in `out`                                                            |
                                                                                         |
    """                                                                                  |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                               |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                               |
                                                                                         |
    # TODO: Implement for Task 3.2.                                                      |
    for n in prange(out_shape[0]):-------------------------------------------------------| #19
        for i in prange(out_shape[1]):---------------------------------------------------| #18
            for j in prange(out_shape[2]):-----------------------------------------------| #17
                a_idx = n * a_batch_stride + i * a_strides[1]                            |
                b_idx = n * b_batch_stride + j * b_strides[2]                            |
                accum = 0.0                                                              |
                                                                                         |
                for _ in range(a_shape[2]):                                              |
                    accum += a_storage[a_idx] * b_storage[b_idx]                         |
                    a_idx += a_strides[2]                                                |
                    b_idx += b_strides[1]                                                |
                out[n * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = (    |
                    accum                                                                |
                )                                                                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #19, #18).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--19 is a parallel loop
   +--18 --> rewritten as a serial loop
      +--17 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--19 (parallel)
   +--18 (parallel)
      +--17 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--19 (parallel)
   +--18 (serial)
      +--17 (serial)



Parallel region 0 (loop #19) had 0 loop(s) fused and 2 loop(s) serialized as
part of the larger parallel loop (#19).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None