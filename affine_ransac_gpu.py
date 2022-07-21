# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/10
#


import numpy as np
from affine_transform import Affine
import numba as nb
from numba import cuda
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32


# The number of iterations in RANSAC
ITER_NUM = 1000
NUM = 3
NUM_2 = NUM * 2
POINT_NUM = 1000

@cuda.jit(device=True)
def transpose(M,MT):
    rows = M.shape[0]
    cols = M.shape[1]
    for i in range(rows):
        for j in range(cols):
            MT[j][i] = M[i][j]

@cuda.jit(device=True)
def matrix_multiply(A, B, C):
    rowsA = A.shape[0]
    colsA = A.shape[1]
    colsB  = B.shape[1]
    for i in range(rowsA):
        for j in range(colsB):
            total = 0
            for ii in range(colsA):
                total += A[i][ii] * B[ii][j]
            C[i][j] = total
    return C

@cuda.jit(device=True)
def determinant_fast(A,):
    """
    Create an upper triangle matrix using row operations.
        Then product of diagonal elements is the determinant
        :param A: the matrix to find the determinant for
        :return: the determinant of the matrix
    """
    # Section 1: Establish n parameter and copy A
    n = A.shape[0]
    AM = cuda.local.array(shape=(NUM_2,NUM_2),dtype=nb.float32)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            AM[i][j] = A[i][j]
    
    # Section 2: Row manipulate A into an upper triangle matrix
    for fd in range(n): # fd stands for focus diagonal
        if AM[fd][fd] == 0:
            AM[fd][fd] = 1.0e-18 # Cheating by adding zero + ~zero
        for i in range(fd+1,n): # skip row with fd in it.
            crScaler = AM[i][fd] / AM[fd][fd] # cr stands for "current row".
            for j in range(n): # cr - crScaler * fdRow, but one element at a time.
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
    # Section 3: Once AM is in upper triangle form ...
    product = 1.0
    for i in range(n):
        product *= AM[i][i] # ... product of diagonals is determinant
    return product

@cuda.jit(device=True)
def check_non_singular(A):
    """
    Ensure matrix is NOT singular
        :param A: The matrix under consideration
        :return: determinant of A - nonzero is positive boolean
                  otherwise, raise ArithmeticError
    """
    det = determinant_fast(A)
    # print(det)
    if det != 0:
        return det
    else:
        # raise "Singular Matrix!"
        return 0

@cuda.jit(device=True)
def check_matrix_equality(A,B,tol=None):
    if A.shape[0] != B.shape[0] or A.shape[1] != B.shape[1]:
        return False
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if round(A[i][j],tol) != round(B[i][j],tol):
                return False
    return True
    

@cuda.jit(device=True)
def solve_equations(A,AM,B,BM,tol):
    """
    Returns the solution of a system of equations in matrix format.
        :param A: The system matrix
        :return: The solution X where AX = B
    """
    # Section 1: Make sure A can be inverted.
    check_non_singular(A)

    # Section 2: Make copies of A & I, AM & IM, to use for row operations
    n = A.shape[0]
    # IM = cuda.local.array(shape=(NUM_2,NUM_2),dtype=nb.int32)
    # for i in range(NUM_2):
    #     IM[i,i] = 1
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            BM[i,j] = B[i,j]
    
    # Section 3: Perform row operations
    indices = cuda.local.array(shape=NUM_2,dtype=nb.int32)
    for i in range(NUM_2):
        indices[i] = i
    for fd in range(n): # fd stands for focus diagonal
        if AM[fd][fd] == 0:
            AM[fd][fd] = 1.0e-18
        fdScaler = 1.0 / AM[fd][fd]
        # FIRST: scale fd row with fd inverse. 
        for j in range(n): # Use j to indicate column looping.
            AM[fd][j] *= fdScaler
        BM[fd][0] *= fdScaler
        # SECOND: operate on all rows except fd row as follows:
        for i in indices: # *** skip row with fd in it.
            if i == fd: continue
            crScaler = AM[i][fd] # cr stands for "current row".
            for j in range(n): # cr - crScaler * fdRow, but one element at a time.
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                # IM[i][j] = IM[i][j] - crScaler * IM[fd][j]
            BM[i][0] = BM[i][0] - crScaler * BM[fd][0]
    

    # # Section 4: Make sure that BM is the solution for X
    res = cuda.local.array(shape=(NUM_2,1),dtype=nb.float32)
    res = matrix_multiply(A,BM,res)
    
    res = check_matrix_equality(B,res,tol)
    return res

@cuda.jit(device=True)
def estimate_affine(source,target,A,t):
    M = cuda.local.array((NUM_2,6),dtype=nb.float32)
    B = cuda.local.array((NUM_2,1),dtype=nb.float32)
    for i in range(NUM):
        M[i*2,0] = source[0,i]
        M[i*2,1] = source[1,i]
        M[i*2,4] = 1
        M[i*2+1,2] = source[0,i]
        M[i*2+1,3] = source[1,i]
        M[i*2+1,5] = 1
    for i in range(NUM):
        B[i*2,0] = target[0,i]
        B[i*2+1,0] = target[1,i]
    
    # np.linalg.lstsq(M,b) with pure python and numba
    MT = cuda.local.array((6,NUM_2),dtype=nb.float32)
    transpose(M,MT) # 转置
    
    MTM = cuda.local.array((NUM_2,NUM_2),dtype=nb.float32)
    matrix_multiply(MT,M,MTM)
    
    MTB = cuda.local.array((NUM_2,1),dtype=nb.float32)
    matrix_multiply(MT,B,MTB)
    
    BM = cuda.local.array((NUM_2,1),dtype=nb.float32)
    MTM_copy = cuda.local.array((NUM_2,NUM_2),dtype=nb.float32)
    MTB_copy = cuda.local.array((NUM_2,1),dtype=nb.float32)
    for i in range(MTM.shape[0]):
        for j in range(MTM.shape[1]):
            MTM_copy[i][j] = MTM[i][j]
    for i in range(MTB.shape[0]):
        for j in range(MTB.shape[1]):
            MTB_copy[i][j] = MTB[i][j]
    res = solve_equations(MTM,MTM_copy,MTB,BM,2)
    
    A[0,0] = BM[0][0]
    A[0,1] = BM[1][0]
    A[1,0] = BM[2][0]
    A[1,1] = BM[3][0]
    t[0] = BM[4][0]
    t[1] = BM[5][0]

@cuda.jit(device=True)
def matrix_dot(A, B, C):
    """
    Returns the product of the matrix A * B
        :param A: The first matrix - ORDER MATTERS!
        :param B: The second matrix
        :return: The product of the two matrices
    """
    # Section 1: Ensure A & B dimensions are correct for multiplication
    rowsA = len(A)
    colsA = len(A[0])
    colsB = len(B[0])
    # Section 2: Store matrix multiplication in a new matrix
    for i in range(rowsA):
        for j in range(colsB):
            total = 0
            for ii in range(colsA):
                total += A[i][ii] * B[ii][j]
            C[i][j] = total



@cuda.jit(device=True)
def residual_lengths(A, t, pts_s, pts_t,residual):
    if not(A is None) and not(t is None):
        # Calculate estimated points:
        # pts_esti = A * pts_s + t
        val = cuda.local.array(shape=(2,POINT_NUM),dtype=nb.float32)
        matrix_dot(A, pts_s, val)
        pts_e = cuda.local.array(shape=(2,POINT_NUM),dtype=nb.float32)
        for i in range(pts_e.shape[1]):
            pts_e[0][i] = val[0][i] + t[0]
            pts_e[1][i] = val[1][i] + t[1]
        # Calculate the residual length between estimated points
        # and target points
        for i in range(pts_t.shape[1]):
            diff_sum = (pts_t[0][i]-pts_e[0][i])**2 + (pts_t[1][i]-pts_e[1][i])**2
            residual[i] = math.sqrt(diff_sum)
        # diff_square = np.power(pts_e - pts_t, 2)
        # residual = np.sqrt(np.sum(diff_square, axis=0))

@cuda.jit()
def ransac_gpu(pts_s,pts_t,threshold,rng_states,index_num_memory,index_val_memory,As_memory,ts_memory):
    index = nb.cuda.grid(1)
    if index < ITER_NUM:
        indexs = cuda.local.array(shape=(NUM),dtype=nb.int32)
        source = cuda.local.array(shape=(2,NUM),dtype=nb.float32)
        target = cuda.local.array(shape=(2,NUM),dtype=nb.float32)
        for i in range(NUM):
            indexs[i] = int(xoroshiro128p_uniform_float32(rng_states, index)*pts_s.shape[1])
            source[0,i] = pts_s[0,indexs[i]]
            source[1,i] = pts_s[1,indexs[i]]
            target[0,i] = pts_t[0,indexs[i]]
            target[1,i] = pts_t[1,indexs[i]]
        A_tmp = cuda.local.array((2,2),dtype=nb.float32)
        t_tmp = cuda.local.array((2),dtype=nb.float32)
        estimate_affine(source,target,A_tmp,t_tmp)
        
        residual = cuda.local.array(shape=(POINT_NUM,),dtype=nb.float32)
        residual_lengths(A_tmp, t_tmp, pts_s, pts_t,residual)
        
        inliers_tmp = cuda.local.array(shape=(POINT_NUM),dtype=nb.int32)
        num = 0
        for i in range(pts_s.shape[1]):
            if residual[i] < threshold:
                inliers_tmp[num] = i
                num += 1
        index_num_memory[index] = num
        for i in range(num):
            index_val_memory[index][i] = inliers_tmp[i]
        As_memory[index][0][0] = A_tmp[0][0]
        As_memory[index][0][1] = A_tmp[0][1]
        As_memory[index][1][0] = A_tmp[1][0]
        As_memory[index][1][1] = A_tmp[1][1]
        ts_memory[index][0] = t_tmp[0]
        ts_memory[index][1] = t_tmp[1]            
        

def gpu_test():
    af = Affine()
    outlier_rate = 0.8
    
    thread_per_grid = 512
    block_per_grid = math.ceil(ITER_NUM / thread_per_grid)
    
    A_true, t_true, pts_s, pts_t = af.create_test_case(outlier_rate,point_num=POINT_NUM)
    print(pts_s.shape,pts_t.shape)
    print("gt_info:\n","A:\n",A_true,"\n t:\n",t_true)
    pts_s_device = cuda.to_device(pts_s)
    pts_t_device = cuda.to_device(pts_t)
    
    threshold = 1
    rng_states = create_xoroshiro128p_states(thread_per_grid * block_per_grid, seed=1)
    
    index_num = np.zeros(shape=(ITER_NUM),dtype=np.int32)
    index_val = np.zeros(shape=(ITER_NUM,POINT_NUM),dtype=np.int32)
    As = np.zeros(shape=(ITER_NUM,2,2),dtype=np.float32)
    ts = np.zeros(shape=(ITER_NUM,2),dtype=np.float32)
    
    index_num_memory = cuda.to_device(index_num)
    index_val_memory = cuda.to_device(index_val)
    As_memory = cuda.to_device(As)
    ts_memory = cuda.to_device(ts)
    
    ransac_gpu[block_per_grid,thread_per_grid](pts_s_device,pts_t_device,threshold,rng_states,index_num_memory,index_val_memory,As_memory,ts_memory)
    
    index_num = index_num_memory.copy_to_host()
    inter_index_val = index_val_memory.copy_to_host()
    As = As_memory.copy_to_host()
    ts = ts_memory.copy_to_host()
    
    max_index = np.argmax(index_num)
    max_val = np.max(index_num)
    inliers = inter_index_val[max_index]
    A = As[max_index]
    t = ts[max_index]
    print("ransac:\n","A:\n",A,"\n t:\n",t)
    
    return A, t, inliers
