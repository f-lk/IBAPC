from scipy.io import loadmat
import os
import torch
# import pytorch3d.ops
# import pytorch3d.utils
from sklearn.neighbors import KDTree

@torch.no_grad()
def eig_vector(data, K):
    device = torch.device("cuda")
    b, n, _ = data.shape
    #_, idx, _ = pytorch3d.ops.knn_points(data, data, K=K)  # idx (b,n,K)
    idx_list = []
    for i in range(len(data)):
        kdtree = KDTree(data[i].detach().cpu().numpy())
        _, idx = kdtree.query(data[i].detach().cpu().numpy(), k=K)
        idx_list.append(idx)

    idx = torch.tensor(idx_list).to(device)

    idx0 = torch.arange(0,b,device=data.device).reshape((b,1)).expand(-1,n*K).reshape((1,b*n*K))
    idx1 = torch.arange(0,n,device=data.device).reshape((1,n,1)).expand(b,n,K).reshape((1,b*n*K))
    idx = idx.reshape((1,b*n*K))
    idx = torch.cat([idx0, idx1, idx], dim=0) # (3, b*n*K)
    # print(b, n, K, idx.shape)
    ones = torch.ones(idx.shape[1], dtype=bool, device=data.device)
    A = torch.sparse_coo_tensor(idx, ones, (b, n, n)).to_dense() # (b,n,n)
    A = A | A.transpose(1, 2)
    A = A.float()  ## 邻接矩阵 A, 1024 * 1024 其中每个元素为 1 或 0， 1 表示对应的两个点有连接关系
    deg = torch.diag_embed(torch.sum(A, dim=2))  ## 对角矩阵
    laplacian = deg - A
    u, v = torch.linalg.eig(laplacian) # (b,n,n) 特征值分解  特征值 u 和 特征向量 v
    return v.real, laplacian, u.real

def GFT(pc_ori, K, factor):
    x = pc_ori.transpose(0,1) #(b,n,3)
    b, n, _ = x.shape
    v = eig_vector(x, K)
    x_ = torch.einsum('bij,bjk->bik',v.transpose(1,2), x) # (b,n,3)
    x_ = torch.einsum('bij,bi->bij', x_, factor)
    x = torch.einsum('bij,bjk->bik',v, x_)
    return x

@torch.no_grad()
def eig_vector_ori(data, K):
    b, n, _ = data.shape
    #_, idx, _ = pytorch3d.ops.knn_points(data, data, K=K)  # idx (b,n,K)
    kdtree = KDTree(data[0])
    _, idx = kdtree.query(data[0], k=K)

    idx = torch.tensor([idx])

    idx0 = torch.arange(0,b,device=data.device).reshape((b,1)).expand(-1,n*K).reshape((1,b*n*K))
    idx1 = torch.arange(0,n,device=data.device).reshape((1,n,1)).expand(b,n,K).reshape((1,b*n*K))
    idx = idx.reshape((1,b*n*K))
    idx = torch.cat([idx0, idx1, idx], dim=0) # (3, b*n*K)
    # print(b, n, K, idx.shape)
    ones = torch.ones(idx.shape[1], dtype=bool, device=data.device)
    A = torch.sparse_coo_tensor(idx, ones, (b, n, n)).to_dense() # (b,n,n)
    A = A | A.transpose(1, 2)
    A = A.float()  ## 邻接矩阵 A, 1024 * 1024 其中每个元素为 1 或 0， 1 表示对应的两个点有连接关系
    deg = torch.diag_embed(torch.sum(A, dim=2))  ## 对角矩阵
    laplacian = deg - A
    u, v = torch.linalg.eig(laplacian) # (b,n,n) 特征值分解  特征值 u 和 特征向量 v
    return v.real, laplacian, u.real











