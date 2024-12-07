import spectral_attack
import numpy as np
import torch
import pyvista as pv


def GFT_opt(point, GFT_noise, v):
    point_gft = torch.einsum('bij,bjk->bik', v.transpose(1, 2), point)  # (b,n,3)
    for i in range(len(point_gft)):
        point_gft[i] = point_gft[i] + GFT_noise

    ### I-GFT
    point2 = torch.einsum('bij,bjk->bik', v, point_gft).transpose(1, 2)
    point2 = point2.transpose(2, 1)
    return point2

def show_pl(pointclouds_pl_adv, special_index):
    pointclouds_pl_adv = pointclouds_pl_adv.squeeze()
    p = pv.Plotter()
    camera = pv.Camera()
    camera.position = (18, 4, -20)
    camera.focal_point = (0, 0, 0)
    for i in range(len(pointclouds_pl_adv)):
        p.add_mesh(pv.PolyData(pointclouds_pl_adv[i]), color=[0, 0, 0], point_size=np.float(11),
                   render_points_as_spheres=True)
    p.camera = camera
    p.show()

def GFT(point, GFT_noise):
    K = 10
    v, laplacian, u = spectral_attack.eig_vector(point, K)  #
    ### GFT
    point_gft = torch.einsum('bij,bjk->bik', v.transpose(1, 2), point)  # (b,n,3)
    point_gft = point_gft + GFT_noise

    ### I-GFT
    point2 = torch.einsum('bij,bjk->bik', v, point_gft).transpose(1, 2)
    point2 = point2.transpose(2, 1)
    return point2




