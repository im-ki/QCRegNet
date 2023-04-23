import os
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from PIL import Image
import matplotlib.pyplot as plt
import time
import torch

def image_meshgen(height, width):
    """
    Inputs:
        height: int
        width: int
    Outputs:
        face : m x 3 index of trangulation connectivity
        vertex : n x 2 vertices coordinates(x, y)
    """    
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    y = y[::-1, :]
    x = x/(width-1)
    y = y/(height-1)
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    vertex = np.hstack((x, y))
    face = np.zeros(((height-1)*(width-1)*2, 3)).astype(np.int)

    # original code with for loop
    #for i in range(height-1):
    #    for j in range(width-1):
    #        face[i*(width-1)*2+j*2:i*(width-1)*2+(j+1)*2] = np.array(((i*width+j, i*width+j+1, (i+1)*width+j),
    #                                                        ((i+1)*width+j+1, i*width+j+1, (i+1)*width+j)))
    #
    # without for loop
    ind = np.arange(height*width).reshape((height, width))
    mid = ind[0:-1, 1:]
    left1 = ind[0:-1, 0:-1]
    left2 = ind[1:, 1:]
    right = ind[1:, 0:-1]
    face[0::2, 0] = left1.reshape(-1)
    face[0::2, 1] = right.reshape(-1)
    face[0::2, 2] = mid.reshape(-1)
    face[1::2, 0] = left2.reshape(-1)
    face[1::2, 1] = mid.reshape(-1)
    face[1::2, 2] = right.reshape(-1)

    return face, vertex

def generalized_laplacian2D(face, vertex, mu, h, w):
    """
    Inputs:
        face : m x 3 index of triangulation connectivity
        vertex : n x 2 vertices coordinates(x, y)
        mu : m x 1 Beltrami coefficients
        h, w: int
    Outputs:
        A : 2-dimensional generalized laplacian operator (h*w, h*w)
        abc : vectors containing the coefficients alpha, beta and gamma (m, 3)
        area : float, area of every triangles in the mesh
    """
    af = (1 - 2 * np.real(mu) + np.abs(mu)**2) / (1 - np.abs(mu)**2)
    bf = -2 * np.imag(mu) / (1 - np.abs(mu)**2)
    gf = (1 + 2 * np.real(mu) + np.abs(mu)**2) / (1 - np.abs(mu)**2)
    abc = np.hstack((af, bf, gf))
    
    f0, f1, f2 = face[:, 0, np.newaxis], face[:, 1, np.newaxis], face[:, 2, np.newaxis]

    uxv0 = vertex[f1,1] - vertex[f2,1]
    uyv0 = vertex[f2,0] - vertex[f1,0]
    uxv1 = vertex[f2,1] - vertex[f0,1]
    uyv1 = vertex[f0,0] - vertex[f2,0] 
    uxv2 = vertex[f0,1] - vertex[f1,1]
    uyv2 = vertex[f1,0] - vertex[f0,0]

    area = (1/(h-1)) * (1/(w-1)) / 2

    v00 = (af * uxv0 * uxv0 + 2 * bf * uxv0 * uyv0 + gf * uyv0 * uyv0) / area;
    v11 = (af * uxv1 * uxv1 + 2 * bf * uxv1 * uyv1 + gf * uyv1 * uyv1) / area;
    v22 = (af * uxv2 * uxv2 + 2 * bf * uxv2 * uyv2 + gf * uyv2 * uyv2) / area;

    v01 = (af * uxv1 * uxv0 + bf * uxv1 * uyv0 + bf * uxv0 * uyv1 + gf * uyv1 * uyv0) / area;
    v12 = (af * uxv2 * uxv1 + bf * uxv2 * uyv1 + bf * uxv1 * uyv2 + gf * uyv2 * uyv1) / area;
    v20 = (af * uxv0 * uxv2 + bf * uxv0 * uyv2 + bf * uxv2 * uyv0 + gf * uyv0 * uyv2) / area;

    I = np.vstack((f0,f1,f2,f0,f1,f1,f2,f2,f0)).reshape(-1)
    J = np.vstack((f0,f1,f2,f1,f0,f2,f1,f0,f2)).reshape(-1)
    nRow = vertex.shape[0]
    V = np.vstack((v00,v11,v22,v01,v01,v12,v12,v20,v20)).reshape(-1) / 2
    A = sps.coo_matrix((-V, (I, J)), shape = (nRow, nRow))

    return A, abc, area

def lbs(mu, h, w):
    """
    Inputs:
        mu : m x 1 Beltrami coefficients
        h, w: int
    Outputs:
        A : 2-dimensional generalized laplacian operator (h*w, h*w)
    """
    face, vertex = image_meshgen(h, w)
    A, abc, area = generalized_laplacian2D(face, vertex, mu, h, w) 
    return A

def batch_mu2reshape(mu_pad0):
    """
    Inputs:
        mu_pad0 : (N, h, w), the elements in the right-most column and bottom row are padded by 0.
    Outputs:
        mu : (N, 2, (h-1)*(w-1)*2), add removed elements back to the input mu_pad0
    """
    N, h, w = mu_pad0.shape
    mu = mu_pad0[:, :h-1, :w-1]

    mu_reshape = np.zeros((N, (h-1), (w-1)*2), dtype = np.complex)
    mu_reshape[:, :, ::2] = mu
    mu_reshape[:, :-1, 1:-1:2] = (mu[:, :-1, :-1] + mu[:, :-1, 1:] + mu[:, 1:, :-1]) / 3
    mu_reshape[:, :-1, -1] = (mu[:, :-1, -1] + mu[:, 1:, -1]) / 2
    mu_reshape[:, -1, 1:-1:2] = (mu[:, -1, :-1] + mu[:, -1, 1:]) / 2
    mu_reshape[:, -1, -1] = mu[:, -1, -1]
    mu = mu_reshape.reshape((N, 1, -1))
    mu_r, mu_i = np.real(mu), np.imag(mu)
    mu = np.concatenate((mu_r, mu_i), axis=1).reshape((N, 2, (h-1)*(w-1)*2))

    #r, i = mu[:, 0].reshape(N, 1, -1), mu[:, 1].reshape(N, 1, -1)
    #print(np.sum(np.abs(r - mu_r)))
    #print(np.sum(np.abs(i - mu_i)))
    #
    #print(mu.shape)
    return mu 

def batch_mu2reshape_torch(mu_pad0, device):
    """
    Inputs:
        mu_pad0 : (N, 2, h, w), the elements in the right-most column and bottom row are padded by 0.
    Outputs:
        mu : (N, 2, (h-1)*(w-1)*2), add removed elements back to the input mu_pad0
    """
    N, _, h, w = mu_pad0.shape
    mu = mu_pad0[:, :, :h-1, :w-1]

    mu_reshape = torch.zeros((N, 2, (h-1), (w-1)*2), device = device)
    mu_reshape[:, :, :, ::2] = mu
    mu_reshape[:, :, :-1, 1:-1:2] = (mu[:, :, :-1, :-1] + mu[:, :, :-1, 1:] + mu[:, :, 1:, :-1]) / 3
    mu_reshape[:, :, :-1, -1] = (mu[:, :, :-1, -1] + mu[:, :, 1:, -1]) / 2
    mu_reshape[:, :, -1, 1:-1:2] = (mu[:, :, -1, :-1] + mu[:, :, -1, 1:]) / 2
    mu_reshape[:, :, -1, -1] = mu[:, :, -1, -1]
    mu = mu_reshape.reshape((N, 2, -1))
    return mu

def detD(device, H, W):
    face, vertex = image_meshgen(H, W)
    #print('mask', boundary_mask, boundary_mask.shape, np.sum(boundary_mask))
    face = torch.from_numpy(face).to(device=device)
    vertex = torch.from_numpy(vertex).to(device=device)
    area = 1/((H-1)*(W-1)*2)
    relu = torch.nn.ReLU()

    def detD_loss(mapping):
        """
        Inputs:
            mapping: (N, 2, h, w), torch tensor
        Outputs:
            loss: (N, (h-1)*(w-1)*2), torch tensor
        """
        N, C, H, W = mapping.shape
        mapping = mapping.reshape((N, C, -1))
        mapping = mapping.permute(0, 2, 1)

        si = mapping[:, face[:, 0], 0]
        sj = mapping[:, face[:, 1], 0]
        sk = mapping[:, face[:, 2], 0]

        ti = mapping[:, face[:, 0], 1]
        tj = mapping[:, face[:, 1], 1]
        tk = mapping[:, face[:, 2], 1]

        sjsi = sj - si
        sksi = sk - si
        tjti = tj - ti
        tkti = tk - ti

        a = (sjsi) / area / 2;
        b = (sksi) / area / 2;
        c = (tjti) / area / 2;
        d = (tkti) / area / 2;

        det = (a*d-c*b)# * boundary_mask
        loss = torch.mean(relu(-det))
        return loss #mu
    return detD_loss

def generalized_laplacian2D_torch(face, vertex, h, w, device):
    """
    Inputs:
        face : m x 3 index of triangulation connectivity
        vertex : n x 2 vertices coordinates(x, y)
        mu : N x 2 x m Beltrami coefficients
        h, w: int
    Outputs:
        A : 2-dimensional generalized laplacian operator (h*w, h*w)
        abc : vectors containing the coefficients alpha, beta and gamma (m, 3)
        area : float, area of every triangles in the mesh
    """
    f0, f1, f2 = face[:, 0], face[:, 1], face[:, 2]

    uxv0 = (vertex[f1,1] - vertex[f2,1]).reshape((1, -1))
    uyv0 = (vertex[f2,0] - vertex[f1,0]).reshape((1, -1))
    uxv1 = (vertex[f2,1] - vertex[f0,1]).reshape((1, -1))
    uyv1 = (vertex[f0,0] - vertex[f2,0]).reshape((1, -1))
    uxv2 = (vertex[f0,1] - vertex[f1,1]).reshape((1, -1))
    uyv2 = (vertex[f1,0] - vertex[f0,0]).reshape((1, -1))

    area = (1/(h-1)) * (1/(w-1)) / 2

    I = torch.cat((f0,f1,f2,f0,f1,f1,f2,f2,f0))
    J = torch.cat((f0,f1,f2,f1,f0,f2,f1,f0,f2))

    def generalized_laplacian2D(mu):
        mu_sqr = torch.sum(mu**2, dim=1)
        af = (1 - 2 * mu[:, 0] + mu_sqr) / (1 - mu_sqr)
        bf = -2 * mu[:, 1] / (1 - mu_sqr)
        gf = (1 + 2 * mu[:, 0] + mu_sqr) / (1 - mu_sqr)
        #abc = np.hstack((af, bf, gf))

        v00 = (af * uxv0 * uxv0 + 2 * bf * uxv0 * uyv0 + gf * uyv0 * uyv0) / area;
        v11 = (af * uxv1 * uxv1 + 2 * bf * uxv1 * uyv1 + gf * uyv1 * uyv1) / area;
        v22 = (af * uxv2 * uxv2 + 2 * bf * uxv2 * uyv2 + gf * uyv2 * uyv2) / area;

        v01 = (af * uxv1 * uxv0 + bf * uxv1 * uyv0 + bf * uxv0 * uyv1 + gf * uyv1 * uyv0) / area;
        v12 = (af * uxv2 * uxv1 + bf * uxv2 * uyv1 + bf * uxv1 * uyv2 + gf * uyv2 * uyv1) / area;
        v20 = (af * uxv0 * uxv2 + bf * uxv0 * uyv2 + bf * uxv2 * uyv0 + gf * uyv0 * uyv2) / area;

        nRow = vertex.shape[0]
        N = mu.shape[0]
        V = torch.cat((v00,v11,v22,v01,v01,v12,v12,v20,v20), dim=1).type(torch.float) / 2
        A = torch.zeros((N, nRow, nRow), dtype=torch.float, device=device)
        A[:, I, J] = -V
        return A, area
    return generalized_laplacian2D

def mu2A_torch(h, w, device):
    face, vertex = image_meshgen(h, w)
    face = torch.from_numpy(face).to(device=device)
    vertex = torch.from_numpy(vertex).to(device=device)
    ind = torch.arange(h*w).reshape((h, w)).to(device=device)
    #ind1 = ind[1:, 1:].reshape(-1)
    ind2 = ind[1:, :].reshape(-1)
    ind3 = ind[1:, :-1].reshape(-1)
    ind4 = ind[:, 1:].reshape(-1)
    ind5 = ind[:, :].reshape(-1)
    ind6 = ind[:, :-1].reshape(-1)
    ind7 = ind[:-1, 1:].reshape(-1)
    ind8 = ind[:-1, :].reshape(-1)
    #ind9 = ind[:-1, :-1].reshape(-1)
    lapla = generalized_laplacian2D_torch(face, vertex, h, w, device)

    def mu2A(mu_pad0):
        """
        Inputs:
            mu_pad0 : (N, 2, h, w), the elements in the right-most column and bottom row are padded by 0.
        Outputs:
            A : 2-dimensional generalized laplacian operator (N, 7, h, w)
        """
        with torch.no_grad():
            mu = batch_mu2reshape_torch(mu_pad0, device)
            A, area = lapla(mu)
            e = torch.zeros((mu.shape[0], 7, h*w), device=device)
            #e[ind1, 0] = A[ind1, ind1-(w+1)]
            e[:, 0, ind2] = A[:, ind2, ind2-w]
            e[:, 1, ind3] = A[:, ind3, ind3-(w-1)]
            e[:, 2, ind4] = A[:, ind4, ind4-1]
            e[:, 3, ind5] = A[:, ind5, ind5]
            e[:, 4, ind6] = A[:, ind6, ind6+1]
            e[:, 5, ind7] = A[:, ind7, ind7+(w-1)]
            e[:, 6, ind8] = A[:, ind8, ind8+w]
            #e[ind9, 8] = A[ind9, ind9+(w+1)]
            A = e.reshape((mu.shape[0], 7, h, w))
        return A
    return mu2A

def mu2A(mu_pad0):
    """
    Inputs:
        mu_pad0 : (h, w), the elements in the right-most column and bottom row are padded by 0.
    Outputs:
        A : 2-dimensional generalized laplacian operator (h*w, h*w)
    """
    # a = time.time()
    h, w = mu_pad0.shape
    mu = mu_pad0[:h-1, :w-1]

    mu_reshape = np.zeros(((h-1), (w-1)*2), dtype = np.complex)
    mu_reshape[:, ::2] = mu
    mu_reshape[:-1, 1:-1:2] = (mu[:-1, :-1] + mu[:-1, 1:] + mu[1:, :-1]) / 3
    mu_reshape[:-1, -1] = (mu[:-1, -1] + mu[1:, -1]) / 2
    mu_reshape[-1, 1:-1:2] = (mu[-1, :-1] + mu[-1, 1:]) / 2
    mu_reshape[-1, -1] = mu[-1, -1]
    mu = mu_reshape.reshape((-1, 1))
    
    A = lbs(mu, h, w);
    #x = mapping[:, 0].reshape((h, w))[np.newaxis, ...]
    #y = mapping[:, 1].reshape((h, w))[np.newaxis, ...]
    #mapping = np.vstack((x, y))
    # b = time.time()
    # print('Time(s) of solving Ax=b:', b-a)
    #return mapping
    return A

def mu2map(mu_pad0):
    """
    Inputs:
        mu_pad0 : (h, w), the elements in the right-most column and bottom row are padded by 0.
    Outputs:
        mapping: (2, h, w)
        Ax, Ay: (h*w, h*w)
        bx, by: (h*w, 1)
    """
    # a = time.time()
    h, w = mu_pad0.shape
    hw = h*w
    Ax = mu2A(mu_pad0).tolil()
    Ay = Ax.copy()
    bx = np.zeros((hw, 1))
    by = bx.copy()

    Edge1 = np.reshape(np.arange((h-1)*w, hw), (w, 1))
    Edge2 = np.reshape(np.arange(w-1, hw, step = w), (h, 1))
    Edge3 = np.reshape(np.arange(w), (w, 1))
    Edge4 = np.reshape(np.arange((h-1)*w+1, step = w), (h, 1))

    landmarkx = np.vstack((Edge4, Edge2))
    targetx = np.vstack((np.zeros_like(Edge4), np.ones_like(Edge2)))
    lmx = landmarkx.reshape(-1)
    bx[lmx] = targetx
    Ax[lmx, :] = 0
    tmp = sps.csr_matrix((np.ones_like(lmx), (np.arange(lmx.shape[0]), lmx)), shape = (lmx.shape[0], hw)).tolil()
    Ax[lmx, :] = tmp
    mapx = spsolve(Ax.tocsc(), bx).reshape((h, w))#.reshape((-1, 1))

    landmarky = np.vstack((Edge1, Edge3))
    targety = np.vstack((np.zeros_like(Edge1), np.ones_like(Edge3)))
    lmy = landmarky.reshape(-1)
    by[lmy] = targety
    Ay[lmy, :] = 0
    tmp = sps.csr_matrix((np.ones_like(lmy), (np.arange(lmy.shape[0]), lmy)), shape = (lmy.shape[0], hw)).tolil()
    Ay[lmy, :] = tmp
    mapy = spsolve(Ay.tocsc(), by).reshape((h, w))#.reshape((-1, 1))

    mapping = np.array((mapx, mapy))
    return mapping, Ax, Ay, bx, by

def plot_map(mapping):
    """
    Inputs:
        mapping: (2, h, w)
    """
    x = mapping[0].reshape((-1, 1))
    y = mapping[1].reshape((-1, 1))
    plt.plot(x, y, 'r.')
    plt.show()
    
#def print_error(pred, Ax, Ay, bx, by):
#    """
#    Inputs:
#        pred: (1, 2, h, w)
#        Ax, Ay: (h*w, h*w)
#        bx, by: (h*w, 1)
#    """
#    pred = pred[0]
#    x_pred = pred[0].reshape((-1, 1))
#    x = Ax.dot(x_pred) - bx.reshape((-1, 1))
#    y_pred = pred[1].reshape((-1, 1))
#    y = Ay.dot(y_pred) - by.reshape((-1, 1))
#    print(np.mean(np.abs(x)) + np.mean(np.abs(y)))

def print_error(pred, Ax, Ay):#, bx=0, by=0):
    """
    Inputs:
        pred: (1, 2, h, w)
        Ax, Ay: (h*w, h*w)
        bx, by: (h*w, 1)
    """
    pred = pred[0]
    x_pred = pred[0].reshape((-1, 1))
    x = Ax.dot(x_pred)
    y_pred = pred[1].reshape((-1, 1))
    y = Ay.dot(y_pred)
    print(np.mean(np.abs(x)) + np.mean(np.abs(y)))

def print_mu_error(pred_map, org_mu):
    """
    Inputs:
        pred_map: (1, 2, h, w), numpy array
        org_mu: (1, 2, h, w), numpy array
    """
    h, w = pred_map.shape[2:]
    pred_mu = bc_metric(torch.from_numpy(pred_map))
    pred_mu = pred_mu[0].numpy().reshape((2, h-1, (w-1)*2))
    pred_mu = pred_mu[0] + pred_mu[1] * 1j
    mu = org_mu[0][:, :-1, :-1].numpy()
    mu = mu[0] + 1j * mu[1]
    mu_reshape = np.zeros((mu.shape[0], mu.shape[1]*2), dtype = np.complex)
    mu_reshape[:, ::2] = mu
    mu_reshape[:-1, 1:-1:2] = (mu[:-1, :-1] + mu[:-1, 1:] + mu[1:, :-1]) / 3
    mu_reshape[:-1, -1] = (mu[:-1, -1] + mu[1:, -1]) / 2
    mu_reshape[-1, 1:-1:2] = (mu[-1, :-1] + mu[-1, 1:]) / 2
    mu_reshape[-1, -1] = mu[-1, -1]
    mu = mu_reshape
    err = np.abs(mu - pred_mu).reshape(-1)
    print('mu_err:', np.mean(err))

def plot_pred_map(pred, mapping, save=False):
    """
    Inputs:
        pred, mapping: (2, h, w)
        save: True or False
    """
    x_pred = pred[0].reshape((-1, 1))
    y_pred = pred[1].reshape((-1, 1))
    x_true = mapping[0].reshape((-1, 1))
    y_true = mapping[1].reshape((-1, 1))

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.plot(x_pred, y_pred, 'r.')#, markersize=3)
    plt.title('Network output')
    fig.add_subplot(1, 2, 2)
    plt.plot(x_true, y_true, 'r.')#, markersize=3)
    plt.title('Target image')
    if save:
        if not os.path.exists('save_plot'):
            os.mkdir('save_plot')
        plt.savefig('./save_plot/'+str(time.time())+'.png',dpi=1200, bbox_inches = 'tight')
    else:
        plt.show()
    #plt.savefig("")

def triplot_pred_map(pred, mapping, save=False):
    """
    Inputs:
        pred, mapping: (2, h, w)
        save: True or False
    """
    h, w = pred.shape[1:]
    face, _ = image_meshgen(h, w)
    x_pred = pred[0].reshape(-1)#.reshape((-1, 1))
    y_pred = pred[1].reshape(-1)#.reshape((-1, 1))
    x_true = mapping[0].reshape(-1)#.reshape((-1, 1))
    y_true = mapping[1].reshape(-1)#.reshape((-1, 1))

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.triplot(x_pred, y_pred, face, 'r.-')#, markersize=3)
    plt.title('Network output')
    fig.add_subplot(1, 2, 2)
    plt.triplot(x_true, y_true, face, 'r.-')#, markersize=3)
    plt.title('Target image')
    if save:
        if not os.path.exists('save_plot'):
            os.mkdir('save_plot')
        plt.savefig('./save_plot/'+str(time.time())+'.png',dpi=1200, bbox_inches = 'tight')
    else:
        plt.show()
    #plt.savefig("")

def bc_metric(mapping):
    """
    Inputs:
        mapping: (N, 2, h, w), torch tensor
    Outputs:
        mu: (N, 2, (h-1)*(w-1)*2), torch tensor
    """
    # The three input variables are pytorch tensors.
    N, C, H, W = mapping.shape
    device = mapping.device
    face, vertex = image_meshgen(H, W)
    face = torch.from_numpy(face).to(device=device)
    vertex = torch.from_numpy(vertex).to(device=device)

    gi = vertex[face[:, 0], 0]
    gj = vertex[face[:, 1], 0]
    gk = vertex[face[:, 2], 0]
    
    hi = vertex[face[:, 0], 1]
    hj = vertex[face[:, 1], 1]
    hk = vertex[face[:, 2], 1]
    
    gjgi = gj - gi
    gkgi = gk - gi
    hjhi = hj - hi
    hkhi = hk - hi
    
    area = (gjgi * hkhi - gkgi * hjhi) / 2

    gigk = -gkgi
    hihj = -hjhi

    mapping = mapping.view((N, C, -1))
    mapping = mapping.permute(0, 2, 1)

    si = mapping[:, face[:, 0], 0]
    sj = mapping[:, face[:, 1], 0]
    sk = mapping[:, face[:, 2], 0]
    
    ti = mapping[:, face[:, 0], 1]
    tj = mapping[:, face[:, 1], 1]
    tk = mapping[:, face[:, 2], 1]
    
    sjsi = sj - si
    sksi = sk - si
    tjti = tj - ti
    tkti = tk - ti
    
    a = (sjsi * hkhi + sksi * hihj) / area / 2;
    b = (sjsi * gigk + sksi * gjgi) / area / 2;
    c = (tjti * hkhi + tkti * hihj) / area / 2;
    d = (tjti * gigk + tkti * gjgi) / area / 2;
    
    down = (a+d)**2 + (c-b)**2 + 1e-8
    up_real = (a**2 - d**2 + c**2 - b**2)
    up_imag = 2*(a*b+c*d)
    real = up_real / down
    imag = up_imag / down

    mu = torch.stack((real, imag), dim=1)
    return mu

def S_Jzdz(mapping, mu):
    """
    Inputs:
        mapping: (N, 2, h, w), torch tensor
        mu : m x 1 Beltrami coefficients, mu=(h-1)*(w-1)*2
    Outputs:
        S : torch float number
    """
    N, C, H, W = mapping.shape
    device = mapping.device
    face, vertex = image_meshgen(H, W)
    _, abc, _ = generalized_laplacian2D(face, vertex, mu, H, W)
    abc = torch.from_numpy(abc).to(device=device)
    af = abc[:, 0]
    bf = abc[:, 1]
    gf = abc[:, 2]

    face = torch.from_numpy(face).to(device=device)
    vertex = torch.from_numpy(vertex).to(device=device)

    gi = vertex[face[:, 0], 0]
    gj = vertex[face[:, 1], 0]
    gk = vertex[face[:, 2], 0]

    hi = vertex[face[:, 0], 1]
    hj = vertex[face[:, 1], 1]
    hk = vertex[face[:, 2], 1]

    gjgi = gj - gi
    gkgi = gk - gi
    hjhi = hj - hi
    hkhi = hk - hi

    area = (gjgi * hkhi - gkgi * hjhi) / 2

    gigk = -gkgi
    hihj = -hjhi

    mapping = mapping.view((N, C, -1))
    mapping = mapping.permute(0, 2, 1)

    si = mapping[:, face[:, 0], 0]
    sj = mapping[:, face[:, 1], 0]
    sk = mapping[:, face[:, 2], 0]

    sjsi = sj - si
    sksi = sk - si

    a = (sjsi * hkhi + sksi * hihj) / area / 2;
    b = (sjsi * gigk + sksi * gjgi) / area / 2;
    S = torch.mean(af*a*a+2*bf*a*b+gf*b*b)    # Using mean because the area of all the triangles are the same.

    return S

def mesh_area(mapping):
    """
    Inputs:
        mapping: (N, 2, h, w), torch tensor
    Outputs:
        S : torch float number
    """
    N, C, H, W = mapping.shape
    device = mapping.device
    face, _ = image_meshgen(H, W)
    face = torch.from_numpy(face).to(device=device)

    mapping = mapping.view((N, C, -1))
    mapping = mapping.permute(0, 2, 1)

    si = mapping[:, face[:, 0], 0]
    sj = mapping[:, face[:, 1], 0]
    sk = mapping[:, face[:, 2], 0]

    ti = mapping[:, face[:, 0], 1]
    tj = mapping[:, face[:, 1], 1]
    tk = mapping[:, face[:, 2], 1]

    sjsi = sj - si
    sksi = sk - si
    tjti = tj - ti
    tkti = tk - ti

    area = (sjsi * tkti - sksi * tjti) / 2
    S = torch.sum(torch.abs(area))
    return S 

# def size2remain_ind(h, w):
#     Edge1 = np.reshape(np.arange((h-1)*w, h*w), (w, 1))
#     Edge2 = np.reshape(np.arange(w-1, h*w, step = w), (h, 1))
#     Edge3 = np.reshape(np.arange(w), (w, 1))
#     Edge4 = np.reshape(np.arange((h-1)*w+1, step = w), (h, 1))
#     #Edge = np.vstack((Edge1, Edge2, Edge3, Edge4))
   
#     landmarkx = np.vstack((Edge4, Edge2)) 
#     rmx = np.setdiff1d(np.arange(h*w), landmarkx.reshape(-1))
   
#     landmarky = np.vstack((Edge1, Edge3)) 
#     rmy = np.setdiff1d(np.arange(h*w), landmarky.reshape(-1))
#     return rmx, rmy
    
def coo_iv2torch(ind, val, h, w):
    """
    Inputs:
        ind: (2, number of sparse elements), torch tensor
        val: (number of sparse elements, ), torch tensor
        h, w: the h, w of mapping
    Outputs:
        [Ax, Ay] : the sparse A matrix without rows correspond to landmarks.
    """
    hw = h * w
    vertex_ind = np.arange(hw).reshape((h, w))
    rmx = vertex_ind[:, 1:-1].reshape(-1)
    rmy = vertex_ind[1:-1, :].reshape(-1)
    ind, val = ind.numpy(), val.numpy()

    A = sps.csr_matrix((val, ind), shape=(hw, hw))
    Ax, Ay = A[rmx].tocoo(), A[rmy].tocoo()
    Ax_val, Ay_val = Ax.data, Ay.data
    Ax_ind, Ay_ind = np.vstack((Ax.row, Ax.col)), np.vstack((Ay.row, Ay.col))

    ind_x = torch.LongTensor(Ax_ind)
    val_x = torch.FloatTensor(Ax_val)
    Ax = torch.sparse.FloatTensor(ind_x, val_x, torch.Size((rmx.shape[0], hw)))

    ind_y = torch.LongTensor(Ay_ind)
    val_y = torch.FloatTensor(Ay_val)
    Ay = torch.sparse.FloatTensor(ind_y, val_y, torch.Size((rmy.shape[0], hw)))
    return [Ax, Ay]

# Since preprocessor code has been modified, the method next_batch has been removed.
def mu_gen(preprocessor, datanum = 5):

    real = preprocessor.next_batch(datanum)
    imag = preprocessor.next_batch(datanum)
    
    mu = real + 1j * imag
    randnum = np.random.rand(mu.shape[0], 1)
    mu = mu/np.max(np.abs(mu)) * randnum[..., np.newaxis]
    print(randnum)
    mu[:, -1, :] = 0
    mu[:, :, -1] = 0
    return mu

def multi_mu2map(mu, proc_num = 2):
    from multiprocessing import Pool
    with Pool(proc_num) as p:
        out = np.array(list(p.map(mu2map, mu)))
    return out

if __name__ == '__main__':
    #f, v = image_meshgen(80, 70)
    #f1, v1 = image_meshgen1(80, 70)
    #print(np.sum(f-f1), np.sum(v-v1))

    #h = 120
    #w = 120
    ## fre_stay = 100;
    #
    ## norm = norm_dis(h, w);
    #
    ## f = rand(fre_stay/2)+ 1i*rand(fre_stay/2);
    ## f = [f, f(:, end:-1:1); f(end:-1:1, :), f(end:-1:1, end:-1:1)];
    ## fre = zeros(h);
    ## fre(h/2-fre_stay/2+1:h/2+fre_stay/2, w/2-fre_stay/2+1:w/2+fre_stay/2) = f;
    ## fre = fre.* norm;
    ## fre = ifftshift(fre);
    ## real = abs(ifft2(fre));
    #
    #real = np.array(Image.open('a.png'))#.resize((w, h)))
    #img = np.array(Image.open('b.jpg'))#.resize((w, h)))
    #
    #real = real[:, :, 1] - np.random.rand() * 255
    #img = img[:, :, 1] - np.random.rand() * 255
    #rh, rw = np.random.randint(real.shape[0]-h), np.random.randint(real.shape[1]-w)
    #ih, iw = np.random.randint(img.shape[0]-h), np.random.randint(img.shape[1]-w)
    #real = real[rh:rh+h, rw:rw+w]
    #img = img[ih:ih+h, iw:iw+w]
    #
    ## imshow(int64(real))
    #
    ## f = rand(fre_stay/2)+ 1i*rand(fre_stay/2);
    ## f = [f, f(:, end:-1:1); f(end:-1:1, :), f(end:-1:1, end:-1:1)];
    ## fre = zeros(h);
    ## fre(h/2-fre_stay/2+1:h/2+fre_stay/2, w/2-fre_stay/2+1:w/2+fre_stay/2) = f;
    ## fre = fre.* norm;
    ## fre = ifftshift(fre);
    ## img = abs(ifft2(fre));
    #
    #
    ## mu = real * np.random.rand() + 1j * img * np.random.rand()
    ## mu = mu/np.max(np.abs(mu))
    ## randnum = np.random.rand()
    ## print(randnum)
    ## mu = mu * randnum
    #
    #randnum = 0.5
    #mu = real * randnum + 1j * img * randnum
    #mu = mu/np.max(np.abs(mu))
    #print(randnum)
    #mu = mu * randnum
    ##mu(30:40, 30:36);
    #
    #mapping = mu2map(mu)
    #plot_map(mapping)
    from scipy.io import loadmat
    mapping = loadmat('map.mat')['map'].T.reshape((1,2,4,4))
    print(mapping)
    mapping = torch.from_numpy(mapping)
    recon_mu = bc_metric(mapping)
    print(recon_mu)
