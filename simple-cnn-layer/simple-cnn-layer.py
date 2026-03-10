import numpy as np

def conv2d(x, W, b):
    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = W.shape
    H_out = H - KH + 1
    W_out = W_in - KW + 1
    
    shape = (N, C_in, H_out, W_out, KH, KW)
    strides = (*x.strides[:2], x.strides[2], x.strides[3], x.strides[2], x.strides[3])
    patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    
    # patches: (N, C_in, H_out, W_out, KH, KW)
    # W:       (C_out, C_in, KH, KW)
    # keep n, h_out, w_out → contract over c_in, kh, kw
    y = np.einsum('nihwuv,oiuv->nohw', patches, W) + b[None, :, None, None]
    
    return y.astype(float)