import torch


def reverse_odd_rows_optimized(x):
    x[:, :, 1::2, :] = torch.flip(x[:, :, 1::2, :], dims=[-1])
    return x
def cross_scan_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=4):
    if in_channel_first:
        B, C, H, W = x.shape
        if scans == 0:
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)
            y[:, 1, :, :] = x.transpose(2, 3).flatten(2, 3)
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
        elif scans == 1:
            y = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        elif scans == 2:
            y = x.view(B, 1, C, H * W)
            y = torch.cat([y, y.flip(dims=[-1])], dim=1)
        elif scans == 3:
            y = x.new_empty((B, 8, C, H * W))
            # 0: row-major
            y[:, 0, :, :] = x.flatten(2, 3)
            # 1: column-major
            y[:, 1, :, :] = x.transpose(2, 3).flatten(2, 3)
            y[:, 2,:, :] = reverse_odd_rows_optimized(x.clone()).flatten(2, 3)
            y[:,3,:,:]=reverse_odd_rows_optimized(x.clone().transpose(2, 3)).flatten(2, 3)

            y[:, 4, :, :] = torch.flip(y[:, 0, :, :], dims=[-1])
            y[:, 5, :, :] = torch.flip(y[:, 1, :, :], dims=[-1])
            y[:, 6, :, :] = torch.flip(y[:, 2, :, :], dims=[-1])
            y[:, 7, :, :] = torch.flip(y[:, 3, :, :], dims=[-1])
        if scans == 4:
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = diagonal_gather_main(x)
            y[:, 1, :, :] = diagonal_gather_anti(x)
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
        if scans == 5:
            y = x.new_empty((B, 12, C, H * W))
            # 0: row-major
            y[:, 0, :, :] = x.flatten(2, 3)
            # 1: column-major
            y[:, 1, :, :] = x.transpose(2, 3).flatten(2, 3)
            y[:, 2,:, :] = reverse_odd_rows_optimized(x.clone()).flatten(2, 3)
            y[:,3,:,:]=reverse_odd_rows_optimized(x.clone().transpose(2, 3)).flatten(2, 3)
            # 对角 
            y[:, 4, :, :] = diagonal_gather_main(x)
            y[:, 5, :, :] = diagonal_gather_anti(x)
            #逆转
            y[:, 6, :, :] = torch.flip(y[:, 0, :, :], dims=[-1])
            y[:, 7, :, :] = torch.flip(y[:, 1, :, :], dims=[-1])
            y[:, 8, :, :] = torch.flip(y[:, 2, :, :], dims=[-1])
            y[:, 9, :, :] = torch.flip(y[:, 3, :, :], dims=[-1])
            y[:, 10, :, :] = torch.flip(y[:, 4, :, :], dims=[-1])
            y[:, 11, :, :] = torch.flip(y[:, 5, :, :], dims=[-1])
    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=3):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)  # [B, K, D, HW]
        if scans == 0:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y[:, 0] + y[:, 1].view(B, D, W, H).transpose(2, 3).contiguous().view(B, D, -1)
        elif scans == 1:
            y = y.sum(1)
        elif scans == 2:
            y = y[:, 0] + y[:, 1].flip(dims=[-1]).view(B, 1, D, -1)
            y = y.sum(1)
        elif scans == 3:
            y = y[:, 0:4] + y[:, 4:8].flip(dims=[-1]).view(B, 4, D, -1)
            y = y[:, 0] + y[:, 1].view(B, D, W, H).transpose(2, 3).contiguous().view(B, D, -1)+reverse_odd_rows_optimized(y[:, 2].view(B, D, W, H)).view(B, D, -1) + reverse_odd_rows_optimized(y[:, 3].view(B, D, W, H)).transpose(2, 3).contiguous().view(B, D, -1)
        elif scans == 4:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            # xs[:, 4] = diagonal_gather(x)
            y = diagonal_scatter_main(y[:, 0], (B,D,H,W)).contiguous().view(B, D, -1)+ diagonal_scatter_anti(y[:, 1], (B,D,H,W)).contiguous().view(B, D, -1)
        elif scans == 5:
            y = y[:, 0:6] + y[:, 6:12].flip(dims=[-1]).view(B, 6, D, -1)
            y = y[:, 0] + y[:, 1].view(B, D, W, H).transpose(2, 3).contiguous().view(B, D, -1)+reverse_odd_rows_optimized(y[:, 2].view(B, D, W, H)).view(B, D, -1) + reverse_odd_rows_optimized(y[:, 3].view(B, D, W, H)).transpose(2, 3).contiguous().view(B, D, -1)+diagonal_scatter_main(y[:, 4], (B,D,H,W)) .contiguous().view(B, D, -1)+ diagonal_scatter_anti(y[:, 5], (B,D,H,W)).contiguous().view(B, D, -1)
    if in_channel_first and (not out_channel_first):    
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()

    return y
def diagonal_gather_anti(tensor):
    B, C, H, W = tensor.size()
    device = tensor.device

    row = torch.arange(H, device=device).unsqueeze(1).expand(H, W)  # [H, W]
    col = torch.arange(W, device=device).unsqueeze(0).expand(H, W)  # [H, W]
    diag_id = (row + col)  

    flat_idx = diag_id.flatten()
    sort_order = flat_idx.argsort()  

    tensor = tensor.view(B, C, -1)  # [B, C, H*W]
    tensor_sorted = tensor[:, :, sort_order]  

    return tensor_sorted  # [B, C, H*W]
def diagonal_gather_main(tensor):
    B, C, H, W = tensor.size()
    device = tensor.device

    row = torch.arange(H, device=device).unsqueeze(1).expand(H, W)  # [H, W]
    col = torch.arange(W, device=device).unsqueeze(0).expand(H, W)  # [H, W]
    diag_id = (col - row + (H - 1))  

    flat_idx = diag_id.flatten()
    sort_order = flat_idx.argsort() 

    tensor = tensor.view(B, C, -1)  # [B, C, H*W]
    tensor_sorted = tensor[:, :, sort_order]  # [B, C, H*W]

    return tensor_sorted
def diagonal_scatter_anti(tensor_flat, original_shape):
    B, C, H, W = original_shape
    device = tensor_flat.device

    row = torch.arange(H, device=device).unsqueeze(1).expand(H, W)
    col = torch.arange(W, device=device).unsqueeze(0).expand(H, W)
    diag_id = (row + col).flatten()
    sort_order = diag_id.argsort()  

    inverse_order = sort_order.argsort()

    tensor_restored = tensor_flat[:, :, inverse_order].reshape(B, C, H, W)
    return tensor_restored

x = torch.tensor([[[[0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8]]]], dtype=torch.float)
def diagonal_scatter_main(tensor_flat, original_shape):
    B, C, H, W = original_shape
    device = tensor_flat.device

    row = torch.arange(H, device=device).unsqueeze(1).expand(H, W)
    col = torch.arange(W, device=device).unsqueeze(0).expand(H, W)
    diag_id = (col - row + (H - 1)).flatten()  # ↘方向：j - i = const，统一加偏移使非负
    sort_order = diag_id.argsort()

    inverse_order = sort_order.argsort()  # 得到反向索引
    tensor_restored = tensor_flat[:, :, inverse_order].reshape(B, C, H, W)
    return tensor_restored


if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9,10,11,12],
                      [13,14,15,16]], dtype=torch.float32, requires_grad=True)
    x = x.view(1, 1, 4, 4)
    y=cross_scan_fwd(x, in_channel_first=True, out_channel_first=True, scans=3)
    print(y)
    y=y.view(1, -1, 1, 4, 4)
    z=cross_merge_fwd(y, in_channel_first=True, out_channel_first=True, scans=3)
    print(z)
