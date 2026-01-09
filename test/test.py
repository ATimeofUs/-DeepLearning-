import torch

def main():
    nx = 3
    ny = 2
    device = "cpu"
    yv, xv = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij"
    )
    grid = torch.stack((xv, yv), dim=-1)
    print("xv:", xv)
    print("yv:", yv)
    print("grid:", grid)
    grid = grid.view(1, 1, ny, nx, 2)

    print("reshaped grid:", grid)

def main2():
    # 假设：B=1, nA=1, ny=2, nx=3，只关心 tw, th
    # raw[..., 2:4] 形状 (1,1,2,3,2)，填一些便于计算的值
    raw = torch.tensor(
        [[[[[ 0.0,  0.0],    # cell (0,0): tw=0,  th=0
            [ 0.5, -0.5],    # cell (0,1): tw=0.5,th=-0.5
            [-1.0,  1.0]],   # cell (0,2): tw=-1, th=1

        [[ 1.0,  1.0],    # cell (1,0): tw=1, th=1
            [ 2.0, -2.0],    # cell (1,1): tw=2, th=-2
            [ 0.0,  0.0]]]]  # cell (1,2): tw=0, th=0
        ], dtype=torch.float32)

    # anchors：与前面的 YOLOLayer 中归一化后的形状一致 (1, nA, 1, 1, 2)
    # 假设单个 anchor 的 (w,h) = (2, 4)（单位：特征图 cell）
    anchors = torch.tensor([[[[[2.0, 4.0]]]]])  # shape (1,1,1,1,2)

    # 计算 wh
    wh = torch.exp(raw[..., 2:4]) * anchors

    print("raw[...,2:4] (tw,th):\n", raw[...,2:4][0,0])
    print("exp(tw,th):\n", torch.exp(raw[...,2:4])[0,0])
    print("anchors:\n", anchors[0,0,0,0])
    print("wh = exp(tw,th) * anchors:\n", wh[0,0])
    

if __name__ == "__main__":
    main2()

"""



"""