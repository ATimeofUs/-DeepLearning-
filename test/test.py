import torch

def main():
    nx = 5
    ny = 5
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

if __name__ == "__main__":
    main()