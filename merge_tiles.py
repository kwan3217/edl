import numpy as np

for level0 in [8,9,10,11]:
    level1=level0+1
    print(f"{level0}-->{level1}")
    #size of a tile in pixels
    pix_size=256
    #size of a tile in m
    tilesize= 2 ** (level0)
    full_x=12*1024
    full_y=16*1024

    n_x0= full_x // tilesize
    n_y0= full_y // tilesize
    n_x1=n_x0//2
    n_y1=n_y0//2

    path="/mnt/big/home/chrisj/workspace/Data/Planet/Mars/Jezero/Topography/"

    for i_x1 in range(n_x1):
        print(f"{i_x1:3}",end='')
        for i_y1 in range(n_y1):
            a=np.zeros((pix_size * 2, pix_size * 2))
            a[:pix_size, :pix_size]=np.load(path + f"{level0:02d}/tile{i_x1 * 2 + 0:03d}x{i_y1 * 2 + 0:03d}.npy")
            a[pix_size:, :pix_size]=np.load(path + f"{level0:02d}/tile{i_x1 * 2 + 0:03d}x{i_y1 * 2 + 1:03d}.npy")
            a[:pix_size, pix_size:]=np.load(path + f"{level0:02d}/tile{i_x1 * 2 + 1:03d}x{i_y1 * 2 + 0:03d}.npy")
            a[pix_size:, pix_size:]=np.load(path + f"{level0:02d}/tile{i_x1 * 2 + 1:03d}x{i_y1 * 2 + 1:03d}.npy")
            if np.any(np.isnan(a)):
                w=np.where(np.isnan(a))
                a[w]=np.nanmin(a)
                print(f"NaN found in input data for tile level {level1}, i_x={i_x1}, i_y={i_y1}, patched {len(w[0])} pixels")
            b =a[0::2,0::2]
            b+=a[0::2,1::2]
            b+=a[1::2,0::2]
            b+=a[1::2,1::2]
            b/=4
            oufn=path+f"{level1:02d}/tile{i_x1:03d}x{i_y1:03d}.npy"
            np.save(oufn,b)
            print('.',end='')
        print()


