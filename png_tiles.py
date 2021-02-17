import numpy as np
import imageio

def linterp(x0,y0,x1,y1,x):
    t=(x-x0)/(x1-x0)
    return (1-t)*y0+t*y1


path = "/mnt/big/home/chrisj/workspace/Data/Planet/Mars/Jezero/Topography/"
# size of a tile in pixels
pix_size = 256

with open(path + "tiles.csv","wt") as csv:
    print("# Tile summary, made by png_tiles.py",file=csv)
    print("# level,i_x,i_y,x0,y0,z0,x1,y1,z1",file=csv)
    for level in [8, 9, 10, 11, 12]:
        print(level)
        #size of a tile in m
        tilesize= 2 ** (level)
        full_x=12*1024
        full_y=16*1024

        n_x= full_x // tilesize
        n_y= full_y // tilesize

        for i_x in range(n_x):
            print(f"{i_x:3}",end='')
            for i_y in range(n_y):
                x0=i_x*tilesize-full_x//2;
                x1=x0+tilesize
                y0=i_y*tilesize-full_y//2;
                y1=y0+tilesize
                a=np.load(path + f"{level:02d}/tile{i_x:03d}x{i_y:03d}.npy")
                if np.any(np.isnan(a)):
                    w=np.where(np.isnan(a))
                    a[w]=np.nanmin(a)
                    print(f"NaN found in input data for tile level {level}, i_x={i_x}, i_y={i_y}, patched {len(w[0])} pixels")
                z0=np.nanmin(a)
                z1=np.nanmax(a)
                #Write everything in km
                print(f"{level},{i_x},{i_y},{x0/1000:25.15e},{y0/1000:25.15e},{z0/1000:25.15e},{x1/1000:25.15e},{y1/1000:25.15e},{z1/1000:25.15e}",file=csv)
                pngfn=path+f"{level:02d}/tile{i_x:03d}x{i_y:03d}.png"
                incfn=path+f"{level:02d}/tile{i_x:03d}x{i_y:03d}.inc"
                with open(incfn,"wt") as incf:
                    print(f"#declare TopoLevel={level};",file=incf)
                    print(f"#declare i_x={i_x};",file=incf)
                    print(f"#declare i_y={i_y};",file=incf)
                    print(f"#declare x0={x0/1000};",file=incf)
                    print(f"#declare x1={x1/1000};",file=incf)
                    print(f"#declare y0={y0/1000};",file=incf)
                    print(f"#declare y1={y1/1000};",file=incf)
                    print(f"#declare z0={z0/1000};",file=incf)
                    print(f"#declare z1={z1/1000};",file=incf)
                b=linterp(z0,0,z1,65535,a).astype(np.uint16)
                imageio.imwrite(pngfn,np.flipud(b))
                print('.',end='')
            print()


