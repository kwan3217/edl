import numpy as np
import re
import vector
from collections import namedtuple

path="/mnt/big/home/chrisj/workspace/Data/Planet/Mars/Jezero/Topography/"

heightTilesReturn=namedtuple("heightTilesReturn","latc lonc tiles")
def read_height_tile_csv():
    tiles={}
    for i in [8,9,10,11,12]:
        n_y=64//2**(i-8)
        n_x=n_y*48//64
        tiles[i]=np.zeros((n_x,n_y,6))
    with open(path+"topo.inc","rt") as incf:
        for line in incf:
            match=re.search("#declare\s+latc\s*=\s*([0-9.]+)\s*;",line)
            if match:
                latc=float(match.group(1))
            match = re.search("#declare\s+lonc\s*=\s*([0-9.]+)\s*;", line)
            if match:
                lonc = float(match.group(1))
    with open(path+"tiles.csv","rt") as csvf:
        for line in csvf:
            line=line.strip()
            if "#" in line:
                line=line[:line.index("#")]
            line=line.strip()
            if line=="":
                continue
            parts=line.split(",")
            coords=np.array([float(x) for x in parts[3:]])
            level=int(parts[0])
            col=int(parts[1])
            row=int(parts[2])
            tiles[level][col,row,:]=coords
    return heightTilesReturn(latc=latc,lonc=lonc,tiles=tiles)

def rotz(theta):
    c=np.cos(theta)
    s=np.sin(theta)
    return np.array([[ c,-s, 0],
                     [ s, c, 0],
                     [ 0, 0, 1]])

def roty(theta):
    c=np.cos(theta)
    s=np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rotx(theta):
    c=np.cos(theta)
    s=np.sin(theta)
    return np.array([[ 1, 0, 0],
                     [ 0, c,-s],
                     [ 0, s, c]])

def rep_inview(x0,y0,z0,x1,y1,z1,location,look_at,camin):
    """
    All coordinates and vector should be in topocentric frame
    :param x0: minimum x of box
    :param y0: minimum y of box
    :param z0: minimum z of box (ignored)
    :param x1: maximum x of box
    :param y1: maximum y of box
    :param z1: maximum z of box
    :param location: location of camera
    :param look_at: look target of camera
    :param camin: cosine of half-angle of field of view. Vectors whose normalized dot product
    are greater than this are in the field of view
    :return: distance to the closest point in view of this tile,
             or none if there aren't any
    """
    test_granularity=4
    x,y=np.meshgrid(np.linspace(x0,x1,test_granularity),np.linspace(y0,y1,test_granularity))
    x=np.ravel(x)
    y=np.ravel(y)
    if location[2,0]>z1:
        z=z1
    elif location[2,0]<z0:
        z=z0
    else:
        z=location[2,0]
        camin=-1 #Effectively ignore the view constraint
    z=x*0+z
    v_terrain=vector.vcomp((x,y,z))
    v_camera=look_at-location
    v_camera=v_camera/vector.vlength(v_camera)
    dist=vector.vlength(v_terrain-location)
    cangle=vector.vdot(v_camera,v_terrain-location)/dist
    w=np.where(cangle>camin)
    if len(w[0])==0:
        return None
    return np.min(dist[w])
    print(x.shape)

def is_tile_close_enough(tiles,location,look_at,camin,level,i_x,i_y,rad_pixel):
    x0,y0,z0,x1,y1,z1=tuple(tiles[level][i_x,i_y,:])
    dist=rep_inview(x0,y0,z0,x1,y1,z1,location,look_at,camin)
    if dist:
        if level==12:
            return True #Early exit for coarsest level
        closest_tilepix_angle = (2 ** (level-8)) / dist
        return closest_tilepix_angle>rad_pixel
    return False

def height_map_tile_level(level,location,look_at,width=1920,height=1080,angle=None,tiles=None):
    """

    :param ouf:
    :param location: Numpy column vector position of camera relative to center of planet
    :param look_at:
    :param width: width of viewport in pixels
    :param height:
    :param angle:
    :return:
    """
    if angle is None:
        right=width/height
        direction=1
        angle=np.degrees(2*np.arctan(right/(2*direction)))
    rad_pixel=np.radians(angle/width)
    angle=np.radians(angle)*np.hypot(width,height)/width
    camin=np.cos(angle/2)
    tile_map=np.zeros(tiles[level].shape[0:2],dtype=np.bool8)
    for i_x in range(tiles[level].shape[0]):
        for i_y in range(tiles[level].shape[1]):
            tile_map[i_x][i_y]=is_tile_close_enough(tiles,location,look_at,camin,level,i_x,i_y,rad_pixel)
    return tile_map

def height_tile_map(location,look_at,*,angle=None,width=1920,height=1080,tiles=None,lonc=None,latc=None):
    if lonc is not None:
        # Rotate location and look_at into topocentric coordinate system:
        #   topocenter -- chosen when merging topographic maps, has latitude latc and longitude lonc
        #   origin - center of Mars
        #   x - local east
        #   y - local north (not polar axis)
        #   z - local vertical (surface is more than 3000km in +z direction)
        # This transformation puts grid east in the +x and grid north in the +y direction as expected
        Mz = rotz(np.radians(-90 - lonc))  # put center on -90E meridian
        Mx = rotx(np.radians(latc - 90))  # put -90E, center lat on north pole
        M = Mx @ Mz
        location=M @ location
        look_at=M @ look_at

    level_maps={level:np.zeros((48//2**(level-8),48//2**(level-8)),dtype=np.bool8) for level in range(8,13)}
    #Check all levels, but if any level has no tiles, don't check any finer
    for level in range(12,7,-1):
        level_maps[level]=height_map_tile_level(level,location,look_at,tiles=tiles,angle=angle,width=width,height=height)
        if not np.any(level_maps[level]):
            break
    #From fine to coarse: If any of the finer tiles covering this tile are used, use all the finer tiles
    for level in range(9,13):
        n_x=48 // 2 ** (level - 8)
        n_y=64 // 2 ** (level - 8)
        for i_x in range(n_x):
            for i_y in range(n_y):
                if np.any(level_maps[level-1][i_x*2:i_x*2+2,i_y*2:i_y*2+2]):
                    level_maps[level - 1][i_x * 2:i_x * 2 + 2, i_y * 2:i_y * 2 + 2]=True
    #From coarse to fine: If any of the finer tiles covering this tile are used, don't use this tile
    for level in range(12,8,-1):
        n_x=48 // 2 ** (level - 8)
        n_y=64 // 2 ** (level - 8)
        for i_x in range(n_x):
            for i_y in range(n_y):
                if np.any(level_maps[level-1][i_x*2:i_x*2+2,i_y*2:i_y*2+2]):
                    level_maps[level][i_x, i_y]=False
    return level_maps

def frame(i_frame,*,Location,Look_at,latc,lonc,tiles,ouf=None):
    """

    :param i_frame:
    :param Location:
    :param Look_at:
    :param latc:
    :param lonc:
    :param tiles:
    :return:
    """
    close_self=ouf is None
    if close_self:
        ouf=open(f"inc/tile{frame:03d}.inc", "wt")
    print("//Generated in tile.py",file=ouf)
    print(f"#declare Location=<{Location[0,0]},{Location[1, 0]},{Location[2, 0]}>;", file=ouf)
    print(f"#declare Look_at=<{Look_at[0, 0]},{Look_at[1, 0]},{Look_at[2, 0]}>;", file=ouf)
    a = height_tile_map(Location, Look_at, tiles=tiles,latc=latc,lonc=lonc)
    for level in a:
        for i_x in range(a[level].shape[0]):
            for i_y in range(a[level].shape[1]):
                if (a[level][i_x, i_y]):
                    print(f"height_tile({level:2d},{i_x:2d},{i_y:2d})", file=ouf)
    if close_self:
        ouf.close()

if __name__=="__main__":
    latc, lonc, tiles = read_height_tile_csv()
    for i_frame in range(0,1000):
        Location=np.array([[0],[-8],[frame*10+3391900]])
        Look_at=np.array([[0],[0],[3391900]])
        frame(frame,Location=Location,Look_at=Look_at,latc=latc,lonc=lonc,tiles=tiles)
        print(".", end='')
        if (i_frame % 100 == 0):
            print(f"{frame}/1000")
