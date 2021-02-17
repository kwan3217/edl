import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from vector import vcomp, vdecomp
from scipy.interpolate import interp2d, RectBivariateSpline, LinearNDInterpolator
import re
import os
import multiprocessing

class DTM:
    def __init__(self,infn=None,filetype='f4',casttype='f4',has_header=True,lat0=None,lon0=None,lat1=None,lon1=None,rows=None,cols=None,blank=-30000,offset=0):
        """

        :param infn:
        :param format:
        :param has_header:
        :param lat0:
        :param lon0:
        :param lat1:
        :param lon1:
        :param rows:
        :param cols:
        :param blank:
        :param offset:

        Note - pixels are referred to as *samples*, so we will interpret each value as a 0-dimensional
        point. We will interpret the minimum and maximum latitude and longitude as those of a sample.
        If there are 101 samples, minimum longitude is 61 degrees and maximum is 62 degrees, we interpret
        the first sample as a vertex at exactly 61deg, the next as at exactly 61.01, etc. If there were
        100 samples, they would not be spaced by 0.01deg, but by exactly 1/99 deg. So the sample spacing
        is (n_samples-1)/(max-min)
        """
        pixbytes=int(filetype[-1])
        self.blank=blank
        self.offset=offset
        if infn is not None:
            if has_header:
                patterns = [[False, re.compile("RECORD_BYTES\s*=\s*([0-9]+)"), 0.0],
                            [False, re.compile("FILE_RECORDS\s*=\s*([0-9]+)"), 0.0],
                            [False, re.compile("MINIMUM_LATITUDE\s*=\s*([0-9]+\.[0-9]+)"), 0.0],
                            [False, re.compile("MAXIMUM_LATITUDE\s*=\s*([0-9]+\.[0-9]+)"), 0.0],
                            [False, re.compile("WESTERNMOST_LONGITUDE\s*=\s*([0-9]+\.[0-9]+)"), 0.0],
                            [False, re.compile("EASTERNMOST_LONGITUDE\s*=\s*([0-9]+\.[0-9]+)"), 0.0]]
                with open(infn,'rt') as inf:
                    for line in inf:
                        has_all = True
                        for pattern in patterns:
                            match=pattern[1].search(line)
                            if match is not None:
                                pattern[2]=float(match.group(1))
                                pattern[0]=True
                            has_all=has_all and pattern[0]
                        if has_all:
                            break
                self.cols=  int(patterns[0][2]) // pixbytes
                self.rows=  int(patterns[1][2])-1
                self.lat0=float(patterns[2][2])
                self.lat1=float(patterns[3][2])
                self.lon0=float(patterns[4][2])
                self.lon1=float(patterns[5][2])
            else:
                self.cols=cols
                self.rows=rows
                self.lat0=lat0
                self.lon0=lon0
                self.lat1=lat1
                self.lon1=lon1
            with open(infn,'rb') as inf:
                if has_header:
                    inf.read(self.cols*pixbytes)
                self.data=np.flipud(np.fromfile(inf,dtype=np.dtype(filetype),count=self.rows*self.cols).reshape(self.rows,self.cols))
                if casttype!=filetype:
                    self.data=self.data.astype(np.dtype(casttype))
                if self.blank is not None:
                    self.data[np.where(self.data<blank)]=blank-1
        else:
            self.cols = cols
            self.rows = rows
            self.lat0 = lat0
            self.lon0 = lon0
            self.lat1 = lat1
            self.lon1 = lon1
            self.data=np.zeros((self.rows,self.cols),dtype=np.dtype(casttype))
            if self.blank is not None:
                self.data+=self.blank-1
        self.data+=offset
    def lat(self):
        result=np.linspace(self.lat0,self.lat1,self.data.shape[0], endpoint=True)
        result+=(result[1]-result[0])/2
        return result
    def lon(self):
        result=np.linspace(self.lon0,self.lon1,self.data.shape[1], endpoint=True)
        result+=(result[1]-result[0])/2
        return result
    def interp(self):
        """
        Return an interp callable that acts like a function
        :return:
        """
        return interp2d(self.lon(),self.lat(),self.data)

class linterp:
    def __init__(self,x0,y0,x1,y1):
        self.x0=x0
        self.x1=x1
        self.y0=y0
        self.y1=y1
    def __call__(self,x,inv=False):
        if not inv:
            t=(x-self.x0)/(self.x1-self.x0)
            return self.y0*(1-t)+self.y1*t
        else:
            t=(y-self.y0)/(self.y1-self.y0)
            return self.x0*(1-t)+self.x1*t

def union(dtm1,dtm2):
    #figure out new lon/lat bounding box
    new_lat0=np.min((dtm1.lat0,dtm2.lat0))
    new_lon0=np.min((dtm1.lon0,dtm2.lon0))
    new_lat1=np.max((dtm1.lat1,dtm2.lat1))
    new_lon1=np.max((dtm1.lon1,dtm2.lon1))
    max1=np.max(dtm1.data)
    max2=np.max(dtm2.data)
    maxx=np.max((max1,max2))
    row_interp=linterp(dtm1.lat0,0,dtm1.lat1,dtm1.rows-1)
    col_interp=linterp(dtm1.lon0,0,dtm1.lon1,dtm1.cols-1)
    new_row0=int(row_interp(new_lat0))
    new_row1=int(row_interp(new_lat1))
    new_rows=new_row1-new_row0+1
    new_col0=int(col_interp(new_lon0))
    new_col1=int(col_interp(new_lon1))
    new_cols=new_col1-new_col0+1
    old_row0=-new_row0
    old_col0=-new_col0
    result=DTM(lat0=new_lat0,lat1=new_lat1,lon0=new_lon0,lon1=new_lon1,rows=new_rows,cols=new_cols,blank=dtm1.blank)
    result.data[old_row0:old_row0+dtm1.data.shape[0],old_col0:old_col0+dtm1.data.shape[1]]=dtm1.data
    del dtm1
    row_interp=linterp(result.lat0,0,result.lat1,result.rows-1)
    col_interp=linterp(result.lon0,0,result.lon1,result.cols-1)
    new_row0=int(row_interp(dtm2.lat0))
    new_row1=int(row_interp(dtm2.lat1))
    lats=result.lat()[new_row0:new_row1]
    new_col0=int(col_interp(dtm2.lon0))
    new_col1=int(col_interp(dtm2.lon1))
    lons=result.lon()[new_col0:new_col1]
    dtm2_blank=dtm2.blank
    result.data[new_row0:new_row1,new_col0:new_col1]=np.maximum(dtm2.interp()(lons,lats),result.data[new_row0:new_row1,new_col0:new_col1])
    w=np.where(result.data>maxx)
    result.data[w]=-3001
    print("done")
    return result

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
mars_r=3396000
#Note that this runs from 0 to 360 longitude, while radius map runs from -180 to 180
areoid=DTM("/mnt/big/home/chrisj/workspace/Data/Planet/Mars/Equipotential/mega90n000eb.img",
           filetype='>i2',has_header=False,lat0=-90+1/32,lon0=0+1/32,lat1=90-1/32,lon1=360-1/32,rows=180*16,cols=360*16,offset=mars_r,blank=None)
#dtm1=DTM("/mnt/big/home/chrisj/workspace/Data/Planet/Mars/Jezero/DTM/DTEEC_023524_1985_023379_1985_U01.IMG")
#dtm1=DTM("/mnt/big/home/chrisj/workspace/Data/Planet/Mars/Jezero/DTM/DTEEC_023247_1985_022957_1985_U01.IMG")
#dtm1=union(dtm1,dtm2)
#del dtm2
dtm1=DTM("/mnt/big/home/chrisj/workspace/Data/Planet/Mars/Jezero/DTM/DTEEC_048842_1985_048908_1985_U01.IMG")
#dtm1=union(dtm1,dtm2)
#del dtm2
dtm2=DTM("/mnt/big/home/chrisj/workspace/Data/Planet/Mars/Jezero/DTM/DTEEC_045994_1985_046060_1985_U01.IMG")
dtm1=union(dtm1,dtm2)
del dtm2
dtm2=DTM("/mnt/big/home/chrisj/workspace/Data/Planet/Mars/Jezero/DTM/DTEEC_002387_1985_003798_1985_A01.IMG")
dtm1=union(dtm1,dtm2)
plot=False

if plot:
    plt.figure('dtm1')
    plt.imshow(dtm1.data[::16,::16],origin='lower')

lon_1d=dtm1.lon().reshape(1,-1)
lat_1d=dtm1.lat().reshape(-1,1)
dtm_latc=(dtm1.lat0+dtm1.lat1)/2
dtm_lonc=(dtm1.lon0+dtm1.lon1)/2
with open("/mnt/big/home/chrisj/workspace/Data/Planet/Mars/Jezero/Topography/topo.inc","wt") as ouf:
    print("""// Generated by python/edl/dtm.py
// This set of files is designed to solve the spherical height field problem
//    by not solving it in POV-Ray, but externally in a Python script. It 
//    generates rectangular tileable height fields from input data in 
//    planetocentric spherical coordinates.  The Python code loads all the
//    appropriate data and converts it to rectangular coordinates in a 
//    topocentric coordinate frame centered on the data.
//     
// This frame has its origin at the center of the planet, +X axis parallel to
//    local east, +Y axis parallel to local north (not the polar axis), and +Z 
//    axis in local vertical. Once properly scaled and rotated, the tiles should
//    perfectly fit and describe the actual topography, including curvature of 
//    the surface.
//
// The data is broken into tiles at several scales -- in level 08, the tiles are
//    256 (=2**8) meters on a side, level 9 is 512m, etc. Each tile is still 256
//    pixels big, so the pixels are twice as big at each next coarser level.
//    Each coarser level is made by taking the right block of 2x2 tiles from the
//    finer level, and averaging together each 2x2 block of pixels in the data 
//    to make a tile of the same pixel size, but twice as much physical size. 
//    Averaging is done in merge_tiles.py
//
// The tiles are 16-bit PNGs, scaled such that the lowest z coordinate in each 
//    tile is value 0 and the highest is 2**16-1 (65535). Each tile is 
//    accompanied by a small .inc file which holds the level and span of the 
//    file. This is done in png_tiles.py
//   
// To use these files:
// 1) Load the appropriate tile as a height field, with increasing columns 
//    towards +X, increasing rows towards +y, and topography towards +z
// 2) Scale and position the tile so that it fits within the x0-x1, y0-y1, z0-z1
//    bound in the corresponding .inc file
// 3) Rotate the tile around the Y axis (right-handed) by 90deg-latc. 
// 3) Rotate the tile around the Z axis (right-handed) by 90deg+lonc. 
""",file=ouf)
    print(f"""    
// Center planetocentric latitude of PNG topography tiles in this folder tree
#declare latc={dtm_latc}; //deg 
// Center east longitude of PNG topography tiles in this folder tree
#declare lonc={dtm_lonc}; //deg
""",file=ouf)
    print("""
#macro height_tile(Level,I_x,I_y)
  #include concat(str(Level,-2,0),"/tile",str(I_x,-3,0),"x",str(I_y,-3,0),".inc")
  height_field {
    png concat(str(Level,-2,0),"/tile",str(I_x,-3,0),"x",str(I_y,-3,0),".png")
    smooth
    rotate x*90
    scale <1,-1,1>
    scale <x1-x0,y1-y0,z1-z0>
    translate <x0,y0,z0>
    rotate y*(90-latc)
    rotate z*(90+lonc)
    pigment {color rgb LevelColor[3/*Level-6*/]}
  }
#end
    """,file=ouf)

#This transformation puts grid east in the +x and grid north in the +y direction as expected
Mz=rotz(np.radians(-90-dtm_lonc)) #put center on -90E meridian
Mx=rotx(np.radians(dtm_latc-90)) #put -90E, center lat on north pole
M=Mx @ Mz
xx = np.cos(np.radians(dtm_lonc)) * np.cos(np.radians(dtm_latc))
yy = np.sin(np.radians(dtm_lonc)) * np.cos(np.radians(dtm_latc))
zz = np.sin(np.radians(dtm_latc))
v=np.array([[xx],[yy],[zz]])
print(Mz @ v)
print(Mx @ Mz @ v)

if plot:
    plt.figure('areoid globe')
    plt.imshow(areoid.data[::16,::16],origin='lower',extent=[0,360,-90,90])
    plt.pause(0.001)

areoid=areoid.interp()(lon_1d.reshape(-1),lat_1d.reshape(-1))
if plot:
    plt.figure('areoid patch')
    plt.imshow(areoid[::16,::16],origin='lower')
    plt.pause(0.001)
w=np.where(dtm1.data<-3000)
dtm1.data+=areoid #dtm1.data is now radius. latitude and longitude have always been planetocentric
del areoid

def loadmap():
    result=DTM(infn=None,lat0=-88+1/256,lat1=88-1/256,lon0=-180+1/256,lon1=+180+1/256,rows=176*128,cols=360*128)
    if plot:
        plt.figure('radius globe')
    for filerow,filelat in enumerate([-44,0,44,88]):
        rowname=f"{np.abs(filelat):02d}{'s' if filelat<0 else 'n'}"
        for filecol,filelon in enumerate([180,270,0,90]):
            colname=f"{filelon:03d}"
            filename=f"/mnt/big/home/chrisj/workspace/Data/Planet/Mars/Radius/megr{rowname}{colname}hb.img"
            maprow=filerow*(44*128)
            mapcol=filecol*(90*128)
            result.data[maprow:maprow+44*128,mapcol:mapcol+90*128]=np.flipud(np.fromfile(filename,dtype=np.dtype('>i2')).reshape(44*128,90*128))
            if plot:
                plt.imshow(result.data[::128,::128],origin='lower',extent=[-180,180,-88,88])
                plt.title(filename)
                plt.pause(0.001)
    result.data+=mars_r
    return result

radius=DTM("/mnt/big/home/chrisj/workspace/Data/Planet/Mars/Radius/megr44n000hb.img",
           filetype='>i2',has_header=False,lat0=0+1/32,lon0=0+1/32,lat1=44-1/32,lon1=90-1/32,rows=44*128,cols=90*128,offset=mars_r,blank=None)


radius=radius.interp()(lon_1d.reshape(-1),lat_1d.reshape(-1))
if plot:
    plt.figure('radius patch')
    plt.imshow(radius[::16,::16],origin='lower',extent=[dtm1.lon0,dtm1.lon1,dtm1.lat0,dtm1.lat1])
    plt.pause(0.001)
dtm1.data[w]=radius[w]
del w

if plot:
    plt.figure('dtm filled')
    plt.imshow(dtm1.data[::16,::16],origin='lower',extent=[dtm1.lon0,dtm1.lon1,dtm1.lat0,dtm1.lat1])
    plt.pause(0.001)

lon=(lat_1d*0+1) @ np.deg2rad(lon_1d)
lat=np.deg2rad(lat_1d) @ (lon_1d*0+1)

if plot:
    plt.figure('lon')
    plt.imshow(lon[::16,::16],origin='lower')
    plt.figure('lat')
    plt.imshow(lat[::16,::16],origin='lower')
    plt.pause(0.001)
print("Converting to rectangular")
x=dtm1.data*np.cos(lon)*np.cos(lat)
y=dtm1.data*np.sin(lon)*np.cos(lat)
del lon
z=dtm1.data            *np.sin(lat)
del dtm1
shape=x.shape
x.shape=(shape[0]*shape[1],)
y.shape=(shape[0]*shape[1],)
z.shape=(shape[0]*shape[1],)
print("Rotating")
print(Mz@v)

(xp,yp,zp)=vdecomp(M @ vcomp((x,y,z)))
del x,y,z
xp.shape=shape
yp.shape=shape
zp.shape=shape
print("Done rotating")

print(np.min(xp),np.max(xp),np.max(xp)-np.min(xp))
print(np.min(yp),np.max(yp),np.max(yp)-np.min(yp))
print(np.min(zp),np.max(zp),np.max(zp)-np.min(zp))
if plot:
    plt.figure('dtm')
    plt.imshow(zp[::16,::16],origin='lower')
    plt.show()

print("Ready to punch out tiles")

def punchout(x0,y0,x1,y1):
    print(f"Tile from x0={x0} to x1={x1}, y0={y0} to y1={y1}")
    print("    Finding data inside bounds")
    w=np.where(np.logical_and(np.logical_and(np.logical_and(xp>x0-1,xp<x1+1),yp>y0-1),yp<y1+1))
    print("    Subsetting data")
    xw=xp[w]
    yw=yp[w]
    zw=zp[w]
    print("    Creating interpolator")
    zinterp=LinearNDInterpolator(list(zip(xw,yw)),zw)
    print("    Executing interpolator")
    x=np.arange(x0,x1)
    y=np.arange(y0,y1)
    x,y=np.meshgrid(x,y)
    zinterp=zinterp(x,y)
    print("    Done")
    return zinterp

tilesize=256

def dotile(xy):
    x0=xy[0]
    y0=xy[1]
    tilefn = f'tile{x0 // tilesize + 6 * 1024 // tilesize:03d}x{y0 // tilesize + 8 * 1024 // tilesize:03d}.npy'
    tilefn = "/mnt/big/home/chrisj/workspace/Data/Planet/Mars/Jezero/Topography/" + tilefn
    if not os.path.exists(tilefn):
        a = punchout(x0, y0, x0 + tilesize, y0 + tilesize)
        # plt.figure('a')
        # plt.imshow(a-np.min(a),origin='lower')
        tilefn = f'tile{x0 // tilesize + 6 * 1024 // tilesize:03d}x{y0 // tilesize + 8 * 1024 // tilesize:03d}.npy'
        # plt.title(tilefn)
        tilefn = "/mnt/big/home/chrisj/workspace/Data/Planet/Mars/Jezero/Topography/" + tilefn
        # plt.pause(0.001)
        np.save(tilefn, a)

inpair=[]
for y0 in range(-8*1024,8*1024,tilesize):
    for x0 in range(-6*1024,6*1024,tilesize):
        inpair.append((x0,y0))

pool=multiprocessing.Pool(processes=12)
out=pool.map(dotile,inpair)


