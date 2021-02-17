import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from vector import vcomp, vdecomp
from scipy.interpolate import interp2d


wholemap=np.zeros((44*128*4,90*128*4),dtype=np.int16)

def loadmap():
    global wholemap
    for filerow,filelat in enumerate([-44,0,44,88]):
        rowname=f"{np.abs(filelat):02d}{'s' if filelat<0 else 'n'}"
        for filecol,filelon in enumerate([180,270,0,90]):
            colname=f"{filelon:03d}"
            filename=f"/mnt/big/home/chrisj/workspace/Data/Planet/Mars/Radius/megr{rowname}{colname}hb.img"
            maprow=filerow*(44*128)
            mapcol=filecol*(90*128)
            wholemap[maprow:maprow+44*128,mapcol:mapcol+90*128]=np.flipud(np.fromfile(filename,dtype=np.dtype('>i2')).reshape(44*128,90*128))
            plt.imshow(wholemap[::128,::128],origin='lower',extent=[0,360,-88,88])
            plt.title(filename)
            plt.pause(0.001)

def lon2col(lon):
    return int((lon+180)*128)

def col2lon(col):
    return col/128-180

def lat2row(lat):
    return int((lat+88)*128)

def row2lat(row):
    return row/128-88

def subsample(lat0,lon0,lat1,lon1,x,y):
    col0=lon2col(lon0)
    col1=lon2col(lon1)
    row0=lat2row(lat0)
    row1=lat2row(lat1)
    zoomfac=(y/(row1-row0),x/(col1-col0))
    return zoom(wholemap[row0:row1,col0:col1],zoom=zoomfac)

def sample(lat,lon):
    return wholemap[lat2row(lat),lon2col(lon)]

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

plt.figure("WholeGlobe")
loadmap()
plt.figure("Jezero")
#lon0=76.5
#lat0=17.5
lon0=77.5
lat0=18.5
r0=3396000
size=128
lat=np.radians(row2lat(np.array(range(lat2row(lat0)-size,lat2row(lat0)+size)))).reshape(-1, 1)
lon=np.radians(col2lon(np.array(range(lon2col(lon0)-size,lon2col(lon0)+size)))).reshape( 1,-1)
r=wholemap[lat2row(lat0)-size:lat2row(lat0)+size,lon2col(lon0)-size:lon2col(lon0)+size]+r0
x=r*np.cos(lon)*np.cos(lat)
y=r*np.sin(lon)*np.cos(lat)
z=r            *np.sin(lat)
print(np.min(x),np.max(x))
print(np.min(y),np.max(y))
print(np.min(z),np.max(z))
M=roty(np.radians(lat0-90)) @ rotz(np.radians(-lon0))
print(M)
(xp,yp,zp)=vdecomp(M @ vcomp((x,y,z)))
print(np.min(xp),np.max(xp))
print(np.min(yp),np.max(yp))
print(np.min(zp),np.max(zp))

plt.imshow(zp,origin='lower')
plt.show()
print("Done")