"""
Variable naming convention:


Vectors and components are named as {name}{comp}{frame} where:
    * name is:
       * r - position
       * v - velocity
       * s - six-component position and velocity
       * a - acceleration
       * ang - non-gravitational acceleration
       * z - z axis
       * n - LVLH north
       * e - LVLH east
       * h - LVLH downrange
       * q - LVLH crossrange
    * comp is:
       * v - vector
       * hat - normalized vector
       * x - x axis
       * y - y axis
       * z - z axis
       * n - LVLH north
       * e - LVLH east
       * h - LVLH downrange
       * q - LVLH crossrange
    * frame is:
       * i - J2000 inertial
       * f - frozen frame -- inertial frame parallel to IAU_MARS at one instant
       * r - IAU_MARS rotating frame
       * missing if the frame doesn't matter, like with zhat

Matrices are named M{sp}{from}2{to} where each of from and to are frames as above. sp
is either 'p' for pxform, which can transform a position between any two frames
and a velocity between two inertial frames, or 's' for sxform, which can transform
a full position and velocity state to and from possibly non-inertial frames

Stacks of vectors are done as [row,stack]. Stacks of matrices are done as [stack,row,col].

To transform a stack of vectors with one matrix, do: rvb=Ma2b @ rva
To transform a stack of vectors with a stack of matrices, it isn't as straightforward. The vector stack
has to be transformed into a [stack,row,1] stack of matrices, then back: rvb=(Ma2b @ (rva.transpose().reshape(-1,3,1)))[:,:,0].transpose()

In POV-Ray, names of variables customarily include capital letters to distinguish from keywords which
are pure lowercase. We use the same name in POV-Ray as in python, except for capitalizing the first letter.
"""


import spiceypy
import numpy as np
import matplotlib.pyplot as plt
from vector import vlength, vcomp, vdecomp, vangle, vdot, vcross

km=1.0 #Transform Spice distance units -- Set to 1.0 to leave units in km. Set to 1000.0 to transform to meters
zhat=np.array([[0],[0],[1]],dtype=np.float64)

#Standard planetary constant kernel, to support the IAU_MARS Mars body fixed frame
spiceypy.furnsh("/mnt/big/home/chrisj/workspace/Data/spice/generic/pck/pck00010.tpc")
spiceypy.furnsh("/mnt/big/home/chrisj/workspace/Data/spice/generic/spk/planets/de430.bsp")
spiceypy.furnsh("/mnt/big/home/chrisj/workspace/Data/spice/generic/spk/satellites/mar097.bsp")
#GM and J2 for Mars, and radii consistent with them
spiceypy.furnsh("/mnt/big/home/chrisj/workspace/Data/spice/generic/pck/mars_jgmro.tpc")
#Spacecraft clock kernel, not used until we have a CK, then must use the one that matches the CK
#spiceypy.furnsh("/mnt/big/home/chrisj/workspace/Data/spice/M20/sclk/m2020.tsc")
#Spacecraft cruise/EDL/surface trajectory
spiceypy.furnsh("/mnt/big/home/chrisj/workspace/Data/spice/M20/spk/m2020_trajCEDLS-6DOF_ops_od020v1_AL23.bsp")

def Mtrans(M,v):
    if len(M.shape)>2:
        return (M @ (v.transpose().reshape(v.shape[1], v.shape[0], 1)))[:, :, 0].transpose()
    else:
        return M @ v

def rv(sv):
    return sv[:3,:]

def vv(sv):
    return sv[3:,:]

def Mr(Ms):
    return Ms[:,:3,:3]

def Mv(Ms):
    return Ms[:,3:,3:]

def xyz2llr(sv):
    x,y,z=vdecomp(rv(sv))
    r=vlength(rv(sv))
    lat=np.arcsin(z/r)
    lon=np.arctan2(y,x)
    return(lon,lat,r)

MarsR=spiceypy.gdpool("BODY499_RADII",0,1)[0]*km #Values in kernel uses km...
MarsGM=spiceypy.gdpool("BODY499_GM",0,1)[0]*km**3   #we use meters internally
MarsJ2=spiceypy.gdpool("BODY499_J2",0,1)[0]
Marsw=np.deg2rad(spiceypy.gdpool("BODY499_PM",1,1)[0])/86400

def J2(svf):
    """
    J2 gravity acceleration
    :param svf: State vector in an inertial equatorial frame (frozen frame is designed to meet this requirement)
    :return: J2 acceleration in (distance units implied by rvf)/(time units implied by constants)s**2 in same frame as rvf

    Constants MarsGM, MarsJ2, and MarsR must be pre-loaded and have distance units consistent
    with rvf, and time units consistent with each other.

    Only position part of state is used, but the velocity part *should* have time units consistent
    with the constants. Time units follow those of the constants, completely ignoring those implied
    by the velocity part
    """
    r=vlength(rv(svf))
    coef=-3*MarsJ2*MarsGM*MarsR**2/(2*r**5)
    x,y,z=vdecomp(rv(svf))
    j2x=x*(1-5*z**2/r**2)
    j2y=y*(1-5*z**2/r**2)
    j2z=z*(3-5*z**2/r**2)
    return (coef*vcomp((j2x,j2y,j2z)))

def TwoBody(svi):
    """
    Two-body gravity acceleration
    :param rv: Position vector in an inertial frame
    :return: Two-body acceleration in (distance units implied by rv)/s**2
    """
    return -MarsGM*rv(svi)/vlength(rv(svi))**3

def ai(ets,svi):
    """
    Measured inertial acceleration
    :param ets: Time scale
    :param svi: State vector in an inertial frame. Time units implied by this vector must match those implied by ets.
    :return: Measured inertial acceleration in (distance units implied by svi)

    Note -- beginning and/or ending points may be patched
            with their nearest neighbors.

            Position part of state is ignored, but *should* be in the same frame
            with the same implied distance units as velocity part.
    """
    result=vv(svi)*0 #Make an array with the same shape as the input vectors
    #Follow formula from https://en.wikipedia.org/wiki/Symmetric_derivative
    fx=vv(svi)[:,1:-1]
    fxp=vv(svi)[:,2:]
    fxm=vv(svi)[:,:-2]
    h2=ets[2:]-ets[:-2]
    result[:,1:-1]=(fxp-fxm)/h2
    result[:,0]=result[:,1]
    result[:,-1]=result[:,-2]
    return result

def ang(ets,svi,avi=None):
    """
    Measured non-gravitational acceleration.
    :param ets: Time values for each vector, s, Spice ET scale will do fine
    :param svi: Inertial position vector in a Mars equatorial frame
    :param avi: Measured total acceleration in the same frame. If not passed, calculated from velocity part of state
    :return: Non-gravitational acceleration in the same frame.

    Calculated as the measured acceleration minus the modeled gravitational accelerations at the measured position
    """
    if avi is None:
        avi=ai(ets,svi)
    return avi-TwoBody(svi)-J2(svi)

#EI definition from https://mars.nasa.gov/mer/mission/timeline/edl/
ei_rad=3522.2*km
et_land=666953092.07640004158020 #Time of first point in post-landing segment.
et0=et_land-420 #Seven minutes of excitement! Entry interface is about 7.25s after this time.
dt=1.0/24.0
et1=et_land+20 #Make sure we are steady on the ground

#grab a frozen copy of IAU_MARS relative to J2000 at et0. Use this frozen frame
#as the inertial frame for doing J2 calculations (and therefore non-gravitational
#accelerations) since it is Mars-equatorial, as needed by J2.
Mpi2f=spiceypy.pxform("J2000","IAU_MARS",et0)
Mpf2i=spiceypy.pxform("IAU_MARS","J2000",et0)
#Calculate a state transform by making a 6x6 matrix and filling in the diagonal
#corners with the position transform
#From https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html#Rotation%20State
#
#    When a parameterized dynamic frame is specified as ``inertial,'' the
#    derivative with respect to time of the transformation between the frame and
#    any inertial frame, for example the J2000 frame, is zero. The rotation
#    between the frame and any inertial frame is still treated as time-dependent.
#    For such a frame F, the call
#
#       sxform_c ( "F", "J2000", t, xform );
#
#    yields a 6x6 state transformation matrix `xform' having the structure
#
#       +-----+-----+
#       | R(t)|  0  |
#       +-----+-----+
#       |  0  | R(t)|
#       +-----+-----+
#
#    where R(t) is the 3x3 rotation matrix that transforms vectors from frame
#    F to the J2000 frame at time `t'. By contrast, when the rotation state
#    of F is ``rotating,'' `xform' has the structure
#
#       +-----+-----+
#       | R(t)|  0  |
#       +-----+-----+
#       |dR/dt| R(t)|
#       +-----+-----+
#
#    So, when the rotation state of frame F is ``inertial,'' velocities are
#    transformed from frame F to J2000 by left-multiplication by R(t); the
#    time derivative of the rotation from F to J2000 is simply ignored.
#
#    Normally the inertial rotation state makes sense only for slowly rotating
#    frames such as the earth mean equator and equinox of date frame.
#
# We have to do this manually because the frozen frame is not
# an official Spice frame
Msi2f=np.zeros((6,6))
Msi2f[:3,:3]=Mpi2f
Msi2f[3:,3:]=Mpi2f
# Do it again for the inverse transform
Msf2i=np.zeros((6,6))
Msf2i[:3,:3]=Mpf2i
Msf2i[3:,3:]=Mpf2i
#Prove that these two matrices are in fact inverses. Their product
#should be the identity matrix
#print(Msf2i@Msi2f)
#Prove that the two matrices are transposes of each other
#print(Msf2i-Msi2f.transpose())

ets=np.arange(et0,et1,dt)
Msi2r=np.zeros((len(ets),6,6))
Msr2i=np.zeros((len(ets),6,6))
svi=np.zeros((6,len(ets)))
svr=np.zeros((6,len(ets)))
sunr=np.zeros((3,len(ets)))
suni=np.zeros((3,len(ets)))
equal=np.zeros(len(ets))
for i,et in enumerate(ets):
    state,_=spiceypy.spkezr("-168",et,"J2000","NONE","499")
    svi[:,i]=state*km
    state,_=spiceypy.spkezr("-168",et,"IAU_MARS","NONE","499")
    svr[:,i]=state*km
    state,_=spiceypy.spkezr("SUN",et,"IAU_MARS","LT+S","499")
    sunr[:,i]=state[:3]*km
    state,_=spiceypy.spkezr("SUN",et,"J2000","LT+S","499")
    suni[:,i]=state[:3]*km
    Msr2i[i, :, :] = spiceypy.sxform("IAU_MARS", "J2000", et)
    Msi2r[i, :, :] = spiceypy.sxform("J2000", "IAU_MARS", et)
    #Prove that the upper left corner of Msr2i is equivalent to Mpr2i
    #Mpr2i[i, :, :] = spiceypy.pxform("IAU_MARS", "J2000", et)
    #Mpi2r[i, :, :] = spiceypy.pxform("J2000", "IAU_MARS", et)
    #equal[i]=1 if np.allclose(Mpi2r[i,:,:],Msi2r[i,:3,:3]) else 0
    #Prove that Msi2r correctly transforms states (including velocity) from inertial to relative
    #equal[i]=1 if np.allclose(svr[:,i],Msi2r[i,:,:]@svi[:,i]) else 0
svf=Msi2f @ svi
#Prove that magnitudes of position and velocity are preserved
#diff_r=vlength(svf[:3,:])-vlength(svi[:3,:])
#diff_v=vlength(svf[3:,:])-vlength(svi[3:,:])
#plt.plot(diff_r)
#plt.plot(diff_v)
#plt.show()

avf=ai(ets,svf)
angvf=ang(ets, svf, avi=avf)
avr=avf*0
Msf2r=Msi2r @ Msf2i
Msr2f=Msi2f @ Msr2i
avr  = Mtrans(Mr(Msf2r),avf)
angvr= Mtrans(Mr(Msf2r),angvf)

"""
plt.figure('diff1')
plt.plot(rvr[0,:]-rvr_f2r[0,:])
plt.plot(rvr[1,:]-rvr_f2r[1,:])
plt.plot(rvr[2,:]-rvr_f2r[2,:])
plt.pause(0.001)
"""

#Local vertical
rhatr=rv(svr)/vlength(rv(svr))
#unit vector in direction of relative velocity
vhatr=vv(svr)/vlength(vv(svr))
#East vector
ehatr=vcross(zhat,rhatr)
ehatr/=vlength(ehatr)
#North vector
nhatr=vcross(rhatr,ehatr)
nhatr/=vlength(nhatr)
#Crossrange vector, perpendicular to vvr and rhat, so in local horizon plane
hhatr=vcross(rhatr, vhatr)
hhatr/=vlength(hhatr)
#Downrange vector, horizontal direction of velocity. Perpendicular to both h and r
qhatr=vcross(hhatr, rhatr)
qhatr/=vlength(qhatr)

def decompose_lvlh(v, rhat, ehat, nhat, hhat, qhat):
    """
    Decompose a vector into components in the local-horizontal, local-vertical frame
    :param v: Vector to decompose
    :param rhat: Local vertical
    :param vhat: Unit vector in direction of velocity
    :param ehat: East vector
    :param nhat: North vector
    :param hhat: Side vector, perpendicular to vhat and rhat
    :param qhat: Downrange vector, horizontal direction of velocity, perpendicular to rhat and hhat
    :return: A tuple of components of input vector v in the vertical, east, north, side, and downrange direction.
       hypot(east,north) should equal hypot(side,downrange) and hypot(vert,east,north) should equal
       hypot(vert,side,downrnage) which should both equal vlength(v)
    """
    return (vdot(v,rhat),
            vdot(v,ehat),
            vdot(v,nhat),
            vdot(v, hhat),
            vdot(v, qhat))

#Components of relative velocity in LVLH
(  vrr,  ver,  vnr,  vhr,  vqr)=decompose_lvlh(vv(svr), rhatr, ehatr, nhatr, hhatr, qhatr)
#Components of non-gravitational acceleration in LVLH
(angrr,anger,angnr,anghr,angqr)=decompose_lvlh( angvr , rhatr, ehatr, nhatr, hhatr, qhatr)

(lon,lat,r)=xyz2llr(svr)

def print_pov(i_frame,*,file):
    with(f"inc/frame_{i_frame:05d}.inc","wt") as file:
    for i,et in enumerate(ets):
        print(f"{{{ets[i]:.6f},",end='',file=file)
        print(f"{svr   [0,i]:10.6f},{svr   [1,i]:10.6f},{svr   [2,i]:10.6f},",end='',file=file)
        print(f"{svr   [3,i]:10.6f},{svr   [4,i]:10.6f},{svr   [5,i]:10.6f},",end='',file=file)
        print(f"{angvr [0,i]:13.6e},{angvr [1,i]:13.6e},{angvr [2,i]:13.6e},",end='',file=file)
        print(f"{sunr  [0,i]:13.6e},{sunr  [1,i]:13.6e},{suni  [2,i]:13.6e},",end='',file=file)
        print(f"}}, //{i:5d} {spiceypy.etcal(ets[i])}",file=file)
    print("}",file=file)
    print(f"""
// generated in python/edl/edl.py
#declare ET  = {ets  [i]}; //{i_frame:5d} {spiceypy.etcal(ets[i])}
#declare R   =<{svr  [0,i_frame]},{svr  [1,i_frame]},{svr  [2,i_frame]}>;
#declare R1  =<{svr  [0,     -1]},{svr  [0,     -1]},{svr  [0,     -1]}>;
#declare V   =<{svr  [3,i_frame]},{svr  [4,i_frame]},{svr  [5,i_frame]}>;
#declare Ang =<{angvr[0,i_frame]},{angvr[1,i_frame]},{angvr[2,i_frame]}>;
#declare Sun =<{sunr [0,i_frame]},{sunr [1,i_frame]},{sunr [2,i_frame]}>;
""",file=file)

with open("m20.pov","wt") as ouf:
#    print_pov()
    print_pov(file=ouf)

"""
plt.figure('rrel')
plt.plot(ets-et0,svr[0,:],'r',label='relative position x component')
plt.plot(ets-et0,svr[1,:],'g',label='relative position y component')
plt.plot(ets-et0,svr[2,:],'b',label='relative position y component')
plt.xlabel(f'Time from {spiceypy.etcal(et0)}/s')
plt.ylabel('position/km')
plt.legend()

plt.figure('lon,lat')
plt.plot(np.degrees(lon),np.degrees(lat))
plt.xlabel('east longitude/deg')
plt.ylabel('latitude/deg')

plt.figure('alt above EI')
plt.plot(ets-et0,r-ei_rad,'*')
plt.xlabel(f'Time from {spiceypy.etcal(et0)}/s')
plt.ylabel('altitude/km')
plt.figure('avr magnitude')
plt.plot(ets-et0,vlength(avr))

plt.figure('arngr, aengr, anngr')
plt.plot(ets-et0,angrr,'r',label='non-gravitational acceleration radial component')
plt.plot(ets-et0,anger,'g',label='non-gravitational acceleration east component')
plt.plot(ets-et0,angnr,'b',label='non-gravitational acceleration north component')
plt.xlabel(f'Time from {spiceypy.etcal(et0)}/s')
plt.ylabel('acceleration/(km/s**2)')
plt.legend()

plt.figure('angrr, anghr, angqr')
plt.plot(ets-et0,angrr,'r',label='non-gravitational acceleration radial component')
plt.plot(ets-et0,anghr,'g',label='non-gravitational acceleration crossrange component')
plt.plot(ets-et0,angqr,'b',label='non-gravitational acceleration downrange component')
plt.xlabel(f'Time from {spiceypy.etcal(et0)}/s')
plt.ylabel('acceleration/(km/s**2)')
plt.legend()

plt.show()
"""