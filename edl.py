"""
Variable naming convention:

There is a naming collision in the word "frame" -- it is both the unit
of time in an animation, and the coordinate system of a vector. We shall
exclusively use the word "frame" for vectors and instead call the
unit of animation time a "step".

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
       * sunr - position of Sun
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
    All position and velocity vectors are relative to the center of Mars in the
    named not-necessarily-inertial frame. This implies that any position vector
    will have the same length, no matter the frame, but not every velocity
    vector will have the same length, since they might not be expressed in
    an inertial frame.

Matrices are named M{sp}{to}{from} where each of from and to are frames as above.
sp is either 'p' for pxform, which can transform a position between any two frames
and a velocity between two inertial frames, or 's' for sxform, which can transform
a full position and velocity state to and from possibly non-inertial frames

We therefore read Mpfi as the position transformation matrix *to* the fixed frame *from* the inertial frame

Older code used M{sp}{from}2{to}. We like the way above, because that means that in a correct
chain of transformations, intermediate frames must be adjacent in consecutive matrices. Also, if reading right-to-left
like you do for transformations, you can read the frames the same way. IE in the old way, we had:

Mpi2r=Mpf2r @ Mpi2f

Where as with the new way we have:

Mpri=Mprf @ Mpfi

We see the two f's for the intermediate f frame are adjacent.

Stacks of vectors are done as [row,stack]. Stacks of matrices are done as [stack,row,col].

To transform a stack of vectors with one matrix, do: rvb=Ma2b @ rva
To transform a stack of vectors with a stack of matrices, it isn't as straightforward. The vector stack
has to be transformed into a [stack,row,1] stack of matrices, then back:

rvb=(Ma2b @ (rva.transpose().reshape(-1,3,1)))[:,:,0].transpose()

In POV-Ray, names of variables customarily include capital letters to distinguish from keywords which
are pure lowercase. We use the same name in POV-Ray as in python, except for capitalizing the first letter.
"""
from collections import namedtuple
import datetime

import pytz
import spiceypy
import numpy as np
from kwanspice.daf import double_array_file
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from vector import vlength, vcomp, vdecomp, vdot, vcross
from mars_atm import mars_atm
import tile
from which_kernel import ls_spice

import kwanspice

def smooth(s,p0,p1):
    """
    Do a boxcar average on a 1D dataset
    :param v:
    :return:
    """
    result=s*0
    count=0
    for p in range(p0,p1):
        result[-p0:-p1]+=s[-p0+p:len(s)-p1+p]
        count+=1
    result/=count
    result[:-p0]=s[:-p0]
    result[-p1:]=s[-p1:]
    return result


#Generic functions, usable in this or other projects. No access to global state
def vnormalize(a):
    """
    Calculate the unit vector in a given direction
    :param a: vector to get direction from
    :return: unit vector in given direction
    """
    return a/vlength(a)

def vncross(a, b):
    """
    Normalized cross product
    :param a: First cross product factor
    :param b: Second cross product factor
    :return: Unit vector in same direction as vcross(a,b)
    """
    return vnormalize(vcross(a, b))

def point_toward(*, p_b, p_r, t_b, t_r):
    """
    Calculate the point-towards transform
    :param p_b: Body frame point vector
    :param p_r: Reference frame point vector
    :param t_b: Body frame toward vector
    :param t_r: Reference toward vector
    :return: Mb2r which makes p_r=Mb2r@p_b true, and simultaneously minimizes vangle(t_r,Mb2r@t_b)
    """
    s_r=vncross(p_r, t_r)
    u_r=vncross(p_r, s_r)
    R=np.stack((vnormalize(p_r).transpose(), s_r.transpose(), u_r.transpose()), axis=2)
    s_b=vncross(p_b, t_b)
    u_b=vncross(p_b, s_b)
    B=np.stack((vnormalize(p_b).transpose(), s_b.transpose(), u_b.transpose()), axis=2)
    return R @ inv(B)

_LocLookReturn=namedtuple("LocLookReturn", "look_at location")
def loc_look(*, location, look_at, sky=np.array([[0], [0], [1]]), direction_b=np.array([[0], [0], [1]]), sky_b=np.array([[0], [1], [0]])):
    """
    Calculate Location-Look_at transform.
    :param location: Location of object in world space
    :param look_at: Point to look at in world space
    :param sky: Toward vector in reference space
    :param direction_b: Point this vector at look_at
    :param sky_b: Point this vector toward sky
    :return: A tuple of Mb2r rotation matrix and location vector
    """
    p_b=direction_b
    p_r=look_at-location
    t_b=sky_b
    t_r=sky
    return _LocLookReturn(look_at=point_toward(p_b=p_b, p_r=p_r, t_b=t_b, t_r=t_r), location=location)

def Mtrans(M,*vs):
    """
    Transform a matrix or stack of matrices against a vector or stack of vectors
    :param M: Matrix [rows,columns] or stack of matrices [stack,rows,columns]
    :param v: Column vector [rows,1] or stack of vectors [rows,stack]
    :return: Transformed vector(s)
    If M and v are both singles, return a single column vector
    If M is single and v is stack, return a stack of vectors, each one transformed against the (one and only) matrix
    If M is stack, v must be stack, return a stack of vectors, each one transformed against the corresponding matrix
    """
    if len(M.shape)>2:
        result=tuple([ (M @ (v.transpose().reshape(v.shape[1], v.shape[0], 1)))[:, :, 0].transpose() for v in vs])
    else:
        result=tuple([M @ v for v in vs])
    if len(vs)==1:
        result=result[0]
    return result

def rv(sv):
    """
    Position part of state vector
    :param sv: Stack of state vectors, can be one in stack IE column vector
    :return: Position part, will be stack matching sv
    """
    return sv[:3,:]

def vv(sv):
    """
    Velocity part of state vector
    :param sv: Stack of state vectors, can be one in stack IE column vector
    :return: Position part, will be stack matching sv
    """
    return sv[3:,:]

def Mr(Ms):
    """
    Get position part of state transformation matrix
    :param Ms: Stack of state transformation matrices
    :return: Position part, IE upper 3x3 of each matrix
    """
    return Ms[...,:3,:3]

def Mv(Ms):
    """
    Get position part of state transformation matrix
    :param Ms: Stack of state transformation matrices
    :return: Position part, IE upper 3x3 of each matrix
    """
    return Ms[...,3:,3:]

def xyz2llr(sv):
    """
    Calculate spherical coordinates of state
    :param sv: State vector, can be stack
    :return: tuple of (lon,lat,r). Each will be an array iff sv is a stack
    """
    x,y,z=vdecomp(rv(sv))
    r=vlength(rv(sv))
    lat=np.arcsin(z/r)
    lon=np.arctan2(y,x)
    return(lon,lat,r)

def llr2xyz(*,latd,lond,r):
    x=r*np.cos(np.radians(latd))*np.cos(np.radians(lond))
    y=r*np.cos(np.radians(latd))*np.sin(np.radians(lond))
    z=r*np.sin(np.radians(latd))
    return vcomp((x,y,z))

def linterp(x0,y0,x1,y1,x):
    """
    Linear interpolation
    :param x0: independent value of variable at one end
    :param y0: dependent value corresponding to x0
    :param x1: independent value at other end, doesn't need to be less than x0
    :param y1: dependent corresponding to x1
    :param x: independent variable, can be outside of range [x0,x1]
    :return: dependent variable at this point
    Note: All values can be arrays subject to Numpy broadcasting
    """
    t=(x-x0)/(x1-x0)
    return (1-t)*y0+t*y1

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

def body_rate(Mrb, dts):
    #Rate of change of Mrb matrix
    Mdotrb= Mrb * 0
    Mdotrb[1:-1, :, :]= (Mrb[2:, :, :] - Mrb[:-2, :, :]) / (2 * dts[1:-1,None,None])
    Mdotrb[0, :, :]= Mdotrb[1, :, :]
    Mdotrb[-1, :, :]= Mdotrb[-2, :, :]
    #extract body rotation rate from matrix derivative
    wtimes= Mrb.transpose((0,2,1)) @ Mdotrb
    zx = wtimes[:,0, 0];mz = wtimes[:,0, 1];py = wtimes[:,0, 2]
    pz = wtimes[:,1, 0];zy = wtimes[:,1, 1];mx = wtimes[:,1, 2]
    my = wtimes[:,2, 0];px = wtimes[:,2, 1];zz = wtimes[:,2, 2]
    omegax = (px - mx) / 2
    omegay = (py - my) / 2
    omegaz = (pz - mz) / 2
    omegarv=vcomp((omegax,omegay,omegaz))
    return omegarv

def aJ2(svf,*,j2,gm,re):
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
    coef=-3*j2*gm*re**2/(2*r**5)
    x,y,z=vdecomp(rv(svf))
    j2x=x*(1-5*z**2/r**2)
    j2y=y*(1-5*z**2/r**2)
    j2z=z*(3-5*z**2/r**2)
    return (coef*vcomp((j2x,j2y,j2z)))

def aTwoBody(svi,*,gm):
    """
    Two-body gravity acceleration
    :param rv: Position vector in an inertial frame
    :return: Two-body acceleration in (distance units implied by rv)/s**2
    """
    return -gm*rv(svi)/vlength(rv(svi))**3


#Event frame numbers
eventTuple=namedtuple("eventTuple","entry0 sufr0 ebm sufr1 chute0 chute1 heatshield backshell skycrane touchdown")
bodyTuple=namedtuple("bodyTuple","pov_suffix et0 a m cd r0vb v0vb",defaults=(np.zeros((3,1)),np.zeros((3,1))))

def et2pdt(t):
    """
    Get Python datetime from Spice ET
    :param t: Time as either a datetime object, string that str2et() can parse, or double-precision Spice ET
    :return: A naive Python datetime object -- naive since it is in ET, not UTC or any other time zone
    """
    cal = spiceypy.timout(t, "YYYY-MM-DD HR:MN:SC.###### ::TDB")
    t = datetime.datetime(int(cal[0:4]), int(cal[5:7]), int(cal[8:10]),
                          int(cal[11:13]), int(cal[14:16]), int(cal[17:19]),
                                                     int(cal[20:26]))
    return t

class SPKLoader:
    """
    Abstract class representing code which can return a densely
    spaced set of points along the trajectory. Two immediate
    classes suggest themselves: One (FrameSPKLoader) which uses
    the spiceypy library to get interpolated positions at
    a chosen uniform frame rate, and one (DAFSPKLoader) which
    bypasses the spiceypy library, reads the kernel directly
    with kwanspice.daf, and gets the exact times and sample
    points in the file.
    """
    def __init__(self):
        pass
    def time(self):
        """
        Calculate the time coverage to be returned

        :return:  numpy array of Spice et values (float64)
        """
        raise NotImplementedError
    def spice(self):
        raise NotImplementedError

class UniformSPKLoader(SPKLoader):
    def __init__(self,et0,et1,dt,spice_sc):
        super().__init__()
        self.et0=et0
        self.et1=et1
        self.dt=dt
        self.spice_sc=spice_sc
    def time(self)->np.array:
        return np.arange(self.et0, self.et1, self.dt)
    def spice(self)->np.array:
        ets=self.time()
        result=np.zeros((6,ets.size))
        for i, et in enumerate(ets):
            result[:,i],_ = spiceypy.spkezr(self.spice_sc, et, "J2000", "NONE", "499")
        return result

class Tableterp:
    def __init__(self,x,y):
        self._x=x
        self._y=y
        if len(self._y.shape)==2:
            self.axis = 1
        else:
            self.axis=0
        self._yi=interp1d(self._x,self._y,axis=self.axis,assume_sorted=True,copy=False)
    def __call__(self,x=None):
        if x is None:
            result=self._y.view()
            result.flags.writeable=False
            return result
        if len(self._y.shape)==2:
            return self._yi(x)[:,None]
        else:
            return self._yi(x)


class Trajectory:
    # Fixed constants, not dependent on any Spice kernel
    km = 1000.0  # Transform Spice distance units -- Set to 1.0 to leave units in km. Set to 1000.0 to transform to meters
    # EI definition from https://mars.nasa.gov/mer/mission/timeline/edl/. Intended to be 125km above surface, but
    # defined this way so as to not depend on latitude or actual radius of Mars
    ei_rad = 3522.2 * km

    def __init__(self, *,loader:SPKLoader,extra_kernels,dropobjs=None,
                 Cheat=0.224e-10,mheat=0.879,nheat=4.22,
                 DIMU_A_ofs=None):
        """

        :param loader:
        :param extra_kernels:
        :param dropobjs:
        :param Cheat: Heating indicator coefficient -- Constant for unit conversion to W/cm**2
        :param mheat: Heating indicator coefficient -- Exponent for rho. mheat=1 would mean heating is proportional to rho
        :param nheat: Heating indicator coefficient -- Exponent for velocity. nheat=2 would make heating just like dynamic pressure
        Default values for heating indicator coefficients are from Edquist et al,
        "Aerothermodynamic design of the Mars Science Laboratory Heatshield",
        https://doi.org/10.2514/6.2009-4075, table 7
        """
        self.dropobjs=dropobjs
        self.loader=loader
        self._furnsh(extra_kernels)
        self.Cheat = Cheat
        self.mheat = mheat
        self.nheat = nheat
        self.DIMU_A_ofs=np.array([[-0.958],[0.641],[1.826]]) if DIMU_A_ofs is None else DIMU_A_ofs
    def _calc(self):
        self._time()
        self._spice()
        self._atm()
        self._vrel()
        self._heat()
        self._acc()
        self._lvlh()
        self._att()
        self._body_rate()
        self._rel()
        self._drops()
    def _time(self):
        """
        Establish the time scales

        :param et0: Spice ET of start of animation
        :param et1: Spice ET of end of animation
        :param dt: Time between steps
        :return: None
        :sets dt: same as passed et
        :sets ets: et of each step
        :sets pdts: python datetime object for each step
        :sets i_steps: step number for each step
        """
        self.ets=self.loader.time()
        self.dts=self.ets*0
        self.dts[1:]=self.ets[1:]-self.ets[:-1]
        self.dts[0]=self.dts[1]
        self.pdts=[et2pdt(et) for et in self.ets]
        self.i_steps=np.arange(len(self.ets))
    def _furnsh(self,extra_kernels):
        """
        Furnish the Spice kernels

        :param extra_kernels: List of extra kernels to furnish
        :return: None, but furnishes all the kernels
        """
        # Leap second kernel, to support spiceypy.timout
        spiceypy.furnsh("/mnt/big/home/chrisj/workspace/Data/spice/generic/lsk/naif0012.tls")
        # Standard planetary constant kernel, to support the IAU_MARS Mars body fixed frame
        spiceypy.furnsh("/mnt/big/home/chrisj/workspace/Data/spice/generic/pck/pck00010.tpc")
        spiceypy.furnsh("/mnt/big/home/chrisj/workspace/Data/spice/generic/spk/planets/de438s.bsp")
        spiceypy.furnsh("/mnt/big/home/chrisj/workspace/Data/spice/generic/spk/satellites/mar097s.bsp")
        # GM and J2 for Mars, and radii consistent with them
        spiceypy.furnsh("/mnt/big/home/chrisj/workspace/Data/spice/generic/pck/mars_jgmro.tpc")
        for kernel in extra_kernels:
            spiceypy.furnsh(kernel)
        ls_spice(verbose=True)
    def _spice(self):
        """
        Extract spacecraft state from Spice

        :uses ets: ephemeris times (set in time())
        :uses km: conversion factor to internal units
        :sets mars_re: equatorial radius of Mars in internal distance units
        :sets mars_gm: gravitational constant of Mars in internal distance units and seconds
        :sets mars_j2: oblate gravity field coefficient, unitless
        :sets mars_omega: rotation rate of Mars in rad/s
        :sets svf: Stack of spacecraft state vectors in frozen frame
        :sets Msfi: Stack of matrices to transform to frozen from inertial
        :sets sunr: Stack of apparent position of Sun in rotating frame, taking LT+S into account
        """

        # Constants pulled from Spice kernels
        self.mars_re    = spiceypy.gdpool("BODY499_RADII", 0, 1)[0] * self.km  # Convert immediately to internal units
        self.mars_gm    = spiceypy.gdpool("BODY499_GM", 0, 1)[0] * self.km ** 3
        self.mars_j2    = spiceypy.gdpool("BODY499_J2", 0, 1)[0]
        self.mars_omega = np.deg2rad(spiceypy.gdpool("BODY499_PM", 1, 1)[0]) / 86400  # Convert deg/day to rad/s

        # grab a frozen copy of IAU_MARS relative to J2000 at et0. Use this frozen frame
        # as the inertial frame for doing J2 calculations (and therefore non-gravitational
        # accelerations) since it is Mars-equatorial, as needed by J2.
        Mpfi = spiceypy.pxform("J2000", "IAU_MARS", self.ets[0]) #Matrix converting position to frozen from inertial
        # Calculate a state transform by making a 6x6 matrix and filling in the diagonal
        # corners with the position transform
        # From https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html#Rotation%20State
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
        self.Msfi = np.zeros((6, 6)) #Matrix converting a state to frozen from inertial
        self.Msfi[:3, :3] = Mpfi
        self.Msfi[3:, 3:] = Mpfi
        # Prove that these two matrices are in fact inverses. Their product
        # should be the identity matrix
        # print(Msif@Msfi)
        # Prove that the two matrices are transposes of each other
        # print(Msif-Msfi.transpose())

        print("Extracting state vectors from Spice...")
        Msri = np.zeros((len(self.ets), 6, 6)) #Stack of matrices converting to rotating from inertial
        svi = np.zeros((6, len(self.ets))) #state vector in inertial frame
        self.sunr = np.zeros((3, len(self.ets))) #sun position vectors in rotating frame
        self.earthr = np.zeros((3, len(self.ets))) #sun position vectors in rotating frame
        self.earthlt = np.zeros(len(self.ets)) #sun position vectors in rotating frame
        svi=self.loader.spice()*self.km
        for i, et in enumerate(self.ets):
            Msri[i, :, :] = spiceypy.sxform("J2000", "IAU_MARS", et)
            # Prove that Msri correctly transforms states (including velocity) from inertial to relative
            # equal[i]=1 if np.allclose(svr[:,i],Msri[i,:,:]@svi[:,i]) else 0
            # Since it does, we don't need these
            # state,_=spiceypy.spkezr(self.spice_sc,et,"IAU_MARS","NONE","499")
            # svr[:,i]=state*km
            # Msir[i, :, :] = spiceypy.sxform("IAU_MARS", "J2000", et)
            state, _ = spiceypy.spkezr("SUN", et, "IAU_MARS", "LT+S", "499")
            self.sunr[:, i] = state[:3] * self.km
            state, lt = spiceypy.spkezr("399", et, "IAU_MARS", "XCN+S", "499")
            self.earthr[:, i] = state[:3] * self.km
            self.earthlt[i]=lt
            # Never use suni
            # state,_=spiceypy.spkezr("SUN",et,"J2000","LT+S","499")
            # suni[:,i]=state[:3]*km
            # Prove that the upper left corner of Msr2i is equivalent to Mpr2i
            # Mpir[i, :, :] = spiceypy.pxform("IAU_MARS", "J2000", et)
            # Mpri[i, :, :] = spiceypy.pxform("J2000", "IAU_MARS", et)
            # equal[i]=1 if np.allclose(Mpri[i,:,:],Msri[i,:3,:3]) else 0
        print("Converting to frozen from inertial...")
        self.svf = self.Msfi @ svi #state vector in frozen frame
        print("Calculating to-rotating-from-frozen transformation")
        self.Msrf = Msri @ self.Msfi.transpose() #Stack of matrices converting to rotating from frozen
        #print("Converting to rotating from frozen...")
        #self.svr=Mtrans(self.Msrf, self.svf)
        # Prove that magnitudes of position and velocity are preserved
        # diff_r=vlength(svf[:3,:])-vlength(svi[:3,:])
        # diff_v=vlength(svf[3:,:])-vlength(svi[3:,:])
        # plt.plot(diff_r)
        # plt.plot(diff_v)
        # plt.show()
    def _atm(self):
        """
        Calculate atmosphere properties

        :return: None
        :uses svf: stack of spacecraft state vector in frozen frame (set by _spice())
        :sets rho: stack of free-air density in kg/m**3 at each step
        :sets P: stack of free-air pressure in N/m**2 at each step
        :sets T: stack of free-air temperature in K at each step
        :sets csoound: stack of free-air speed of sound in m/s at each step
        """
        print("Calculating atmosphere...")
        atm_vals = mars_atm(vlength(rv(self.svf)))
        self.rho=atm_vals.rho
        self.P=atm_vals.P
        self.T=atm_vals.T
        self.csound=atm_vals.csound
    def _wind(self,svf):
        """
        Calculate velocity vector of the rotating frame at this point in the frozen frame
        :param svf: stack of state vector in frozen frame
        :uses mars_omega: rotation rate of Mars, rad/s
        :return: "Wind" vector. If you are fixed in frozen frame,
        the atmosphere is attached to the rotating frame, and you will feel a wind blowing
        past you at this velocity. Subtract this wind vector from your frozen frame velocity
        to get your velocity relative to the atmosphere
        """
        result = vcross(np.array([[0], [0], [self.mars_omega]]), rv(svf))
        return result
    def _vrel(self):
        """
        Calculate spacecraft relative velocity

        :uses svf: stack of spacecraft state vectors in frozen frame
        :sets vrelf: stack of spacecraft atmosphere-relative velocity vectors in frozen frame
        :return: None
        """
        self.vrelf = vv(self.svf)-self._wind(self.svf)  # Atmosphere-relative speed in frozen frame
    def _heat(self):
        """
        Calculate spacecraft heating and dynamic pressure

        :uses vrelf: stack of spacecraft atmosphere-relative velocity vectors in frozen frame (from _vrel())
        :uses rho: stack of free-air atmospheric density in kg/m**3 (from _atm())
        :sets dynHeat: stack of dynamic heating rate indicator, in W/cm**2. This is intended to be proportional
            to the actual heating rate that the spacecraft feels, but not necessarily equal to the heating rate
            at any point on the heat shield
        :sets qbar: stack of dynamic pressure in N/m**2
        :sets scorch: stack of integrated heating in J/cm**2, from entry interface to current time, for each step.
            Useful for things like how scorched to draw the heatshield and backshell.
        :return: None
        """
        print("Integrating entry heating")
        self.dynHeat = self.Cheat * vlength(self.vrelf) ** self.nheat * self.rho ** self.mheat  # Heating rate indicator, W/cm**2
        self.qbar = vlength(self.vrelf) ** 2 * self.rho / 2
        self.scorch = np.cumsum(self.dynHeat * self.dts)  # Integrated heating, J/cm**2
    def _acc(self):
        """
        Calculate non-gravitational acceleration on spacecraft. This is what would be
        measured by an accelerometer on board the spacecraft. It is calculated by:
          1. Figuring the total acceleration from the state vectors by finite difference.
             This includes both gravitational and non-gravitational accelerations.
          2. Figuring the gravitational acceleration from the position vector and the
             gravity model. An accelerometer would not measure this component.
          3. Subtracting off the gravitational acceleration from the total acceleration.

        :uses ets: Stack of ephemeris times (from _time())
        :uses svf: Stack of spacecraft state vectors in frozen frame (from _spice())
        :uses mars_gm: Mars gravitational constant (from _spice())
        :uses mars_j2: Mars largest gravity field coefficient (from _spice())
        :uses mars_re: Mars equatorial radius (from _spice())
        :sets angvf: Stack of non-gravitational acceleration vectors in the frozen frame
        """
        print("Calculating acceleration...")
        avf=ai(self.ets,self.svf) #Total acceleration vector in frozen frame
        agf=aTwoBody(self.svf, gm=self.mars_gm) + aJ2(self.svf, j2=self.mars_j2, gm=self.mars_gm, re=self.mars_re)
        self.angvf=avf-agf #Stack of non-gravitational acceleration vectors in frozen frame
    def _lvlh(self):
        """
        Determine a bunch of unit vectors

        :uses svf: spacecraft state in frozen frame
        :uses vrelf: spacecraft atmosphere-relative velocity in frozen frame
        :sets rhatf: planetocentric local vertical in frozen frame
        :sets vhatf: Direction of travel relative to atmosphere in frozen frame
        :sets ehatf: East vector in frozen frame -- in horizon plane, towards direction of motion of rotating frame at this point
        :sets nhatf: North vector in frozen frame -- in horizon plane, perpendicular to east (not parallel to polar axis)
        :sets hhatf: Crossrange direction in frozen frame, perpendicular to direction of travel in horizon plane
        :sets qhatf: Downrange direction in frozen frame, direction of travel projected onto horizontal plane
        :sets khatf: normal direction in frozen frame, perpendicular to direction of travel in vertical plane
        :sets LH: component of non-grav in h direction "sideways lift"
        :sets LK: component of non-grav in k direction "vertical lift"
        :sets LD: component of non-grav in v direction "drag" (will be negative for normal drag)
        :sets lhatf: direction of lift vector in frozen frame, component of non-grav perpendicular to travel direction
        """
        print("Calculating local basis vectors...")
        zhat = np.array([[0], [0], [1]], dtype=np.float64)
        self.rhatf = vnormalize(rv(self.svf))
        self.vhatf = vnormalize(self.vrelf)
        self.ehatf = vncross(     zhat , self.rhatf)
        self.nhatf = vncross(self.rhatf, self.ehatf)
        self.hhatf = vncross(self.rhatf, self.vhatf)
        self.qhatf = vncross(self.hhatf, self.rhatf)
        self.khatf = vncross(self.vhatf, self.hhatf)
        self.LH = vdot(self.angvf, self.hhatf)
        self.LK = vdot(self.angvf, self.khatf)
        self.LD = vdot(self.angvf, self.vhatf)
        self.lhatf = vnormalize(self.LH * self.hhatf + self.LK * self.khatf)
    def _att(self):
        raise NotImplementedError
    def _cam(self):
        """
        Calculate the camera position

        :sets location: Location of camera in rotating frame
        :sets look_at: look_at point in rotating frame
        :sets sky: Camera sky vector in rotating frame

        Note that neither location nor look_at are relative to the spacecraft -- if
        you want to look at the spacecraft, put the spacecraft position in look_at.

        """
        raise NotImplementedError
    def _body_rate(self):
        """
        Calculate body rotation rate by finite difference

        :uses Mpfb: Matrix transforming position to frozen from body (set by _att())
        :sets omega: rotation rate vector in body frame
        """
        print("Calculating body rates")
        self.omega = body_rate(self.Mpfb, self.dts)
        self.omegadot=self.omega*0
        self.omegadot[:,1:-1]=(self.omega[:,2:]-self.omega[:,:-2])/self.dts[1:-1]
        self.omegadot[:,0]=self.omegadot[:,1]
        self.omegadot[:,-1]=self.omegadot[:,-2]
    def _drop(self, dropobj):
        """

        :param et0: Initial step -- at this point, the state of the dropped object will still exactly match
          that of the main spacecraft
        :param svf: State vector of spacecraft in frozen frame
        :param Mpfb: Transformation to frozen from body
        :param omega: body rate vector
        :param r0vb: position of object in body frame at time of drop
        :param v0vb: Velocity of object in body frame at time of drop (ejection velocity)
        :param a: aerodynamic area
        :param m: mass
        :param Cd: drag coefficient
        :return: A complete state vector series for the object in the frozen frame.
          Position before the drop will be offset from the spacecraft by r_b
          transformed to the frozen frame.
        """
        result=self.svf*0
        #Figure stuff prior to drop. We are specifically including
        #step0, as this calculates our initial condition

        # Prior to drop, position of object is just position of spacecraft
        # plus trnasformed initial position.
        w=np.where(self.ets <= dropobj.et0)[0]
        result[:3,w]=self.svf[:3,w]+Mtrans(self.Mpfb[w,:,:], dropobj.r0vb)
        #Velocity of object in body frame is cross product of body rate and
        #position in body frame. Doesn't really mean anything before drop,
        #but the velocity at step0 will be our initial condition
        result[3:,w]=self.svf[3:,w]+Mtrans(self.Mpfb[w,:,:], vcross(self.omega[:,w], dropobj.r0vb))
        #i_step0 - time step where ejection occurs. Exactly at i_step0, the ejection is a step
        #change in object speed. The position at i_step0 is exactly what it would have
        #been if the ejection hadn't taken place, but the velocity has the ejection speed added.
        i_step0=np.max(w)

        #transform the ejection speed from body and add it to the state at ejection. We already
        #have the object speed due to spacecraft spin from the above.
        result[3:, None,i_step0]+=Mtrans(self.Mpfb[i_step0, :, :], dropobj.v0vb)
        #integrate to the end of the animation
        for i in range(i_step0, len(self.ets) - 1):
            #Propagate from current step i to next step i+1 with Euler integrator
            #This is good enough for visual purposes.
            dt=self.ets[i+1]-self.ets[i]
            smvf=result[:,None,i] #State at current time. m is for "minus" because variables
                                  #*before* an update are often marked \vec{s}^-_f
            vmrelf=vv(smvf)-self._wind(smvf) #Atmosphere-relative velocity
            (rhom,_,_,_)=mars_atm(vlength(rv(smvf)))
            Fd= -rhom * vlength(vmrelf) ** 2 * dropobj.cd * dropobj.a * vnormalize(vmrelf) / 2 #Drag force
            advf= Fd / dropobj.m #Drag acceleration
            agvf= aTwoBody(smvf, gm=self.mars_gm) + aJ2(smvf, j2=self.mars_j2, gm=self.mars_gm, re=self.mars_re)
            avf=agvf+advf #Total acceleration
            dsvf=smvf*0   #State derivative
            dsvf[:3,:]=smvf[3:,:]  #Derivative of position is velocity, copied from velocity part of state
            dsvf[3:,:]=avf         #Derivative of velocity is acceleration, calculated from physics above
            spvf=smvf+dt*dsvf      #Update the state. p is for "plus" for *after* update, would be notated as \vec{s}^+_f
            result[:,None,i+1]=spvf #Store updated stat in result array
        return result
    def _drops(self):
        if self.dropobjs is not None:
            self.droptraj = {}
            droptrajResult = namedtuple("droptrajResult", "drop svf svr")
            for name, dropobj in self.dropobjs.items():
                print(f"Propagating dropped object: {name}")
                bsvf = self._drop(dropobj=dropobj)
                bsvr = Mtrans(self.Msrf, bsvf)
                self.droptraj[name] = droptrajResult(drop=dropobj, svf=bsvf, svr=bsvr)
    def _rel(self):
        self.svr=Mtrans(self.Msrf,self.svf)

        self.rhatr,self.ehatr,self.nhatr           =Mtrans(Mr(self.Msrf),self.rhatf,self.ehatf,self.nhatf)
        self.vhatr,self.hhatr,self.qhatr,self.khatr=Mtrans(Mr(self.Msrf),self.vhatf,self.hhatf,self.qhatf,self.khatf)
        self.lhatr                                 =Mtrans(Mr(self.Msrf),self.lhatf)
        (self.lon, self.lat, self.r) = xyz2llr(rv(self.svr))
    def _tabulate(self):
        self.t_svr=Tableterp(self.ets,self.svr)
        self.t_angvr=Tableterp(self.ets,Mtrans(Mr(self.Msrf),self.angvf))
        self.t_sunr = Tableterp(self.ets, self.sunr)
        self.t_earthr = Tableterp(self.ets, self.earthr)
        self.t_earthlt = Tableterp(self.ets, self.earthlt)
        self.t_Mprb=Tableterp(self.ets,Mr(self.Msrf) @ self.Mpfb)
        self.t_csound=Tableterp(self.ets,self.csound)
        self.t_rho=Tableterp(self.ets,self.rho)
        self.t_P=Tableterp(self.ets,self.P)
        self.t_T=Tableterp(self.ets,self.T)
        self.t_scorch=Tableterp(self.ets,self.scorch)
        self.t_omega=Tableterp(self.ets,self.omega)
        self.t_omegadot=Tableterp(self.ets,smooth(self.omegadot,-50,50))
        self.t_droptraj={}
        for name, drop in self.droptraj.items():
            self.t_droptraj[name]=(Tableterp(self.ets,drop.svr),drop.drop.pov_suffix)

    @staticmethod
    def print_vector(name, *, v=None, formula=None, comment=None, file):
        print(f"#declare {name}=", file=file, end='')
        if v is not None:
            print(f"<{v[0, 0]},{v[1, 0]},{v[2, 0]}>;", file=file, end='')
        else:
            print(f"{formula};", file=file, end='')
        if comment is not None:
            print(f"  // {comment}", file=file, end='')
        print(file=file)
    @staticmethod
    def print_scalar(name, *, v=None, formula=None, comment=None, file):
        print(f"#declare {name}=", file=file, end='')
        if v is not None:
            print(f"{v};", file=file, end='')
        else:
            print(f"{formula};", file=file, end='')
        if comment is not None:
            print(f"  // {comment}", file=file, end='')
        print(file=file)

    def _print_pov(self, et_step, i_step, tiles=None, file=None):
        print(f"// generated in python/edl/edl.py",file=file)
        print(f"// All vectors are in IAU_MARS frame. This is centered on and rotates with Mars.",file=file)
        print(f'// All units are consistent with {"kilo" if self.km<500 else ""}meters and seconds',file=file)
        print(f"#declare ET  = {et_step}; //step {i_step:5d}, {spiceypy.etcal(et_step)}",file=file)
        R1=self.svr[:3,-1,None]
        self.print_vector("Rland",v=R1,comment="Landing site position vector",file=file)
        Ang=self.t_angvr(et_step)
        self.print_vector("Ang",v=Ang,comment="Non-gravitational acceleration vector",file=file)
        Sun=self.t_sunr(et_step)
        self.print_vector("Sun",v=Sun,comment="Position of Sun",file=file)
        self.print_vector("Earth",v=self.t_earthr(et_step),comment="Position of Earth (with XCN+S correction, where to aim photons to get from Mars to Earth, leaving Mars now)",file=file)
        print(f"#declare LT={self.t_earthlt(et_step)}; //Travel time of photons from Mars to Earth, leaving Mars now",file=file)
        print( "/*#declare LanderOrient=transform{matrix<",file=file)
        Mprb=self.t_Mprb(et_step)
        print(f"  {Mprb[0,0]},{Mprb[1,0]},{Mprb[2,0]},",file=file)
        print(f"  {Mprb[0,1]},{Mprb[1,1]},{Mprb[2,1]},",file=file)
        print(f"  {Mprb[0,2]},{Mprb[1,2]},{Mprb[2,2]},",file=file)
        print( "  0,0,0>};*/",file=file)
        print( "//Matrix which transforms from DIMU-A body frame to IAU_MARS rotating",file=file)
        print( "//frame at the current instant. Note that POV-Ray wants",file=file)
        print( "//the transpose of this matrix for ",file=file)
        print( "#declare LanderOrientM=array[3][3] {",file=file)
        print(f"  {{{Mprb[0,0]},{Mprb[0,1]},{Mprb[0,2]}}},",file=file)
        print(f"  {{{Mprb[1,0]},{Mprb[1,1]},{Mprb[1,2]}}},",file=file)
        print(f"  {{{Mprb[2,0]},{Mprb[2,1]},{Mprb[2,2]}}},",file=file)
        print( "};",file=file)
        self.print_vector("Omega",v=self.t_omega(et_step),comment="rotation rate in body frame",file=file)
        self.print_vector("Omegadot",v=self.t_omega(et_step),comment="rotation acceleration in body frame",file=file)
        self.print_scalar("Csound",v=self.t_csound(et_step),comment="Free-stream speed of sound, m/s",file=file)
        self.print_scalar("Rho",v=self.t_rho(et_step),comment="Free-stream Density in kg/m**3",file=file)
        self.print_scalar("P",v=self.t_P(et_step),comment="Free-stream Static pressure in Pa",file=file)
        self.print_scalar("T",v=self.t_T(et_step),comment="Free-stream temperature in K",file=file)
        self.print_scalar("Scorch_n",v=self.t_scorch(et_step),comment="integrated dynamic heating, in energy per unit area",file=file)
        self.print_scalar("Scorch_max",v=np.max(self.scorch),comment="Maximum integrated dynamic heating",file=file)
        self.print_scalar("dynHeat_max",v=np.max(self.dynHeat),comment="Maximum dynamic heating rate",file=file)
        self.print_vector("Rds",v=self.t_svr(et_step)[:3,:],comment="Position vector of DIMU-A",file=file)
        self.print_vector("Vds",v=self.t_svr(et_step)[3:,:],comment="Velocity vector of DIMU-A",file=file)
        if self.droptraj is not None:
            for name,(t_drop,pov_suffix) in self.t_droptraj.items():
                Rhs=t_drop(et_step)[:3,:]
                Vhs=t_drop(et_step)[3:,:]
                self.print_vector(f"R{pov_suffix}", v=Rhs, comment=f"Position vector of {name}", file=file)
                self.print_vector(f"V{pov_suffix}", v=Vhs, comment=f"Velocity vector of {name}", file=file)
        if tiles is not None:
            location = self.t_location(et_step)
            look_at = self.t_look_at(et_step)
            sky = self.t_sky(et_step)
            tile.frame(i_step, Location=location, Look_at=look_at, latc=tiles.latc, lonc=tiles.lonc, tiles=tiles.tiles, ouf=file)

    @staticmethod
    def format_time(et, et0):
        """

        :param et: time to print
        :param t0: reference time
        :return: string in form [+-]MM:SS.###
        """
        if et > et0:
            sign = "+"
            dt = et - et0
        else:
            sign = "-"
            dt = et0 - et
        dt_m = int(dt) // 60
        dt_s = dt % 60
        return f"{sign}{dt_m:02d}:{dt_s:06.3f}"
    def write_events(self):
        print("Writing events...")
        with open(f"events_{self.name}.inc", "wt") as ouf:
            print("// generated in edl/edl.py", file=ouf)
            print("// Event list and constants", file=ouf)
            print("// DIMU-A offset -- offset between DIMU-A, the actual object tracked in the kernel, and\n"
                  "// the origin of the rover model frame in body coordinates",file=ouf)
            self.print_vector("DIMU_A_ofs",v=self.DIMU_A_ofs,file=ouf)
            self.print_scalar("Cheat",v=self.Cheat,comment="Heating indicator coefficient -- Constant for unit conversion to W/cm**2",file=ouf)
            self.print_scalar("mheat",v=self.mheat,comment="Heating indicator coefficient -- Exponent for rho. mheat=1 would mean heating is proportional to rho",file=ouf)
            self.print_scalar("nheat",v=self.nheat,comment="Heating indicator coefficient -- Exponent for velocity. nheat=2 would make heating just like dynamic pressure",file=ouf)
            print(f"#declare Events=array[{len(self.events)}];",file=ouf)
            print(f"#declare EventNames=array[{len(self.events)}];",file=ouf)
            for i_event, (name, et) in enumerate(self.events.items()):
                print(f"\n//Event {name:18s}",file=ouf)
                print(f"//SCET ET  {spiceypy.etcal(et)}",file=ouf)
                print(f"//SCET UTC {spiceypy.timout(et,'YYYY MON DD HR:MN:SC.### UTC ::UTC')}",file=ouf)
                """
                print(f"//ERT  ET  {spiceypy.etcal(et+self.t_earthlt(et))}",file=ouf)
                print(f"//ERT  UTC {spiceypy.timout(et+self.t_earthlt(et),'YYYY MON DD HR:MN:SC.### UTC ::UTC')}",file=ouf)
                """
                print(f"//EI{self.format_time(et,self.events['ei'])} L{self.format_time(et,self.events['land'])} ", file=ouf)
                print(f"#declare Event_{name:18s}={i_event:3d};", file=ouf)
                print(f"#declare ET_{name:18s}   ={et:16.6f};", file=ouf)
                print(f'#declare Events[{i_event}]={et:16.6f};#declare EventNames[{i_event}]="{name}";', file=ouf)
    def write_frames(self, *, et0:float,et1:float,fps:float,step0:int=0, step1:int=None, do_tiles:bool=False):
        print("Interpolating trajectory...")
        self._tabulate()
        print("Writing frames...")
        if do_tiles:
            tiles = tile.read_height_tile_csv()
        else:
            tiles=None
        if step1 is None:
            step1=int((et1-et0)*fps)
        n_steps=step1-step0+1
        for i_step in range(step0,step1):
            et_step=et0+i_step/fps
            with open(f"inc_{self.name}/step_{i_step:05d}.inc", "wt") as file:
                self._print_pov(et_step, i_step, tiles=tiles,file=file)
            print('.',end='')
            if i_step%100==0:
                print(f"{i_step}/{n_steps}")
    def plot_events(self,y0,y1):
        for k,et in self.events.items():
            if et>=(self.ets[0]-20) and et<(self.ets[-1]+20):
                plt.plot([et,et],[y0,y1],'k-')
                plt.text(et,0,k,rotation=90)
        plt.xlabel('Spacecraft Event Time/(ET seconds)')
    def plot_rrel(self):
        plt.figure('rrel')
        plt.plot(self.ets,self.svr[0,:],'r',label='relative position x component')
        plt.plot(self.ets,self.svr[1,:],'g',label='relative position y component')
        plt.plot(self.ets,self.svr[2,:],'b',label='relative position y component')
        plt.xlabel('Spacecraft Event Time/(ET seconds)')
        plt.ylabel('position/km')
        plt.legend()
        plt.pause(0.001)
    def plot_lon_lat(self):
        plt.figure('lon,lat')
        plt.plot(np.degrees(self.lon),np.degrees(self.lat))
        plt.xlabel('east longitude/deg')
        plt.ylabel('latitude/deg')
    def plot_alt(self):
        plt.figure('alt above ei')
        r=vlength(rv(self.svf))
        r_ei=3522200 #m from center of Mars, official EI altitude
        r_land=r[-1] #m from center of Mars, last data point
        r0=r_land
        max=500
        plt.plot(self.ets,r-r_ei,'+-',label=f"{self.name} altitude above EI/m")
        plt.plot(self.ets,r-r_land,'+-',label=f"{self.name} altitude above landing site/m")
        plt.plot(self.ets,vdot(self.rhatf,vv(self.svf))*100,label=f"{self.name} hdot/(cm/s)")
        self.plot_events(-max,max)
        plt.ylabel('altitude/m  hdot cm/s')
        plt.xlabel('Spacecraft Event Time/(ET seconds)')
        plt.legend()
        plt.pause(0.001)
    def plot_att(self):
        plt.figure("body rates")
        plt.plot(self.ets,self.omega[0,:],'r',label='body x rate')
        plt.plot(self.ets,self.omega[1,:],'g',label='body y rate')
        plt.plot(self.ets,self.omega[2,:],'b',label='body z rate')
        self.plot_events(-0.5,0.5)
        plt.ylabel('Body rate/(rad/s)')
        plt.xlabel('Spacecraft Event Time/(ET seconds)')
        plt.legend()

        plt.figure("body accs")
        plt.plot(self.ets,       self.omegadot[0,:]        ,color='#808080')
        plt.plot(self.ets,       self.omegadot[1,:]        ,color='#808080')
        plt.plot(self.ets,       self.omegadot[2,:]        ,color='#808080')
        plt.plot(self.ets,smooth(self.omegadot[0,:],-50,50),'r',label='body x acc')
        plt.plot(self.ets,smooth(self.omegadot[1,:],-50,50),'g',label='body y acc')
        plt.plot(self.ets,smooth(self.omegadot[2,:],-50,50),'b',label='body z acc')
        self.plot_events(-0.5,0.5)
        plt.ylabel('Body acc/(rad/s**2)')
        plt.xlabel('Spacecraft Event Time/(ET seconds)')
        plt.legend()

        plt.pause(0.001)
    def plot_acc(self):
        plt.figure('ang magnitude')
        smoothing=50
        max=np.max(vlength(self.angvf))
        plt.plot(self.ets,       vlength(self.angvf),                      color="#808080")
        plt.plot(self.ets,smooth(vlength(self.angvf),-smoothing,smoothing),'-')
        self.plot_events(0,max)
        plt.xlabel('Spacecraft Event Time/(ET seconds)')
        plt.legend()

        plt.figure('arngr, aengr, anngr')
        plt.plot(self.ets,       vdot(self.rhatf,self.angvf)      ,color="#808080")
        plt.plot(self.ets,       vdot(self.ehatf,self.angvf)      ,color="#808080")
        plt.plot(self.ets,       vdot(self.nhatf,self.angvf)      ,color="#808080")
        plt.plot(self.ets,smooth(vdot(self.rhatf,self.angvf),-smoothing,smoothing),'r',label='non-gravitational acceleration radial component')
        plt.plot(self.ets,smooth(vdot(self.ehatf,self.angvf),-smoothing,smoothing),'g',label='non-gravitational acceleration east component')
        plt.plot(self.ets,smooth(vdot(self.nhatf,self.angvf),-smoothing,smoothing),'b',label='non-gravitational acceleration north component')
        self.plot_events(-max,max)
        plt.ylabel('acceleration/(m/s**2)')
        plt.xlabel('Spacecraft Event Time/(ET seconds)')
        plt.legend()

        plt.figure('angrr, anghr, angqr')
        plt.plot(self.ets,       vdot(self.rhatf,self.angvf)      ,color="#808080")
        plt.plot(self.ets,       vdot(self.hhatf,self.angvf)      ,color="#808080")
        plt.plot(self.ets,       vdot(self.qhatf,self.angvf)      ,color="#808080")
        plt.plot(self.ets,smooth(vdot(self.rhatf,self.angvf),-smoothing,smoothing),'r',label='non-gravitational acceleration radial component')
        plt.plot(self.ets,smooth(vdot(self.hhatf,self.angvf),-smoothing,smoothing),'g',label='non-gravitational acceleration crossrange component')
        plt.plot(self.ets,smooth(vdot(self.qhatf,self.angvf),-smoothing,smoothing),'b',label='non-gravitational acceleration downrange component')
        self.plot_events(-max,max)
        plt.ylabel('acceleration/(m/s**2)')
        plt.xlabel('Spacecraft Event Time/(ET seconds)')
        plt.legend()

        self.LH = vdot(self.angvf, self.hhatf)
        self.LK = vdot(self.angvf, self.khatf)
        self.LD = vdot(self.angvf, self.vhatf)
        plt.figure('LH, LK, LD')
        plt.plot(self.ets,       self.LH      ,color="#808080")
        plt.plot(self.ets,       self.LK      ,color="#808080")
        plt.plot(self.ets,       self.LD      ,color="#808080")
        plt.plot(self.ets,smooth(self.LH,-smoothing,smoothing),'r',label='horizontal lift component')
        plt.plot(self.ets,smooth(self.LK,-smoothing,smoothing),'g',label='vertical lift component')
        plt.plot(self.ets,smooth(self.LD,-smoothing,smoothing),'b',label='drag component')
        self.plot_events(-max,max)
        plt.ylabel('acceleration/(m/s**2)')
        plt.xlabel('Spacecraft Event Time/(ET seconds)')
        plt.legend()
        plt.pause(0.001)
    def plot_vel(self):
        plt.figure('vrr, ver, vnr')
        plt.plot(self.ets,vdot(vv(self.vrelf),self.rhatf),'r',label='Radial relative velocity')
        plt.plot(self.ets,vdot(vv(self.vrelf),self.nhatf),'g',label='North relative velocity')
        plt.plot(self.ets,vdot(vv(self.vrelf),self.ehatf),'b',label='East relative velocity')
        plt.xlabel('Spacecraft Event Time/(ET seconds)')
        plt.ylabel('speed/(m/s)')
        plt.legend()
        plt.pause(0.001)
    def plot_qbar(self):
        plt.figure('scorch')
        plt.plot(self.ets,self.dynHeat/np.max(self.dynHeat),label='dynHeat/max(dynheat)')
        plt.plot(self.ets,self.scorch /np.max(self.scorch ),label='scorch/max(scorch)')
        plt.plot(self.ets,self.qbar   /np.max(self.qbar   ),label='dynP/max(dynP)')
        plt.xlabel('Spacecraft Event Time/(ET seconds)')
        plt.legend()
        plt.pause(0.001)
    def plot(self):
        self.plot_rrel()
        self.plot_lon_lat()
        self.plot_alt()
        self.plot_att()
        self.plot_acc()
        self.plot_vel()
        self.plot_qbar()

class CKTrajectory(Trajectory):
    def __init__(self, *,loader:SPKLoader,extra_kernels,ckobj,dropobjs=None):
        self.ckobj=ckobj
        super().__init__(loader=loader,extra_kernels=extra_kernels,dropobjs=dropobjs)
    def _att(self):
        print("Calculating attitude...")
        self.roll= self.i_steps * 0.0
        self.pitch= self.i_steps * 0.0
        Mpib = np.zeros((len(self.ets), 3, 3))
        for i_et,et in enumerate(self.ets):
            Mpib[i_et,:,:]=spiceypy.pxform(self.ckobj,"J2000",et)
        self.Mpfb=Mr(self.Msfi) @ Mpib

class DAFSPKLoader(SPKLoader):
    def __init__(self,spk,spice_sc,reject=[]):
        super().__init__()
        self.spk=spk
        self.spice_sc=int(spice_sc)
        self.reject=reject
    def time(self):
        result=None
        with double_array_file(self.spk) as daf:
            for i,sr in enumerate(daf):
                print(f"Summary record {i}")
                #print(str(sr))
                for j,sum in enumerate(sr):
                    #print(f"Summary {j}")
                    #print(str(sum))
                    #print(str(sum.segment()))
                    if sum.target!=self.spice_sc or j in self.reject:
                        continue
                    this_result=[]

                    for k,line in enumerate(sum.segment()):
                        this_result.append(line.et)
                    this_result=np.array(this_result)
                    if result is None:
                        result=this_result
                    else:
                        result=result[np.where(np.logical_or(result<this_result[0],result>this_result[-1]))]
                        result=np.hstack((result,this_result))
        return result
    def spice(self):
        result=None
        ets=None
        with double_array_file(self.spk) as daf:
            for i,sr in enumerate(daf):
                #print(f"Summary record {i}")
                #print(str(sr))
                for j,sum in enumerate(sr):
                    #print(f"Summary j")
                    #print(str(sum))
                    #print(str(sum.segment()))
                    if sum.target!=self.spice_sc:
                        continue
                    this_result=[]
                    this_ets=[]
                    for k,line in enumerate(sum.segment()):
                        this_ets.append(line.et)
                        this_result.append(np.array((line.x,line.y,line.z,line.dxdt,line.dydt,line.dzdt)))
                    this_ets=np.array(this_ets)
                    this_result=np.array(this_result).transpose()
                    if result is None:
                        result=this_result
                        ets=this_ets
                    else:
                        w=np.where(np.logical_or(ets<this_ets[0],ets>this_ets[-1]))
                        ets=np.hstack((ets[w],this_ets))
                        result=np.hstack((result[:,w[0]],this_result))
        return result




