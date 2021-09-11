import spiceypy

from edl import *
import numpy as np
import scipy.interpolate
from sound import sound

class MSLTrajectory(CKTrajectory):
    def __init__(self,name="msl",bodyid="MSL"):
        self.name=name
        extra_kernels=(
            # Spacecraft clock kernel, not used until we have a CK, then must use the one that matches the CK
            "/mnt/big/home/chrisj/workspace/Data/spice/MSL/sclk/MSL_76_SCLKSCET.00017.tsc",
            #Need additional frames to use the spk
            "/mnt/big/home/chrisj/workspace/Data/spice/MSL/fk/msl_v08.tf",
            # Spacecraft EDL/surface trajectory
            "/mnt/big/home/chrisj/workspace/Data/spice/MSL/spk/msl_edl_v01.bsp",
            # Landing site coordinates
            "/mnt/big/home/chrisj/workspace/Data/spice/MSL/spk/msl_ls_ops120808_iau2000_v1.bsp",
            # Spacecraft orientation -- needed since some
            "/mnt/big/home/chrisj/workspace/Data/spice/MSL/ck/msl_edl_v01.bc"
            )
        et0 = 397501912.0  # first whole second before EI
        print(spiceypy.etcal(et0))
        et1 = 397502345.0  # last whole second before end of kernel
        print(spiceypy.etcal(et1))
        self.spice_sc=bodyid
        self.spice_bodyframe="MSL_ROVER"
        events = eventTuple(entry0=None,
                            sufr0=5783,  # 1 - First frame of Straighten up and Fly right
                            ebm=(5876, 5876, 5924, 5924, 5972, 5972), #Released in pairs at two second intervals. Two seps seen in smoothed ang data presume first pair is two seconds before
                            # 2 - First points of disturbance for each EBM jettison
                            sufr1=6134,  # 3 - Last frame of Straighten up and Fly right
                            chute0=6241,  # 4 - First frame of parachute deploy (mortar fire)
                            chute1=6285,  # 5 - Last frame of parachute deploy (first peak decelleration)
                            heatshield=6721,  # 6 - Heat shield jettison
                            backshell=9042,  # 7 - Backshell jettison
                            skycrane=9930,  # 8 - rover skycrane
                            touchdown=10363  # 8 - Touchdown
                            )
        mass = {
            "Rover": 899,
            "DescentStage": {
                "Prop": 387,
                "Dry": 983
            },
            "Backshell": {
                "Parachute": 54,
                "Backshell": 295
            },
            "EntryBalanceMass": {
                "Mass0": 25,
                "Mass1": 25,
                "Mass2": 25,
                "Mass3": 25,
                "Mass4": 25,
                "Mass5": 25
            },
            "HeatShield": 385,
            "CruiseBalanceMass": {
                "Mass0": 75,
                "Mass1": 75
            },
            "CruiseStage": {
                "Prop": 79,
                "Dry": 460
            }
        }
        londs=[30,-30,20,-20,10,-10]
        r0ebmvb=[llr2xyz(latd=8.5,lond=lond,r=2.06)-np.array([[0],[0],[0.35]]) for lond in londs]
        v0ebmvb=[llr2xyz(latd=8.5,lond=lond,r=1) for lond in londs]
        dropobjs = {
            "Heatshield": bodyTuple(pov_suffix="hs", frame0=events.heatshield, a=3.14 * 2.25 ** 2, m=mass["HeatShield"],
                                    v0vb=np.array([[0], [0], [2.7]]), cd=1.5),
            "Backshell": bodyTuple(pov_suffix="bs", frame0=events.backshell, a=3.14 * 10 ** 2,
                                   m=mass["Backshell"]["Backshell"] + mass["Backshell"]["Parachute"], cd=2.5)}
        for i in range(len(londs)):
            dropobjs[f"EBM{i}"]=bodyTuple(pov_suffix=f"ebm{i}", frame0=events.ebm[i], a=0.0193,
                                   m=mass["EntryBalanceMass"][f"Mass{i}"], cd=1,r0vb=r0ebmvb[i],v0vb=v0ebmvb[i])
        super().__init__(et0=et0,et1=et1,dt=1/24,extra_kernels=extra_kernels,events=events,dropobjs=dropobjs,ckobj="MSL_ROVER")
    def _init_cam(self):
        self.xbasis = self.qhatr * 1  # First horizontal basis vector (reference point for angle=0). Default to qhatr
        self.ybasis = self.hhatr * 1  # Second horizontal basis vector (direction of positive angle). Default to hhatr
        self.zbasis = self.rhatr * 1  # Vertical basis vector (direction of height)
        self.angle = np.zeros(self.i_steps.size)  # Angle from xbasis in degrees, positive towards ybasis
        self.dist = np.zeros(self.i_steps.size) + 20  # Horizontal distance from rv(self.svr)
        self.height = np.zeros(self.i_steps.size)  # Vertical distance above horizontal plane
        self.look_at = rv(self.svr)  # Look-at point, defaults to rv(svr)
    def _keyframe_w(self,frame0,frame1=None):
        if frame1 is None:
            frame1=frame0
            frame0= self.i_steps[self._w[-1]] + 1
        self._w=np.where(np.logical_and(self.i_steps >= frame0, self.i_steps < frame1))[0]
    def _keyframe(self,y,y0,y1=None):
        if y1 is None:
            y1=y0*1
        try:
            if y0.shape[-1]>1:
                y0=y0[...,self._w]
        except:
            pass
        try:
            if y1.shape[-1]>1:
                y1=y1[...,self._w]
        except:
            pass
        y[...,self._w]=linterp(self.i_steps[self._w][0], y0, self.i_steps[self._w][-1] + 1, y1, self.i_steps[self._w])
    def _keyframe_nvector(self,y,y0,y1):
        self._keyframe(y,y0,y1)
        y[...,self._w]=vnormalize(y[...,self._w])
    def _cam(self):
        self._init_cam()
        #Before parachute deploy
        self._keyframe_w(0,self.events.chute0 - 100)
        self._keyframe(self.angle,180)
        self._keyframe(self.dist,20)
        self._keyframe(self.height,12)
        self._keyframe(self.xbasis,self.qhatr)
        self._keyframe(self.ybasis,self.hhatr)
        self._keyframe(self.zbasis,self.rhatr)

        #Pan to capture parachute deploy
        self._keyframe_w(self.events.chute0 - 50)
        self._keyframe(self.angle,180,90)
        self._keyframe(self.dist,20,10)
        self._keyframe(self.height,12,0)
        self._keyframe(self.xbasis,self.qhatr)
        self._keyframe(self.ybasis,self.hhatr)
        self._keyframe(self.zbasis,self.rhatr)

        #Watch parachute deploy
        self._keyframe_w(self.events.heatshield+100)
        self._keyframe(self.angle,90)
        self._keyframe(self.dist,10)
        self._keyframe(self.height,0)
        self._keyframe(self.xbasis,self.qhatr)
        self._keyframe(self.ybasis,self.hhatr)
        self._keyframe(self.zbasis,self.rhatr)

        #Change from velocity basis to east-north basis
        self._keyframe_w(self.events.heatshield+150)
        self._keyframe(self.angle,90)
        self._keyframe(self.dist,10)
        self._keyframe(self.height,0)
        self._keyframe_nvector(self.xbasis,self.qhatr,self.ehatr)
        self._keyframe_nvector(self.ybasis,self.hhatr,self.nhatr)
        self._keyframe(self.zbasis,self.rhatr)

        self._keyframe_w(99999)
        self._keyframe(self.angle,90)
        self._keyframe(self.dist,10)
        self._keyframe(self.height,0)
        self._keyframe(self.xbasis,self.ehatr)
        self._keyframe(self.ybasis,self.nhatr)
        self._keyframe(self.zbasis,self.rhatr)

        self.location = (self.look_at
                         + self.dist * np.cos(np.radians(self.angle)) * self.xbasis
                         + self.dist * np.sin(np.radians(self.angle)) * self.ybasis
                         + self.height                                * self.zbasis)
        # Sky=self.Mprb[i_frame,:,np.newaxis,0]
        self.sky = self.rhatr

if __name__=="__main__":
    MSL=MSLTrajectory(bodyid="-76031")
#    MSL.plot_att()
#    MSL.plot_alt()
#    MSL.plot_acc()
#    plt.legend()
    #MSL.write_frames(do_tiles=False)
    sound(A=lambda x:32767,hz=scipy.interpolate.interp1d(MSL.ets-MSL.ets[0],linterp(0,220,6000,1760,vlength(MSL.vrelf))),t0=0,t1=MSL.ets[-1]-MSL.ets[0],samplerate=48000,oufn="speed.wav")
#    plt.show()
