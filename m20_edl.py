from edl import Trajectory, eventTuple, bodyTuple, llr2xyz, Mr, linterp, vnormalize
import numpy as np

class PredictTrajectory(Trajectory):
    def __init__(self, *,et0, et1, dt,extra_kernels,dropobjs=None):
        super().__init__(et0=et0,et1=et1,dt=dt,extra_kernels=extra_kernels,dropobjs=dropobjs)
    def _att(self):
        """
        Calculate the attitude of the main spacecraft
        :param svf: State vector series
        :return: Tuple of
          * roll in degrees
          * pitch in degrees
          * Matrix transforming body to IAU_MARS rotating frame
          * Position of spacecraft in IAU_MARS rotating frame
        """
        print("Calculating attitude...")
        self.roll= self.i_steps * 0.0
        self.pitch= self.i_steps * 0.0
        #default p_b is body -z (towards heat shield)
        p_b=self.rhatf*0
        p_b[2,:]=-1
        #Default t_b is body +x (towards EBM)
        t_b=self.rhatf*0
        t_b[0,:]=1
        #No defaults for p_r and t_r -- must select a vector every time
        p_f=self.rhatf*0
        t_f=self.rhatf*0

        # Before entry - copy the first frame of entry
        w=np.where(self.i_steps < self.events[0])[0]
        self.roll[w]=np.degrees(np.arctan2(self.LH[self.events[0]],self.LK[self.events[0]]))
        self.pitch[w]=16
        p_f[:,w]=self.vhatf[:,w]
        t_f[:,w]=np.sin(np.radians(self.roll[w]))*self.hhatf[:,w]+np.cos(np.radians(self.roll[w]))*self.khatf[:,w]
        #Entry to SUFR - Point reference is velocity vector,
        #                point body determined by pitch,
        #                toward vector is lift vector
        w=np.where(np.logical_and(self.i_steps >= self.events[0], self.i_steps < self.events[1]))[0]
        self.roll[w]=np.degrees(np.arctan2(self.LH[w],self.LK[w]))
        self.pitch[w]=16
        p_f[:,w]=self.vhatf[:,w]
        t_f[:,w]=np.sin(np.radians(self.roll[w]))*self.hhatf[:,w]+np.cos(np.radians(self.roll[w]))*self.khatf[:,w]
        #During SUFR - Point reference changes smoothly from velocity to acceleration
        #              pitch changes from 16 to 0
        #              roll changes from lift at beginning of SUFR to 180
        w=np.where(np.logical_and(self.i_steps >= self.events[1], self.i_steps < self.events[3]))[0]
        t=linterp(self.events[1], 0, self.events[3], 1, self.i_steps[w])
        p_f[:,w]=(1-t)*self.vhatf[:,w]
        p_f[:,w]+= t * vnormalize(-self.angvf[:, w])
        p_f[:,w]=vnormalize(p_f[:, w])
        self.roll[w]=linterp(self.events[1], self.roll[self.events[1]-1], self.events[3], 180, self.i_steps[w])
        self.pitch[w]=linterp(self.events[1], 16, self.events[3], 0, self.i_steps[w])
        t_f[:,w]=np.sin(np.radians(self.roll[w]))*self.hhatf[:,w]+np.cos(np.radians(self.roll[w]))*self.khatf[:,w]
        #From end of SUFR to one second (24 frames) before parachute deploy
        #t_r moves from lift frame to hhat/qhat at one second before parachute deploy
        w=np.where(np.logical_and(self.i_steps >= self.events[3], self.i_steps < self.events[4] - 24))[0]
        self.roll[w]=180
        self.pitch[w]=0
        p_f[:,w]=vnormalize(-self.angvf[:, w])
        t=linterp(self.events[3], 0, self.events[4] - 24, 1, self.i_steps[w])
        t_f[:,w]=(1-t)*(np.sin(np.radians(self.roll[w]))*self.hhatf[:,w]+np.cos(np.radians(self.roll[w]))*self.khatf[:,w])
        t_f[:,w]+=t*(np.sin(np.radians(self.roll[w]))*self.hhatf[:,None,self.events[4]-24]
                    +np.cos(np.radians(self.roll[w]))*self.qhatf[:,None,self.events[4]-24])
        t_f[:,w]=vnormalize(t_f[:, w])
        #For the rest of the flight, same reference vectors as end of previous segment
        w=np.where(self.i_steps >= self.events[4] - 24)[0]
        self.roll[w]=180
        self.pitch[w]=0
        p_f[:,w]=vnormalize(-self.angvf[:, w])
        t_f[:,w]=(np.sin(np.radians(self.roll[w]))*self.hhatf[:,None,self.events[4]-24]
                 +np.cos(np.radians(self.roll[w]))*self.qhatf[:,None,self.events[4]-24])
        p_b=vcomp((np.sin(np.radians(self.pitch)),self.pitch*0,-np.cos(np.radians(self.pitch))))
        self.Mpfb=point_toward(p_b=p_b, p_r=p_f, t_b=t_b, t_r=t_f)

class M20Trajectory(Trajectory):
    def __init__(self):
        self.name="m20"
        extra_kernels=(
            # Spacecraft clock kernel, not used until we have a CK, then must use the one that matches the CK
            # spiceypy.furnsh("/mnt/big/home/chrisj/workspace/Data/spice/M20/sclk/m2020.tsc")
            # Spacecraft cruise/EDL/surface trajectory
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/spk/predict/m2020_trajCEDLS-6DOF_ops_od020v1_AL23.bsp",
        )
        # Trajectory values. Will probably need new values if we get a different trajectory
        self.et_land = 666953092.07640004158020  # Time of first point in post-landing segment.
        et0 = self.et_land - 420  # Seven minutes of excitement! Entry interface is about 7.25s after this time.
        et1 = self.et_land + 20  # Make sure we are steady on the ground
        self.spice_sc="-168"
        self.spice_bodyframe=None
        events = eventTuple(entry0=1490,
                            # 0 - Before this frame, not enough lift to properly measure roll, so use roll *at* this frame for *before* this frame
                            sufr0=5564,  # 1 - First frame of Straighten up and Fly right
                            ebm=(5570, 5618, 5665, 5714, 5762, 5810),
                            # 2 - First points of disturbance for each EBM jettison
                            sufr1=5928,  # 3 - Last frame of Straighten up and Fly right
                            chute0=5976,  # 4 - First frame of parachute deploy (mortar fire)
                            chute1=6010,  # 5 - Last frame of parachute deploy (first peak decelleration)
                            heatshield=6463,  # 6 - Heat shield jettison
                            backshell=8544,  # 7 - Backshell jettison
                            skycrane=9630,  # 8 - rover skycrane
                            touchdown=10080  # 8 - Touchdown
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
        r0ebmvb=[llr2xyz(latd=-8.5,lond=lond,r=2.06)+np.array([[0],[0],[0.35]]) for lond in londs]
        v0ebmvb=[llr2xyz(latd=-8.5,lond=lond,r=1) for lond in londs]
        dropobjs = {
            "Heatshield": bodyTuple(pov_suffix="hs", frame0=events.heatshield, a=3.14 * 2.25 ** 2, m=mass["HeatShield"],
                                    v0vb=np.array([[0], [0], [-2.7]]), cd=1.5),
            "Backshell": bodyTuple(pov_suffix="bs", frame0=events.backshell, a=3.14 * 10 ** 2,
                                   m=mass["Backshell"]["Backshell"] + mass["Backshell"]["Parachute"], cd=2.5)}
        for i in range(len(londs)):
            dropobjs[f"EBM{i}"]=bodyTuple(pov_suffix=f"ebm{i}", frame0=events.ebm[i], a=0.0193,
                                   m=mass["EntryBalanceMass"][f"Mass{i}"], cd=1,r0vb=r0ebmvb[i],v0vb=v0ebmvb[i])
        super().__init__(et0=et0,et1=et1,dt=1/24,extra_kernels=extra_kernels,events=events,dropobjs=dropobjs)
    def _cam(self,i_frame):
        Mprf = Mr(self.Msrf[i_frame, :, :])
        Rhat = Mprf @ self.rhatf[:, None, i_frame]
        Vhat = Mprf @ self.vhatf[:, None, i_frame]
        Ehat = Mprf @ self.ehatf[:, None, i_frame]
        Nhat = Mprf @ self.nhatf[:, None, i_frame]
        Hhat = Mprf @ self.hhatf[:, None, i_frame]
        Qhat = Mprf @ self.qhatf[:, None, i_frame]
        Khat = Mprf @ self.khatf[:, None, i_frame]
        R    = self.svr[:3, None, i_frame]
        R1   = self.svr[:3, None, -1]
        V    = self.svr[3:, None, i_frame]
        Ang  = self.angvr[:, None, i_frame]
        Sun  = self.sunr[:, None, i_frame]

        if i_frame < self.events[4] - 100:
            angle = 0
            dist = 20
            height = 12
            qbasis = Qhat
            hbasis = Hhat
            rbasis = Rhat
        elif i_frame < self.events[4] - 50:
            angle = linterp(self.events[4] - 100, 0, self.events[4] - 50, np.pi / 2, i_frame)
            dist = linterp(self.events[4] - 100, 20, self.events[4] - 50, 10, i_frame)
            height = linterp(self.events[4] - 100, 12, self.events[4] - 50, 0, i_frame)
            qbasis = Qhat
            hbasis = Hhat
            rbasis = Rhat
        elif i_frame < (self.events[6] + 100):
            angle = np.pi / 2
            dist = 10
            height = 0
            qbasis = Qhat
            hbasis = Hhat
            rbasis = Rhat
        elif i_frame < (self.events[6] + 150):
            angle = np.pi / 2
            dist = 10
            height = 0
            qbasis = vnormalize(linterp(self.events[6] + 100, Qhat, self.events[6] + 150, Ehat, i_frame))
            hbasis = vnormalize(linterp(self.events[6] + 100, Hhat, self.events[6] + 150, Nhat, i_frame))
            rbasis = Rhat
        else:
            angle = np.pi / 2
            dist = 10
            height = 0
            qbasis = Ehat
            hbasis = Nhat
            rbasis = Rhat
        Location = R - dist * np.cos(angle) * qbasis - dist * np.sin(angle) * hbasis + height * rbasis
        Look_at = R
        # Sky=self.Mprb[i_frame,:,np.newaxis,0]
        Sky = Rhat


if __name__=="__main__":
    M20=M20Trajectory()
    M20.plot_att()
    M20.write_frames(do_tiles=True)
