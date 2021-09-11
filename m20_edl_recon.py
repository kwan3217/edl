from edl import Trajectory, eventTuple, bodyTuple, llr2xyz, Mr, linterp, vnormalize
import numpy as np

class M20Trajectory(Trajectory):
    def __init__(self):
        self.name="m20_recon"
        extra_kernels=(
            # Spacecraft clock kernel, not used until we have a CK, then must use the one that matches the CK
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/sclk/M2020_168_SCLKSCET.00007.tsc"
            # Spacecraft frame kernel, must use the one referenced in the CK comments
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/fk/m2020_v04.tf"
            
            #Spacecraft cruise trajectory
            
            # Spacecraft EDL surface trajectory
            
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/spk/m2020_edl_v01.bsp",
            #Cruise pointing (maybe don't need, but we have it, so we can will use it).
            #Follow the recommended order in the comments
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/ck/m2020_cruise_recon_nospin_v1.bc",
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/ck/m2020_cruise_recon_rawrt_v1.bc",
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/ck/m2020_cruise_recon_raweng_v1.bc",
            #EDL pointing
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/ck/m2020_edl_v01.bc",
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
