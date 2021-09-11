from edl import Trajectory, eventTuple, bodyTuple, llr2xyz
import numpy as np

class M20Trajectory(Trajectory):
    def __init__(self):
        extra_kernels=(
            # Spacecraft clock kernel, not used until we have a CK, then must use the one that matches the CK
            # spiceypy.furnsh("/mnt/big/home/chrisj/workspace/Data/spice/M20/sclk/m2020.tsc")
            # Spacecraft cruise/EDL/surface trajectory
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/spk/m2020_trajCEDLS-6DOF_ops_od020v1_AL23.bsp",
            )
        # Trajectory values. Will probably need new values if we get a different trajectory
        self.et_land = 666953092.07640004158020  # Time of first point in post-landing segment.
        et0 = self.et_land - 420  # Seven minutes of excitement! Entry interface is about 7.25s after this time.
        et1 = self.et_land + 20  # Make sure we are steady on the ground
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

if __name__=="__main__":
    M20=M20Trajectory()
    M20.plot_att()
    M20.write_frames(do_tiles=True)
