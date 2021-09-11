from edl import CKTrajectory, eventTuple, bodyTuple, llr2xyz, Mr, rv, linterp, vnormalize, Mtrans, \
    SPKLoader, UniformSPKLoader, DAFSPKLoader, Tableterp, smooth
import numpy as np
import matplotlib.pyplot as plt

from vector import vlength
from which_kernel import which_kernel, ls_spice
from kwanspice.daf import double_array_file
import matplotlib.pyplot as plt
import spiceypy


class M20ReconTrajectory(CKTrajectory):
    def __init__(self):
        self.name="m20_recon"
        # Spacecraft EDL surface trajectory
        edl_spk="/mnt/big/home/chrisj/workspace/Data/spice/M20/spk/m2020_edl_v01.bsp"
        # Spacecraft clock kernel, not used until we have a CK, then must use the one that matches the CK
        edl_sclk="/mnt/big/home/chrisj/workspace/Data/spice/M20/sclk/M2020_168_SCLKSCET.00007.tsc",
        extra_kernels=(
            edl_sclk,
            # Spacecraft frame kernel, must use the one referenced in the CK comments
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/fk/m2020_v04.tf",
            #Spacecraft structures kernel. This is mostly about where the various mechanisms
            #are. You would think that this would have something to do with skycrane,
            #and the separation between the descent stage and the rover, but that
            #is actually in the edl bsp below.
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/spk/m2020_struct_v01.bsp",

            #Spacecraft cruise trajectory
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/spk/m2020_cruise_od138_v1.bsp",

            # Spacecraft EDL surface trajectory
            edl_spk,

            #Spacecraft landing site
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/spk/m2020_ls_ops210303_iau2000_v1.bsp",

            #Cruise pointing (maybe don't need, but we have it, so we can will use it).
            #Follow the recommended order in the comments
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/ck/m2020_cruise_recon_nospin_v1.bc",
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/ck/m2020_cruise_recon_rawrt_v1.bc",
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/ck/m2020_cruise_recon_raweng_v1.bc",
            #EDL pointing
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/ck/m2020_edl_v01.bc",
        )
        loader=DAFSPKLoader(spk=edl_spk, spice_sc=-168031)
        super().__init__(loader=loader,extra_kernels=extra_kernels,ckobj="M2020_ROVER", dropobjs=None)
        # Trajectory values. Manually figured from plots of acceleration and body rates
        #self.et_ei=666952679.25992 #Actual time of last Spice point before EI (about 4m above, next point is about 3m below)
        self.events={}
        self.events["ei"]         =666952679.262727 #Interpolated EI time
        self.events["peakheat"]   =666952758.91 #peak heating
        self.events["rollrev10"]  =666952761.09  #Start of first roll reversal
        self.events["maxg"]       =666952766.306 #Smoothed maximum g, 104.874 m/s**2 (10.694 earth g)
        self.events["maxq"]       =666952766.82 #maximum dynamic pressure
        self.events["rollrev11"]  =666952772.37  #End of first roll reversal
        self.events["rollrev20"]  =666952781.43  #Start of second roll reversal
        self.events["rollrev21"]  =666952791.31  #End of second roll reversal
        self.events["rollrev30"]  =666952809.43  #Start of third roll reversal
        self.events["rollrev31"]  =666952828.10  #End of third roll reversal
        self.events["sufr0"]      =666952903.05  #Start of straighten up and fly right (SUFR)
        self.events["ebm0"]       =666952908.20  #Start of straighten up and fly right (SUFR)
        dt_ebm0=2.0 #(nominal) Time between EBM jettisons. EBMs were jettisoned in pairs

        et_ebm=[self.events["ebm0"]+dt_ebm0*0,self.events["ebm0"]+dt_ebm0*0,
                self.events["ebm0"]+dt_ebm0*1,self.events["ebm0"]+dt_ebm0*1,
                self.events["ebm0"]+dt_ebm0*2,self.events["ebm0"]+dt_ebm0*2]
        for i,et in enumerate(et_ebm):
            self.events[f"ebm{i}"]=et
        self.events["sufr1"]      =666952917.09  #End of SUFR
        self.events["mortar"]     =666952919.895 #Mortar firing
        self.events["linestretch"]=666952920.936 #Line stretch jolt
        self.events["chuteinf1"]  =666952921.55  #First chute inflation peak, 49.02 m/s**2 (4.999 earth g)
        self.events["chuteinf2"]  =666952922.13  #Second chute inflation peak, 53.67 m/s**2
        self.events["heatshield"] =666952942.527  #Heat shield jettison
        self.events["backshell"]  =666953036.562  #Backshell separation
        self.events["pdi"]        =666953037.598  #Powered descent initiation
        self.events["cvel0"]       =666953068.837  #Constant velocity phase begin, hdot=-32m/s
        self.events["cvel1"]       =666953073.077  #Constant velocity phase end
        self.events["skycrane"]    =666953078.99   #Skycrane start, hdot=-0.75m/s
        self.events["rappel0"]     =666953081.6396100   #Rover rapell begin
        self.events["rappel1"]     =666953087.1396100   #Rover rappel end
        #This seems to be the point where the rover actually makes contact with the ground
        self.events["contact"]     =666953097.33
        #This seems to be the point where the rover *declares* touchdown.
        self.events["land"]       =666953098.82832849025726  # Time of first point in post-landing segment.
        self.events["last"]       =666953098.828360 #Last point in DIMU segment
        self.events["rocker"]          =self.events["rappel0"]+0.7
        #self.events["rappel1"]         =self.events["rappel0"]+5.5
        self.events["bogie"]           =self.events["rappel0"]+6.0

        #sort the event list
        self.events=dict(sorted(self.events.items(),key=lambda x:x[1]))

        et0 = self.events["land"] - 420  # Seven minutes of excitement! Entry interface is about 7.25s after this time.
        et1 = self.events["land"] + 20  # Make sure we are steady on the ground

        #loader=UniformSPKLoader(et0=et0,et1=et1,dt=1/24,spice_sc="-168")
        loader_ets=loader.time()
        self.i_step={}
        for k,v in self.events.items():
            try:
                self.i_step[k]=np.min(np.where(v<loader_ets))
            except ValueError:
                self.i_step[k]=len(loader_ets)-1
            ei_dt=v-self.events['ei']
            ei_dt_m=int(ei_dt)//60
            ei_dt_s=ei_dt%60
            l_dt= self.events['land'] - v
            l_dt_m=int(l_dt)//60
            l_dt_s=l_dt%60
            print(f"Event {k:15}: E+{ei_dt_m}:{ei_dt_s:06.3f} L-{l_dt_m}:{l_dt_s:06.3f} ET{v:.3f} ({spiceypy.etcal(v)}) i_step {self.i_step[k]:6d}")
        self.spice_sc="-168"
        self.spice_bodyframe=None
        self.mass = {
            "Rover": 899,
            "DescentStage": {
                "RCSProp": 12.276,
                "MLEProp": 387-12.276,
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
        r0ebmvb=[llr2xyz(latd=-8.5,lond=lond,r=2.06)+np.array([[0],[0],[-0.65]])+self.DIMU_A_ofs for lond in londs]
        v0ebmvb=[llr2xyz(latd=-8.5,lond=lond,r=1) for lond in londs]
        if False:
            dropobjs={}
        else:
            londs=[30,-30,20,-20,10,-10]
            dropobjs = {
                "Heatshield": bodyTuple(pov_suffix="hs", et0=self.events["heatshield"], a=3.14 * 2.25 ** 2, m=self.mass["HeatShield"],
                                        v0vb=np.array([[0], [0], [2.7]]), cd=1.5),
                "Backshell": bodyTuple(pov_suffix="bs", et0=self.events["backshell"], a=3.14 * 10 ** 2,
                                       m=self.mass["Backshell"]["Backshell"] + self.mass["Backshell"]["Parachute"], cd=2.5)}
            for i in range(len(londs)):
                dropobjs[f"EBM{i}"]=bodyTuple(pov_suffix=f"ebm{i}", et0=et_ebm[i], a=0.0193,
                                              m=self.mass["EntryBalanceMass"][f"Mass{i}"], cd=1, r0vb=r0ebmvb[i], v0vb=v0ebmvb[i])
        self.dropobjs=dropobjs
        self.write_events()
        self._calc()
    def _spice(self):
        super()._spice()
        print("Extracting state vectors for rover from Spice...")
        self.rovsvr=np.zeros((6,self.ets.size))
        for i,et in enumerate(self.ets):
            #try:
                self.rovsvr[:,None,i]=(self.Msrf[i,:,] @ self.Msfi@spiceypy.spkezr("-168",et,"J2000","NONE","499")[0])[:,None]
            #except Exception:
            #    print(f"Couldn't figure spice for timestep {i}, et {et} ({spiceypy.etcal(et)})")
        self.rovsvr*=self.km
    def _acc(self):
        super()._acc()
        self._thrust()
    def _thrust(self):
        """
        Calculate thrust needed to perform powered descent,
        :sets pdv_mass: PDV mass, only valid after backshell separation
        :sets delta_v: accumulated delta-v
        :sets throttle_straight: throttle level for straight engines
        :sets throttle_cant: throttle level for canted engines
        """
        # Assume constant Isp across all throttle settings. For a more realistic
        # Isp vs thrust, see Baker 2014 doi:10.2514/1.A32788, fig 22.
        print("Calculating thrust and throttle...")
        mle_ve=2200 #m/s,
        self.pdv_mass=np.zeros(self.ets.size)
        self.delta_v=np.zeros(self.ets.size)
        self.throttle_straight=np.zeros(self.ets.size)
        self.throttle_cant=np.zeros(self.ets.size)
        et_pdi=self.events["pdi"]
        et_skycrane0=self.events["skycrane"]
        w=np.where(self.ets<et_pdi)
        self.pdv_mass[w]=self.mass["Rover"]+self.mass["DescentStage"]["Dry"]+self.mass["DescentStage"]["MLEProp"]
        self.prop_used=np.zeros(self.ets.size)
        self.Fvert=np.zeros(self.ets.size)
        self.Ftot=np.zeros(self.ets.size)
        for i,et in enumerate(self.ets):
            if et<et_pdi:
                continue
            vrel=vlength(self.rovsvr[3:,i])
            drag=41*(vrel/32)**2 #drag is 41N in CVA (32m/s) and varies with square of vrel
            m=self.pdv_mass[i-1]
            a=vlength(self.angvf[:,i-1])
            Fvert=m*a-drag #effective vertical thrust
            n_mle=8 if et<et_skycrane0 else 4
            F_mle=Fvert/n_mle #vertical thrust on each engine
            self.Fvert[i]=Fvert
            #Actual thrust taking into account cosine losses, assumes
            #that canted engines have higher total thrust to make up for
            #their cantedness to give equal vertical thrust
            Fstraight=0 if n_mle<8 else F_mle/np.cos(np.radians(5))
            Fcant=F_mle/np.cos(np.radians(25))
            self.throttle_straight[i]=Fstraight
            self.throttle_cant[i]=Fcant
            Ftot=Fstraight*4+Fcant*4
            self.Ftot[i]=Ftot
            mdot=Ftot/mle_ve #kg/s fuel flow from all active engines
            dt=et-self.ets[i-1]
            dv=(Fvert/m)*dt #Only include effect of vertical thrust, not drag
            self.pdv_mass[i]=m-dt*mdot
            self.prop_used[i]=self.prop_used[i-1]+dt*mdot
            self.delta_v[i]=self.delta_v[i-1]+dv
        self.Ftot=smooth(self.Ftot,-50,50)
        self.Fvert=smooth(self.Fvert,-50,50)
        self.throttle_straight=smooth(self.throttle_straight,-50,50)
        self.throttle_cant=smooth(self.throttle_cant,-50,50)
        self.plot_throttle()
    def _tabulate(self):
        super()._tabulate()
        self.t_rovsvr=Tableterp(self.ets,self.rovsvr)
        self.t_Ftot=Tableterp(self.ets,self.Ftot)
        self.t_Fvert=Tableterp(self.ets,self.Fvert)
        self.t_throttle_straight=Tableterp(self.ets,self.throttle_straight)
        self.t_throttle_cant=Tableterp(self.ets,self.throttle_cant)
        self.t_pdv_mass=Tableterp(self.ets,self.pdv_mass)
        self.t_delta_v=Tableterp(self.ets,self.delta_v)
        self.t_prop_used = Tableterp(self.ets, self.prop_used)

    def _print_pov(self, et_step, i_step, tiles=None, file=None):
        super()._print_pov(et_step, i_step, tiles=tiles, file=file)
        self.print_vector("Rrover",v=self.t_rovsvr(et_step)[:3],comment="Rover position vector",file=file)
        self.print_vector("Vrover",v=self.t_rovsvr(et_step)[3:],comment="Rover velocity vector",file=file)
        self.print_scalar("Fvert",v=self.t_Fvert(et_step),comment="Vertical thrust",file=file)
        self.print_scalar("Ftot",v=self.t_Ftot(et_step),comment="Total thrust",file=file)
        self.print_scalar("Fstraight",v=self.t_throttle_straight(et_step),comment="Thrust on each straight engine",file=file)
        self.print_scalar("Fcant",v=self.t_throttle_cant(et_step),comment="Thrust on each canted engine",file=file)
        self.print_scalar("Prop_used",v=self.t_prop_used(et_step),comment="Propellant used",file=file)
        self.print_scalar("PDV_mass",v=self.t_pdv_mass(et_step),comment="Powered Descent Vehicle wet mass",file=file)
        self.print_scalar("DeltaV", v=self.t_delta_v(et_step), comment="Effective Delta-V", file=file)

    def plot_throttle(self):
        plt.figure('mass')
        max=np.max(self.pdv_mass)
        plt.plot(self.ets,self.pdv_mass,'-',label="PDV wet mass")
        plt.plot(self.ets,self.prop_used,'-',label="PDV prop used")
        plt.legend()
        plt.ylabel('mass/kg')
        self.plot_events(0,max)
        plt.legend()

        plt.figure('thrust')
        F=self.throttle_cant*4+self.throttle_straight*4
        max=np.max(F)
        plt.plot(self.ets,F,'-',label="Total thrust")
        plt.plot(self.ets,self.throttle_cant,'-',label="Canted engine thrust/each")
        plt.plot(self.ets,self.throttle_straight,'-',label="Straight engine thrust/each")
        plt.legend()
        plt.ylabel('thrust/N')
        self.plot_events(0,max)
        plt.legend()

        plt.figure('DeltaV')
        max=np.max(self.delta_v)
        plt.plot(self.ets,self.delta_v,'-',label="Accumulated Delta-V")
        plt.legend()
        plt.ylabel('DeltaV/(m/s)')
        self.plot_events(0,max)
        plt.legend()
        plt.pause(0.001)


if __name__=="__main__":
    M20=M20ReconTrajectory()
    if False:
        M20.plot()
        plt.show()
    else:
        M20.write_frames(et0=M20.events["ei"]-5,et1=M20.events["land"]+5,fps=30,do_tiles=False)
