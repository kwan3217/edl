from edl import CKTrajectory, eventTuple, bodyTuple, SPKLoader, UniformSPKLoader, DAFSPKLoader, et2pdt
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from kwanmath.vector import vlength, vnormalize, vncross, vcross, rv
from kwanmath.matrix import Mr,  Mtrans
from kwanmath.interp import linterp, smooth, tableterp
from kwanmath.geodesy import llr2xyz, aTwoBody, aJ2
from kwanspice.which_kernel import which_kernel, ls_spice
import matplotlib.pyplot as plt
import spiceypy
from hud import HUD
from picturebox import PictureBox


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
        self.events["bridle"]       =666953098.828360 #Last point in DIMU segment, assumed bridle cut
        self.events["rocker"]          =self.events["rappel0"]+0.7
        #self.events["rappel1"]         =self.events["rappel0"]+5.5
        self.events["bogie"]           =self.events["rappel0"]+6.0
        #Beginning and end of animation
        self.events["anim_et0"] = self.events["land"] - 420  # Seven minutes of excitement! Entry interface is about 0.43s after this time.
        self.events["anim_et1"] = self.events["land"] + 20  # Make sure we are steady on the ground

        #sort the event list
        self.events=dict(sorted(self.events.items(),key=lambda x:x[1]))


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
            print(f"Event {k:15}: E{self.format_time(v,self.events['ei'])} L{self.format_time(v,self.events['land'])} ET{v:.3f} ({spiceypy.etcal(v)}) i_step {self.i_step[k]:6d}")
        self.spice_sc="-168"
        self.spice_bodyframe=None
        prop_ei=399.42
        prop_backshell=387.49
        prop_bridle=95.65
        self.mass = {
            "Rover": 899,
            "DescentStage": {
                "RCSProp": prop_ei-prop_backshell,
                "MLEProp": prop_backshell-prop_bridle,
                "PostLandProp": prop_bridle,
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
        self.cache_fields = self.cache_fields + ["rovsvr", "Ftot", "Fvert", "throttle_straight", "throttle_cant",
                                               "pdv_mass", "delta_v", "prop_used"]
        # Assume constant Isp across all throttle settings. For a more realistic
        # Isp vs thrust, see Baker 2014 doi:10.2514/1.A32788, fig 22.
        # Had to add a fudge factor to hit realtime-telemetered data (387.49kg at PDI, 95.65 after bridle release)
        self.mle_ve=2200*1.180875 #m/s, corrected to hit actual end-of-PD prop mass
        #Aerodynamics of descent vehicle, Baker2014 p1225 "This effective lift area yields a lift force of 41N at CVA
        #[constant velocity accordion, 32m/s] and zero just before touchdown."
        self.rho0 = 0.0385
        q0 = self.rho0 * 32 ** 2 / 2
        D0 = 41
        self.CdA = D0 / q0
        self._calc()
    def _spice(self,sps=200):
        super()._spice()
        print("Extracting state vectors for rover from Spice...")
        self.rovsvr=np.zeros((6,self.ets.size))
        for i,et in enumerate(self.ets):
            #try:
                self.rovsvr[:,None,i]=(self.Msrf[i,:,] @ self.Msfi @ spiceypy.spkezr("-168",et,"J2000","NONE","499")[0])[:,None]
            #except Exception:
            #    print(f"Couldn't figure spice for timestep {i}, et {et} ({spiceypy.etcal(et)})")
        self.rovsvr*=self.km
        self._offset_rover()
        self._flyaway(sps=sps)
    def _offset_rover(self):
        print("Offset things so that the rover ends up at its final resting place...")
        #Figure the location on the in-motion track at the moment of contact,
        #in the rotating frame (where landing point is fixed)
        t_rovsvr=tableterp(self.ets,self.rovsvr)
        r1vr_motion=t_rovsvr(self.events["contact"])[0:3,:]
        r1vr_station=t_rovsvr(self.events["land"])[0:3,:]
        dr1vr=r1vr_station-r1vr_motion
        #Figure out which points on rover trajectory are before or after contact
        w_before=np.where(self.ets<self.events["contact"])[0]
        w_after=np.where(self.ets>=self.events["contact"])[0]
        #Offset the rover rotating position by the delta
        self.rovsvr[:3,w_before]+=dr1vr
        self.rovsvr[:3,w_after]=r1vr_station
        #Offset the descent stage frozen-frame position by the delta rotated into the frozen frame
        self.svf[:3,:]+=Mtrans(Mr(self.Msrf).transpose((0,2,1)),dr1vr)
    def _flyaway(self,sps=200):
        dt=1/sps
        et0=self.events["bridle"]
        #Figure out what direction to fly the descent stage
        t_Msrf=tableterp(self.ets,self.Msrf)
        t_svr=tableterp(self.ets,Mtrans(self.Msrf,self.svf))
        Mds1sib=spiceypy.sxform(self.ckobj,"J2000",et0) #J2000 from body at moment of bridle disconnect
        Mds1sfb=self.Msfi @ Mds1sib #Frozen frame from body
        #Mds1srb=t_Msrf(et0) @ self.Msfi @ Mds1sib #Rotating frame from body
        zhat=np.array([[0], [0], [1]])
        ds1rhatr = vnormalize(self.Msrf[-1,:,:] @ self.svf[:,-1])
        ds1ehatr = vncross(   zhat ,ds1rhatr)
        ds1nhatr = vncross(ds1rhatr,ds1ehatr)
        #Mds1rtr=np.hstack((ds1ehatr,ds1nhatr,ds1rhatr)).T
        #Mds1rtb=Mds1rtr @ Mr(Mds1srb) #topocentric from body at descent stage location at bridle cut
        #Fly the descent stage off to its final resting place (rest in pieces)
        r0v=self.svf[:3,None,-1]
        v0v=self.svf[3:,None,-1]
        r0=vlength(r0v)
        dsrvfs=[r0v]
        dsvvfs=[v0v]
        ets=[et0]
        m=self.mass["DescentStage"]["Dry"]+self.mass["DescentStage"]["PostLandProp"]
        i=1
        done=False
        while not done:
            et=et0+i/sps
            #Pitch down from vertical
            pitchdown=np.deg2rad(linterp(0.7,0,1.7,45,et-self.events["bridle"],bound=True))
            if et-self.events["bridle"]<0.7:
                #hover
                F=4*2000
            elif et-self.events["bridle"]<6.7:
                F=4*3000
            else:
                F=0
            #Force direction at current time, in body frame at bridle cut time
            fhatb1=np.array([[0],[-np.sin(pitchdown)],[-np.cos(pitchdown)]])
            #Cosine loss -- we are flying on the canted engines
            fhatb1*=np.cos(np.rad2deg(25))
            #Transform to frozen frame
            fhatf=Mr(Mds1sfb) @ fhatb1
            dsrvf=dsrvfs[i-1]
            dsvvf=dsvvfs[i-1]
            windf=vcross(np.array([[0], [0], [self.mars_omega]]), dsrvf)
            dsuvf=dsvvf-windf #air-relative motion in frozen frame. Drag vector is antiparallel to this vector
            q=self.rho0*vlength(dsuvf)**2/2
            Dv=-vnormalize(dsuvf)*q*self.CdA
            Fv=F*fhatf
            mdot=-F/self.mle_ve
            agvf= aTwoBody(dsrvf, gm=self.mars_gm) + aJ2(dsrvf, j2=self.mars_j2, gm=self.mars_gm, re=self.mars_re)
            dsavf=(Fv+Dv)/m+agvf
            dsrvfs.append(dsrvf+dsvvf*dt)
            dsvvfs.append(dsvvf+dsavf*dt)
            ets.append(et)
            m=m+mdot*dt
            i+=1
            done=vlength(dsrvf)<(r0-7)
        self.events["flyaway1"] = et  # Landing phase is over when kinetic energy of all hardware relative to surface is zero
        self.i_step["flyaway1"]=i-1+self.ets.size
        #sort the event list
        self.events=dict(sorted(self.events.items(),key=lambda x:x[1]))
        self.write_events()
        k="flyaway1"
        v=self.events[k]
        print(f"Event {k:15}: E{self.format_time(v, self.events['ei'])} L{self.format_time(v, self.events['land'])} ET{v:.3f} ({spiceypy.etcal(v)}) i_step {self.i_step[k]:6d}")
        print(f"Done integrating flyaway. Steps: {i}")
        self._extend(dsrvfs,dsvvfs,ets)
    def _extend(self,dsrvfs,dsvvfs,ets):
        dsrvfs=np.array(dsrvfs)[:,:,0].T
        dsvvfs=np.array(dsvvfs)[:,:,0].T
        self.svf=np.hstack((self.svf,np.vstack((dsrvfs,dsvvfs))))
        n_ets_old = self.ets.size
        self.ets=np.hstack((self.ets,np.array(ets)))
        print("Extend the et list to anim_et1...")
        n_ets_new = len(ets)
        self.dts=self.ets*0
        self.dts[1:]=self.ets[1:]-self.ets[:-1]
        self.dts[0]=self.dts[1]
        self.pdts=self.pdts+[et2pdt(et) for et in ets]
        self.i_steps=np.arange(len(self.ets))
        print("Extend the vector stacks to cover the new et range...")
        self.Msrf = np.concatenate((self.Msrf, np.zeros((n_ets_new, 6, 6))), axis=0)
        self.sunr = np.concatenate((self.sunr, np.zeros((3, n_ets_new))), axis=1)
        self.earthr = np.concatenate((self.earthr, np.zeros((3, n_ets_new))), axis=1)
        self.earthlt = np.hstack((self.earthlt, np.zeros(n_ets_new)))
        self.rovsvr = np.concatenate((self.rovsvr, np.zeros((6, n_ets_new))), axis=1)
        for i_new, et in enumerate(ets):
            i = n_ets_old + i_new
            Msri = spiceypy.sxform("J2000", "IAU_MARS", et)  # Stack of matrices converting to rotating from inertial
            self.Msrf[i, :, :] = Msri @ self.Msfi.transpose()
            state, _ = spiceypy.spkezr("SUN", et, "IAU_MARS", "LT+S", "499")
            self.sunr[:, i] = state[:3] * self.km
            state, lt = spiceypy.spkezr("399", et, "IAU_MARS", "XCN+S", "499")
            self.earthr[:, i] = state[:3] * self.km
            self.earthlt[i] = lt
        self.rovsvr[:,n_ets_old:]=self.rovsvr[:,None,n_ets_old-1]

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
        print("Calculating thrust and throttle...")
        self.pdv_mass=np.zeros(self.ets.size)
        self.delta_v=np.zeros(self.ets.size)
        self.throttle_straight=np.zeros(self.ets.size)
        self.throttle_cant=np.zeros(self.ets.size)
        et_pdi=self.events["pdi"]
        et_skycrane0=self.events["skycrane"]
        w=np.where(self.ets<et_pdi)
        self.pdv_mass[w]=self.mass["Rover"]+self.mass["DescentStage"]["Dry"]+self.mass["DescentStage"]["MLEProp"]+self.mass["DescentStage"]["PostLandProp"]
        self.prop_used=np.zeros(self.ets.size)
        self.Fvert=np.zeros(self.ets.size)
        self.Ftot=np.zeros(self.ets.size)
        for i,et in enumerate(self.ets):
            if et<et_pdi:
                continue
            vrel=vlength(self.rovsvr[3:,i])
            drag=41*(vrel/32)**2 #drag is 41N in CVA (32m/s) and varies with square of vrel
            if et<self.events["bridle"]:
                m=self.mass["Rover"]+self.mass["DescentStage"]["Dry"]+self.mass["DescentStage"]["PostLandProp"]
            else:
                m = self.mass["DescentStage"]["Dry"] + self.mass["DescentStage"]["PostLandProp"]
            m-=self.prop_used[-1]
            a=vlength(self.angvf[:,i-1])
            Fvert=m*a-drag if m*a>drag else 0 #effective vertical thrust
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
            mdot=Ftot/self.mle_ve
            if np.abs(mdot)>10000:
                print("Hey!")#kg/s fuel flow from all active engines
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
        return None
    def tabulate(self):
        super().tabulate()
        self.t_rovsvr=tableterp(self.ets,self.rovsvr)
        self.t_Ftot=tableterp(self.ets,self.Ftot)
        self.t_Fvert=tableterp(self.ets,self.Fvert)
        self.t_throttle_straight=tableterp(self.ets,self.throttle_straight)
        self.t_throttle_cant=tableterp(self.ets,self.throttle_cant)
        self.t_pdv_mass=tableterp(self.ets,self.pdv_mass)
        self.t_delta_v=tableterp(self.ets,self.delta_v)
        self.t_prop_used = tableterp(self.ets, self.prop_used)

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

class JPLHUD(HUD):
    def draw_meter(self,pb:PictureBox,*,xc:float,v0:float,v1:float,v:float,name:str,units:str):
        scl=1
        r=70*scl
        xc=xc*scl
        yc=960*scl
        lw=8*scl
        pb.arc(xc,yc,r,90,315,color='w',linewidth=lw,fill=False,alpha=0.5)
        pb.arc(xc,yc,r,90,linterp(v0,90,v1,315,v1 if v>v1 else v0 if v<v0 else v),color='w',linewidth=lw,fill=False,alpha=1.0)
        pb.text(xc+2,yc+r,f"{v0:.2f}",alpha=0.5,color='w')
        pb.text(xc+36*scl, yc-36*scl, f"{v1:.2f}", alpha=0.5, color='w')
        pb.text(xc,yc-r-10,name,color='w',ha='center')
        pb.text(xc,yc,f"{v:.2f}",color='w',fontsize=20,ha='center',va='center')
        pb.text(xc,yc+20, units, color='w', alpha=0.5, ha='center', va='center')

    def draw_frame(self,i,et,pb:PictureBox):
        m0=730
        m1=954
        dm=m1-m0
        pos=self.trajectory.t_rovsvr(et)[:3]
        alt=vlength(pos)-vlength(self.trajectory.svr[:3,-1,None])
        alt_km=alt/1000
        vrel=vlength(self.trajectory.t_svr(et)[3:])
        mach=vrel/self.trajectory.t_csound(et)
        acc=vlength(self.trajectory.t_angvr(et))
        lon=np.rad2deg(np.arctan2(pos[1],pos[0]))[0]
        lat=np.rad2deg(np.arctan2(pos[2],np.sqrt(pos[1]**2+pos[0]**2)))[0]
        fuel=self.trajectory.mass["DescentStage"]["MLEProp"]-self.trajectory.t_prop_used(et)
        if et<self.trajectory.events["backshell"]:
            self.draw_meter(pb,xc=m0+0*dm,v0=0,v1=1500,name="ALTITUDE"    ,units="KM"     ,v=alt_km)
        else:
            self.draw_meter(pb,xc=m0+0*dm,v0=0,v1=1500,name="ALTITUDE"    ,units="M"      ,v=alt)
        self.draw_meter(pb,xc=m0+1*dm,v0=0,v1=   50   ,name="MACH"        ,units="MACH"   ,v=mach)
        self.draw_meter(pb,xc=m0+2*dm,v0=0,v1=10000   ,name="SPEED"       ,units="M/S"    ,v=vrel)
        self.draw_meter(pb,xc=m0+3*dm,v0=0,v1=  200   ,name="ACCELERATION",units="M/S$^2$",v=acc)
        self.draw_meter(pb,xc=m0+4*dm,v0=0,v1=  401.33,name="FUEL"        ,units="KG"     ,v=fuel)
        pb.text(1035,34,spiceypy.timout(et+self.trajectory.t_earthlt(et),"UTC: YYYY MON DD HR:MN:SC::UTC"),color='w',alpha=0.5,va='baseline',fontsize=13)
        pb.text(1325,34,spiceypy.timout(et,"SCET: YYYY MON DD HR:MN:SC::UTC"),color='w',alpha=0.5,va='baseline',fontsize=13)
        pb.text( 795,34,f"LAT/LON: {lat:.1f}$^\circ$,{lon:.1f}$^\circ$",color='w',alpha=0.5,va='baseline',fontsize=13)
        pb.text(959,810,"Spacecraft Position and Configuration Based on Reconstructed Trajectory",color='w',va='baseline',fontsize=9)

if __name__=="__main__":
    M20=M20ReconTrajectory()
    if False:
        M20.plot()
        plt.show()
    else:
        hud=JPLHUD(M20,"Frames_recon_hud/hud_%05d.png")
        M20.write_frames(et0=M20.events["ei"]-5,et1=M20.events["flyaway1"],fps=30,do_tiles=False)
        hud.write_frames(et0=M20.events["ei"]-5,et1=M20.events["flyaway1"],fps=30)
