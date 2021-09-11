from edl import CKTrajectory, eventTuple, bodyTuple, llr2xyz, Mr, rv, linterp, vnormalize, Mtrans, \
    SPKLoader,UniformSPKLoader
import numpy as np
import matplotlib.pyplot as plt
from which_kernel import which_kernel, ls_spice
from kwanspice.daf import double_array_file
import matplotlib.pyplot as plt
import spiceypy

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


class M20ReconTrajectory(CKTrajectory):
    def __init__(self):
        self.name="m20_recon"
        # Spacecraft EDL surface trajectory
        edl_spk="/mnt/big/home/chrisj/workspace/Data/spice/M20/spk/m2020_edl_v01.bsp"
        extra_kernels=(
            # Spacecraft clock kernel, not used until we have a CK, then must use the one that matches the CK
            "/mnt/big/home/chrisj/workspace/Data/spice/M20/sclk/M2020_168_SCLKSCET.00007.tsc",
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
        self.events["land"]       =666953098.82832849025726  # Time of first point in post-landing segment.
        self.events["last"]       =666953098.828360 #Last point in DIMU segment
        et0 = self.events["land"] - 420  # Seven minutes of excitement! Entry interface is about 7.25s after this time.
        et1 = self.events["land"] + 20  # Make sure we are steady on the ground
        loader=DAFSPKLoader(spk=edl_spk, spice_sc=-168031)
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
        dropobjs={}
        """
        dropobjs = {
            "Heatshield": bodyTuple(pov_suffix="hs", et0=self.events["heatshield"], a=3.14 * 2.25 ** 2, m=mass["HeatShield"],
                                    v0vb=np.array([[0], [0], [2.7]]), cd=1.5),
            "Backshell": bodyTuple(pov_suffix="bs", et0=self.events["backshell"], a=3.14 * 10 ** 2,
                                   m=mass["Backshell"]["Backshell"] + mass["Backshell"]["Parachute"], cd=2.5)}
        for i in range(len(londs)):
            dropobjs[f"EBM{i}"]=bodyTuple(pov_suffix=f"ebm{i}", et0=self.events["sufr0"] + 2 + 2 * i, a=0.0193,
                                          m=mass["EntryBalanceMass"][f"Mass{i}"], cd=1, r0vb=r0ebmvb[i], v0vb=v0ebmvb[i])
        """
        super().__init__(loader=loader,extra_kernels=extra_kernels,ckobj="M2020_ROVER", dropobjs=dropobjs)
    def _cam(self):
        """
        Calculate camera position

        :sets location: Location vector of camera in frozen frame relative to center of Mars
        :sets look_at:  Look_at vector of camera in frozen frame relative to center of Mars
        :sets sky:      Sky direction of camera in frozen frame
        """
        Mprf = Mr(self.Msrf)
        Rhat = Mtrans(Mprf,self.rhatf)
        Vhat = Mtrans(Mprf,self.vhatf)
        Ehat = Mtrans(Mprf,self.ehatf)
        Nhat = Mtrans(Mprf,self.nhatf)
        Hhat = Mtrans(Mprf,self.hhatf)
        Qhat = Mtrans(Mprf,self.qhatf)
        Khat = Mtrans(Mprf,self.khatf)
        Ang  = Mtrans(Mprf,self.angvf)
        self.location=Ang*0.0
        self.look_at=Ang*0.0
        self.sky=Ang*0.0
        R = rv(self.svr)
        done1a=False
        done1b=False
        done2a=False
        done2b=False

        for i_step in self.i_steps:

            if self.ets[i_step] < self.events["mortar"] - 100:
                angle = 0
                dist = 20
                height = 12
                qbasis = Qhat[:,i_step]
                hbasis = Hhat[:,i_step]
                rbasis = Rhat[:,i_step]
            elif self.ets[i_step] < self.events["mortar"] - 50:
                if not done1a:
                    print(i_step,self.ets[i_step],"blank")
                    done1a=True
                angle = linterp(self.events["mortar"] - 100, 0, self.events["mortar"] - 50, np.pi / 2, self.ets[i_step])
                dist = linterp(self.events["mortar"] - 100, 20, self.events["mortar"] - 50, 10, self.ets[i_step])
                height = linterp(self.events["mortar"] - 100, 12, self.events["mortar"] - 50, 0, self.ets[i_step])
                qbasis = Qhat[:,i_step]
                hbasis = Hhat[:,i_step]
                rbasis = Rhat[:,i_step]
            elif self.ets[i_step] < (self.events["heatshield"] + 100):
                if not done1b:
                    print(i_step,self.ets[i_step],"blank")
                    done1b=True
                angle = np.pi / 2
                dist = 10
                height = 0
                qbasis = Qhat[:,i_step]
                hbasis = Hhat[:,i_step]
                rbasis = Rhat[:,i_step]
            elif self.ets[i_step] < (self.events["heatshield"] + 150):
                if not done2a:
                    print(i_step,self.ets[i_step],"blank")
                    done2a=True
                angle = np.pi / 2
                dist = 10
                height = 0
                qbasis = vnormalize(linterp(self.events["heatshield"] + 100, Qhat[:,i_step], self.events["heatshield"] + 150, Ehat[:,i_step], self.ets[i_step]))
                hbasis = vnormalize(linterp(self.events["heatshield"] + 100, Hhat[:,i_step], self.events["heatshield"] + 150, Nhat[:,i_step], self.ets[i_step]))
                rbasis = Rhat[:,i_step]
            else:
                if not done2b:
                    print(i_step,self.ets[i_step],"blank")
                    done2b=True
                angle = np.pi / 2
                dist = 10
                height = 0
                qbasis = Ehat[:,i_step]
                hbasis = Nhat[:,i_step]
                rbasis = Rhat[:,i_step]
            self.location[:,i_step] = R[:,i_step] - dist * np.cos(angle) * qbasis - dist * np.sin(angle) * hbasis + height * rbasis
            self.look_at[:,i_step] = R[:,i_step]
            self.sky[:,i_step] = Rhat[:,i_step]


if __name__=="__main__":
    M20=M20ReconTrajectory()
    M20.plot()
    #M20.write_frames(et0=M20.events["ei"]-5,et1=M20.events["land"]+5,fps=30,do_tiles=False)
    plt.show()
