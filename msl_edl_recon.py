from edl import CKTrajectory, eventTuple, bodyTuple, llr2xyz, Mr, rv, linterp, vnormalize, Mtrans, \
    SPKLoader, UniformSPKLoader, DAFSPKLoader
import numpy as np
import matplotlib.pyplot as plt
from which_kernel import which_kernel, ls_spice
from kwanspice.daf import double_array_file
import matplotlib.pyplot as plt
import spiceypy


class MSLReconTrajectory(CKTrajectory):
    def __init__(self):
        self.name="msl_recon"
        # Spacecraft EDL surface trajectory
        edl_spk="/mnt/big/home/chrisj/workspace/Data/spice/MSL/spk/msl_edl_v01.bsp"
        edl_sclk="/mnt/big/home/chrisj/workspace/Data/spice/MSL/sclk/MSL_76_SCLKSCET.00017.tsc"
        extra_kernels=(
            # Spacecraft clock kernel, not used until we have a CK, then must use the one that matches the CK
            edl_sclk,
            # Spacecraft frame kernel, must use the one referenced in the CK comments
            "/mnt/big/home/chrisj/workspace/Data/spice/MSL/fk/msl_v08.tf",

            # Spacecraft EDL surface trajectory
            edl_spk,

            #Spacecraft landing site
            "/mnt/big/home/chrisj/workspace/Data/spice/MSL/spk/msl_ls_ops120808_iau2000_v1.bsp",

            #EDL pointing
            "/mnt/big/home/chrisj/workspace/Data/spice/MSL/ck/msl_edl_v01.bc",
        )
        #Load sclk stuff specifically for scs2e
        spiceypy.furnsh(edl_sclk)
        spiceypy.furnsh("/mnt/big/home/chrisj/workspace/Data/spice/generic/lsk/naif0012.tls")
        # Trajectory values. Manually figured from plots of acceleration and body rates
        #self.et_ei=666952679.25992 #Actual time of last Spice point before EI (about 4m above, next point is about 3m below)
        self.events={}
        t0=397501174.997338 #t0 in SCLK (*not* Spice ET), 540s before predicted EI, from Kalgard 2014 https://doi.org/10.2514/1.A32770
        t0_s=int(t0)
        t0_ss=int((t0-t0_s)*65536)
        self.events["t0"]=spiceypy.scs2e(-76,f"{t0_s:010d}-{t0_ss:05d}")
        t0_baker=397500000
        self.events["t0_baker"]=spiceypy.scs2e(-76,f"{t0_baker:010d}-00000")
        #clear sclk stuff so that they can be reloaded fresh
        spiceypy.kclear()
        self.events["ei"]         =397501912.83806 #Interpolated EI time
        self.events["rollrev10"]  =397501985.997  #Start of first roll reversal
        self.events["peakheat"]   =397501986.28 #peak heating
        self.events["maxg"]       =397501993.187 #Smoothed maximum g, 123.3857 m/s**2
        self.events["maxq"]       =397501993.48 #maximum dynamic pressure
        self.events["rollrev11"]  =397501998.232  #End of first roll reversal
        self.events["rollrev20"]  =397502006.99  #Start of second roll reversal
        self.events["rollrev21"]  =397502017.47  #End of second roll reversal
        self.events["rollrev30"]  =397502036.499  #Start of third roll reversal
        self.events["rollrev31"]  =397502046.817  #End of third roll reversal
        self.events["sufr0"]      =397502152.96  #Start of straighten up and fly right (SUFR)
        self.events["ebm0"]       =397502158.86 # First EBM jostle during SUFR maximum rate
        self.events["sufr1"]      =397502167.00  #End of SUFR
        self.events["mortar"]     =397502172.114 #Mortar firing
        self.events["linestretch"]=397502173.236 #Line stretch jolt
        self.events["chuteinf1"]  =397502173.874  #First chute inflation peak (raw), 58.342 m/s**2
        self.events["chuteinf2"]  =397502174.433 #Second chute inflation peak (smoothed), 41.439 m/s**2
        self.events["heatshield"] =397502191.848  #Heat shield jettison
        self.events["backshell"]  =397502288.745  #Backshell separation
        self.events["pdi"]        =397502289.681  #Powered descent initiation
        self.events["cvel0"]      =397502313.044  #Constant velocity phase begin, hdot=-32m/s
        self.events["cvel1"]      =397502316.104  #Constant velocity phase end
        #self.events["skycrane"]   =666953078.99   #Skycrane start, hdot=-0.75m/s, switch from 8 to 4 engines
        self.events["rappel0"]    =397502325.89685   #Time point from kernel dimu-rover segment
        self.events["rappel1"]    =397502330.59689   #Rover rappel end
        self.events["land"]       =397502344.475040018  # Time of first point in post-landing segment.
        self.events["last"]       =397502345.268023193  #Last point in DIMU segment

        self.events["ei_k2014"]        =self.events["t0"]+540.0
        self.events["guidestart_k2014"]=self.events["t0"]+585.88
        self.events["rollrev1_k2014"]  =self.events["t0"]+612.88
        self.events["maxg_k2014"]      =self.events["t0"]+620.33
        self.events["rollrev2_k2014"]  =self.events["t0"]+633.88
        self.events["rollrev3_k2014"]  =self.events["t0"]+663.38
        self.events["headalign_k2014"] =self.events["t0"]+675.63
        self.events["EBM_k2014"]       =self.events["t0"]+779.87
        self.events["chutewait_k2014"] =self.events["t0"]+793.87
        self.events["chute_k2014"]     =self.events["t0"]+799.12
        self.events["lastMEADS_k2014"] =self.events["t0"]+808.86
        self.events["heatshield_k2014"]=self.events["t0"]+818.87
        self.events["radarlock_k2014"] =self.events["t0"]+837.12
        self.events["MLEprime_k2014"]  =self.events["t0"]+899.63
        self.events["backshell_k2014"] =self.events["t0"]+915.92
        self.events["pdi_k2014"]       =self.events["t0"]+918.38
        self.events["rappel0_k2014"]   =self.events["t0"]+952.92
        self.events["tdready_k2014"]   =self.events["t0"]+961.86
        self.events["land_k2014"]      =self.events["t0"]+971.52
        self.events["flyaway_k2014"]   =self.events["t0"]+972.31

        #Event times from Baker et al 2014 DOI 10.2514/1.A32788
        #Table 2
        self.events["DRCS_CATBED_B2014"]=self.events["t0_baker"]-9085.16 # DRCS CATBED heaters are turned on (prime string only)
        self.events["PV1a_1b_B2014"]    =self.events["t0_baker"]+1099.95 # Prime DRCS feed lines
        self.events["PV2a_2b_B2014"]    =self.events["t0_baker"]+1110.95
        self.events["CSS_B2014"]        =self.events["t0_baker"]+1114.95
        self.events["40ms_B2014"]       =self.events["t0_baker"]+1175.00
        self.events["detumble_B2014"]   =self.events["t0_baker"]+1195.63
        self.events["PV3a_B2014"]       =self.events["t0_baker"]+1714.45
        self.events["PV3b_B2014"]       =self.events["t0_baker"]+1714.58
        self.events["PV4a_B2014"]       =self.events["t0_baker"]+1714.73
        self.events["PV4b_B2014"]       =self.events["t0_baker"]+1714.86
        self.events["guidestart_B2014"] =self.events["t0_baker"]+1761.00
        self.events["EBM_B2014"]        =self.events["t0_baker"]+1954.88
        self.events["sufr1_B2014"]      =self.events["t0_baker"]+1968.88
        self.events["chute_B2014"]      =self.events["t0_baker"]+1974.13
        #self.events["DRCSstop_B2014"]   =self.events["t0_baker"]+1993.88 #Duplicate of heatshield sep
        #Table 4 (duplicates removed)
        #MLE CATBED heaters
        #DS tank fuel-side (same as DRCS_CATBED)
        #PV3a
        #PV3b
        #PV4a
        #PV4b
        self.events["heatshield_B2014"]=self.events["t0_baker"]+1993.88
        self.events["thrstart_B2014"]  =self.events["t0_baker"]+1994.16
        self.events["PV5a_B2014"]      =self.events["t0_baker"]+2074.63
        self.events["PV5b_B2014"]      =self.events["t0_baker"]+2074.66
        self.events["backshell_B2014"] =self.events["t0_baker"]+2090.77
        self.events["PV6a_B2014"]      =self.events["t0_baker"]+2091.70
        self.events["PV6a_B2014"]      =self.events["t0_baker"]+2091.83
        self.events["MLEstart_B2014"]  =self.events["t0_baker"]+2091.89
        self.events["cvel0_B2014"]     =self.events["t0_baker"]+2115.03
        self.events["cvel1_B2014"]     =self.events["t0_baker"]+2118.08
        self.events["skycrane_B2014"]  =self.events["t0_baker"]+2125.36
        self.events["land_B2014"]      =self.events["t0_baker"]+2146.52
        self.events["bridle_B2014"]    =self.events["t0_baker"]+2147.11
        self.events["flyaway0_B2014"]  =self.events["t0_baker"]+2147.14
        self.events["flyaway1_B2014"]  =self.events["t0_baker"]+2153.80
        self.events["dsimpact_B2014"]  =self.events["t0_baker"]+2153.80+20

        self.events["rocker_s2014"]          =self.events["rappel0_k2014"]+0.7
        self.events["rappel1_s2014"]         =self.events["rappel0_k2014"]+5.0
        self.events["bogie_s2014"]           =self.events["rappel0_k2014"]+6.0



        #sort the Kalgaard timings in with the others
        self.events=dict(sorted(self.events.items(),key=lambda x:x[1]))



        et0 = self.events["land"] - 420  # Seven minutes of excitement! Entry interface is about 7.25s after this time.
        et1 = self.events["land"] + 20  # Make sure we are steady on the ground
        loader=DAFSPKLoader(spk=edl_spk, spice_sc=-76031)
        #loader=UniformSPKLoader(et0=et0,et1=et1,dt=1/24,spice_sc="-168")
        loader_ets=loader.time()
        self.i_step={}
        for k,v in self.events.items():
            try:
                self.i_step[k]=np.min(np.where(v<loader_ets))
            except ValueError:
                self.i_step[k]=len(loader_ets)-1
            except TypeError:
                self.i_step[k]=-1
            t0_dt=v-self.events['t0']
            t0_dt_m=int(t0_dt)//60
            t0_dt_s=t0_dt%60
            ei_dt=v-self.events['ei']
            ei_dt_m=int(ei_dt)//60
            ei_dt_s=ei_dt%60
            l_dt= self.events['land'] - v
            l_dt_m=int(l_dt)//60
            l_dt_s=l_dt%60
            print(f"Event {k:18}: T0{print_time(v,self.events['t0'])} E{print_time(v,self.events['ei'])} L{print_time(v,self.events['land'])} ET{v:.3f} ({spiceypy.etcal(v)}) i_step {self.i_step[k]:6d}")
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
        dt_ebm0=2.0 #(nominal) Time between EBM jettisons. EBMs were jettisoned in pairs
        et_ebm=[self.events["ebm0"]+dt_ebm0*0,self.events["ebm0"]+dt_ebm0*0,
                self.events["ebm0"]+dt_ebm0*1,self.events["ebm0"]+dt_ebm0*1,
                self.events["ebm0"]+dt_ebm0*2,self.events["ebm0"]+dt_ebm0*2]
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
        super().__init__(loader=loader,extra_kernels=extra_kernels,ckobj="MSL_ROVER", dropobjs=dropobjs)
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

            if self.ets[i_step] < self.events["mortar"] - 20:
                angle = 0
                dist = 20
                height = 12
                qbasis = Qhat[:,i_step]
                hbasis = Hhat[:,i_step]
                rbasis = Rhat[:,i_step]
            elif self.ets[i_step] < self.events["mortar"] - 15:
                if not done1a:
                    print(i_step,self.ets[i_step],"blank")
                    done1a=True
                angle = linterp(self.events["mortar"] - 20, 0, self.events["mortar"] - 15, np.pi / 2, self.ets[i_step])
                dist = linterp(self.events["mortar"] - 20, 20, self.events["mortar"] - 15, 10, self.ets[i_step])
                height = linterp(self.events["mortar"] - 20, 12, self.events["mortar"] - 15, 0, self.ets[i_step])
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
    MSL=MSLReconTrajectory()
    MSL.plot()
    #M.write_frames(et0=M20.events["ei"]-5,et1=M20.events["land"]+5,fps=30,do_tiles=False)
    plt.show()
