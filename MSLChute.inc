#include "metals.inc"

#declare D0=19.7;
#declare S0=pi*D0*D0/4;
#declare DDisk=0.72*D0;
#declare Dp=13.5; //Projected diameter Dp range ~12.7m to 14m, dependent on dynamic pressure
#declare Rp=Dp/2;
#declare Dv=0.5;  //Disk vent
#declare Rv=Dv/2;
#declare Hp=2.4;  //Height of parachute disk part, m
#declare Hg=0.83; //Heigth of gap, m
#declare Hb=2.38; //Height of band, m
#declare Ls=33.49; //Length of suspension lines, m
#declare Hs=sqrt(Ls*Ls-Rp*Rp); //Height of suspension lines, m
#declare Hr=10.14; //Height of riser, m
#declare Hbr=1.37;  //Height of bridle, m
#declare Dbr=0.4;   //Diameter of bridle circle, m
#declare Rbr=Dbr/2;  //Radius of bridle circle, m
#declare Rbl=0.005; //Radius of lines, m
#declare Xrl=0.01;
#declare Yrl=0.10;
#declare Rsl=Rbl;

#declare MSLChuteBridle=union {
  #local I=0;
  #while(I<3)
    union {
      cylinder {
        <0,Rbr-Rbl,0>,<0,Rbr-Rbl,Hbr*2/3>,Rbl
      }            
      sphere {
        <0,Rbr-Rbl,Hbr*2/3>,Rbl
      }
      cylinder {
        <0,Rbr-Rbl,Hbr*2/3>,<0,0,Hbr>,Rbl
      }
      rotate z*120*I
      pigment {color rgb <1,1,0.5>}
    } 
    #local I=I+1;
  #end
  difference {
    union {
      cylinder {
        <0,0,Hbr/3-0.05>,<0,0,Hbr/3>,Rbr
      }
      cylinder {
        <0,0,Hbr*2/3-0.05>,<0,0,Hbr*2/3>,Rbr
      }
    }
    cylinder {
      <0,0,0>,<0,0,Hbr>,Dbr/2-(Rbl*2)
    }
    texture {T_Chrome_3C}
  }
}  
#declare MSLChuteRiser=box {
  <-Xrl/2,-Yrl/2,0>,<Xrl/2,Yrl/2,Hr>
  pigment {color rgb <1,1,0.5>}
}

#declare MSLChuteSuspension=union {
  #local I=0;
  #while(I<40)
    cylinder {
      <0,-Yrl/2,0>,<Rp*cos(radians(I*180/40)),-Rp*sin(radians(I*180/40)),Hs>,Rsl
    }
    cylinder {
      <0,Yrl/2,0>,<Rp*cos(radians(I*180/40)),Rp*sin(radians(I*180/40)),Hs>,Rsl
    }
    #local I=I+1;
  #end
  pigment {color rgb <1,1,0.5>}
}

#declare MSLChuteGap=union {
  #local I=0;
  #while(I<80)
    cylinder {
      <Rp,0,0>,<Rp,0,Hg>,Rsl
      rotate I*360/80*z
    }
    #local I=I+1;
  #end
  pigment {color rgb <1,1,0.5>}
}
                                
#declare Vst=50; //m/s, line stretch speed
#declare Lp=sqrt(Rp*Rp+Hp*Hp);
#declare Afcrit=1.05;
#declare A0=pi*Rp*Rp;
#declare Acrit=A0*Afcrit;
#declare Rcrit=sqrt(Acrit/pi);
#declare Hcrit=sqrt(Lp*Lp-Rcrit*Rcrit);
#declare Hmax=(Hbr+Hr+Hs+Hb+Hg+Lp);
#declare Tst=ET_linestretch-ET_mortar;
#declare Tp=ET_mortar;
#declare Tinflate=ET_linestretch;
#declare TI1=Tinflate;
#declare TI2=Tinflate+2;

#declare TAcc0=Tinflate-2;                                
#declare Acc0=50;

#macro MSLChuteAf(TT) 
  #switch(TT)
    #range(-99999,Tp)
      #local Af=0.00001;
      #break
    #range(Tp,TI1)
      #local Fd=max(0,vlength(Ang)-Acc0)*3200;
      #local CdA=Fd/Qbar;
      #local Cd=1.222;
      #local AA=CdA/Cd;
      #local Af=AA/A0;
      #break
    #range(TI1,TI2)
      #local Fd=max(0,vlength(Ang)-Acc0)*3200;
      #local CdA=Fd/Qbar;
      #local Cd=1.222;
      #local AA=Linterp(TI1,CdA/Cd,TI2,A0,TT);
      #local Af=AA/A0;
      #break
    #range(TI2,99999999999)
      #local Af=1;
      #break
  #end  
  PrintNumber("TT: ",TT)
  #ifdef(Fd)
  PrintNumber("Acc0:  ",Acc0)
  PrintNumber("Fd: ",Fd)
  PrintNumber("CdA: ",CdA)
  PrintNumber("A0:  ",A0)
  PrintNumber("AA:  ",AA)
  #end
  PrintNumber("Af: ",Af)
  
  (Af)
#end

#declare Wedge0=intersection {
  plane {
    x,0
  }
  plane {
    -x,0
    rotate z*(360/80)
  }
  plane {
    y,0
    rotate z*45
    translate y*3
  }
}

#declare Wedge1=intersection {
  plane {
    x,0
  }
  plane {
    -x,0
    rotate z*(360/80)
  }
  plane {
    -y,0
    rotate z*45
    translate y*3
  }
  plane {
    y,0
    rotate z*45
    translate y*6
  }

}

#declare Wedge2=intersection {
  plane {
    x,0
  }
  plane {
    -x,0
    rotate z*(360/80)
  }
  plane {
    -y,0
    rotate z*45
    translate y*6
  }
}

#declare Wedge3=intersection {
  plane {
    x,0
  }
  plane {
    -x,0
    rotate z*(360/80)
  }
}

#ifndef(Message0)
  #declare Message0=array[8] {
  asc("D")-asc("A")+1,
  asc("A")-asc("A")+1,
  asc("R")-asc("A")+1,
  asc("E")-asc("A")+1,
  127,1023,1023,1023}
#end

#ifndef(Message1)
  #declare Message1=array[8] {
  asc("M")-asc("A")+1,
  asc("I")-asc("A")+1,
  asc("G")-asc("A")+1,
  asc("H")-asc("A")+1,
  asc("T")-asc("A")+1,
  asc("Y")-asc("A")+1,
  127,1023}
#end

#ifndef(Message2)
  #declare Message2=array[8] {
  asc("T")-asc("A")+1,
  asc("H")-asc("A")+1,
  asc("I")-asc("A")+1,
  asc("N")-asc("A")+1,
  asc("G")-asc("A")+1,
  asc("S")-asc("A")+1,
  127,1023}
#end

#ifndef(Message3)
  #declare Message3=array[8] {
  34,11,58,
  asc("N")-asc("A")+1,
  118,10,31,
  asc("W")-asc("A")+1
  }
#end

#macro Ring(Message,Object,extrarot)
  #local I=0;
  #while(I<8)
    #local Char=Message[I];
    PrintNumber("Char: ",Char)
    #local J=0;
    #local Bit=512;
    #while(Bit>0)
      PrintNumber("Bit: ",Bit)
      #if(bitwise_and(Char,Bit))
        #debug "Bit\n"
        object {Object
          #if(mod(J,2))
            scale <-1,1,1>
            rotate z*(360/80)
          #end
          rotate z*((I+extrarot)*10+J)*(360/80)
        }
      #end
      #local Bit=div(Bit,2);
      #local J=J+1;
    #end
    #local I=I+1;
  #end
#end

#declare WedgesDisk=union {
  Ring(Message0,Wedge0,0)
  Ring(Message1,Wedge1,4)
  Ring(Message2,Wedge2,10)
}

#declare WedgesBand=union {
  Ring(Message3,Wedge3,16)
}

#declare MSLChuteBand=cylinder {
  <0,0,0>,<0,0,Hb>,Rp
  open
  pigment { object {WedgesBand color rgb <1,1,1> color rgb <186/255,22/255,12/255>}}
}

#declare MSLChuteDisk= sphere {
  <0,0,0>,1
  scale <Rp,Rp,Hp>
  clipped_by {
    union {
      plane {
        z,0
      }
      cylinder {
        <0,0,0>,<0,0,Hp+0.5>,Rv
      }
      inverse
    }
  }
  pigment { object {WedgesDisk color rgb <1,1,1> color rgb <186/255,22/255,12/255>}}
}

#macro MSLChute(TT)
  PrintNumber("MSLChute(TT=",TT)
  union {
    #local Af=MSLChuteAf(TT);
    #local this_A=A0*Af;   
    #local this_Rp=max(0.25,sqrt(this_A/pi));
    #if(Af<Afcrit)
      #local this_Rp2=this_Rp;
    #else
      #local this_Rp2=Rcrit;
    #end
    #local this_Hp=sqrt(Lp*Lp-this_Rp2*this_Rp2);
    object {MSLChuteBridle}
    object {MSLChuteRiser translate z*(Hbr)}  
    object {MSLChuteSuspension translate z*(Hbr+Hr)}
    object {MSLChuteBand translate z*(Hbr+Hr+Hs)}
    object {MSLChuteGap translate z*(Hbr+Hr+Hs+Hb)}
    union {MSLChuteDisk scale <1,1,this_Hp/Hp> translate z*(Hbr+Hr+Hs+Hb+Hg)}
    scale <this_Rp/Rp,this_Rp/Rp,max(0.00001,min(1,(TT-Tp)/Tst))>
    rotate x*180
    translate DIMU_A_ofs
  }
#end

  
