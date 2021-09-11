#include "KwanMath.inc"
#include "loc_look.inc"
#include "topo.inc"
#declare LevelColor=array[10] {
<0.2,0.2,0.2>,
<0.5,0.25,0>,
<1,0,0>,
<1,0.5,0>,
<1,1,0>,
<0,1,0>,
<0,0,1>,
<0.75,0,1>,
<0.75,0.75,0.75>,
<1,1,1>}

#furnsh "/mnt/big/home/chrisj/workspace/Data/spice/generic/lsk/naif0012.tls"
#include "events_m20_recon.inc"
#include concat("inc_m20_recon/step_",str(frame_number,-5,0),".inc")
#include "MSLRover.inc"
#include "MSLAeroshell2.inc"
#include "MSLDescentStage.inc"

#declare Vrel=vlength(Vds);
PrintNumber("Vrel: ",Vrel)
PrintNumber("Rho:  ",Rho)
#declare Mach=vlength(Vds)/Csound;  // Mach number
PrintNumber("Mach: ",Mach)
#declare Qbar=1/2*Rho*Vrel*Vrel;
#declare Scorch=Scorch_n/Scorch_max;  // Normalized integrated heating
PrintNumber("Scorch: ",Scorch)
PrintNumber("Qbar: ",Qbar)
#declare dynHeat=Cheat*pow(Vrel,nheat)*pow(Rho,mheat);
PrintNumber("dynHeat: ",dynHeat)
#declare Heat=dynHeat/dynHeat_max;
#declare EntryAlpha=radians(15);

#declare LanderOrient=MVT(LanderOrientM,<0,0,0>);
#declare Rcenter=Vtransform(LanderOrientM,Rds,<DIMU_A_ofs.x,DIMU_A_ofs.y,0.5>);
//#declare V=Vds;

//Calculate frame vectors from state, given above in step_%05d.inc
//All are in IAU_MARS (Mars rotating frame), which is parallel to
//the POV global frame (IAU_MARS is Mars-centered, POV global frame
//will be centered on R)
#declare Rhat=vnormalize(Rds);                 //Local vertical
#declare Vhat=vnormalize(Vds);                 //Normalized velocity
#declare Ehat=vnormalize(vcross(z   ,Rhat)); //Local horizontal east
#declare Nhat=vnormalize(vcross(Rhat,Ehat)); //Local horizontal north
#declare Hhat=vnormalize(vcross(Rhat,Vhat)); //Crossrange vector
#declare Qhat=vnormalize(vcross(Hhat,Rhat)); //Downrange vector
#declare Khat=vnormalize(vcross(Vhat,Hhat)); //Normal vector
#declare LH=vdot(Ang,Hhat);                  //component of non-gravitational acceleration in H direction
#declare LK=vdot(Ang,Khat);                  //component of non-gravitational acceleration in K direction
#declare Lhat=vnormalize(LH*Hhat+LK*Khat);   //Lift vector
#declare MarsR=vlength(Rland)-1000; //Sphere radius used for where there is no terrain model

#switch(ET)
  #range(0,ET_mortar - 20)
    #declare Angle = 0;
    #declare Dist = 20;
    #declare Height = 12;
    #declare Qbasis = Qhat;
    #declare Hbasis = Hhat;
    #declare Rbasis = Rhat;
    #break
  #range(ET_mortar - 20,ET_mortar - 15)
    #declare Angle = Linterp(ET_mortar - 20, 0, ET_mortar - 15, -pi / 2, ET);
    #declare Dist = Linterp(ET_mortar - 20, 20, ET_mortar - 15, 10, ET);
    #declare Height = Linterp(ET_mortar - 20, 12, ET_mortar - 15, 0, ET);
    #declare Qbasis = Qhat;
    #declare Hbasis = Hhat;
    #declare Rbasis = Rhat;
    #break
  #range(ET_mortar-15,ET_heatshield+100)
    #declare Angle = -pi / 2;
    #declare Dist = 10;
    #declare Height = 0;
    #declare Qbasis = Qhat;
    #declare Hbasis = Hhat;
    #declare Rbasis = Rhat;
    #break
  #range(ET_heatshield+100,ET_heatshield + 150)
    #declare Angle = -pi / 2;
    #declare Dist = 10;
    #declare Height = 0;
    #declare Qbasis = vnormalize(Linterp(ET_heatshield + 100, Qhat, ET_heatshield + 150, Ehat, ET));
    #declare Hbasis = vnormalize(Linterp(ET_heatshield + 100, Hhat, ET_heatshield + 150, Nhat, ET));
    #declare Rbasis = Rhat;
    #break
  #else
    #declare Angle = -pi / 2;
    #declare Dist = 10;
    #declare Height = 0;
    #declare Qbasis = Ehat;
    #declare Hbasis = Nhat;
    #declare Rbasis = Rhat;
#end
#declare Look_at = Rcenter;
#declare Location = Look_at - Dist * cos(Angle) * Qbasis - Dist * sin(Angle) * Hbasis + Height * Rbasis;
#declare Sky = Rhat;

PrintVector("Location:  ",Location)
PrintVector("LookAt:    ",Look_at)
PrintVector("Sky:       ",Sky)


sphere {
  0,MarsR
  pigment {image_map {png "MarsMapHuge.png" map_type spherical} scale <-1,1,1> rotate x*90}
  translate -Rcenter
}      

cone {
     -vnormalize(Vds)*0.3,0.1,
  Vds+vnormalize(Vds)*0.3,0.0
  pigment {color rgbt <1,0,1,0.5>}
  translate Rcenter-Rcenter
}

cylinder {
  Hhat*0.3,Hhat*1.3,0.01
  pigment {color rgbt <0,1,0>}
  translate Rcenter-Rcenter
}

cylinder {
  Khat*0.3,Khat*1.3,0.01
  pigment {color rgbt <0,0,1>}
  translate Rcenter-Rcenter
}

cylinder {
  Lhat*0.3,Lhat*1.3,0.01
  pigment {color rgbt <1,1,0>}
  translate Rcenter-Rcenter
}

cone {
  -vnormalize(Ang)*0.3    ,0.1,
  +vnormalize(Ang)*0.3+Ang/9.80665,0.0
  pigment {color rgb <1,0.5,0>}
  translate Rcenter-Rcenter
}

#if(frame_number<6000)
cone {
  vnormalize(Rland)*1000,0,vnormalize(Rland)*5000,1000
  pigment {color rgb <0,0,1>}
  translate Rland-Rcenter
}
#end

cone {
  Rland+Ehat*1000,0,Rland+Ehat*5000,1000
  pigment {color rgb <1,0,0>}
  translate -Rcenter
}

cone {
  Rland-Ehat*1000,0,Rland-Ehat*5000,1000
  pigment {color rgb <0,1,1>}
  translate -Rcenter
}

cone {
  Rland+Nhat*1000,0,Rland+Nhat*5000,1000
  pigment {color rgb <0,1,0>}
  translate -Rcenter
}

cone {
  Rland-Nhat*1000,0,Rland-Nhat*5000,1000
  pigment {color rgb <1,0,1>}
  translate -Rcenter
}

union {
  object {EntryBalanceMass(0)} translate -0.35*z rotate -8.5*y translate -2.06*x rotate (BLinterp(0,0,10,-3600,ET-ET_ebm0-0.2)+8.5)*y
  rotate z*30
  transform {LanderOrient} translate Rebm0-Rcenter
}
cone {Rebm0,0.1,Rebm0+(Vebm0-Vds),0 pigment {color rgb <0.2,0.2,0.2>} translate -Rcenter}

union {
  object {EntryBalanceMass(0)} translate -0.35*z rotate -8.5*y translate -2.06*x rotate (BLinterp(0,0,10,-3600,ET-ET_ebm1-0.2)+8.5)*y
  rotate z*-30
  transform {LanderOrient} translate Rebm1-Rcenter
}
cone {Rebm1,0.1,Rebm1+(Vebm1-Vds),0 pigment {color rgb <0.5,0.25,0>} translate -Rcenter}

union {
  object {EntryBalanceMass(0)} translate -0.35*z rotate -8.5*y translate -2.06*x rotate (BLinterp(0,0,10,-3600,ET-ET_ebm2-0.2)+8.5)*y
  rotate z*20
  transform {LanderOrient} translate Rebm2-Rcenter
}
cone {Rebm2,0.1,Rebm2+(Vebm2-Vds),0 pigment {color rgb <1,0,0>} translate -Rcenter}

union {
  object {EntryBalanceMass(0)} translate -0.35*z rotate -8.5*y translate -2.06*x rotate (BLinterp(0,0,10,-3600,ET-ET_ebm3-0.2)+8.5)*y
  rotate z*-20
  transform {LanderOrient} translate Rebm3-Rcenter
}
cone {Rebm3,0.1,Rebm3+(Vebm3-Vds),0 pigment {color rgb <1,0.5,0>} translate -Rcenter}

union {
  object {EntryBalanceMass(0)} translate -0.35*z rotate -8.5*y translate -2.06*x rotate (BLinterp(0,0,10,-3600,ET-ET_ebm4-0.2)+8.5)*y
  rotate z*10
  transform {LanderOrient} translate Rebm4-Rcenter
}
cone {Rebm4,0.1,Rebm4+(Vebm4-Vds),0 pigment {color rgb <1,1,0>} translate -Rcenter}

union {
  object {EntryBalanceMass(0)} translate -0.35*z rotate -8.5*y translate -2.06*x rotate (BLinterp(0,0,10,-3600,ET-ET_ebm5-0.2)+8.5)*y
  rotate z*-10
  transform {LanderOrient} translate Rebm5-Rcenter
}
cone {Rebm5,0.1,Rebm5+(Vebm5-Vds),0 pigment {color rgb <0,1,0>} translate -Rcenter}

union {
  Backshell(Scorch)
  cylinder {0,x*3,0.005 pigment {color rgb x}}
  cylinder {0,y*3,0.005 pigment {color rgb y}}
  cylinder {0,z*3,0.005 pigment {color rgb z}}
  transform {LanderOrient}
  translate Rbs-Rcenter
}

union {
  Heatshield(Scorch)
  transform {LanderOrient}
  translate Rhs-Rcenter
}

#if(Heat>0.01)
#include "EntryFlame.inc"

union {
  object {EntryFlame}
  transform {LanderOrient}
  translate Rbs-Rcenter
}
#end

#declare RappelTime=ET-ET_rappel0;
#declare RoverCf=BLinterp(5,1,6,0,RappelTime);
#declare RoverCa=BLinterp(5,1,6,0,RappelTime);
#declare RoverAf=BLinterp(7,1,7.5,0,RappelTime);
#declare LookOffset=BLinterp(0,0,(ET_rappel1-ET_rappel0),-3.5*Rhat,RappelTime);
#declare Location=Location+LookOffset;
#declare Look_at=Look_at+LookOffset;

union {
  MSLRover(1,1,1)
  transform {LanderOrient}
  translate Rrover-Rcenter
}

union {
  MSLDescentStage(Fstraight/3000,Fcant/3000)
  transform {LanderOrient}
  translate Rds-Rcenter
}

#declare HudOrient=LocLookT(Location,Look_at,Sky);

#macro HUDtex(Color)
  texture {
    pigment {color rgbf Color+<0,0,0,0.7>}
    finish {ambient 1 diffuse 0}
  }
#end

#declare TapeCount=0;
#macro TapeMeter(Label,Vtxt,Val0,Val1,Val,ColorP,ColorN)
  #local Color_dim=0.3;
  #local X0=TapeCount*0.05-0.85;
  #declare TapeCount=TapeCount+1;
  #local X1=X0+0.04;
  #local Y0=-0.45;
  #local Y1= 0.45;
  #local Ytxt=Linterp(0,-0.48,1,0.45,Vtxt);
  #local Ylbl=Linterp(0,-0.465,1,0.465,Vtxt);
  #if(Val>0)
    box {<X0,Y0                          ,0>,<X1,Linterp(Val0,Y0,Val1,Y1,0  ),0> HUDtex(ColorN*Color_dim)}
    box {<X0,Linterp(Val0,Y0,Val1,Y1,0)  ,0>,<X1,Linterp(Val0,Y0,Val1,Y1,Val),0> HUDtex(ColorP)}
    box {<X0,Linterp(Val0,Y0,Val1,Y1,Val),0>,<X1,Y1                          ,0> HUDtex(ColorP*Color_dim)}
    text {ttf "verdana.ttf" str(abs(Val),0,0) 0 0 scale 0.02 translate <X0,Ytxt,0> HUDtex(ColorP)}
    text {ttf "verdana.ttf" Label             0 0 scale 0.02 translate <X0,Ylbl,0> HUDtex(ColorP)}
  #else
    box {<X0,Y0                          ,0>,<X1,Linterp(Val0,Y0,Val1,Y1,Val),0> HUDtex(ColorN*Color_dim)}
    box {<X0,Linterp(Val0,Y0,Val1,Y1,Val),0>,<X1,Linterp(Val0,Y0,Val1,Y1,0  ),0> HUDtex(ColorN)}
    box {<X0,Linterp(Val0,Y0,Val1,Y1,0  ),0>,<X1,Y1                          ,0> HUDtex(ColorP*Color_dim)}
    text {ttf "verdana.ttf" str(abs(Val),0,0) 0 0 scale 0.02 translate <X0,Ytxt,0> HUDtex(ColorN)}
    text {ttf "verdana.ttf" Label             0 0 scale 0.02 translate <X0,Ylbl,0> HUDtex(ColorN)}
  #end
#end

#macro PtMeter(Val0,Val1,Val,Color)
  #local X0=TapeCount*0.05-0.85;
  #local X1=X0+0.04;
  #local Y0=-0.45;
  #local Y1= 0.45;
  box {<X0,Linterp(Val0,Y0,Val1,Y1,Val)-0.001,-0.001>,<X1,Linterp(Val0,Y0,Val1,Y1,Val)+0.001,0.001>HUDtex(Color)}
#end

PrintVector("Location: ",Location)
PrintVector("Look_at: ",Look_at)
PrintVector("Sky: ",Sky)
#declare HudOrientMatrix=array[3][3];
LocLook(Location,Look_at,Sky,HudOrientMatrix)
PrintMatrix("HudOrientMatrix",HudOrientMatrix)

object {
  object {
    union {
      TapeMeter("Alt" ,0,    0,150000,vlength(Rds)-vlength(Rland),<1   ,1  ,1>,0)
      TapeMeter("Vrel",1,    0,  5500,vlength(Vds)            ,<0.75,0  ,1>,0)
      #ifdef(Ang)
      TapeMeter("Acc" ,0,    0,   120,vlength(Ang)          ,<1   ,0.5,0>,0)
      #end
        PtMeter(             0,dynHeat_max,dynHeat,<1,0.5,0>)
      TapeMeter("Heat" ,0,    0,  100,Scorch*100          ,<0.5   ,0.25,0>,0)
      TapeMeter("hdot",1,-1800,   200,vdot(Vds,Rhat)          ,<0   ,1  ,0>,<1,1,1>)
      #if(frame_number<6000)
      TapeMeter("fpa" ,0,  -18,     2,degrees(asin(vdot(Vds,Rhat)/vlength(Vds))),<0   ,1  ,0>,<0,1,1>)
      #end
      text {ttf "verdana.ttf" concat("ERT:   ",timout(ET+LT,"YYYY MON DD HR:MN:SC.### UTC ::UTC")) 0 0 scale 0.02 translate <0.5,-0.495,0> HUDtex(<1,1,1>)}
      text {ttf "verdana.ttf" concat("SCET: ",timout(ET,"YYYY MON DD HR:MN:SC.### UTC ::UTC")) 0 0 scale 0.02 translate <0.5,-0.48,0> HUDtex(<1,1,1>)}
      text {ttf "verdana.ttf" str(frame_number,5,0) 0 0 scale 0.02 translate <0.65,-0.465,0> HUDtex(<1,1,1>)}
      scale <-1,1,1>
      translate z
    }
    no_shadow
    transform{HudOrient}
  }
  translate -Rcenter
}

light_source {
  Sun-Rcenter color rgb 1.5
}

//#declare Location=Look_at+Rhat*7;

light_source {
  Location-Rcenter
  color rgb 1
}

#local Sky=Rhat;

camera {
  up y
  right -x*image_width/image_height
  sky Sky
  //angle 20
  location Location-Rcenter
  look_at Look_at-Rcenter
}

