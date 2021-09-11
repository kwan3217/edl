#include "KwanMath.inc"
#include "topo.inc"
#include concat("inc_m20/frame_",str(frame_number,-5,0),".inc")

//Calculate frame vectors from state, given above in frame_%05d.inc
#declare Rhat=vnormalize(R);                 //Local vertical
#declare Vhat=vnormalize(V);                 //Normalized velocity
#declare Ehat=vnormalize(vcross(z   ,Rhat)); //Local horizontal east
#declare Nhat=vnormalize(vcross(Rhat,Ehat)); //Local horizontal north
#declare Hhat=vnormalize(vcross(Rhat,Vhat)); //Crossrange vector
#declare Qhat=vnormalize(vcross(Hhat,Rhat)); //Downrange vector
#declare Khat=vnormalize(vcross(Vhat,Hhat)); //Normal vector
//#declare LH=vdot(Ang,Hhat);                  //component of non-gravitational acceleration in H direction
//#declare LK=vdot(Ang,Khat);                  //component of non-gravitational acceleration in K direction
//#declare Lhat=vnormalize(LH*Hhat+LK*Khat);   //Lift vector
#declare MarsR=vlength(R1)-1000; //Sphere radius used for where there is no terrain model

#declare Mach=vlength(V)/Csound;  // Mach number
#declare Cheat=2.24e-11;  // Heating coefficient
#declare mheat=0.879;  // Heating rho power
#declare nheat=4.22;  // Heating vrel power
#declare Qbar=Rho*pow(vlength(V),2)/2;  // Dynamic pressure in Pa
#declare dynHeat=Cheat*pow(Rho,mheat)*pow(vlength(V),nheat);  // Heating rate, in power per unit area
#declare Scorch=Scorch_n/Scorch_max;  // Normalized integrated heating

#include "MSLAeroshell2.inc"
#include "MSLDescentStage.inc"
#include "MSLRover.inc"

sphere {
  0,MarsR
  pigment {image_map {png "MarsMapHuge.png" map_type spherical} scale <-1,1,1> rotate x*90}
  translate -R
}      
/*
cone {
  R  -vnormalize(V)*0.3,0.1,
  R+V+vnormalize(V)*0.3,0.0
  pigment {color rgbt <1,0,1,0.5>}
  translate -R
}
*/

cylinder {
  R+Hhat*0.3,R+Hhat*1.3,0.01
  pigment {color rgbt <0,1,0>}
  translate -R
}

cylinder {
  R+Khat*0.3,R+Khat*1.3,0.01
  pigment {color rgbt <0,0,1>}
  translate -R
}

/*
cylinder {
  R+Lhat*0.3,R+Lhat*1.3,0.01
  pigment {color rgbt <1,1,0>}
  translate -R
}
*/

#if(frame_number<6000)
cone {
  R1+vnormalize(R1)*1000,0,R1+vnormalize(R1)*5000,1000
  pigment {color rgb <0,0,1>}
  translate -R
}
#end

cone {
  R1+Ehat*1000,0,R1+Ehat*5000,1000
  pigment {color rgb <1,0,0>}
  translate -R
}

cone {
  R1-Ehat*1000,0,R1-Ehat*5000,1000
  pigment {color rgb <0,1,1>}
  translate -R
}

cone {
  R1+Nhat*1000,0,R1+Nhat*5000,1000
  pigment {color rgb <0,1,0>}
  translate -R
}

cone {
  R1-Nhat*1000,0,R1-Nhat*5000,1000
  pigment {color rgb <1,0,1>}
  translate -R
}

/*
cone {
  -vnormalize(Ang)*0.3    ,0.1,
  +vnormalize(Ang)*0.3+Ang/9.80665,0.0
  pigment {color rgb <1,0.5,0>}
  translate -R
}
*/

/*
union {
  object {MSLDescentStage(0,0)}
  transform {LanderOrient}
}
*/

union {
  object {EntryBalanceMass(0)} translate -0.35*z rotate -8.5*y translate -2.06*x rotate 8.5*y
  rotate z*30
  transform {LanderOrient} translate Rebm0 translate -R
}
cone {Rebm0,0.1,Rebm0+(Vebm0-V),0 pigment {color rgb <0.2,0.2,0.2>} translate -R}

union {
  object {EntryBalanceMass(0)} translate -0.35*z rotate -8.5*y translate -2.06*x rotate 8.5*y
  rotate z*-30
  transform {LanderOrient} translate Rebm1 translate -R
}
cone {Rebm1,0.1,Rebm1+(Vebm1-V),0 pigment {color rgb <0.5,0.25,0>} translate -R}

union {
  object {EntryBalanceMass(0)} translate -0.35*z rotate -8.5*y translate -2.06*x rotate 8.5*y
  rotate z*20
  transform {LanderOrient} translate Rebm2 translate -R
}
cone {Rebm2,0.1,Rebm2+(Vebm2-V),0 pigment {color rgb <1,0,0>} translate -R}

union {
  object {EntryBalanceMass(0)} translate -0.35*z rotate -8.5*y translate -2.06*x rotate 8.5*y
  rotate z*-20
  transform {LanderOrient} translate Rebm3 translate -R
}
cone {Rebm3,0.1,Rebm3+(Vebm3-V),0 pigment {color rgb <1,0.5,0>} translate -R}

union {
  object {EntryBalanceMass(0)} translate -0.35*z rotate -8.5*y translate -2.06*x rotate 8.5*y
  rotate z*10
  transform {LanderOrient} translate Rebm4 translate -R
}
cone {Rebm4,0.1,Rebm4+(Vebm4-V),0 pigment {color rgb <1,1,0>} translate -R}

union {
  object {EntryBalanceMass(0)} translate -0.35*z rotate -8.5*y translate -2.06*x rotate 8.5*y
  rotate z*-10
  transform {LanderOrient} translate Rebm5 translate -R
}
cone {Rebm5,0.1,Rebm5+(Vebm5-V),0 pigment {color rgb <0,1,0>} translate -R}

union {
  object {Backshell}
  cylinder {0,x*3,0.005 pigment {color rgb x}}
  cylinder {0,y*3,0.005 pigment {color rgb y}}
  cylinder {0,z*3,0.005 pigment {color rgb z}}
  transform {LanderOrient}
  translate Rbs
  translate -R
}

union {
  object {Heatshield transform {LanderOrient}}
  translate Rhs
  translate -R
}

#declare SkycraneStartFrame=9630;
#declare RappelTime=(frame_number-SkycraneStartFrame)/24;
#declare MaxRappelDist=7.0;
#declare RappelDist=BLinterp(0,0,7.5,MaxRappelDist,RappelTime);
#declare RoverCf=BLinterp(5,1,6,0,RappelTime);
#declare RoverCa=BLinterp(5,1,6,0,RappelTime);
#declare RoverAf=BLinterp(7,1,7.5,0,RappelTime);

union {
  MSLRover(1,1,1)
  translate -z*RappelDist
  transform {LanderOrient}
}

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

object {
  object {
    union {
      TapeMeter("Alt" ,0,    0,150000,vlength(R)-vlength(R1),<1   ,1  ,1>,0)
      TapeMeter("Vrel",1,    0,  5500,vlength(V)            ,<0.75,0  ,1>,0)
      #ifdef(Ang)
      TapeMeter("Acc" ,0,    0,   120,vlength(Ang)          ,<1   ,0.5,0>,0)
      #end
        PtMeter(             0,dynHeat_max,dynHeat,<1,0.5,0>)
      TapeMeter("Heat" ,0,    0,  100,Scorch*100          ,<0.5   ,0.25,0>,0)
      TapeMeter("hdot",1,-1800,   200,vdot(V,Rhat)          ,<0   ,1  ,0>,<1,1,1>)
      #if(frame_number<6000)
      TapeMeter("fpa" ,0,  -18,     2,degrees(asin(vdot(V,Rhat)/vlength(V))),<0   ,1  ,0>,<0,1,1>)
      #end
      text {ttf "verdana.ttf" etcal(ET) 0 0 scale 0.02 translate <0.6,-0.48,0> HUDtex(<1,1,1>)}
      text {ttf "verdana.ttf" str(frame_number,5,0) 0 0 scale 0.02 translate <0.65,-0.465,0> HUDtex(<1,1,1>)}
      scale <-1,1,1>
      translate z
    }
    no_shadow
    transform{HudOrient}
  }
  translate -R
}

light_source {
  Sun-R color rgb 1.5
}

//#declare Location=Look_at+Rhat*7;

light_source {
  Location-R
  color rgb 1
}

camera {
  up y
  right -x*image_width/image_height
  sky Sky
  //angle 20
  location Location-R
  look_at Look_at-R
}

