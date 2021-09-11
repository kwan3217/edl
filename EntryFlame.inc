#local R1=1.25; //Radius of nose cap
#local R2=2.25; //Outer radius of main body
#local R3=0.5; //minor radius of shoulder
#local L4=50; //Length factor of trail
#declare Heat=0.15+Heat*1.6;
#local Theta=radians(70);
#local Theta2=min(Theta,atan(1/((Mach)-1)));
#local Axis=x;
#declare FlameGradient=pigment {
  gradient -x
  color_map {
    [0.1 color rgbf <1,1,1,0.1>]
    [0.11 color rgbf <1,1,0,0.25>]
    [0.12 color rgbf <1,0.5,0,0.75>]
    [0.18 color rgbf <1,1,1,1>]
  }
  translate x*(0.09+0.1*(1-Heat))
}
#declare EntryFlame=merge {
  sphere {
    0,R1
  }
  #local ConeFront=R1*sin(Theta);
  #local ConeBack=ConeFront-R2*cos(Theta);
  cone {
    Axis*ConeFront,R1*cos(Theta),Axis*ConeBack,R2   
  }
  #local TorusCenter=ConeBack-R3*(sin(Theta));
  #local TorusMajorRad=R2-R3*(cos(Theta));
  torus {
    TorusMajorRad,R3
    rotate z*90
    translate Axis*TorusCenter
  }
  #local ConeFront=TorusCenter+R3*sin(Theta2);
  #local ConeBack=ConeFront-L4;
  cone {
    0,TorusMajorRad+R3*cos(Theta2),Axis*ConeBack,L4*tan(Theta2)+R3*cos(Theta2)+TorusMajorRad
    matrix <cos(EntryAlpha),0,sin(EntryAlpha),
            0,1,0,
            0,0,1,
            0,0,0>
    translate Axis*ConeFront            
  }
/*  clipped_by {
    plane {
      -y,0
    }
  } */
  finish {ambient 1 diffuse 0}
  pigment {
    FlameGradient
    translate x*(ET-ET_ei)*1000/(L4*10)
    warp { 
      turbulence 0.02
    }
    translate -x*(ET-ET_ei)*1000/(L4*10)
    scale <(L4+10),1,1>
    translate x*R1
  }
  no_shadow
  rotate y*90
  rotate x*180
  translate -z*0.85
  translate DIMU_A_ofs
//  translate -z*0.65
}
