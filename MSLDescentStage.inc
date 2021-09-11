#include "metals.inc"
#declare MLESeed=seed(frame_number);
#local coreR1=0.75;
#local coreR2=0.65;
#local ST=sqrt(3)/2;
#declare Core=prism {
  linear_spline
  0,0.8,14
  coreR1*<ST, 0.5>,coreR1*<0, 1>,coreR1*<-ST, 0.5>
  coreR1*<-ST,-0.5>,coreR1*<0,-1>,coreR1*<ST,-0.5>,coreR1*<ST, 0.5>
  coreR2*<ST, 0.5>,coreR2*<0, 1>,coreR2*<-ST, 0.5>
  coreR2*<-ST,-0.5>,coreR2*<0,-1>,coreR2*<ST,-0.5>,coreR2*<ST, 0.5>
  rotate x*90
}
#declare OutriggerOut=prism {
  linear_spline
  0,0.05,5
  <0,0>,<0.0,0.75>,<0.72,0.30>,<0.72,0.22>,<0,0>
//  rotate -y*90
  rotate z*60
  translate <coreR1*ST,coreR1*0.5,0>
}

#declare OutriggerIn=prism {
  linear_spline
  -0.05,0.0,5
  <0,0>,<0.0,0.75>,<0.78,0.30>,<0.78,0.22>,<0,0>
//  rotate -y*90
  rotate z*44
  translate <0,coreR1*1,0>
}

#macro MLE(Throttle,FlameProgress)
  union {
    cylinder {<0,0,0>,<0,0,-0.23>,0.075 texture {T_Chrome_5E}}
    sphere {
      <0,0,-0.27>,0.3
      scale <1,1,10>
      clipped_by {plane {-z,0}}
      scale 0.5
      translate -z*0.37
      texture {T_Chrome_1A}
    }
  #if(Throttle>0)
    sphere {
      0,0.15
      translate -z*0.15
      finish {ambient 1 diffuse 0}
      pigment {
        gradient -z
        color_map {
          [0.1 color rgbt <1,1,1,1-0.9*Throttle>]
          [0.2 color rgbt <1,1,0,1-0.75*Throttle>]
          [0.4 color rgbt <1,1,1,1-0*Throttle>]
        }
        translate  z*0.1
        translate  z*FlameProgress*3
        warp { 
          turbulence 0.1*<0,0,1> octaves 6 lambda 4 omega 0.7
        }
        translate -z*FlameProgress*3
        scale <1,1,BLinterp(0,0.5,1,0.75,Throttle)>
      }
      scale <1,1,8>
      no_shadow
      translate -z*(0.25)
    }
  #end
  }
#end


#declare OutriggerPlatform=prism {
  linear_spline
  0.22,0.30,5
  <0.46,1.14>,<0.88,0.87>,<0.98,1.02>,<0.59,1.27>,<0.46,1.14>
  rotate x*90
  scale <1,-1,1>
}
#declare PropTank=union {
  sphere {
    0,0.28 translate z*0.3
  }
  sphere {
    0,0.28 translate z*0.5
  }
  cylinder {
    <0,0,0.3>,<0,0,0.5>,0.28
  }
  texture {T_Chrome_5E normal {wrinkles 1 scale 0.03}}
}

#macro Outrigger(StraightThrottle,CantThrottle)
  union {
    object {OutriggerOut}
    object {OutriggerIn}
    object {OutriggerPlatform}
    object {MLE(CantThrottle,frame_number+100) rotate x*15 translate <0.81,1,0.18>}
    object {MLE(StraightThrottle,frame_number) translate <0.61,1.14,0.18>}
  }
#end

#local PlanReference=box {
  <0,0,0>,<1,1,0>
//  pigment {image_map {png "Descent Stage Plan.png"}}
  finish {ambient 1 diffuse 0 specular 0}
  scale <-1192/1042,1,1>
  translate <752/1042,-522/1042,0>
  scale 2.7
}

#local ElevReference=box {
  <0,0,0>,<1,1,0>
//  pigment {image_map {png "Descent Stage Elevation.png"}}
  finish {ambient 1 diffuse 0 specular 0}
  scale <-1547/882,1,1>
  rotate x*90
  translate <918/882,0,-402/882>
  scale 1.66
}

#macro MSLDescentStage(StraightThrottle,CantThrottle) 
  union {
    object {Core}
    object {Outrigger(StraightThrottle,CantThrottle)}
    object {Outrigger(StraightThrottle,CantThrottle) scale <-1, 1,1>}
    object {Outrigger(StraightThrottle,CantThrottle) scale <-1,-1,1>}
    object {Outrigger(StraightThrottle,CantThrottle) scale < 1,-1,1>}
    object {PropTank translate x*0.6}
    object {PropTank translate x*0.6 rotate  z*120}
    object {PropTank translate x*0.6 rotate -z*120}
    object {PropTank scale 0.72 translate x*0.86 rotate  z*52}
    object {PropTank scale 0.72 translate x*0.86 rotate -z*52}
//    cylinder {0,x*5,0.03 pigment {color rgb x}}
//    cylinder {0,y*5,0.03 pigment {color rgb y}}
//    cylinder {0,-z*5,0.03 pigment {color rgb z}}
//  object {PlanReference translate  z*100}
//  object {ElevReference translate -y*100}
    texture {T_Chrome_5A}
    translate z*0.5
    rotate x*180
    translate -z*0.95
    translate DIMU_A_ofs
  }
#end

