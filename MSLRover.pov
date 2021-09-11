//Show off the MSL rover model by itself
#include "KwanMath.inc"
#include "MSLRover.inc"
#include "MSLDescentStage.inc"
#declare Scorch=0;
#include "MSLAeroshell2.inc"

camera {
  orthographic
  up y
  right -x*image_width/image_height
  angle 55
  sky -z
  location <0,-5,-1>
  look_at <0,0,-1>
}


/*
plane {
  x,0
  pigment {checker}
}
*/


cylinder {0,x,0.1 pigment {color rgb x}}
cylinder {0,y,0.1 pigment {color rgb y}}
cylinder {0,-z,0.1 pigment {color rgb 1-z}}

light_source {
  <0,-20,0>*1000
  color rgb 1.5
}

MSLRover(1,1,1)
union {
  MSLDescentStage(0,0)
  translate -DIMU_A_ofs
}

object {Backshell(0) translate -DIMU_A_ofs}
object {Heatshield(0) translate -DIMU_A_ofs}
