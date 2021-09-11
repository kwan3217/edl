#include "topo.inc"

#include concat("inc/tile",str(frame_number,-3,0),".inc")

cylinder {0,x*1000,0.01 pigment {color rgb x}}
cylinder {0,y*1000,0.01 pigment {color rgb y}}
cylinder {0,z*1000,0.01 pigment {color rgb z}}

light_source {
  <-20,-20,20>*10000
  color rgb 1
}

camera {
  up y
  right -x*image_width/image_height
  sky z
  location Location
  look_at Look_at
}


