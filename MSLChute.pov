#include "KwanMath.inc"
#include "events_m20_recon.inc"
#include "MSLChute.inc"
#include "inc_m20_recon/step_07500.inc"

object {
  MSLChute(ET)
}

camera {
  up y
  right -x*image_width/image_height
  sky x

  location -z*20
  look_at -z*100
}

light_source {
  <20,20,20>*10000
  color rgb 1
}



