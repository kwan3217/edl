#include "metals.inc"

#declare ScorchPig=pigment {
  gradient z
  #local Stretch=1;
  #local F=0.0;
  color_map {
    [0.91061/2.876                      color rgbf Linterp(0,<1,1,1,F>,1,<0.5,0.375,0.25,F>,Scorch)]
    [0.91061/2.876+Scorch*1/25/Stretch  color rgbf Linterp(0,<1,1,1,F>,1,<1.0,0.750,0.50,F>,Scorch)]
    [0.91061/2.876+Scorch*2/25/Stretch  color rgbf <1.0,1.000,1.00,F>]
  }
  scale <1,1,2.876>
  translate -z*0.91061
  warp { turbulence <0,0,0.1> octaves 6 lambda 2.0 omega 0.8 }
  scale <1,1,5*Stretch>
  translate  z*0.91061
}

#declare Heatshield = union {
   //*PMName Heatshield
   
   difference {
      //*PMName Heatshield Shoulder
      
      torus {
         2.124, 0.126
         rotate x*90
         translate z*0.83496
      }
      
      cylinder {
         <0, 0, 0.91061>, <0, 0, 1>, 2.25
         //pigment {color rgb <0.3,0.3,0.3>}
      }
   }
   
   cone {
      //*PMName Heatshield Cone
      <0, 0, 0.71656>, 2.1671,
      <0, 0, 0.06785>, 0.38477
   }
   
   intersection {
      cylinder {
         <0, 0, 0>, <0, 0, 0.07>, 0.38477
         scale 1
         rotate <0, 0, 0>
         translate <0, 0, 0>
      }
      
      sphere {
         <0, 0, 1.125>, 1.125
      }
   }
   
   material {
      texture {
         pigment {
            color rgb BLinterp(0,<164,150,115>/255,1,<0.1,0.1,0.1>,Scorch)
         }
         
      }
   }
   translate z*(-1)
}

#declare EBMCyl=cylinder {
  <1.96265,0,0>,<2.21301,0,0>,0.108/2
  rotate y*8.5
  translate z*1.35
}

#macro EntryBalanceMass(Eject)
  intersection {
    cone {
      <0, 0, 1.878>, 1.5005,
      <0, 0, 0.91061>, 2.2248
    }
    object {EBMCyl}
    translate -z
    texture {T_Chrome_1A}
  }
#end

#declare SepSpringDoor=intersection {
  union {
    cylinder {<2.5,0,0  >,<-2.5,0,0  >,0.1}
    cylinder {<2.5,0,0.1>,<-2.5,0,0.1>,0.1}
    box {<2.5,-0.1,0>,<-2.5,0.1,0.1>}
    rotate -y*45
    translate <2,0,1>
  }
  cone {
    //*PMName FwdBackshell
    <0, 0, 1.878>, 1.5105,
    <0, 0, 0.91061>, 2.2348
  }
  texture {T_Chrome_5A}
}

#declare Backshell = difference {
   union {
     #local I=0;
     #while(I<360)
       object {SepSpringDoor rotate z*I}
       #local I=I+40;
     #end
      //*PMName Outer shell
      
      cone {
         //*PMName AftBackshell
         <0, 0, 2.296>, 0.7645,
         <0, 0, 1.878>, 1.5005
      }
      intersection {
        cone {
         <0, 0, 2.306>, 0.7645,
         <0, 0, 1.888>, 1.5005
        }
        #local X=0.9;
        union {
          cylinder {
            < X, X,0>,< X, X,2.296>,0.15
          }
          cylinder {
            < X,-X,0>,< X,-X,2.296>,0.15
          }
          cylinder {
            <-X,-X,0>,<-X,-X,2.296>,0.15
          }
          cylinder {
            <-X, X,0>,<-X, X,2.296>,0.15
          }
        }
        texture {T_Chrome_2A}
      }
      
      difference {
         cone {
           //*PMName FwdBackshell
           <0, 0, 1.878>, 1.5005,
           <0, 0, 0.91061>, 2.2248
         }
         object {
           EBMCyl
           rotate z*10
         }
         object {
           EBMCyl
           rotate z*20
         }
         object {
           EBMCyl
           rotate z*30
         }
         object {
           EBMCyl
           rotate -z*10
         }
         object {
           EBMCyl
           rotate -z*20
         }
         object {
           EBMCyl
           rotate -z*30
         }
      }
      
      cone {
         //*PMName Wraparound Antenna
         <0, 0, 2.875>, 0.3655,
         <0, 0, 2.356>, 0.6535
      }
      
      cone {
         //*PMName Cruise Stage Interface
         <0, 0, 2.356>, 0.6535,
         <0, 0, 2.296>, 0.7645
         
         material {
            texture {
               finish {
                  phong 1
                  metallic 1
               }
               
               pigment {
                  color rgb <1, 0.992157, 0.411765>
               }
            }
         }
      }
   }
   
   union {
      //*PMName Inner shell
      
      cone {
         //*PMName AftBackshell
         <0, 0, 2.296>, 0.7545,
         <0, 0, 1.8779>, 1.4905
      }
      
      cone {
         //*PMName FwdBackshell
         <0, 0, 1.8781>, 1.4905,
         <0, 0, 0.91051>, 2.2148
      }
      
      cone {
         //*PMName Wraparound Antenna
         <0, 0, 2.874>, 0.3555,
         <0, 0, 2.3559>, 0.6435
      }
      
      cone {
         //*PMName Cruise Stage Interface
         <0, 0, 2.3561>, 0.6435,
         <0, 0, 2.2959>, 0.7545
      }
   }
   
   material {
      texture {
         pigment {ScorchPig}
         
         finish {
            phong 0 //1
            phong_size 60
         }
      }
   }
   translate z*(-1)
}
