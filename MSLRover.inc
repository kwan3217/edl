#version 3.6;

//Stowed offset between descent stage IMU (DIMU-A) and rover
//origin before rappel. Add this offset to DIMU-A position
//to get rover position, or subtract from rover to get DIMU-A
#declare DIMU_A_ofs=<-0.958,0.641,1.826>;


#macro half_torus(r_maj,r_min) 
  intersection {
    torus {r_maj,r_min rotate x*90}
    plane {-x,0}
  }
#end
    
#macro quarter_torus(r_maj,r_min)
  intersection {
    torus {r_maj,r_min rotate x*90}
    plane {-x,0}
    plane {-y,0}
  }
#end

#macro trapezoid(xx,yy,z1,z2) 
  prism { 0 ,yy , 4
    <0,0>,
    <xx,0>,
    <xx,z2>,
    <0,z1>
//    scale <1,-1,1>
//    rotate x*90
  }
#end

#declare tailpiece=union {
  trapezoid(1.31,0.78,1.96,1.96+1.31)
  union {
    object {
      trapezoid(0.4,0.78,0.2,0.01)
      scale <1,1,-1>              
    }
    cylinder {
      <0,0,0>,<0,0,0.01>,0.78/2
      translate <0.78/2,0.78/2,0>
    }
    translate <1.31,0.0,1.96+1.31>
  }
}

#macro Cube(V)
  box {-V/2,V/2}
#end

#declare rtg_radiator_panel=union {
  union {
    union {
      half_torus(0.2,0.02)
      object {
        quarter_torus(0.2,0.02)
        translate <0.05,0.05,0>
      }
      object {
        quarter_torus(0.2,0.02)
        rotate -z*90
        translate <0.05,-0.05,0>
      }
      cylinder {
        <0.25,-0.05,0>,<0.25,0.05,0>,0.02
      }
      translate <3.2,0,0>  
    
    }
    cylinder {<0,-0.25,0>,<3.25,-0.25,0>,0.02}
    cylinder {<0, 0.25,0>,<3.25, 0.25,0>,0.02}
    cylinder {<0,-0.2,0>,<3.2,-0.2,0>,0.02}
    cylinder {<0, 0.2,0>,<3.2, 0.2,0>,0.02}
    pigment {color rgb <1,1,0>}
  }
  box {
    <0,0,0>,<3.6,0.85,0.02>
    translate <0.01,-0.85/2,-0.01>
  }
}

#declare rtg_radiator_half=union {
  object {rtg_radiator_panel translate <0, 0.85/2,0> }
  object {rtg_radiator_panel translate <0,-0.85/2,0> }
  object {rtg_radiator_panel translate <0, 0.85/2,0> rotate  x*45 translate <0, 0.85,0> }
  object {rtg_radiator_panel translate <0,-0.85/2,0> rotate -x*45 translate <0,-0.85,0> }
  translate <0,0,-1.5>
}  

#declare rtg_radiator=union {
  union {
    object {rtg_radiator_half scale <1,1,-1>}
    object {rtg_radiator_half}
    rotate x*90
  }
  intersection {
    Cube(<0.01,3.0,3.0>)
    object { Cube(<0.03,3.30,3.30>) rotate x*45 }
    translate <3.6,0,0>
  }
}

#declare rtg=union {
  cylinder {<0,0,0>,<3.41,0,0>,1.255/2}
  #declare I=0;
  #while(I<=359) 
    #declare J=0;
    #while(J<=3)
      box {<0,0,0>,<0.64,1.225+0.66*2,0.02> translate <0.77+0.65*J,-(1.225+0.66*2)/2,-0.01> rotate x*(22.5+I)}
      #declare J=J+1;
    #end
    #declare I=I+45;
  #end
}

#declare rtg_asy =union {
  object {rtg}
  object {rtg_radiator}
  translate <0,0,-0.8501>
  rotate -y*23
}

#macro pancam_mast(rx) 
  box {<0,0,0>,<0.66,0.66,0.39> translate <-0.33,-0.33,0> }
  cylinder {<-0.33,0,0.39>,<0.33,0,0.39>,0.33 }
  union {
    cylinder {<0,0,0.39>,<0,0,0.39+1.85>,0.33}
    cylinder {<0,0,0.39+1.85>,<0,0,0.39+1.85+0.26>,1.18/2}
    cylinder {<0,0,0.39+1.85+0.26>,<0,0,0.39+1.85+0.26+0.65>,0.33}
    object {Cube(<1.63,1.11,1.11>) translate <0,0,0.39+1.85+0.26+0.65+1.11/2> }
    translate <0,0,-0.39>
    rotate x*rx
    translate <0,0,0.39>
  }
#end

#declare wheel=difference {
  cylinder {<0,0,0>,<0,1.31,0>,1.06}
  cylinder {<0,-0.01,0>,<0,1.32,0>,1.0}
  pigment {color rgb <1,1,1>*0.1}
}
                    
//cf - center pivot, front part 
//ca - center pivot, aft part                    
//a  - aft pivot, aft part
//all values run from 0 (deployed) to 1 (stowed)
#macro suspension(cf,ca,a)
  union {
    #declare center_joint=(< 4.56,2.31,-0.24>+< 4.63,1.94,-0.61>)/2;
    #declare aft_joint=(< 8.07,0,-1.45>+< 7.95,0,-1.80>)/2;
    //Center joint
    cylinder {center_joint-<0,0.37,0>,center_joint+<0,0.37,0>,0.33}
    //Rod 5  
    union {
      cylinder {(<0.72,0.81,-1.08>+<0.80,0.44,-1.47>)/2,center_joint,0.20}
      object {wheel translate (< 0.72,0.99,-3.90>+< 0.80,0.83,-3.95>)/2-y*1} 
      translate -center_joint
      rotate y*cf*40
      translate center_joint
    }
    union {
      //Rod 4
      cylinder {center_joint,(< 8.07,2.31,-1.45>+< 7.95,1.94,-1.80>)/2,0.20}
      //aft joint
      cylinder {aft_joint+y*1.17,aft_joint+y*2.35,0.33}
      union {
        //Rod 2
        cylinder {(<8.07,1.76,-1.45>+<7.95,1.36,-1.80>)/2,(< 5.99,1.76,-2.55>+< 6.18,1.36,-2.87>)/2,0.20}
        //Rod 3
        cylinder {(<5.99,1.76,-2.55>+<6.18,1.36,-2.87>)/2,(< 5.72,0.99,-3.90>+< 6.11,0.83,-3.95>)/2,0.20}
        //Rod 1
        cylinder {(<8.07,1.76,-1.45>+<7.95,1.36,-1.80>)/2,(<10.76,0.88,-1.22>+<10.81,0.51,-1.60>)/2,0.20}
        //center axle
        cylinder {(< 5.72,0.99,-3.90>+< 6.11,0.83,-3.95>)/2,(< 5.72,0.99,-3.90>+< 6.11,0.83,-3.95>)/2+<0,-0.91,0>,0.125}
        object {wheel translate (< 5.72,0.99,-3.90>+< 6.11,0.83,-3.95>)/2-y*1} 
        object {wheel translate (<10.76,0.99,-3.90>+<10.81,0.83,-3.95>)/2-y*1} 
        translate -aft_joint
        rotate y*40*a 
        translate aft_joint
      }
      translate -center_joint
      rotate -y*50*ca
      translate center_joint
    }
    translate <3.00,-2.5,1.11>-<4.56,0,-0.24>
    pigment {color rgb 0.5*<1,1,1>}
  } 
#end

#macro MSLRover(cf,ca,a)
  union {
      union {
        box {<0,0,0>,<6.54,4.71,1.96>}
        cylinder{<0,0,0>,<0,0,1.05>, 0.26 translate <6.54+1.31+0.4,4.71-0.78/2,1.96+1.31>}
        box {<0,0,0>,<1.18,0.91,0.46> translate <5.23,3.66,1.96> }
        object {tailpiece translate(<6.54,0,0>)}
        object {tailpiece translate(<6.54,4.71-0.78,0>)}

        object {rtg_asy translate <6.54,4.71/2,1.96>}
        union {pancam_mast(90) translate(<0.33,0.33,0>) rotate(<0,0,35>) translate(<0.36,3.79,1.96>)}

        object {suspension(cf,ca,a)}
        object {suspension(cf,ca,a) scale <1,-1,1> translate <0,4.71,0>}
        scale <1,-1,1> //right to left conversion
        translate y*4.71/2
        scale 0.18
        rotate z*180
        translate <0.8,0,-0.20>
        pigment {color rgb <1,1,1>}
        rotate x*180
        translate <0,0,-0.853>
      }
      sphere {
          -DIMU_A_ofs,0.01
          pigment {color rgb x}
      }
  }
#end


