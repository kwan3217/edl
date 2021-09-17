module torus(r_maj,r_min,fn_lat,fn_lon) {
  rotate_extrude(convexity=4,$fn=fn_lon) 
  translate([r_maj,0,0]) 
  circle(r=r_min,$fn=fn_lat);
}

module half_torus(r_maj,r_min,fn_lat,fn_lon) {
  intersection() {
    torus(r_maj,r_min,fn_lat,fn_lon*2);
    translate([0,-r_maj-r_min,-r_min])
    cube([r_maj+r_min+0.01,2*(r_maj+r_min)+0.01,2*r_min+0.01]);
  }
}
    
module quarter_torus(r_maj,r_min,fn_lat,fn_lon) {
  intersection() {
    torus(r_maj,r_min,fn_lat,fn_lon*4);
    translate([0,0,-r_min])
    cube([r_maj+r_min+0.01,r_maj+r_min+0.01,2*r_min+0.01]);
  }
}

module cyl_2v(p1,p2,r,fn) {
  length=sqrt(pow(p2[0]-p1[0],2)+pow(p2[1]-p1[1],2)+pow(p2[2]-p1[2],2));
  clat=90-asin((p2[2]-p1[2])/length);
  lon=atan2(p2[1]-p1[1],p2[0]-p1[0]);
  translate(p1)
  rotate([0,0,lon])
  rotate([0,clat,0])
  cylinder(h=length,r1=r,r2=r,$fn=fn);
}

module box(x1,y1,z1,x2,y2,z2) {
  translate([x1,y1,z1]) cube([x2-x1,y2-y1,z2-z1]);
}

module trapezoid(x,y,z1,z2) {
  mirror([0,1,0])
  rotate([90,0,0])
  linear_extrude(height=y,center=false)
  polygon(points=[[0,0],[x,0],[x,z2],[0,z1]],
          paths=[[0,1,2,3]]);
}

module tailpiece() {
  trapezoid(1.31,0.78,1.96,1.96+1.31);
  translate([1.31,0.0,1.96+1.31]) {
    mirror([0,0,1]) {
      trapezoid(0.4,0.78,0.2,0.01);
    }
    translate([0.78/2,0.78/2,0]) cylinder(h=0.01,r1=0.78/2,r2=0.78/2,center=false,$fn=24);
  }
}

module rtg_radiator_panel() {
  translate([3.2,0,0]) {
    half_torus(r_maj=0.2,r_min=0.02,fn_lat=12,fn_lon=12);
    translate([0.05,0.05,0])
    quarter_torus(r_maj=0.2,r_min=0.02,fn_lat=12,fn_lon=6);
    translate([0.05,-0.05,0])
    rotate([0,0,-90])
    quarter_torus(r_maj=0.2,r_min=0.02,fn_lat=12,fn_lon=6);
    cyl_2v([0.25,-0.05,0],[0.25,0.05,0],0.02,fn=12);
  }
  cyl_2v([0,-0.25,0],[3.25,-0.25,0],0.02,fn=12);
  cyl_2v([0, 0.25,0],[3.25, 0.25,0],0.02,fn=12);
  cyl_2v([0,-0.2,0],[3.2,-0.2,0],0.02,fn=12);
  cyl_2v([0, 0.2,0],[3.2, 0.2,0],0.02,fn=12);
  translate([0.01,-0.85/2,-0.01]) cube([3.6,0.85,0.02]);
}

module rtg_radiator_half() {
  translate([0,0,-1.5]) {
    translate([0,0.85/2,0]) rtg_radiator_panel();
    translate([0,-0.85/2,0]) rtg_radiator_panel();
    translate([0,0.85,0]) rotate([45,0,0]) translate([0,0.85/2,0]) rtg_radiator_panel();
    translate([0,-0.85,0]) rotate([-45,0,0]) translate([0,-0.85/2,0]) rtg_radiator_panel();
  }
}  

module rtg_radiator() {
  rotate([90,0,0]) {
    mirror([0,0,1]) rtg_radiator_half();
                    rtg_radiator_half();
  }
  translate([3.6,0,0]) intersection() {
    cube([0.01,3.0,3.0],center=true);
    rotate([45,0,0]) translate([-0.01,0,0]) cube([0.03,3.30,3.30],center=true);
  }
}

module rtg() {
  rotate([22.5,0,0]) cyl_2v([0,0,0],[3.41,0,0],r=1.255/2,fn=8);
  for(i=[0:45:359]) {
    for(j=[0:1:3]) {
      rotate([22.5+i,0,0]) translate([0.77+0.65*j,-(1.225+0.66*2)/2,-0.01]) cube([0.64,1.225+0.66*2,0.02]);
    }
  }
}

module rtg_asy() {
  rotate([0,-23,0]) translate([0,0,-0.8501]) {
    rtg();
    rtg_radiator();
  }
}

module pancam_mast(rx) {
  translate([-0.33,-0.33,0]) cube([0.66,0.66,0.39]);
  cyl_2v([-0.33,0,0.39],[0.33,0,0.39],r=0.33,fn=24);
  translate([0,0,0.39]) rotate([rx,0,0]) translate([0,0,-0.39]) {
    cyl_2v([0,0,0.39],[0,0,0.39+1.85],r=0.33,fn=24);
    cyl_2v([0,0,0.39+1.85],[0,0,0.39+1.85+0.26],r=1.18/2,fn=24);
    cyl_2v([0,0,0.39+1.85+0.26],[0,0,0.39+1.85+0.26+0.65],r=0.33,fn=24);
    translate([0,0,0.39+1.85+0.26+0.65+1.11/2]) cube([1.63,1.11,1.11],center=true);
  }
}  

module suspension() {
  center_joint=([ 4.56,2.31,-0.24]+[ 4.63,1.94,-0.61])/2;
  translate([3.00,-2.5,1.11]-[4.56,0,-0.24]) {
    //Rod 5
    cyl_2v(([0.72,0.81,-1.08]+[0.80,0.44,-1.47])/2,center_joint,r=0.20,fn=12);
    //Rod 4
    cyl_2v(center_joint,([ 8.07,2.31,-1.45]+[ 7.95,1.94,-1.80])/2,r=0.20,fn=12);
    //Rod 2
    cyl_2v(([8.07,1.76,-1.45]+[7.95,1.36,-1.80])/2,([ 5.99,1.76,-2.55]+[ 6.18,1.36,-2.87])/2,r=0.20,fn=12);
    //Rod 3
    cyl_2v(([5.99,1.76,-2.55]+[6.18,1.36,-2.87])/2,([ 5.72,0.99,-3.90]+[ 6.11,0.83,-3.95])/2,r=0.20,fn=12);
    //Rod 1
    cyl_2v(([8.07,1.76,-1.45]+[7.95,1.36,-1.80])/2,([10.76,0.88,-1.22]+[10.81,0.51,-1.60])/2,r=0.20,fn=12);
    //Center joint
    cyl_2v(center_joint-[0,0.37,0],center_joint+[0,0.37,0],r=0.33,fn=12);
    //aft joint
    cyl_2v(([ 8.07,1.17,-1.45]+[ 7.95,1.17,-1.80])/2,([ 8.07,2.35,-1.45]+[ 7.95,2.35,-1.80])/2,r=0.33,fn=12);
    //center axle
    cyl_2v(([ 5.72,0.99,-3.90]+[ 6.11,0.83,-3.95])/2,([ 5.72,0.99,-3.90]+[ 6.11,0.83,-3.95])/2+[0,-0.91,0],r=0.125,fn=6);
  }
}

cube([6.54,4.71,1.96]);
translate([6.54+1.31+0.4,4.71-0.78/2,1.96+1.31]) cylinder(height=1.05, r=0.26,$fn=24);
translate([5.23,3.66,1.96]) cube([1.18,0.91,0.46]);
translate([6.54,0,0]) tailpiece();
translate([6.54,4.71-0.78,0]) tailpiece();

translate([6.54,4.71/2,1.96]) rtg_asy();
translate([0.36,3.79,1.96]) rotate([0,0,35]) translate([0.33,0.33,0]) pancam_mast(90);

suspension();
translate([0,4.71,0]) mirror([0,1,0]) suspension();
