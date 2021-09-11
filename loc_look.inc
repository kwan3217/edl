#macro Vncross(a,b)
  #local c=vcross(a,b);
  #local c=c/vlength(c);
  (c)
#end

#macro MxM(a,b,c)
  #local m=dimension_size(a,1); //number of rows in A (should equal number of rows in C)
  #local n=dimension_size(a,2); //number of cols in A (should equal number of rows in B)
  #local p=dimension_size(b,2); //number of cols in B (should equal number of cols in C)
  #local i=0; //row index in C
  //PrintMatrix("a: ",a)
  //PrintMatrix("b: ",b)
  #while(i<m)
    #local j=0; //col index in C
    #while(j<p)
      #local k=0; //dot product index
      //calculate dot product between row a_i and col b_j
      #declare c[i][j]=0;
      #while(k<n)
        //PrintNumber("i:              ",i)
        //PrintNumber("j:              ",j)
        //PrintNumber("k:              ",k)
        //PrintNumber("a[i][k]:        ",a[i][k])
        //PrintNumber("b[k][j]:        ",b[k][j])
        //PrintNumber("c[i][j] before: ",c[i][j])
        #declare c[i][j]=c[i][j]+a[i][k]*b[k][j];
        //PrintNumber("c[i][j] after:  ",c[i][j])
        #local k=k+1;
      #end
      #local j=j+1;
    #end
    #local i=i+1;
  #end
#end

#macro Mrow(V0n,V1n,V2n,Result)
  #declare Result[0][0]=V0n.x;#declare Result[0][1]=V0n.y;#declare Result[0][2]=V0n.z;
  #declare Result[1][0]=V1n.x;#declare Result[1][1]=V1n.y;#declare Result[1][2]=V1n.z;
  #declare Result[2][0]=V2n.x;#declare Result[2][1]=V2n.y;#declare Result[2][2]=V2n.z;
#end

#macro Mcol(Vm0,Vm1,Vm2,Result)
  #declare Result[0][0]=Vm0.x;#declare Result[0][1]=Vm1.x;#declare Result[0][2]=Vm2.x;
  #declare Result[1][0]=Vm0.y;#declare Result[1][1]=Vm1.y;#declare Result[1][2]=Vm2.y;
  #declare Result[2][0]=Vm0.z;#declare Result[2][1]=Vm1.z;#declare Result[2][2]=Vm2.z;
#end

#macro PointToward(p_b, p_r, t_b, t_r, Mrb)
  /*
  Calculate the point-towards transform
  :param p_b: Body frame point vector
  :param p_r: Reference frame point vector
  :param t_b: Body frame toward vector
  :param t_r: Reference toward vector
  :return: Mrb which makes p_r=MxV(Mrb,p_b) true, and simultaneously minimizes vangle(t_r,MxV(Mrb,t_b))
           in the form of a 3x3 array (row index first)
  */
  #local s_r=Vncross(p_r, t_r);
  #local u_r=Vncross(p_r, s_r);
  #local n_r=p_r/vlength(p_r);
  //Do matrix math manually. Matrix is 2D 3x3 array with row index first
  #local R=array[3][3];
  Mcol(n_r,s_r,u_r,R)
  #local s_b=Vncross(p_b, t_b);
  #local u_b=Vncross(p_b, s_b);
  #local n_b=p_b/vlength(p_b);
  #local BT=array[3][3];
  Mrow(n_b,s_b,u_b,BT)
  MxM(R,BT,Mrb)
#end

#macro PrintMatrix(label,a)
  #debug concat(label,"=\n")
  #debug concat("[",str(a[0][0],10,6),",",str(a[0][1],10,6),",",str(a[0][2],0,6),"]\n")
  #debug concat("[",str(a[1][0],10,6),",",str(a[1][1],10,6),",",str(a[1][2],0,6),"]\n")
  #debug concat("[",str(a[2][0],10,6),",",str(a[2][1],10,6),",",str(a[2][2],0,6),"]\n")
#end

#macro PrintMatrixT(label,a)
  #debug concat(label,"=\n")
  #debug concat("[",str(a[0][0],10,6),",",str(a[1][0],10,6),",",str(a[2][0],0,6),"]\n")
  #debug concat("[",str(a[0][1],10,6),",",str(a[1][1],10,6),",",str(a[2][1],0,6),"]\n")
  #debug concat("[",str(a[0][2],10,6),",",str(a[1][2],10,6),",",str(a[2][2],0,6),"]\n")
#end

#macro MVT(M,V)
  transform{matrix <
    M[0][0],M[1][0],M[2][0],
    M[0][1],M[1][1],M[2][1],
    M[0][2],M[1][2],M[2][2],
    V.x,V.y,V.z
  >}
#end

#macro MxV(M,V)
  (<M[0][0]*V.x+M[0][1]*V.y+M[0][2]*V.z,
    M[1][0]*V.x+M[1][1]*V.y+M[1][2]*V.z,
    M[2][0]*V.x+M[2][1]*V.y+M[2][2]*V.z>)
#end

#macro Vtransform(M,T,V)
  (MxV(M,V)+T)
#end

#macro LocLook(Location,LookAt,Sky,result)
  #local Direction_b=z;
  #local Sky_b=y;
  #local P_b=Direction_b;
  #local P_r=LookAt-Location;
  #local T_b=Sky_b;
  #local T_r=Sky;
  #local result=array[3][3];
  PointToward(P_b,P_r,T_b,T_r,result)
#end

#macro LocLookT(Location,LookAt,Sky)
  #local Mrb=array[3][3];
  LocLook(Location,LookAt,Sky,Mrb)
  PrintMatrix("Mrb",Mrb)
  PrintVector("Location",Location)
  MVT(Mrb,Location)
#end

