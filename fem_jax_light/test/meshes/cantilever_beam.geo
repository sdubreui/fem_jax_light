// Gmsh project created on Tue Apr 28 16:19:56 2026
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 10, 0, 1.0};
//+
Point(3) = {1, 10, 0, 1.0};
//+
Point(4) = {1, 0, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Physical Surface("beam", 1) = {1};
//+
Transfinite Curve {4, 2} = 7 Using Progression 1;
//+
Transfinite Curve {1, 3} = 21 Using Progression 1;
//+
Transfinite Surface {1} = {1, 2, 3, 4};
//+
Transfinite Curve {1, 3} = 41 Using Progression 1;
//+
Transfinite Curve {4, 2} = 5 Using Progression 1;
//+
Transfinite Curve {1, 3} = 21 Using Progression 1;
