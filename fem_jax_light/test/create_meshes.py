import numpy as np 
import gmsh
from scipy import interpolate, integrate

x_area = 0.5
x_aspect_ratio = 0.5
#1) create the FEM mesh according to the input geometry
#plan form of the wing
S = 2.0+(19.0-2.0)*x_area
AR = 6.0+(56.0-6.0)*x_aspect_ratio 
b1 = np.sqrt(AR*S)
b = b1/2
c1 = S/(0.78*b1)
c2 = 0.8*c1
c3 = 0.4*c1
b = b
#we compute the mean aerodynamic chord
chord = interpolate.interp1d(np.array([0.0,0.6*b,b]),np.array([c1,c2,c3]))
def chord_sqr(x):
    y = chord(x)**2
    return y
res = integrate.quad(chord_sqr,0,b)
MAC = res[0]*2/S
TR = c3/c1

#### FEM mesh
#we assume a NACA0013 airfoil, a single aluminium spar and wooden ribs
#naca 0013 airfoil eq 
def naca0013(x,c):
    y = 0.13/0.2*(0.2969*np.sqrt(x/c)-0.1260*(x/c)-0.3516*(x/c)**2+0.2843*(x/c)**3-0.1015*(x/c)**4)
    return y

#Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize()
# Next we add a new model named "wing" (if gmsh.model.add() is not called a new
# unnamed model will be created on the fly, if necessary):
# print("gmsh model",gmsh.model.list())

gmsh.option.setNumber("General.Terminal", 0)

gmsh.model.add("FEM_wing")
#reference airfoil
c_ref = 1
xsk1 = np.linspace(0,0.30*c_ref,30)
xsp =  np.linspace(0.30*c_ref,0.40*c_ref,10)
xsk2 = np.linspace(0.40*c_ref,c_ref,60)
Ysk1_ref = naca0013(xsk1,c_ref)
Ysp_ref = naca0013(xsp,c_ref)
Ysk2_ref = naca0013(xsk2,c_ref)

#mesh discretization
#chordwise leading edge to begining of spar
nx_1 = 3
#chordwise begining of spar to end of spar
nx_2 = 2
#chordwise end of spar to trailing edge
nx_3 = 5
#spanwise in between ribs
ny = 3
#mesh size at point (not used but must be specified)
lc = 0.1
#list for the definition of the mesh
transfinite_lines_1 = [] #before the spar
transfinite_lines_2 = [] #the spar
transfinite_lines_3 = [] #after the spar
transfinite_lines_4 = [] # mesh size with respect to y

#list of physical group
skin = []
ribs = []
spar = []

#root
y_sect = 0
c = c1
xsk1 = np.linspace(0,0.30*c,30)
xsp =  np.linspace(0.30*c,0.40*c,10)
xsk2 = np.linspace(0.40*c,c,60)
Ysk1 = c/c_ref * Ysk1_ref
Ysp = c/c_ref * Ysp_ref
Ysk2 = c/c_ref * Ysk2_ref
n = 1
n_init = n
n_init_m1 = n_init
for i in range(len(xsk1)):
    gmsh.model.geo.addPoint(xsk1[i], y_sect, Ysk1[i], lc, n)
    n+=1
n_start_spar_top = n-1
n_start_spar_top_m1 = n-1        
for i in range(len(xsp)-1):
    gmsh.model.geo.addPoint(xsp[i+1], y_sect, Ysp[i+1], lc, n)
    n+=1
n_end_spar_top = n-1 
n_end_spar_top_m1 = n-1

for i in range(len(xsk2)-1):    
    gmsh.model.geo.addPoint(xsk2[i+1], y_sect, Ysk2[i+1], lc, n)
    n+=1   

#we add a point to close the airfoil
gmsh.model.geo.addPoint(xsk2[-1]+1e-3, y_sect, 0.0, lc, n)
n+=1
n_end = n-1
n_end_m1 = n-1 


for i in range(len(xsk1)-1):
    gmsh.model.geo.addPoint(xsk1[i+1], y_sect, -Ysk1[i+1], lc, n)
    n+=1
n_start_spar_bot = n-1
n_start_spar_bot_m1 = n-1    
for i in range(len(xsp)-1):
    gmsh.model.geo.addPoint(xsp[i+1], y_sect, -Ysp[i+1], lc, n)
    n+=1
n_end_spar_bot = n-1 
n_end_spar_bot_m1 = n-1 

for i in range(len(xsk2)-1):    
    gmsh.model.geo.addPoint(xsk2[i+1], y_sect, -Ysk2[i+1], lc, n)
    n+=1 


#lines, Splines
l = 1
gmsh.model.geo.addSpline(list(range(n_init,n_start_spar_top+1)),l)
l_top_1 = l
l_top_1_m1 = l
transfinite_lines_1.append(l)
l+=1
gmsh.model.geo.addSpline(list(range(n_start_spar_top,n_end_spar_top+1)),l)
transfinite_lines_2.append(l)
l_top_spar = l
l_top_spar_m1 = l
l+=1
gmsh.model.geo.addSpline(list(range(n_end_spar_top,n_end+1)),l)
transfinite_lines_3.append(l)
l_top_2 = l
l_top_2_m1 = l
l+=1

gmsh.model.geo.addSpline(list([n_init]+list(range(n_end+1,n_start_spar_bot+1))),l)
l_bot_1 = l
l_bot_1_m1 = l
transfinite_lines_1.append(l)
l+=1
gmsh.model.geo.addSpline(list(range(n_start_spar_bot,n_end_spar_bot+1)),l)
transfinite_lines_2.append(l)
l_bot_spar = l
l_bot_spar_m1 = l
l+=1

gmsh.model.geo.addSpline(list(list(range(n_end_spar_bot,n))+[n_end]),l)
transfinite_lines_3.append(l)
l_bot_2 = l
l_bot_2_m1 = l
l+=1

gmsh.model.geo.addLine(n_start_spar_top, n_start_spar_bot,l)
transfinite_lines_4.append(l)
l_spar_1 = l
l_spar_1_m1 = l
l+=1
gmsh.model.geo.addLine(n_end_spar_top, n_end_spar_bot,l)
transfinite_lines_4.append(l)
l_spar_2 = l
l_spar_2_m1 = l
l+=1

cl = 1
s = 1
gmsh.model.geo.addCurveLoop([-l_top_1,l_bot_1,-l_spar_1],cl)
gmsh.model.geo.addSurfaceFilling([cl],s)
ribs.append(s)
cl += 1
s+=1

gmsh.model.geo.addCurveLoop([-l_top_2,l_spar_2,l_bot_2],cl)
gmsh.model.geo.addSurfaceFilling([cl],s)
ribs.append(s)
cl+=1
s+=1

#next ribs 
n_ribs = 30

ribs_spacing = b/n_ribs
#equation of the leading and trailing edges
le_1 = lambda x : 0.1*c1/(0.6*b)*x
te_1 = lambda x : c1 - 0.1*c1/(0.6*b)*x
le_2 = lambda x : (5/6-0.5)*c1/(0.4*b)*x+(0.1*c1-(0.6*(5/6-0.5)*c1)/0.4)
te_2 = lambda x : c1 - 0.1*c1/(0.6*b)*x
for n_r in range(n_ribs): 
    #we create the point of this ribs starting from the last one
    n_init = n 
    #calculate the chord and the chord position
    y_sect = (n_r+1)*ribs_spacing
    if y_sect<0.6*b:
        x_start = le_1(y_sect)
        x_stop = te_1(y_sect)
    else:
        x_start =le_2(y_sect)
        x_stop = te_2(y_sect)            
    c = x_stop-x_start
    xsk1 = np.linspace(0,0.30*c,30)
    xsp =  np.linspace(0.30*c,0.40*c,10)
    xsk2 = np.linspace(0.40*c,c,60)
    Ysk1 = c/c_ref * Ysk1_ref
    Ysp = c/c_ref * Ysp_ref
    Ysk2 = c/c_ref * Ysk2_ref
    for i in range(len(xsk1)):
        gmsh.model.geo.addPoint(x_start+xsk1[i], y_sect, Ysk1[i], lc, n)
        n+=1
    n_start_spar_top = n-1        
    for i in range(len(xsp)-1):
        gmsh.model.geo.addPoint(x_start+xsp[i+1], y_sect, Ysp[i+1], lc, n)
        n+=1
    n_end_spar_top = n-1    
    for i in range(len(xsk2)-1):    
        gmsh.model.geo.addPoint(x_start+xsk2[i+1], y_sect, Ysk2[i+1], lc, n)
        n+=1   
    #we add a point to close the airfoil
    gmsh.model.geo.addPoint(x_start+xsk2[-1]+1e-3, y_sect, 0.0, lc, n)
    n_end = n
    n+=1
    for i in range(len(xsk1)-1):
        gmsh.model.geo.addPoint(x_start+xsk1[i+1], y_sect, -Ysk1[i+1], lc, n)
        n+=1
    n_start_spar_bot = n-1    
    for i in range(len(xsp)-1):
        gmsh.model.geo.addPoint(x_start+xsp[i+1], y_sect, -Ysp[i+1], lc, n)
        n+=1
    n_end_spar_bot = n-1     
    for i in range(len(xsk2)-1):    
        gmsh.model.geo.addPoint(x_start+xsk2[i+1], y_sect, -Ysk2[i+1], lc, n)
        n+=1 

    #lines, Splines
    gmsh.model.geo.addSpline(list(range(n_init,n_start_spar_top+1)),l)
    transfinite_lines_1.append(l)
    l_top_1 = l
    l+=1
    gmsh.model.geo.addSpline(list(range(n_start_spar_top,n_end_spar_top+1)),l)
    transfinite_lines_2.append(l)
    l_top_spar = l
    l+=1

    gmsh.model.geo.addSpline(list(range(n_end_spar_top,n_end+1)),l)
    transfinite_lines_3.append(l)
    l_top_2 = l
    l+=1

    gmsh.model.geo.addSpline(list([n_init]+list(range(n_end+1,n_start_spar_bot+1))),l)
    transfinite_lines_1.append(l)
    l_bot_1 = l
    l+=1
    gmsh.model.geo.addSpline(list(range(n_start_spar_bot,n_end_spar_bot+1)),l)
    transfinite_lines_2.append(l)
    l_bot_spar = l
    l+=1

    gmsh.model.geo.addSpline(list(list(range(n_end_spar_bot,n))+[n_end]),l)
    transfinite_lines_3.append(l)
    l_bot_2 = l
    l+=1

    gmsh.model.geo.addLine(n_start_spar_top, n_start_spar_bot,l)
    transfinite_lines_4.append(l)
    l_spar_1 = l
    l+=1
    gmsh.model.geo.addLine(n_end_spar_top, n_end_spar_bot,l)
    transfinite_lines_4.append(l)
    l_spar_2 = l
    l+=1

    #curve loops and surfaces
    gmsh.model.geo.addCurveLoop([-l_top_1,l_bot_1,-l_spar_1],cl)
    gmsh.model.geo.addSurfaceFilling([cl],s)
    ribs.append(s)
    cl += 1
    s+=1

    # gmsh.model.geo.addCurveLoop([-l_top_2,l_spar_2,l_bot_2,-l_end],cl)
    gmsh.model.geo.addCurveLoop([-l_top_2,l_spar_2,l_bot_2],cl)
    gmsh.model.geo.addSurfaceFilling([cl],s)
    ribs.append(s)
    cl+=1
    s+=1

    #we define the 6 lines that create the box between two ribs
    ly_1 = gmsh.model.geo.addLine(n_init_m1, n_init,l)
    transfinite_lines_4.append(l)
    l+=1
    ly_2 = gmsh.model.geo.addLine(n_start_spar_top_m1, n_start_spar_top,l)
    transfinite_lines_4.append(l)
    l+=1
    ly_3 = gmsh.model.geo.addLine(n_end_spar_top_m1, n_end_spar_top,l)
    transfinite_lines_4.append(l)
    l+=1
    ly_4 = gmsh.model.geo.addLine(n_start_spar_bot_m1, n_start_spar_bot,l)
    transfinite_lines_4.append(l)
    l+=1
    ly_5 = gmsh.model.geo.addLine(n_end_spar_bot_m1, n_end_spar_bot,l)
    transfinite_lines_4.append(l)
    l+=1
    ly_6 = gmsh.model.geo.addLine(n_end_m1, n_end,l)
    transfinite_lines_4.append(l)
    l+=1

    n_init_m1 = n_init
    n_start_spar_top_m1 = n_start_spar_top
    n_end_spar_top_m1 = n_end_spar_top
    n_start_spar_bot_m1 = n_start_spar_bot
    n_end_spar_bot_m1 = n_end_spar_bot
    n_end_m1 = n_end

    # #and the 8 curve loops and surfaces
    gmsh.model.geo.addCurveLoop([l_top_1_m1,ly_2,-l_top_1,-ly_1],cl)
    gmsh.model.geo.addSurfaceFilling([cl],s)
    skin.append(s)
    cl+=1
    s+=1
    gmsh.model.geo.addCurveLoop([l_top_spar_m1,ly_3,-l_top_spar,-ly_2],cl)
    gmsh.model.geo.addSurfaceFilling([cl],s)
    spar.append(s)
    cl+=1
    s+=1    
    gmsh.model.geo.addCurveLoop([l_top_2_m1,ly_6,-l_top_2,-ly_3],cl)
    gmsh.model.geo.addSurfaceFilling([cl],s)
    skin.append(s)
    cl+=1
    s+=1

    gmsh.model.geo.addCurveLoop([l_bot_1_m1,ly_4,-l_bot_1,-ly_1],cl)
    gmsh.model.geo.addSurfaceFilling([cl],s)
    skin.append(s)
    cl+=1
    s+=1
    gmsh.model.geo.addCurveLoop([l_bot_spar_m1,ly_5,-l_bot_spar,-ly_4],cl)
    gmsh.model.geo.addSurfaceFilling([cl],s)
    spar.append(s)
    cl+=1
    s+=1

    gmsh.model.geo.addCurveLoop([l_bot_2_m1,ly_6,-l_bot_2,-ly_5],cl)
    gmsh.model.geo.addSurfaceFilling([cl],s)
    skin.append(s)
    cl+=1
    s+=1

    gmsh.model.geo.addCurveLoop([l_spar_1_m1,ly_4,-l_spar_1,-ly_2],cl)
    gmsh.model.geo.addSurfaceFilling([cl],s)
    spar.append(s)
    cl+=1
    s+=1
    gmsh.model.geo.addCurveLoop([l_spar_2_m1,ly_5,-l_spar_2,-ly_3],cl)
    gmsh.model.geo.addSurfaceFilling([cl],s)
    spar.append(s)
    cl+=1
    s+=1


    l_top_1_m1 = l_top_1
    l_top_spar_m1 = l_top_spar
    l_top_2_m1 = l_top_2
    l_bot_1_m1 = l_bot_1
    l_bot_spar_m1 =l_bot_spar
    l_bot_2_m1 = l_bot_2
    l_spar_1_m1 = l_spar_1
    l_spar_2_m1 = l_spar_2
    
gmsh.model.geo.synchronize()


for i in range(len(transfinite_lines_1)):
    gmsh.model.mesh.setTransfiniteCurve(transfinite_lines_1[i],nx_1)
    gmsh.model.mesh.setTransfiniteCurve(transfinite_lines_3[i],nx_3)
for i in range(len(transfinite_lines_2)):
    gmsh.model.mesh.setTransfiniteCurve(transfinite_lines_2[i],nx_2)
for i in range(len(transfinite_lines_4)):
    gmsh.model.mesh.setTransfiniteCurve(transfinite_lines_4[i],ny)     
for i in range(s-1):
    gmsh.model.mesh.setTransfiniteSurface(i+1)    


# skin= gmsh.model.addPhysicalGroup(2, skin, 1)
# gmsh.model.setPhysicalName(2,skin,'skin')
# ribs = gmsh.model.addPhysicalGroup(2, ribs, 2)
# gmsh.model.setPhysicalName(2,ribs,'ribs')
#we split the spar in section between ribs to define thickness optimization variables
for i in range(n_ribs):
    spar_i = gmsh.model.addPhysicalGroup(2, spar[i*4:(i+1)*4], 1+i) #start at 1 if only spar, 3 if ribs and skin are also physical groups
    gmsh.model.setPhysicalName(2,spar_i,'spar_'+str(i+1))
for i in range(n_ribs):
    ribs_i = gmsh.model.addPhysicalGroup(2, ribs[i*2:(i+1)*2], 1+n_ribs+i)
    gmsh.model.setPhysicalName(2,ribs_i,'ribs_'+str(i+1))
for i in range(n_ribs):
    skin_i = gmsh.model.addPhysicalGroup(2, skin[i*4:(i+1)*4], 1+2*n_ribs+i)
    gmsh.model.setPhysicalName(2,skin_i,'skin_'+str(i+1))    
# We can then generate a 2D mesh...
gmsh.model.mesh.generate(2)
# ... and save it to disk
gmsh.write("meshes/wing_full.msh")
gmsh.model.remove()