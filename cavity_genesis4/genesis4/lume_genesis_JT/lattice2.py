## Author: J. Tang
## Date: 3/25/2024
import numpy as np

def taper(n_und, Kstart, dKbyK, order = 2):
    """
    Compute start and end K for each undulator section.
    order :  1 for linear taper, 2 for quadratic taper
    """
    Klist = []
    Kend = Kstart*(1-dKbyK)
    if order == 1:
        Ktaper = np.linspace(Kstart, Kend, n_und + 1)
    elif order == 2:
        alpha = (Kend - Kstart)/n_und*2
        Ktaper = Kstart + alpha*np.arange(n_und + 1)**2
       
    for i in range(n_und):
        Klist.append([Ktaper[i], Ktaper[i + 1]])

    return Klist
            

def write_linear_taper_sec(usegname, Kstart, Kend, nwig, uperiod):
    """
    Within an undulator, linearly change the taper of each period
    """
    Klist = np.linspace(Kstart, Kend, nwig)
    und_line = []
    ele_names = []
    for i in range(nwig):
        ele_name = usegname + str(i)
        line = usegname + str(i) + ': Undulator = {lambdau=' + str(uperiod) + ', nwig=1, aw=' + str(Klist[i]) + '};\n'
        und_line.append(line)
        ele_names.append(ele_name)
    return und_line, ele_names

def write_constant_sec(usegname, K, nwig, uperiod):
    line = usegname + ': Undulator = {lambdau=' + str(uperiod) + ', nwig=' + str(nwig) + ', aw=' + str(K) + '};\n'
    return [line]

def write_quad(usegname, quad_length, quad_strength):
    line = usegname + ':  Quadrupole = { l=' + str(quad_length) + ', k1=' + str(quad_strength) + '};\n'
    return [line]

def write_drift(usegname, drift_length):
    line = usegname + ': Drift = { l=' + str(drift_length) + '};\n'
    return [line]

def write_phaseshifter(usegname, ps_length, ps_phase):
    line = usegname + ': Phaseshifter = {l=' + str(ps_length) + ', phi=' + str(ps_phase) +  '};\n'
    return [line]

def write_chicane(usegname, l, lb, ld, delay):
    line = usegname + ": Chicane = {l=" + str(l) + ", lb=" + str(lb) + ", ld=" + str(ld) + ", delay=" + str(delay) + "};\n"
    return [line]

def write_undulator(Klist, nwig, uperiod, tag):
    lines = []    
    usegname_list = []
    for count, Kpar in enumerate(Klist):
        usegname = 'UD' + tag + str(count)
        if len(np.shape(Kpar)) == 0:
            lines.extend(write_constant_sec(usegname, Kpar, nwig, uperiod))
        else:
            und_line, ele_names = write_linear_taper_sec(usegname+ '_s', Kpar[0], Kpar[1], nwig, uperiod)
            lines.extend(und_line)
            line = usegname + ': Line = {' + ele_names[0]
            for ele_name in ele_names[1:]:
                line += ', ' + ele_name 
            line += '};\n'
            lines.append(line)
        usegname_list.append(usegname)
    return lines, usegname_list




def make_lattice(undKs, und_period, und_nperiods,  fodo_length, 
                 quad_length, quad_grad, 
                 phaseShifts = None, phaseshift_length = 0.0, 
                 use_chicane = False, chicane_pos = 8, l = 3.4, lb = 0.1, ld = 1.2, delay = 15e-15*3e8,
                 latticefilepath = 'lattice.lat', linename = 'myline', append = False, tag = '',
                 apply_taper = False, dKbyK = 0.0, ustart = 0, ustop = 0, order = 2): 
    """
    generate a new lattice file
    if apply_taper = True, apply linear or quadratic taper, starting from K = undKs[ustart], in ustart:ustop
    else, lattice defined by undKs. 
    If len(undKs[i]) = 1, generate a constant undulator section. 
    else if len(undKs[i]) = 2, generate a linear tapered undulator section starting from K = undKs[i][0] and ending at K = undKs[i][1]
    """
    output_lines = []

    #if apply_taper = True, overwrite undKs
    if apply_taper:
        assert ustart <= ustop, f"Error: ustop is smaller than ustart!" 
        if ustart < len(undKs):
            #remove existing ones
            while len(undKs) > ustart:
                undKs.pop()

            # get the Kstart
            Kpar = undKs[-1]
            if len(np.shape(Kpar)) == 0:
                Kstart =  Kpar
            else:
                Kstart =  Kpar[1]
            #calculate tapered undKs 
            Klist = taper(n_und = ustop - ustart, Kstart = Kstart, dKbyK = dKbyK, order = 2)
            undKs.extend(Klist)

            

    #make undulator elements
    lines, undname_list = write_undulator(Klist = undKs, nwig = und_nperiods, uperiod = und_period, tag = tag)
    output_lines.extend(lines)
     
    #make drift
    l_und = und_period * und_nperiods
    fill_length = (fodo_length/2 - l_und - quad_length)/2
    if phaseShifts is None:   #no phase shifter
        drift_length1 = drift_length2 = fill_length
    else:
        drift_length1 = fill_length - phaseshift_length
        drift_length2 = fill_length

   

    line = write_drift(usegname = 'DR1' + tag, drift_length = drift_length1)
    output_lines.extend(line)
    line = write_drift(usegname = 'DR2' + tag, drift_length = drift_length2)
    output_lines.extend(line)   

    #make phase shifter
    if phaseShifts is not None:
        print(len(undKs))
        assert len(phaseShifts) == len(undKs), f"Error: the number of phase Shifts must be equal to the number of undKs"
        assert phaseshift_length > 0, f"The length of phase shifter must be greater than zero"
        psname_list = []
        for count, phase in enumerate(phaseShifts):
            usegname = 'PS' + tag + str(count)
            psname_list.append(usegname)
            line = write_phaseshifter(usegname = usegname, ps_length = phaseshift_length, ps_phase = phase)
            output_lines.extend(line)

    #make quads
    quadname_list = []
    for i in range(len(undKs)):
        usegname = 'QD' + tag +  str(i) +'a'
        quadname_list.append(usegname)
        line = write_quad(usegname = usegname, quad_length = quad_length/2, quad_strength = quad_grad*(-1)**i)
        output_lines.extend(line)
        
        usegname = 'QD' + tag +  str(i) +'b'
        quadname_list.append(usegname)
        line = write_quad(usegname = usegname, quad_length = quad_length/2, quad_strength = quad_grad*(-1)**i*(-1))
        output_lines.extend(line)


    #make chicane
    if use_chicane:
        usegname = 'CH' + tag
        line = write_chicane(usegname = usegname, l = l, lb = lb, ld = ld, delay = delay)
        output_lines.extend(line)
    
    #make line
    line = linename + ': LINE = {'
    for i in range(len(undKs)):
        line += quadname_list[2*i] + ', '
        line += 'DR1' + tag + ', '
        if phaseShifts is not None:
            line += psname_list[i] + ', '
        line += undname_list[i] + ', '
        line += 'DR2' + tag + ','
        line += quadname_list[2*i + 1]+',\n'

        if use_chicane and (i == chicane_pos):
            line += 'CH' + tag + ',\n'

    line = line[:-2] + '};\n'
    output_lines.extend(line)
    
    if append:
        with open(latticefilepath ,'a') as tfile:
        	tfile.write(''.join(output_lines))
    else:
        with open(latticefilepath ,'w') as tfile:
        	tfile.write(''.join(output_lines))

    return output_lines
    
