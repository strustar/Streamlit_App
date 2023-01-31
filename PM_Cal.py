def Cal(In, PM):
    import numpy as np
    import streamlit as st
    # Input Data
    Reinforcement_Type = In.Reinforcement_Type
    RC_Code, FRP_Code, Column_Type, Section_Type = In.RC_Code, In.FRP_Code, In.Column_Type,  In.Section_Type
    fck, fy, ffu, Ec, Es, Ef = In.fck, In.fy, In.ffu, In.Ec, In.Es, In.Ef
    b, h, D = In.b, In.h, In.D
    Layer, dia, dc, nh, nb, nD = In.Layer, In.dia, In.dc, In.nh, In.nb, In.nD
    # Input Data

    Es, Ef = Es*1e3, Ef*1e3
    ep_y, ep_fu = fy/Es, ffu/Ef
    if 'FRP' in Reinforcement_Type:
        Es, fy, ep_y = Ef, ffu, ep_fu

    # Coefficient
    if 'KDS-2021' in RC_Code and 'RC' in Reinforcement_Type:
        n, ep_co, ep_cu = 2, 0.002, 0.0033
        if fck > 40:
            n = 1.2 + 1.5*((100 - fck)/60)**4
            ep_co = 0.002 + (fck - 40)/1e5
            ep_cu = 0.0033 - (fck - 40)/1e5
        
        if n >= 2: n = 2
        n = round(n*100)/100
        
        alpha = 1 - 1/(1 + n)*(ep_co/ep_cu)
        temp = 1/(1 + n)/(2 + n)*(ep_co/ep_cu)**2
        if fck <= 40: alpha = 0.8
        beta = 1 - (0.5 - temp)/alpha
        if fck <= 50: beta = 0.4

        alpha, beta = round(alpha*100)/100, round(beta*100)/100
        beta1 = 2*beta;  eta = alpha/beta1;  eta = round(eta*100)/100
        if fck == 50: eta = 0.97
        if fck == 80: eta = 0.87

    else:  #if strncmpi(RC_Code,'KCI-2012',3) == 1  ||  strncmpi(FRP_Code,'AASHTO-2018',3) == 1
        ep_cu, eta = 0.003, 1.
        beta1 = 0.85 if fck <= 28 else 0.85 - 0.007*(fck - 28)        
        if beta1 < 0.65: beta1 = 0.65
    
    alpha, phi0 = 0.80, 0.65
    if 'Spiral' in Column_Type:
        alpha, phi0 = 0.85, 0.70
    # Coefficient

    # Preparation for Calculation
    nst, ni, Ast = np.zeros(Layer), np.zeros(Layer), np.pi*dia**2/4
    if 'Rectangle' in Section_Type:
        hD, Ag, ni = h, b*h, nh        
        for L in range(Layer):
            for i in range(nh[L]):
                for j in range(nb[L]):
                    if (i > 0 and i < nh[L]-1) and (j > 0 and j < nb[L]-1): continue
                    nst[L] = nst[L] + 1        
    elif 'Circle' in Section_Type:
        nst = nD
        hD, Ag, ni = D, np.pi*D**2/4, np.int32(np.floor(nst/2) + 1)

    Ast[0:Layer] = nst[0:Layer]*Ast[0:Layer];  rho = np.sum(Ast)/Ag
    if 'Rectangle' in Section_Type:
        A1 = np.sum(Ast)/np.sum(nst)*nb[0];  A2 = np.sum(Ast)/2 - A1    # for ACI 440.1R-15(06), Zadeh & Nanni

    dsi, Asi = np.zeros((Layer, np.max(ni))), np.zeros((Layer, np.max(ni)))
    for L in range(Layer):
        for i in range(ni[L]):
            if 'Rectangle' in Section_Type:
                dsi[L][i] = dc[L] + i*(h - 2*dc[L])/(ni[L] - 1)
                Asi[L][i] = 2*Ast[L]/nst[L]
                Asi[L][1] = nb[L]*Ast[L]/nst[L]
                Asi[L][ni[L]-1] = Asi[L][1]
            elif 'Circle' in Section_Type:
                # theta = i*2*pi/nst[L]
                # r = D/2 - dc[L]
                # dsi[L,i] = D/2 - r*cos[theta]
                pass

    print(ni)
    print(dsi)
    print(Asi)
    # st.write('dd')
    # print(Layer,'ll')

    # Preparation for Calculation
    

    PM.ep_y, PM.ep_fu, PM.ep_cu, = ep_y, ep_fu, ep_cu
    PM.beta1, PM.eta, PM.alpha, PM.phi0 = beta1, eta, alpha, phi0

    # return PM

