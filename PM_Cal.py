import numpy as np
import pandas as pd
import streamlit as st

def Cal(In, PM):
    ### Input Data ###
    Reinforcement_Type = In.Reinforcement_Type
    [RC_Code, FRP_Code, Column_Type, Section_Type] = [In.RC_Code, In.FRP_Code, In.Column_Type,  In.Section_Type]
    [fck, fy, ffu, Ec, Es, Ef] = [In.fck, In.fy, In.ffu, In.Ec, In.Es, In.Ef]
    [b, h, D] = [In.b, In.h, In.D]
    [Layer, dia, dc, nh, nb, nD] = [In.Layer, In.dia, In.dc, In.nh, In.nb, In.nD]
    ### Input Data ###    
    [Es, Ef] = [Es*1e3, Ef*1e3]
    [ep_y, ep_fu] = [fy/Es, ffu/Ef]
    if 'FRP' in Reinforcement_Type:
        [Es, fy, ep_y] = [Ef, ffu, ep_fu]

    ### Coefficient ###
    if 'KDS-2021' in RC_Code and 'RC' in Reinforcement_Type:
        [n, ep_co, ep_cu] = [2, 0.002, 0.0033]
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

        [alpha, beta] = [round(alpha*100)/100, round(beta*100)/100]
        beta1 = 2*beta;  eta = alpha/beta1;  eta = round(eta*100)/100
        if fck == 50: eta = 0.97
        if fck == 80: eta = 0.87
    else:  #if strncmpi(RC_Code,'KCI-2012',3) == 1  ||  strncmpi(FRP_Code,'AASHTO-2018',3) == 1
        [ep_cu, eta] = [0.003, 1.]
        beta1 = 0.85 if fck <= 28 else 0.85 - 0.007*(fck - 28)        
        if beta1 < 0.65: beta1 = 0.65
    
    [alpha, phi0] = [0.80, 0.65]
    if 'Spiral' in Column_Type:
        [alpha, phi0] = [0.85, 0.70]
    ### Coefficient ###

    ### Preparation for Calculation ###
    [nst, ni, Ast] = [np.zeros(Layer), np.zeros(Layer), np.pi*dia**2/4]
    if 'Rectangle' in Section_Type:
        [hD, Ag, ni] = [h, b*h, nh]      
        for L in range(Layer):
            for i in range(nh[L]):
                for j in range(nb[L]):
                    if (i > 0 and i < nh[L]-1) and (j > 0 and j < nb[L]-1): continue
                    nst[L] = nst[L] + 1        
    elif 'Circle' in Section_Type:
        nst = nD
        [hD, Ag, ni] = [D, np.pi*D**2/4, np.int32(np.floor(nst/2) + 1)]

    Ast = np.multiply(nst, Ast);  rho = np.sum(Ast)/Ag
    if 'Rectangle' in Section_Type:
        A1 = np.sum(Ast)/np.sum(nst)*nb[0];  A2 = np.sum(Ast)/2 - A1    # for ACI 440.1R-15(06), Zadeh & Nanni

    dsi, Asi = np.zeros((Layer, np.max(ni))), np.zeros((Layer, np.max(ni)))  #initial_rotation = 0 for Circle Section
    for L in range(Layer):
        for i in range(ni[L]):
            if 'Rectangle' in Section_Type:
                dsi[L, i] = dc[L] + i*(h - 2*dc[L])/(ni[L] - 1)
                Asi[L, i] = 2*Ast[L]/nst[L]
                Asi[L, 0] = nb[L]*Ast[L]/nst[L]
                Asi[L, ni[L]-1] = Asi[L, 0]
            elif 'Circle' in Section_Type:
                [r, theta] = [D/2 - dc[L], i*2*np.pi/nst[L]]
                dsi[L, i] = D/2 - r*np.cos(theta)
                Asi[L, i] = 2*Ast[L]/nst[L]
                Asi[L, 0] = Ast[L]/nst[L]
                Asi[L, ni[L]-1] = Asi[L, 0]        # 짝수개 및 0도 배열
    ### Preparation for Calculation ###
    
    d = np.max(dsi); gamma = d/hD
    [cc, Pn, Mn, ee] = [np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8)]
    [ep_s, fs] = [np.zeros((8, 2)), np.zeros((8, 2))]
    ### Calculation Point A(1-1) : Pure Comppression (e = 0, c = inf, Mn = 0) ###
    z1 = 1-1;  cc[z1] = np.inf;  Mn[z1] = 0
    Rein = 0 if 'FRP' in Reinforcement_Type else fy*Ast.sum()
    Pn[z1] = 0.85*fck*Ag/1e3 if [('FRP' in Reinforcement_Type) and ('ACI 440.1' in FRP_Code)] else (eta*0.85*fck*(Ag - Ast.sum()) + Rein)/1e3
    ee[z1] = Mn[z1]/Pn[z1]*1e3;  ep_s[z1, 0:2] = ep_cu;  fs[z1, 0:2] = fy
    ### Calculation Point A(1-1) : Pure Comppression (e = 0, c = inf, Mn = 0) ###

    ### Calculation Point G(7-1) : Pure Tension(e = 0, c = inf, Mn = 0)
    z1 = 7-1;  cc[z1] = -np.inf;  Mn[z1] = 0
    Pn[z1] = -Ast.sum()*fy/1e3
    ee[z1] = Mn[z1]/Pn[z1]*1e3;  ep_s[z1, 0:2] = -ep_y;  fs[z1, 0:2] = -fy
    ### Calculation Point G(7-1) : Pure Tension(e = 0, c = inf, Mn = 0)

    ### Calculation Point C(3-1), D(4-1) (Zero Tension, Balacne Point) and E(5) &&  Z = eps/ep_cu  0.1~9.1?
    for zz in [1, 2]:
        [zz1, zz2] = [3-1, 5-1]
        if zz == 2:
            [zz1, zz2] = [0, 90]
        for z1 in range(zz1, zz2+1):
            [ep_si, fsi] = [np.zeros((Layer, max(ni))), np.zeros((Layer, max(ni)))]            
            if 'FRP' in Reinforcement_Type and 'ACI 440.1' in FRP_Code:             ############# for ACI 440.1**
                xb = d*ep_cu/(ep_cu + ep_fu)  # (xb <= x <= d & d < x <=h)
                if zz == 1:
                    if z1 == 3-1: x = d                               # C x = d    Zero Tension (ep_s(end) = 0)
                    if z1 == 4-1: x = xb                              # D x = xb   Balance Point (ep_t = ep_fu)
                    if z1 == 5-1: x = d*ep_cu/(ep_cu + 0.8*ep_fu)     # E x = 0.8*xb (ep_t = 0.8*ep_fu)
                else:
                    # x = xb + (z1 - 1)*(d - xb)/(zz2 - 1)
                    x = xb + z1*(d - xb)/zz2

                [alp, c] = [x/d, x]
                if 'Rectangle' in Section_Type:
                    Cc = 0.85*alp*beta1*gamma*fck*b*h;                      Lc = (1 - alp*beta1*gamma)*h/2
                    T1 = (1 - alp)/alp*ep_cu*Ef*A1;                         L1 = (2*gamma - 1)*h/2
                    T2 = gamma/(2*gamma - 1)*(1 - alp)**2/alp*ep_cu*Ef*A2;  L2 = (2/3*(2 + alp)*gamma - 1)*h/2
                    P = (Cc - T1 - T2)/1e3;                                 M = (Cc*Lc + T1*L1 + T2*L2)/1e6
                elif 'Circle' in Section_Type:
                    [P, M] = ACI440_Circle(alp, beta1, gamma, fck, ep_cu, Ef, Ast, D)
                    # print(P, M)

                for L in range(Layer):
                    for i in range(ni[L]):
                        ep_si[L, i] = ep_cu*(c - dsi[L, i])/c
                        fsi[L, i] = Es*ep_si[L,i]
            else:                                                                   ############# for AASHTO & RC
                pass

        
        print(zz2)
    ### Calculation Point C(3-1), D(4-1) (Zero Tension, Balacne Point) and E(5) &&  Z = eps/ep_cu  0.1~9.1?



    ### %%% Calculation Point x = c = 0(8) : for Only ACI 440.1
    # if strncmpi(Reinforcement,'FRP',3) == 1  &&  strncmpi(FRP_Code,'ACI 440.1',9) == 1  % for ACI 440.1R**
    #     M = (2*gamma - 1)**2/(2*gamma) *(A1 + A2/3)*f_fu*hD;
    #     if strncmpi(Section, 'Circle', 3) == 1;  M = (2*gamma - 1)**2/(8*gamma) *sum(Ast)*f_fu*hD;  end
    #     z1 = 8;  cc(z1) = 0;  Pn(z1) =-sum(Ast)/(2*gamma)*f_fu/1e3;  Mn(z1) = M/1e6;
    #     ee(z1) = Mn(z1)/Pn(z1)*1e3;  ep_s(z1,1:2) =-ep_y;  fs(z1,1:2) =-fy;
    # end
    ### %%% Calculation Point x = c = 0(8) : for Only ACI 440.1


    # creating a DataFrame
    df = pd.DataFrame()
    # df = pd.DataFrame(
    #     np.random.randn(5, 10),
    #     columns=('col %d' % i for i in range(10)))

    # displaying the DataFrame
    # df = df.style.highlight_max(axis = 0, color = 'red').set_caption('tsfasd').format(precision=2)
    st.dataframe(df)
    st.write('# Pylance : vs code 확장팩 설치')
        


    PM.ep_y, PM.ep_fu, PM.ep_cu, = ep_y, ep_fu, ep_cu
    PM.beta1, PM.eta, PM.alpha, PM.phi0 = beta1, eta, alpha, phi0
    
    if 'Rectangle' in Section_Type: PM.A1, PM.A2 = A1, A2
    PM.Ag = Ag;  PM.Ast = Ast;  PM.nst = nst;  PM.dsi = dsi;  PM.hD = hD;  PM.rho = rho

    # return PM
    
def ACI440_Circle(alp, beta1, gamma, fck, ep_cu, Ef, Ast, D):
    t = np.arccos(1 - 2*alp*beta1*gamma)
    if t < 0 or t > np.pi: print(t)
    Cc = 0.85*fck*(t - np.sin(t)*np.cos(t))*D**2/4
    Mc = 0.85*fck*(np.sin(t))**3*D**3/12

    t = np.arccos((1 - 2*alp*gamma)/(2*gamma - 1))
    if t < 0  or  t > np.pi:  print(t)
    if np.iscomplex(t) or  np.abs(t - np.pi) < 1e-2:   # if np.abs(t - np.pi) < 1e-2:  T = 0;  Mt = 0
        [T, Mt] = [0, 0]
    else:
        T = (np.pi*np.cos(t) - t*np.cos(t) + np.sin(t))/(np.pi*(1 + np.cos(t))) *(1 - alp)/alp *ep_cu*Ef*Ast.sum();
        Mt = (np.pi - t + np.sin(t)*np.cos(t))/(2*np.pi*(1 + np.cos(t))) *(1 - alp)/alp *(gamma - 1/2) *ep_cu*Ef*Ast.sum()*D;
    
    [P, M] = [(Cc - T)/1e3, (Mc + Mt)/1e6]
    return P, M
    

