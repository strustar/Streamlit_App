import numpy as np
import pandas as pd
import streamlit as st

def Cal(In):
    ###* Input Data ###
    Reinforcement_Type = In.Reinforcement_Type
    [RC_Code, FRP_Code, Column_Type, Section_Type] = [In.RC_Code, In.FRP_Code, In.Column_Type,  In.Section_Type]
    [fck, fy, ffu, Ec, Es, Ef] = [In.fck, In.fy, In.ffu, In.Ec, In.Es, In.Ef]
    [b, h, D] = [In.b, In.h, In.D]
    [Layer, dia, dc, nh, nb, nD] = [In.Layer, In.dia, In.dc, In.nh, In.nb, In.nD]
    ###* Input Data ###    
    [Es, Ef] = [Es*1e3, Ef*1e3]
    [ep_y, ep_fu] = [fy/Es, ffu/Ef]
    if 'FRP' in Reinforcement_Type:
        [Es, fy, ep_y] = [Ef, ffu, ep_fu]

    ###* Coefficient ###
    if 'KDS-2021' in RC_Code and 'RC' in Reinforcement_Type:
        [n, ep_co, ep_cu] = [2, 0.002, 0.0033]
        if fck > 40:
            n = 1.2 + 1.5*((100 - fck)/60)**4
            ep_co = 0.002 + (fck - 40)/1e5
            ep_cu = 0.0033 - (fck - 40)/1e5        
        if n >= 2: n = 2
        n = round(n*100)/100
        
        alpha = 1 - 1/(1 + n)*(ep_co/ep_cu);  temp = 1/(1 + n)/(2 + n)*(ep_co/ep_cu)**2
        if fck <= 40: alpha = 0.8
        beta = 1 - (0.5 - temp)/alpha
        if fck <= 50: beta = 0.4

        [alpha, beta] = [round(alpha*100)/100, round(beta*100)/100]
        beta1 = 2*beta;  eta = alpha/beta1;  eta = round(eta*100)/100
        if fck == 50: eta = 0.97
        if fck == 80: eta = 0.87
    else:
        [ep_cu, eta] = [0.003, 1.]
        beta1 = 0.85 if fck <= 28 else 0.85 - 0.007*(fck - 28)        
        if beta1 < 0.65: beta1 = 0.65
    
    if 'Tied' in Column_Type:   [alpha, phi0] = [0.80, 0.65]
    if 'Spiral' in Column_Type: [alpha, phi0] = [0.85, 0.70]
    ###* Coefficient ###

    ###* Preparation for Calculation ###
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
    ###* Preparation for Calculation ###
    
    d = np.max(dsi);  gamma = d/hD;  n = 7
    [cc, Pn, Mn, ee] = [np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)]
    [ep_s, fs] = [np.zeros((n, 2)), np.zeros((n, 2))]
    ###* Calculation Point A(1-1) : Pure Comppression (e = 0, c = inf, Mn = 0) ###
    z1 = 1-1;  cc[z1] = np.inf;  Mn[z1] = 0
    Rein = 0 if 'FRP' in Reinforcement_Type else fy*np.sum(Ast)
    Pn[z1] = (eta*0.85*fck*(Ag - np.sum(Ast)) + Rein)/1e3
    if ('FRP' in Reinforcement_Type) and ('ACI 440.1' in FRP_Code): Pn[z1] = 0.85*fck*Ag/1e3
    ee[z1] = Mn[z1]/Pn[z1]*1e3;  ep_s[z1, 0:2] = ep_cu;  fs[z1, 0:2] = fy
    ###* Calculation Point A(1-1) : Pure Comppression (e = 0, c = inf, Mn = 0) ###

    ###* Calculation Point G(7-1) : Pure Tension(e = 0, c = inf, Mn = 0)
    z1 = 7-1;  cc[z1] = -np.inf;  Mn[z1] = 0
    Pn[z1] = -np.sum(Ast)*fy/1e3;  ee[z1] = Mn[z1]/Pn[z1]*1e3
    ep_s[z1, 0:2] = -ep_y;  fs[z1, 0:2] = -fy
    ###* Calculation Point G(7-1) : Pure Tension(e = 0, c = inf, Mn = 0)

    ###* Calculation Point C(3-1), D(4-1) (Zero Tension, Balacne Point) and E(5-1) &&  Z = eps/ep_cu  0.1~9.?
    [ep_si, fsi, Fsi] = [np.zeros((Layer, np.max(ni))), np.zeros((Layer, np.max(ni))), np.zeros((Layer, np.max(ni)))]
    for zz in [1, 2]:
        if zz == 1: [zz1, zz2] = [3-1, 5-1]
        if zz == 2: [zz1, zz2] = [0, 90]
        [Zcc, ZMn, ZPn, Zee] = [np.zeros(zz2+1), np.zeros(zz2+1), np.zeros(zz2+1), np.zeros(zz2+1)]
        [Zep_s, Zfs] = [np.zeros((zz2+1, 2)), np.zeros((zz2+1, 2))]
        for z1 in range(zz1, zz2+1):            
            if ('FRP' in Reinforcement_Type) and ('ACI 440.1' in FRP_Code):             #############! for ACI 440.1**
                xb = d*ep_cu/(ep_cu + ep_fu)  # (xb <= x <= d & d < x <=h)
                if zz == 1:
                    if z1 == 3-1: x = d                               # C x = d    Zero Tension (ep_s(end) = 0)
                    if z1 == 4-1: x = xb                              # D x = xb   Balance Point (ep_t = ep_fu)
                    if z1 == 5-1: x = d*ep_cu/(ep_cu + 0.8*ep_fu)     # E x = 0.8*xb (ep_t = 0.8*ep_fu)
                elif zz == 2:
                    # x = xb + (z1 - 1)*(d - xb)/(zz2 - 1)
                    x = xb + z1*(d - xb)/zz2

                [alp, c] = [x/d, x]
                if 'Rectangle' in Section_Type:
                    [P, M] = ACI440_Rectangle(alp, beta1, gamma, fck, ep_cu, Ef, A1, A2, b, h)
                elif 'Circle' in Section_Type:
                    [P, M] = ACI440_Circle(alp, beta1, gamma, fck, ep_cu, Ef, Ast, D)

                for L in range(Layer):
                    for i in range(ni[L]):
                        ep_si[L, i] = ep_cu*(c - dsi[L, i])/c
                        fsi[L, i] = Es*ep_si[L, i]
            else:                                                                       #############! for RC & AASHTO
                if zz == 1:
                    if z1 == 3-1: eps = 0                                                       # C   0% fy or ffu  Zero Tension (ep_s(end) = 0)
                    if z1 == 4-1: eps = 1.00*ep_y                                               # D 100% fy or ffu  Balance Point
                    if z1 == 5-1: eps = 2.50*ep_y if 'RC' in Reinforcement_Type else 0.8*ep_y   # E 250% fy or 80% ffu                        
                elif zz == 2:
                    Z = z1/10;  eps = Z*ep_cu

                c = d*ep_cu/(ep_cu + eps)
                if 'Rectangle' in Section_Type: bhD = [hD, b, h]
                elif 'Circle' in Section_Type:  bhD = [hD, D]
                [P, M] = RC_and_AASHTO(Section_Type, Reinforcement_Type, beta1, c, eta, fck, Layer, ni, ep_si, ep_cu, dsi, fsi, Es, fy, Fsi, Asi, *bhD)

            if zz == 1:
                cc[z1] = c;  Mn[z1] = M;  Pn[z1] = P;  ee[z1] = Mn[z1]/Pn[z1]*1e3
                ep_s[z1, 0] = ep_si[0, 0];  ep_s[z1, 1] = ep_si[0, -1];  fs[z1, 0] = fsi[0, 0];  fs[z1, 1] = fsi[0, -1]
            elif zz == 2:                
                Zcc[z1] = c;  ZMn[z1] = M;  ZPn[z1] = P;  Zee[z1] = ZMn[z1]/ZPn[z1]*1e3
                Zep_s[z1, 0] = ep_si[0, 0];  Zep_s[z1, 1] = ep_si[0, -1];  Zfs[z1, 0] = fsi[0, 0];  Zfs[z1, 1] = fsi[0, -1]
    ###* Calculation Point C(3-1), D(4-1) (Zero Tension, Balacne Point) and E(5-1) &&  Z = eps/ep_cu  0.1~9.?

    ###* Calculation Point B(2-1), F(6-1) : Minimum Eccentricity (e = min), Pure Moment(Pn = 0)
    [iter, toler, temp] = [1e5, 1e-1, np.zeros(1000)]
    for z1 in [2-1, 6-1]:        
        if z1 == 2-1: Pnn = alpha*Pn[1-1]
        if z1 == 6-1: Pnn = 0
        if ('FRP' in Reinforcement_Type) and ('ACI 440.1' in FRP_Code):             #############! for ACI 440.1**            
            if z1 == 2-1:
                if 'Rectangle' in Section_Type:
                    x = alpha/beta1*h;  alp = x/d;                          c = x
                    Cc = Pnn;           Lc = (1 - alp*beta1*gamma)*h/2;     M = Cc*Lc/1e3
                elif 'Circle' in Section_Type:
                    tol = toler
                    for k3 in [1, 2, 3]:
                        for k1 in np.arange(1, iter):
                            x = hD/2 + (k1 - 1)*hD/iter;  alp = x/d;  c = x
                            [P, M] = ACI440_Circle(alp, beta1, gamma, fck, ep_cu, Ef, Ast, D)
                            if np.abs(Pnn - P) < tol: break
                        if k1 == iter: tol = 10*tol   #! 수렴이 안될 경우
                        else: break
            elif z1 == 6-1 and Pn[4-1] < 0:        #! Pn(4-1) : D, Balance Point < 0
                if 'Rectangle' in Section_Type:
                    tol = toler
                    for k3 in [1, 2, 3]:
                        for k1 in np.arange(1, iter):
                            x = hD/100 + (k1 - 1)*hD/2/iter;  alp = x/d;  c = x
                            [P, M] = ACI440_Rectangle(alp, beta1, gamma, fck, ep_cu, Ef, A1, A2, b, h)
                            if np.abs(P - Pnn) <= tol: break
                        if k1 == iter: tol = 10*tol   #! 수렴이 안될 경우
                        else: break
                elif 'Circle' in Section_Type:
                    tol = toler
                    for k3 in [1, 2, 3]:          #! 처음부터 고려한 경우  ##########################
                        for k1 in np.arange(1, iter):
                            x = hD/100 + (k1 - 1)*hD/2/iter;  alp = x/d;  c = x
                            [P, M] = ACI440_Circle(alp, beta1, gamma, fck, ep_cu, Ef, Ast, D)
                            if np.abs(Pnn - P) < tol: break
                        if k1 == iter: tol = 10*tol   #! 수렴이 안될 경우
                        else: break
            for L in range(Layer):
                for i in np.arange(ni[L]):
                    ep_si[L, i] = ep_cu*(c - dsi[L, i])/c
                    fsi[L, i] = Es*ep_si[L, i]
        else:                                                                       #############! for RC & AASHTO
            tol = toler;  k7 = 0;  c = hD/2
            for k1 in np.arange(2, 1000):
                if 'Rectangle' in Section_Type: bhD = [hD, b, h]
                elif 'Circle' in Section_Type: bhD = [hD, D]
                [P, M] = RC_and_AASHTO(Section_Type, Reinforcement_Type, beta1, c, eta, fck, Layer, ni, ep_si, ep_cu, dsi, fsi, Es, fy, Fsi, Asi, *bhD)

                temp[k1] = Pnn - P
                sgn1 = np.sign(temp[k1-1]);  sgn2 = np.sign(temp[k1]);  sgn = sgn1*sgn2
                if sgn == -1: k7 = k7 + 1
                c = c + sgn2*100/10**k7
                if abs(temp[k1]) <= 1e-1: break

        cc[z1] = c;  Mn[z1] = M;  Pn[z1] = Pnn
        ee[z1] = np.inf if z1 == 6-1 else Mn[z1]/Pn[z1]*1e3
        ep_s[z1, 0] = ep_si[0, 0];  ep_s[z1, 1] = ep_si[0, -1];  fs[z1, 0] = fsi[0, 0];  fs[z1, 1] = fsi[0, -1]
    ###* Calculation Point B(2-1), F(6-1) : Minimum Eccentricity (e = min), Pure Moment(Pn = 0)

    ##* %%% Calculation Point x = c = 0(8-1) : for Only ACI 440.1    
    if 'FRP' in Reinforcement_Type and 'ACI 440.1' in FRP_Code:    #! for ACI 440.1R**  Only Only   
        if 'Rectangle' in Section_Type: M = (2*gamma - 1)**2/(2*gamma) *(A1 + A2/3)*ffu*hD/1e6
        if 'Circle' in Section_Type:    M = (2*gamma - 1)**2/(8*gamma) *np.sum(Ast)*ffu*hD/1e6
        P = -np.sum(Ast)/(2*gamma)*ffu/1e3
        [Pn8, Mn8, Pd8, Md8] = [P, M, 0.55*P, 0.55*M]  # c = 0, ep_s = -ep_y, fs = -fy
    ##* %%% Calculation Point x = c = 0(8-1) : for Only ACI 440.1

    ###*%% Sorting    
    Ze = np.concatenate((ee, Zee));   Zc = np.concatenate((cc, Zcc));         ZPn = np.concatenate((Pn, ZPn))
    ZMn = np.concatenate((Mn, ZMn));  Zep_s = np.concatenate((ep_s, Zep_s));  Zfs = np.concatenate((fs, Zfs))
    loc = np.argsort(ZPn)
    Ze = Ze[loc][::-1];    Zc = Zc[loc][::-1];        ZPn = ZPn[loc][::-1]
    ZMn = ZMn[loc][::-1];  Zep_s = Zep_s[loc][::-1];  Zfs = Zfs[loc][::-1]
    ###*%% Sorting
    
    ### 강도감소계수 (phi)    
    for zz in [1, 2]:
        if zz == 1:
            n = len(Pn);  phi = np.zeros(n)
        elif zz == 2:
            n = len(ZPn);  Zphi = np.zeros(n)
        for z1 in range(n-1):
            if zz == 1: eps = -ep_s[z1, 1]      # 압축 (+)
            if zz == 2: eps = -Zep_s[z1, 1]
            if 'RC' in Reinforcement_Type:                  #! RC : KDS-2021 & KCI-2012
                [ep_tccl, ep_ttcl] = [ep_y, 0.005]
                if fy >= 400: ep_ttcl = 2.5*ep_y
                ph = phi0 + (eps - ep_tccl)*(0.85 - phi0)/(ep_ttcl - ep_tccl)
                if eps <= ep_tccl: ph = phi0
                if eps >= ep_ttcl: ph = 0.85
                
            elif 'FRP' in Reinforcement_Type:               #! FRP : AASHTO & ACI  (ACI 440.1R**은 약간 다른데 거의 비슷해서 여기에 포함)
                if 'AASHTO' in FRP_Code:
                    a1 = 1.55;  a2 = 1;  a3 = 0.75
                elif 'ACI 440.1' in FRP_Code:
                    a1 = 1.05;  a2 = 0.5;  a3 = 0.65                
                ph = a1 - a2*eps/ep_fu
                if eps <= 0.8*ep_fu: ph = a3
                if eps >= ep_fu: ph = 0.55
            if zz == 1: phi[z1] = ph
            if zz == 2: Zphi[z1] = ph    
    if 'RC' in Reinforcement_Type:  [phi[-1], Zphi[-1]] = [0.85, 0.85]
    if 'FRP' in Reinforcement_Type: [phi[-1], Zphi[-1]] = [0.55, 0.55]
    ### 강도감소계수 (phi)    
    Pd = phi*Pn;  Md = phi*Mn;  ZPd = Zphi*ZPn;  ZMd = Zphi*ZMn

    class PM: pass
    PM.ep_y, PM.ep_fu, PM.ep_cu, = ep_y, ep_fu, ep_cu
    PM.beta1, PM.eta, PM.alpha, PM.phi0 = beta1, eta, alpha, phi0
    
    if 'Rectangle' in Section_Type: PM.A1, PM.A2 = A1, A2
    PM.Ag = Ag;  PM.Ast = Ast;  PM.nst = nst;  PM.dsi = dsi;  PM.hD = hD;  PM.rho = rho
    
    PM.e = ee;  PM.c = cc;  PM.Pn = Pn;  PM.Mn = Mn;  PM.ep_s = ep_s;  PM.fs = fs
    PM.phi = phi;  PM.Pd = Pd;  PM.Md = Md
    PM.Ze = Ze;  PM.Zc = Zc;  PM.ZPn = ZPn;  PM.ZMn = ZMn;  PM.Zep_s = Zep_s;  PM.Zfs = Zfs
    PM.Zphi = Zphi;  PM.ZPd = ZPd;  PM.ZMd = ZMd
    if 'FRP' in Reinforcement_Type and 'ACI 440.1' in FRP_Code:    #! for ACI 440.1R**  Only Only
        PM.Pn8 = Pn8;  PM.Mn8 = Mn8;  PM.Pd8 = Pd8;  PM.Md8 = Md8
    return PM


def ACI440_Circle(alp, beta1, gamma, fck, ep_cu, Ef, Ast, D):
    t = np.arccos(1 - 2*alp*beta1*gamma)
    if t < 0 or t > np.pi: print('ACI440_Circle theta', t)
    Cc = 0.85*fck*(t - np.sin(t)*np.cos(t))*D**2/4
    Mc = 0.85*fck*(np.sin(t))**3*D**3/12

    t = np.arccos((1 - 2*alp*gamma)/(2*gamma - 1))
    if t < 0  or  t > np.pi:  print('ACI440_Circle theta', t)
    if np.iscomplex(t) or  np.abs(t - np.pi) < 1e-2:   # if np.abs(t - np.pi) < 1e-2:  T = 0;  Mt = 0
        [T, Mt] = [0, 0]
    else:
        T = (np.pi*np.cos(t) - t*np.cos(t) + np.sin(t))/(np.pi*(1 + np.cos(t))) *(1 - alp)/alp *ep_cu*Ef*np.sum(Ast);
        Mt = (np.pi - t + np.sin(t)*np.cos(t))/(2*np.pi*(1 + np.cos(t))) *(1 - alp)/alp *(gamma - 1/2) *ep_cu*Ef*np.sum(Ast)*D;
    
    [P, M] = [(Cc - T)/1e3, (Mc + Mt)/1e6]
    return P, M


def ACI440_Rectangle(alp, beta1, gamma, fck, ep_cu, Ef, A1, A2, b, h):
    Cc = 0.85*alp*beta1*gamma*fck*b*h;                      Lc = (1 - alp*beta1*gamma)*h/2
    T1 = (1 - alp)/alp*ep_cu*Ef*A1;                         L1 = (2*gamma - 1)*h/2
    T2 = gamma/(2*gamma - 1)*(1 - alp)**2/alp*ep_cu*Ef*A2;  L2 = (2/3*(2 + alp)*gamma - 1)*h/2

    [P, M] = [(Cc - T1 - T2)/1e3, (Cc*Lc + T1*L1 + T2*L2)/1e6]
    return P, M


def RC_and_AASHTO(Section_Type, Reinforcement_Type, beta1, c, eta, fck, Layer, ni, ep_si, ep_cu, dsi, fsi, Es, fy, Fsi, Asi, *bhD):
    a = beta1*c
    if 'Rectangle' in Section_Type:
        [hD, b, h] = bhD
        Ac = a*b if a < h else h*b
        y_bar = (h - a)/2
    elif 'Circle' in Section_Type:
        [hD, D] = bhD
        if a <= D/2: t = np.arccos((D/2 - a)/(D/2))
        elif a <= D: t = np.pi - np.arccos((a - D/2)/(D/2))
        else: t = np.pi                                     # a가 D보다 큰 경우 원으로 한정하기 위해서
        Ac = D**2/4*(t - np.sin(t)*np.cos(t))
        y_bar = (D*np.sin(t))**3/(12*Ac)
    
    [Cc, M] = [eta*(0.85*fck)*Ac/1e3, 0]
    for L in range(Layer):
        for i in range(ni[L]):
            if c == 0: continue
            ep_si[L, i] = ep_cu*(c - dsi[L, i])/c            
            fsi[L, i] = Es*ep_si[L, i]
            if fsi[L, i] >= fy:  fsi[L, i] = fy
            if fsi[L, i] <=-fy:  fsi[L, i] =-fy
            if 'RC' in Reinforcement_Type:
                if c >= dsi[L, i]:  Fsi[L, i] = Asi[L, i]*(fsi[L, i] - eta*0.85*fck)/1e3    # 압축 철근  -0.85*fck            
                elif c < dsi[L, i]: Fsi[L, i] = Asi[L, i]*fsi[L, i]/1e3                     #! 인장 철근 (c로 판단) ccccc
            elif 'FRP' in Reinforcement_Type:
                if c >= dsi[L, i]:  Fsi[L, i] = 0
                elif c < dsi[L, i]: Fsi[L, i] = Asi[L, i]*fsi[L, i]/1e3                       # 인장 FRP만 계산 (c로 판단)
            M = M + Fsi[L, i]*(hD/2 - dsi[L, i])

    [P, M] = [np.sum(Fsi) + Cc, (M + Cc*y_bar)/1e3]    # Mc =  Cc*y_bar
    return P, M

