import streamlit as st
import numpy as np

def KCI_Rectangle():   # 민창식 Example 9.5~9.6 KCI-2012, 직사각형 단면    
    st.session_state.RC_Code = 'KCI-2012'
    st.session_state.FRP_Code = 'AASHTO-2018'
    st.session_state.Column_Type = 'Tied Column'    

    st.session_state.fck = 27
    st.session_state.Ec = 30
    st.session_state.fy = 400
    st.session_state.Es = 200
    st.session_state.ffu = 1000
    st.session_state.Ef = 100

    st.session_state.Section_Type = 'Rectangle'
    st.session_state.b = 400
    st.session_state.h = 400

    st.session_state.Layer = 'Layer 1'
    st.session_state.dia1 = 19.1
    st.session_state.dc1 = 59.1
    st.session_state.nh1 = 3
    st.session_state.nb1 = 3

def KDS_Rectangle():   # 이정윤 Example 11.1 KDS-2021, 직사각형 단면    
    st.session_state.RC_Code = 'KDS-2021'
    st.session_state.FRP_Code = 'AASHTO-2018'
    st.session_state.Column_Type = 'Tied Column'    

    st.session_state.fck = 35
    st.session_state.Ec = 30
    st.session_state.fy = 600
    st.session_state.Es = 200
    st.session_state.ffu = 1000
    st.session_state.Ef = 100

    st.session_state.Section_Type = 'Rectangle'
    st.session_state.b = 600
    st.session_state.h = 600

    st.session_state.Layer = 'Layer 1'
    st.session_state.dia1 = 22.2
    st.session_state.dc1 = 60
    st.session_state.nh1 = 5
    st.session_state.nb1 = 5

def KDS_Circle():   # 이정윤 Example 11.5, KDS-2021, 원형 단면
    st.session_state.RC_Code = 'KDS-2021'
    st.session_state.FRP_Code = 'AASHTO-2018'
    st.session_state.Column_Type = 'Spiral Column'    

    st.session_state.fck = 30
    st.session_state.Ec = 30
    st.session_state.fy = 500
    st.session_state.Es = 200
    st.session_state.ffu = 1000
    st.session_state.Ef = 100

    st.session_state.Section_Type = 'Circle'
    st.session_state.D = 600    

    st.session_state.Layer = 'Layer 1'
    st.session_state.dia1 = 22.2
    st.session_state.dc1 = 60
    st.session_state.nD1 = 8    

def AASHTO_Rectangle():   # GFRP-Reinforced Concrete Design Training Course, AASHTO-2018, 직사각형 단면 [단위 환산 등으로 인한 오차 발생 (문서에 값 오류도 있음), 압축 FRP를 판단힐 때, a보다 c를 사용하는 것이 F(Pure Moment)점에서 부드럽게 수렴하는 것을 확인하였음 
    st.session_state.RC_Code = 'KDS-2021'
    st.session_state.FRP_Code = 'AASHTO-2018'
    st.session_state.Column_Type = 'Tied Column'    

    psi = 0.006895;  inch = 25.4
    st.session_state.fck = 5000*psi
    st.session_state.Ec = 30
    st.session_state.fy = 400
    st.session_state.Es = 200
    st.session_state.ffu = 408.287425
    st.session_state.Ef = 6500*psi

    st.session_state.Section_Type = 'Rectangle'
    st.session_state.b = 18*inch
    st.session_state.h = 18*inch

    st.session_state.Layer = 'Layer 1'
    st.session_state.dia1 = np.sqrt(4*0.79/np.pi)*inch
    st.session_state.dc1 = 3*inch
    st.session_state.nh1 = 4
    st.session_state.nb1 = 4

def ACI_Rectangle():   # Design of RC Columns Using Glass FRP Reinforcement by Zadeh and Nanni (2013), 직사각형 단면
    st.session_state.RC_Code = 'KDS-2021'
    st.session_state.FRP_Code = 'ACI 440.1R-06(15)'
    st.session_state.Column_Type = 'Tied Column'
    
    st.session_state.fck = 28
    st.session_state.Ec = 30
    st.session_state.fy = 413
    st.session_state.Es = 200
    st.session_state.ffu = 400
    st.session_state.Ef = 40

    st.session_state.Section_Type = 'Rectangle'
    st.session_state.b = 600
    st.session_state.h = 600

    st.session_state.Layer = 'Layer 1'
    st.session_state.dia1 = 25
    st.session_state.dc1 = 63
    st.session_state.nh1 = 4
    st.session_state.nb1 = 4

def ACI_Circle():   # Design of RC Columns Using Glass FRP Reinforcement by Zadeh and Nanni (2013), 원형 단면
    st.session_state.RC_Code = 'KDS-2021'
    st.session_state.FRP_Code = 'ACI 440.1R-06(15)'
    st.session_state.Column_Type = 'Tied Column'
    
    st.session_state.fck = 28
    st.session_state.Ec = 30
    st.session_state.fy = 413
    st.session_state.Es = 200
    st.session_state.ffu = 400
    st.session_state.Ef = 40

    st.session_state.Section_Type = 'Circle'
    st.session_state.D = 600    

    st.session_state.Layer = 'Layer 1'
    st.session_state.dia1 = 25
    st.session_state.dc1 = 63
    st.session_state.nD1 = 12

def ACI_Circle2():   # Design of RC Columns Using Glass FRP Reinforcement by Zadeh and Nanni (2013), 2단 레이어, 원형 단면
    st.session_state.RC_Code = 'KDS-2021'
    st.session_state.FRP_Code = 'ACI 440.1R-06(15)'
    st.session_state.Column_Type = 'Tied Column'
    
    st.session_state.fck = 40
    st.session_state.Ec = 30
    st.session_state.fy = 400
    st.session_state.Es = 200
    st.session_state.ffu = 560
    st.session_state.Ef = 45

    st.session_state.Section_Type = 'Circle'
    st.session_state.D = 2000    

    st.session_state.Layer = 'Layer 2'
    st.session_state.dia1 = 28.6
    st.session_state.dc1 = 100.
    st.session_state.nD1 = 44
    st.session_state.dia2 = 28.6
    st.session_state.dc2 = 200.
    st.session_state.nD2 = 44
    
