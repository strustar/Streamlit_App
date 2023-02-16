import os
os.system('cls')
import time
import streamlit as st
import Column_Examples as ex
# import streamlit_nested_layout  # nested columns
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.patches as patches
# from streamlit_message import message
# from io import BytesIO

###! PM_plot function   ###################
fs = 14
def PM_plot(loc):
    fig, ax = plt.subplots(layout = 'constrained')  # tight_layout = True
    txt1 = r'$\rm P_n-M_n$ Diagram';  txt2 = r'$\rm \phi P_n-\phi M_n$ Diagram'  # txt2 = r'$P_d-M_d$ Diagram'
    txt3 = RC_Code + ' (RC)';     txt4 = FRP_Code + ' (FRP)'
    # st.markdown("<h3 style='text-align: center; color: red;'> Some title</h3>", unsafe_allow_html=True)  # 문자 센터 정렬

    if 'RC' in PM_Type:
        ax.set_xlabel(r'$\rm M_{n}$ or $\rm \phi M_{n}$ [kN$\cdot$m]', fontdict = {'size': fs})
        ax.set_ylabel(r'$\rm P_{n}$ or $\rm \phi P_{n}$ [kN]', fontdict = {'size': fs})
        if 'left' in loc:
            PM_x1 = R.ZMn;  PM_y1 = R.ZPn;  PM_x2 = R.ZMd;  PM_y2 = R.ZPd;  PM_x7 = R.Mn;  PM_y7 = R.Pn;  PM_x8 = R.Md;  PM_y8 = R.Pd            
            c1 = 'red';  c2 = 'green';  ls1 = '--';  ls2 = '-';  lb1 = txt1;  lb2 = txt2;  txt_title = txt3
        elif 'right' in loc:
            PM_x1 = F.ZMn;  PM_y1 = F.ZPn;  PM_x2 = F.ZMd;  PM_y2 = F.ZPd;  PM_x7 = F.Mn;  PM_y7 = F.Pn;  PM_x8 = F.Md;  PM_y8 = F.Pd
            c1 = 'magenta';  c2 = 'cyan';  ls1 = '--';  ls2 = '-';  lb1 = txt1;  lb2 = txt2;  txt_title = txt4
    elif 'Pn' in PM_Type:
        if 'left' in loc:
            ax.set_xlabel(r'$\rm M_{n}$ [kN$\cdot$m]', fontdict = {'size': fs})
            ax.set_ylabel(r'$\rm P_{n}$ [kN]', fontdict = {'size': fs})
            PM_x1 = R.ZMn;  PM_y1 = R.ZPn;  PM_x2 = F.ZMn;  PM_y2 = F.ZPn;  PM_x7 = R.Mn;  PM_y7 = R.Pn;  PM_x8 = F.Mn;  PM_y8 = F.Pn
            c1 = 'red';  c2 = 'magenta';  ls1 = '--';  ls2 = '--';  lb1 = txt3;  lb2 = txt4;  txt_title = txt1
        elif 'right' in loc:
            ax.set_xlabel(r'$\rm \phi M_{n}$ [kN$\cdot$m]', fontdict = {'size': fs})
            ax.set_ylabel(r'$\rm \phi P_{n}$ [kN]', fontdict = {'size': fs})
            PM_x1 = R.ZMd;  PM_y1 = R.ZPd;  PM_x2 = F.ZMd;  PM_y2 = F.ZPd;  PM_x7 = R.Md;  PM_y7 = R.Pd;  PM_x8 = F.Md;  PM_y8 = F.Pd
            c1 = 'green';  c2 = 'cyan';  ls1 = '-';  ls2 = '-';  lb1 = txt3;  lb2 = txt4;  txt_title = txt2
    
    xmax = 1.15*np.max(PM_x1);  ymin = 1.25*np.min([np.min(PM_y1), np.min(PM_y2)]);  ymax = 1.15*np.max(PM_y1)
    ax.set(xlim = (0, xmax), ylim = (ymin, ymax))    
    for i in [1, 2]:      # x, y ticks
        if i == 1: [mn, mx] = [0, xmax]
        if i == 2: [mn, mx] = [ymin, ymax]
        n = len(str(round(mx)))            # 숫자의 자리수 확인
        r = 10 if n <= 3 else 10**(n-2)
        s = np.ceil((mx-mn)/10/r)*r        # 10은 표시될 tick의 수 (9~11)
        if i == 1: ax.set_xticks(np.arange(0, xmax, s))
        if i == 2: ax.set_yticks(np.arange(s*(ymin//s+1), ymax, s))

    [lw1, lw2] = [1.2, 2.]
    ax.plot([0, xmax],[0, 0], linewidth = lw1, color = 'black')     # x축 (y축 = 0)
    ax.set_title(txt_title, pad = 12, backgroundcolor = 'orange', fontdict = {'size': fs+4})
    ax.plot(PM_x1,PM_y1, linewidth = lw2, color = c1, linestyle = ls1, label = lb1)
    ax.plot(PM_x2,PM_y2, linewidth = lw2, color = c2, linestyle = ls2, label = lb2)
    ax.legend(loc = 'upper right', prop = {'size': fs})
    ax.grid(linestyle = '--', linewidth = 0.4)
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])   # 천단위 (,)
    current_values = plt.gca().get_xticks()
    plt.gca().set_xticklabels(['{:,.0f}'.format(x) for x in current_values])   # 천단위 (,)

    # e_min(B), Zero Tension(C), e_b(D), phi*Pn(max)
    if 'RC' in PM_Type:   #! PM_Type = RC 일때만 Only
        if 'left' in loc:  [x2, y2, c] = [R.Md[2-1], R.Pd[2-1], 'green']
        if 'right' in loc: [x2, y2, c] = [F.Md[2-1], F.Pd[2-1], 'cyan']
        ax.plot([0, x2],[y2, y2], linewidth = lw2, color = c)  # phi*Pn(max)
        txt = r'$\bf\phi P_{n(max)} =' + str(round(y2, 1)) + 'kN$'
        ax.text(x2/2, 0.97*y2, txt, ha = 'center', va = 'top')
        
        for i in range(1, 4):
            if 'left' in loc:  [x, y] = [R.Mn[i], R.Pn[i]]
            if 'right' in loc: [x, y] = [F.Mn[i], F.Pn[i]]
            x1 = [0, x];  y1 = [0, y]
            ax.plot(x1, y1, linewidth = lw1, color = 'black')

            if 'left' in loc:  val = R.e[i]
            if 'right' in loc: val = F.e[i]
            ha ='left' if i == 3-1 else 'right'
            c = 'blue' if val > 0 else 'red'
            if i == 2-1:  txt = r'$\bf e_{min} =' + str(round(val, 1)) + 'mm$'
            if i == 3-1:  txt = txt = r'$\bf e_{0} =' + str(round(val, 1)) + 'mm$'
            if i == 4-1:  txt = r'$\bf e_{b} =' + str(round(val, 1)) + 'mm$'
            ax.text(0.4*x, 0.4*y, txt, ha = ha, color = c, backgroundcolor = 'yellow')

    # 테이블 연동 플롯 (직선)
    if 'one' not in selected_row and 'RC' in PM_Type:
        lw = 2
        if 'A ' in selected_row: n = 0;  c = 'red';  lw = 4
        if 'B ' in selected_row: n = 1;  c = 'green'
        if 'C ' in selected_row: n = 2;  c = 'blue'
        if 'D ' in selected_row: n = 3;  c = 'cyan'
        if 'E ' in selected_row: n = 4;  c = 'magenta'
        if 'F ' in selected_row: n = 5;  c = 'yellow'
        if 'G ' in selected_row: n = 6;  c = 'darkred';  lw = 4
        if 'left' in loc:  [x1, y1] = [R.Mn[n], R.Pn[n]]
        if 'right' in loc: [x1, y1] = [F.Mn[n], F.Pn[n]]
        ax.plot([0, x1],[0, y1], linewidth = lw, color = 'blue')

    # A, B, C, D, E, F, G 점 찍기
    for i in [1, 2]:     # 1 : Pn-Mn,  2 : Pd-Md
        for z1 in range(len(PM_x7)):
            if i == 1:
                [x1, y1] = [PM_x7[z1], PM_y7[z1]]
                if z1 == len(PM_x7) - 1: txt = 'c = 0'
                if z1 == 1-1: txt = r'$\bf A \; (Pure \; Compression)$'   # \;, \quad : 공백 넣기
                if z1 == 2-1: txt = r'$\bf B \; (e_{min}$)'
                if z1 == 3-1: txt = r'$\bf C \; (Zero \; Tension)$'
                if z1 == 4-1: txt = r'D ($e_b$)' + '\n' + r'Balance Point'
                if z1 == 5-1: txt = r'$\bf E \; (\epsilon_t = 2.5\epsilon_y)$'
                if z1 == 6-1: txt = r'$\bf F \; (Pure \; Moment)$'
                if z1 == 7-1: txt = r'$\bf G \; (Pure \; Tension)$'

                if 'left' in loc:
                    bc = 'lightgreen' if z1 == 4-1 else 'none'
                if 'right' in loc:
                    bc = 'cyan' if z1 == 4-1 else 'none'
                sgnx = 1;  ha = 'left';  va = 'center'
                sgny = -1 if z1 >= 4-1 else 1
                if 'right' in loc:
                    # if z1 == 4-1: ha = 'center'
                    if z1 == 5-1: txt = r'$\bf E \; (\epsilon_t = 0.8\epsilon_{fu})$'
                    if z1 == 6-1: [sgnx, ha] = [0, 'right']
                if 'RC' not in PM_Type:
                    if z1 == 5-1: txt = r'$\bf E$'

                x = x1 + sgnx*0.02*max(PM_x1);  y = y1 + sgny*0.02*max(PM_y1)
                ax.text(x, y, txt, ha = ha, va = va, backgroundcolor = bc)
            elif i == 2:
                [x1, y1] = [PM_x8[z1], PM_y8[z1]]

            if z1 == len(PM_x7) -1:  c = 'k'
            if z1 == 1-1: c = 'red'
            if z1 == 2-1: c = 'green'
            if z1 == 3-1: c = 'blue'
            if z1 == 4-1: c = 'cyan'
            if z1 == 5-1: c = 'magenta'
            if z1 == 6-1: c = 'yellow'
            if z1 == 7-1: c = 'darkred'
            ax.plot(x1, y1, 'o', markersize = 8, markeredgecolor = 'black', linewidth = lw1, color = c)
    return fig
###! PM_plot function   ###################

### * -- Set page config
# emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
# https://zzsza.github.io/mlops/2021/02/07/python-streamlit-dashboard/  유용한 사이트
st.set_page_config(page_title = "P-M Diagram", page_icon = ":star2:", layout = "wide",    # centered, wide
                    # runOnSave = True,
                    menu_items = {                                   #   initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
                        'Get Help': 'https://www.extremelycoolapp.com/help',
                        'Report a bug': "https://www.extremelycoolapp.com/bug",
                        'About': "# This is a header. This is an *extremely* cool app!"
                    })
### * -- Set page config

### * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main setting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # web app 설정
# with st.expander("왼쪽 사이드바(sidebar)를 적당한 크기로 하시고, 화면은 다크모드(어둡게)로 설정하세요. 클릭하세요"):
#     st.write("#### :blue[Edge browser : 설정 >> 브라우저 디스플레이 (다크모드로 변경)] ####")
#     st.write("#### :blue[Chrome browser : 설정 >> 모양 (다크모드로 변경)] ####")

########? Plot
col_left, col_center, col_right = st.columns([1.6, 1, 1.6], gap = "small")
with col_left:
    container_left = st.container()
    col = st.columns([3, 2])
    with col[0]:
        st.markdown('#### :blue[Design Code]')
    with col[1]:
        st.markdown('#### :blue[Load Case (LC)]')

    col = st.columns([1.5, 1.5, 1, 1])    
    with col[0]:
        RC_Code = st.selectbox('**:green[RC Code]**', ('KCI-2012', 'KDS-2021'), key = 'RC_Code', index = 1)
    with col[1]:
        FRP_Code = st.selectbox('**:green[FRP Code]**', ('AASHTO-2018', 'ACI 440.1R-06(15)', 'ACI 440.11-22'), key = 'FRP_Code')
    with col[2]:
        Pu = st.number_input('**:green[$P_{u}$ [kN]]**', min_value = 10., value = 1500.0, step = 100., format = '%f', key = 'Pu')
    with col[3]:
        Mu = st.number_input('**:green[$M_{u}$ [kN $\cdot$ m]]**', min_value = 10., value = 100.0, step = 10., format = '%f', key = 'Mu')

    col = st.columns([3, 2])
    with col[0]:
        st.markdown('#### :blue[Column Type]')
        Column_Type = st.radio('Choose one', ('Tied Column', 'Spiral Column'), horizontal = True, label_visibility = 'collapsed', key = 'Column_Type')
    with col[1]:
        st.markdown('#### :blue[PM Diagram Option]')
        PM_Type = st.radio('PM Type', ('RC vs. FRP', 'Pn-Mn vs. Pd-Md'), horizontal = True, label_visibility = 'collapsed', key = 'PM_Type')

    st.markdown('#### :blue[RC Examples]')
    col = st.columns(4)
    with col[0]:
        st.button('KCI Rectangle', help = '철근콘크리트 공학(민창식, 예제 9.6~9.7) [교재에서 압축철근을 c가 아닌 a로 판단하여 오차 발생]', on_click = ex.KCI_Rectangle)
    with col[1]:
        st.button('KDS Rectangle', help = '철근콘크리트 구조(이정윤, 예제 11-1)', on_click = ex.KDS_Rectangle)
    with col[2]:
        st.button('KDS Circle', help = '철근콘크리트 구조(이정윤, 예제 11-5) [교재에서 4번째 철근의 위치를 130.3mm가 아닌 180mm로 계산하여 오차 발생]', on_click = ex.KDS_Circle)

with col_right:
    container_right = st.container()
    col = st.columns([3, 2])
    with col[0]:
        st.markdown('#### :blue[Material Properties]')
    with col[1]:
        st.markdown('#### :blue[Section Type]')
        [b, h, D] = [[], [], []]

    col = st.columns([1, 1, 1, 1, 1])
    with col[0]:
        fck = st.number_input('**:green[$f_{ck}$ [MPa]]**', min_value = 0.1, value = 27.0, step = 1., format = '%f', key = 'fck')
        Ec = st.number_input('**:green[$E_{c}$ [GPa]]**', min_value = 0.1, value = 30.0, step = 1., format = '%f', disabled = True, key = 'Ec')
    with col[1]:
        fy = st.number_input('**:green[$f_{y}$ [MPa]]**', min_value = 0.1, value = 400.0, step = 10., format = '%f', key = 'fy')
        Es = st.number_input('**:green[$E_{s}$ [GPa]]**', min_value = 0.1, value = 200.0, step = 10., format = '%f', key = 'Es')
    with col[2]:
        ffu = st.number_input('**:green[$f_{fu}$ [MPa]]**', min_value = 0.1, value = 1000.0, step = 10., format = '%f', key = 'ffu')
        Ef = st.number_input('**:green[$E_{f}$ [GPa]]**', min_value = 0.1, value = 100.0, step = 10., format = '%f', key = 'Ef')
    with col[3]:
        Section_Type = st.radio('Section Type', ('Rectangle', 'Circle'), horizontal = True, label_visibility = 'collapsed', key = 'Section_Type')
    with col[4]:
        if "Rectangle" in Section_Type:
            b = st.number_input('**:green[$b$ [mm]]**', min_value = 0.1, value = 400., step = 10., format = '%f', key = 'b')
        else:
            D = st.number_input('**:green[$D$ [mm]]**', min_value = 0.1, value = 500., step = 10., format = '%f', key = 'D')
        if "Rectangle" in Section_Type:
            h = st.number_input('**:green[$h$ [mm]]**', min_value = 0.1, value = 400., step = 10., format = '%f', key = 'h')
        
    st.markdown('#### :blue[FRP Examples]')
    col = st.columns(4)
    with col[0]:
        st.button('AASHTO_Rectangle', help = 'GFRP-Reinforced Concrete Design Training Course [단위 환산 등으로 인한 오차 발생 (문서에 값 오류도 있음), 압축 FRP를 판단할 때, a보다 c를 사용하는 것이 F(Pure Moment)점에서 부드럽게 수렴하는 것을 확인하였음]', on_click = ex.AASHTO_Rectangle)
    with col[1]:
        st.button('ACI Rectangle', help = 'Design of RC Columns Using Glass FRP Reinforcement by Zadeh and Nanni (2013)', on_click = ex.ACI_Rectangle)
    with col[2]:
        st.button('ACI Circle', help = 'Design of RC Columns Using Glass FRP Reinforcement by Zadeh and Nanni (2013)', on_click = ex.ACI_Circle)
    with col[3]:
        st.button('ACI Circle Layer 2', help = 'Design of RC Columns Using Glass FRP Reinforcement by Zadeh and Nanni (2013) [2단 레이어]', on_click = ex.ACI_Circle2)

with col_center:
    container_center = st.container()
########? Plot
if 'Pn' in PM_Type:
    st.session_state.selected_row = 'Select one below'    
    # st.info('먼저 PM Diagram Option을 RC vs. FRP로 설정하세요', icon = "ℹ️")

#######? Table
col1, col2, col3 = st.columns([2.5, 0.74, 0.96])
with col1:
    container_table = st.container()

with col2:
    st.markdown('#### :blue[Select one from A to G below]')
    st.markdown('###### :blue[PM Diagram Option : RC vs. FRP을 선택했을 경우]')
    selected_row = st.radio('**:green[Select One]**', ('Select one below', 'A (Pure Compression)', 'B (Minimum Eccentricity)', 'C (Zero Tension)', 'D (Balance Point)', 'E (ε_t = 2.5*ε_y or 0.8*ε_fu)', 'F (Pure Moment)', 'G (Pure Tension)'), label_visibility = 'collapsed', key = 'selected_row')    

with col3:
    st.markdown('#### :blue[Reinforcement Layer (Rebar & FRP)]')
    Layer = st.radio('The number of layer', ('Layer 1', 'Layer 2', 'Layer 3'), horizontal = True, help = '보강층의 수', key = 'Layer', index = 0)
    if 'Layer 1' in Layer: Layer = 1
    elif 'Layer 2' in Layer: Layer = 2
    elif 'Layer 3' in Layer: Layer = 3

    col = st.columns(3)
    [dia, dc, nh, nb, nD] = [[], [], [], [], []]   # dia = []; dc = []; nh = []; nb = []; nD = []
    for i in range(Layer):
        key_dia = 'dia' + str(i+1)
        key_dc = 'dc' + str(i+1)
        key_nh = 'nh' + str(i+1)
        key_nb = 'nb' + str(i+1)
        key_nD = 'nD' + str(i+1)

        with col[i]:
            dia.append( st.number_input('**:green[dia [mm]]**', min_value = 0.1, value = 19.1, step = 1., format = '%f', help = '보강재 직경', key = key_dia))
            if i == 0:
                dc.append(st.number_input('**:green[$d_c$ [mm]]**', min_value = 0.1, value = 59.1 + 40*i, step = 1., format = '%f', help = '피복 두께', key = key_dc))
            elif i > 0:
                dc.append(st.number_input('**:green[$d_c$ [mm]]**', min_value = 0.1, value = 59.1 + 40*i, step = 1., format = '%f', help = '피복 두께', key = key_dc))
                # dc.append(st.number_input('**:green[$d_c$ [mm]]**', min_value = dc[i-1], value = 59.1 + 40*i, step = 1., format = '%f', help = '피복 두께', key = key_dc))

            if "Rectangle" in Section_Type:
                nh.append(st.number_input('**:green[$n_h$ [EA]]**', min_value = 2, value = 3, step = 1, format = '%d', help = 'h방향 보강재 개수', key = key_nh))
                nb.append(st.number_input('**:green[$n_b$ [EA]]**', min_value = 2, value = 3, step = 1, format = '%d', help = 'b방향 보강재 개수', key = key_nb))
            else:
                nD.append(st.number_input('**:green[$n_D$ [EA]]**', min_value = 2, value = 8, step = 1, format = '%d', help = '원형 단면 총 보강재 개수', key = key_nD))
[dia, dc, nh, nb, nD] = [np.array(dia), np.array(dc), np.array(nh), np.array(nb), np.array(nD)]
#######? Table
### * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main setting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Input 변수 설정
class In: pass
[In.RC_Code, In.FRP_Code, In.Column_Type, In.Section_Type] = [RC_Code, FRP_Code, Column_Type, Section_Type]
[In.fck, In.fy, In.ffu, In.Ec, In.Es, In.Ef] = [fck, fy, ffu, Ec, Es, Ef]
[In.b, In.h, In.D] = [b, h, D]
[In.Layer, In.dia, In.dc, In.nh, In.nb, In.nD] = [Layer, dia, dc, nh, nb, nD]
# In.selected_row = selected_row


###? Data Import  #############################################################################################################
import PM_Cal
# label = ['A (Pure Compression)', r"$b^{2}$", '$\sqrt{x^2+y^2}=1$', r'\sqrt{x^2+y^2}=1', '\alpha', '\u03C6 \u03BC', 'G', 'H']
label = ['A (Pure Compression)', 'B (Minimum Eccentricity)', 'C (Zero Tension)', 'D (Balance Point)', 'E (ε_t = 2.5*ε_y or 0.8*ε_fu)', 'F (Pure Moment)', 'G (Pure Tension)']
index = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
In.Reinforcement_Type = 'RC';  PM = PM_Cal.Cal(In);  R = PM()
dataR = pd.DataFrame({'e':PM.e, 'c':PM.c, 'Pn':PM.Pn, 'Mn':PM.Mn, 'φ':PM.phi, 'Pd':PM.Pd, 'Md':PM.Md, 'Summary':label})
In.Reinforcement_Type = 'FRP';  PM = PM_Cal.Cal(In);  F = PM()
dataF = pd.DataFrame({'e':PM.e, 'c':PM.c, 'Pn':PM.Pn, 'Mn':PM.Mn, 'φ':PM.phi, 'Pd':PM.Pd, 'Md':PM.Md, 'Summary':label})
df = pd.merge(dataR, dataF, on = 'Summary', suffixes = (' ',''))
df.index = index
# df.style.hide(axis = 'index')

if 'FRP' in In.Reinforcement_Type and 'ACI 440.1' in FRP_Code:    #! for ACI 440.1R**  Only Only
    F.ZPn = np.insert(F.ZPn, -1, F.Pn8);  F.ZMn = np.insert(F.ZMn, -1, F.Mn8)
    F.ZPd = np.insert(F.ZPd, -1, F.Pd8);  F.ZMd = np.insert(F.ZMd, -1, F.Md8)
    F.Pn = np.append(F.Pn, F.Pn8);  F.Mn = np.append(F.Mn, F.Mn8)
    F.Pd = np.append(F.Pd, F.Pd8);  F.Md = np.append(F.Md, F.Md8)
###? Data Import  #############################################################################################################


###? Plot  #############################################################################################################
plt.style.use('default')  # 'dark_background'  'grayscale'
plt.style.use('grayscale')
# plt.style.use('classic') # plt.style.use("fast") # plt.style.available
plt.rcParams['figure.figsize'] = (10, 8)  # 13in.*23in. (27in. Monitor 모니터)
plt.rcParams['figure.dpi'] = 200  # plt.rcParams['figure.facecolor'] = 'gainsboro' # plt.rcParams['axes.facecolor'] = 'green'
plt.rcParams['font.size'] = fs

fig = PM_plot('left');  container_left.pyplot(fig)
fig = PM_plot('right');  container_right.pyplot(fig)

fig, ax = plt.subplots(figsize = (8, 2*8)) #, layout ='constrained')
# ax.set_facecolor('red')
xlim = 50.;  ylim = 2*xlim
ax.set(xlim = (0, xlim), ylim = (0, ylim))
plt.axis('off')  # plt.axis('equal')
ax.add_patch(patches.Rectangle((0, 0), xlim, ylim, facecolor = 'ghostwhite', edgecolor = 'black', linewidth = 0))

### 콘크리트 단면
base_scale = 0.5*xlim
if 'Rectangle' in Section_Type:
    mx = np.max([In.b, In.h]);  mr = base_scale/mx
    w = In.h*mr;  h = In.b*mr;  x = (xlim - w)/2;  y = (ylim - h)/2
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor = 'yellow', edgecolor = 'black', linewidth = 1))
if 'Circle' in Section_Type:
    mx = In.D;  mr = base_scale/mx
    w = In.D*mr;  h = w;  cx = xlim/2;  cy = ylim/2;  x = (xlim - w)/2;  y = (ylim - h)/2
    ax.add_patch(patches.Circle((cx, cy), w/2, facecolor = 'yellow', edgecolor = 'black', linewidth = 1))

# 가로 치수
x1 = x;  x2 = x + w;  y1 = y - 4;  al = 1.4;  aw = al/4;  # 화살표 길이, 폭
if 'Rectangle' in Section_Type: txt = r'$\bf h = ' + str(round(In.hD)) + ' mm$'
if 'Circle'    in Section_Type: txt = r'$\bf D = ' + str(round(In.hD)) + ' mm$'
plt.plot([x1, x2], [y1, y1], color = 'black')
plt.plot([x1, x1], [y1-al, y1+al], color = 'black')
plt.plot([x2, x2], [y1-al, y1+al], color = 'black')
ax.add_patch(patches.Polygon(((x1, y1), (x1+al, y1-aw), (x1+al, y1+aw)), closed = True, facecolor = 'black'))  # 삼각형
ax.add_patch(patches.Polygon(((x2, y1), (x2-al, y1-aw), (x2-al, y1+aw)), closed = True, facecolor = 'black'))  # 삼각형
ax.text((x1+x2)/2, y1+aw, txt, ha = 'center', va = 'bottom') #, fontweight = 'bold') #, fontsize = fs)

# 세로 치수
if 'Rectangle' in Section_Type:
    x1 = x - 4;  y1 = y;  y2 = y + h
    txt = r'$\bf b = ' + str(round(In.b)) + '$'
    plt.plot([x1, x1], [y1, y2], color = 'black')
    plt.plot([x1-al, x1+al], [y1, y1], color = 'black')
    plt.plot([x1-al, x1+al], [y2, y2], color = 'black')
    ax.add_patch(patches.Polygon(((x1, y1), (x1-aw, y1+al), (x1+aw, y1+al)), closed = True, facecolor = 'black'))  # 삼각형
    ax.add_patch(patches.Polygon(((x1, y2), (x1-aw, y2-al), (x1+aw, y2-al)), closed = True, facecolor = 'black'))  # 삼각형
    ax.text(x1-aw, (y1+y2)/2, txt, ha = 'right', va = 'center')
### 콘크리트 단면

### 철근
for L in range(Layer):
    if L == 0: c = 'red'
    if L == 1: c = 'green'
    if L == 2: c = 'blue'
    dc = In.dc[L]*mr;  dia = In.dia[L]*mr;  cr = 1.2*dia/2    # 원래 크기보다 20% 크게~ 보이게
    
    for i in range(In.nhD[L]):    # 전체 원형 등
        if 'Rectangle' in Section_Type:
            for j in range(In.nb[L]):
                if (i > 0 and i < In.nhD[L]-1) and (j > 0 and j < In.nb[L]-1): continue
                cx = x + w - dc - i*(w - 2*dc)/(In.nhD[L] - 1)
                cy = y + h - dc - j*(h - 2*dc)/(In.nb[L] - 1)
                ax.add_patch(patches.Circle((cx, cy), cr, facecolor = c, edgecolor = 'black', linewidth = 1))
        if 'Circle' in Section_Type:
            theta = i*2*np.pi/In.nhD[L];  r = In.D*mr/2 - dc
            cx = x + w/2 + r*np.cos(theta);  cy = y + h/2 + r*np.sin(theta)
            ax.add_patch(patches.Circle((cx, cy), cr, facecolor = c, edgecolor = 'black', linewidth = 1))

    # 띠 또는 나선철근
    if 'Rectangle' in Section_Type:
        x1 = x + dc - cr;  y1 = y + dc - cr
        w1 = w - 2*dc + 2*cr;  h1 = h - 2*dc + 2*cr
        ax.add_patch(patches.Rectangle((x1, y1), w1, h1, facecolor = 'none', edgecolor = 'black'))
    if 'Circle' in Section_Type:
        ax.add_patch(patches.Circle((xlim/2, ylim/2), In.D*mr/2-dc+cr, facecolor = 'none', edgecolor = 'black', linewidth = 1))

    # 가로 치수 (dc)
    y1 = y + h + 2.5 + 4*L
    for i in [1, 2]:        
        if i == 1:
            x1 = x + w - dc;  x2 = x1 + dc
            txt = r'$\bf' + str(In.dc[L]) + '$'
        elif i == 2:    # 가로 치수 (dc2)  % 원형 단면이고 홀수 개 철근 배근일때 양쪽 dc가 다름
            if 'Circle' in Section_Type and In.nD[L]%2 == 1:   # %나머지 : 홀수
                dc2 = np.max(R.dsi[L,:])
                x1 = x;  x2 = x1 + w - dc2*mr
                txt = r'$\bf' + str(round((In.D-dc2), 1)) + '$'

        plt.plot([x1, x2], [y1, y1], color = 'black')
        plt.plot([x1, x1], [y1-al, y1+al], color = 'black')
        plt.plot([x2, x2], [y1-al, y1+al], color = 'black')
        ax.add_patch(patches.Polygon(((x1, y1), (x1+al, y1-aw), (x1+al, y1+aw)), closed = True, facecolor = 'black'))  # 삼각형
        ax.add_patch(patches.Polygon(((x2, y1), (x2-al, y1-aw), (x2-al, y1+aw)), closed = True, facecolor = 'black'))  # 삼각형
        ax.text((x1+x2)/2, y1+3*aw, txt, ha = 'center', va = 'center', color = c)

    # rho, 철근량, 보강량 정보 (A1, A2, A3)
    txt = r'$\bf A_' + str(L+1) + ' = ' + str(round(R.Ast[L])) + 'mm^2$'
    ax.text(0, y1, txt, ha = 'left', va = 'bottom', color = c)
    
    if L == 1-1:  # Only one
        txt = r'$\bf \rho = ' + str(round(R.rho, 4)) + '$'
        ax.text(xlim, y1, txt, ha = 'right', va = 'center', color = 'black')
#### 철근

#### 기둥
y_col = 9;  xc = x + w/2
plt.plot([x, x], [0, y_col], color = 'black')
plt.plot([x+w, x+w], [0, y_col], color = 'black')
plt.plot([x, x+w], [y_col, y_col], color = 'black')
plt.plot([xc, xc], [0, 28], color = 'black', linestyle = '--')  # Center Line

c = 'black'
for PM in [1, 2]:  # Pn, e
    n = 0;  txt = selected_row  # n = 0: A 또는 선택하지 않을때, 해석을 위해서    
    if 'A ' in selected_row: n = 0
    if 'B ' in selected_row: n = 1
    if 'C ' in selected_row: n = 2
    if 'D ' in selected_row: n = 3
    if 'E ' in selected_row: n = 4;  txt = r'E ($\bf \epsilon_t = 2.5 \epsilon_y \; or \; 0.8 \epsilon_{fu}$)'  # \;, \quad : 공백 넣기
    if 'F ' in selected_row: n = 5
    if 'G ' in selected_row: n = 6    
    if PM == 1: y_gap =  0;  bc = 'lightgreen';  ee = R.e[n];  Pd = R.Pd[n]
    if PM == 2: y_gap = 10;  bc = 'cyan';        ee = F.e[n];  Pd = F.Pd[n]

    if 'one' in selected_row:    # 초기값 (startupFcn)
        txt = 'Select one \n from A to G below'        
        txtPn = r'$\bf \phi P_n$';  txte = r'$\bf e$'
        xee = x + (3 + PM)*R.hD*mr/6
    elif 'one' not in selected_row and 'RC' in PM_Type:
        txtPn = r'$\bf' + str(abs(round(Pd,1))) + 'kN$'
        if ee == np.inf: txte = 'inf'
        else: txte = r'$\bf' + str(abs(round(ee,1))) + 'mm$'
        if ee == 0: txte = ''

        xee = xc + ee*mr
        if ee > 0.8*R.hD: xee = xc + 0.8*R.hD*mr
        if ee < -0.8*R.hD: xee = xc - 0.8*R.hD*mr
        # if Pd >= 0: c = 'black'
        if Pd < 0:  c = 'red'

    ax.text(xc, y_col/2, txt, ha = 'center', va = 'center', color = 'blue', backgroundcolor = 'white', fontweight = 'bold')
    x1 = xee;  y1 = y_col + y_gap;  y2 = y1 + 8;  al = 2.2;  aw = al/4;  # 화살표 길이, 폭
    plt.plot([x1, x1], [y1+1, y2+1], color = c, linewidth = 2)
    if Pd >= 0: ax.add_patch(patches.Polygon(((x1, y1), (x1-aw, y1+al), (x1+aw, y1+al)), closed = True, facecolor = c))  # 삼각형
    if Pd < 0:  ax.add_patch(patches.Polygon(((x1, y2), (x1-aw, y2-al), (x1+aw, y2-al)), closed = True, facecolor = c))  # 삼각형
    ha = 'center'
    if ee > 0.1: ha = 'right'
    if ee <-0.1: ha = 'left'
    ax.text(x1, y2, txtPn, ha = 'center', va = 'bottom', fontweight = 'bold', backgroundcolor = bc, color = c)

    plt.plot([x1, xc], [(y1+y2)/2, (y1+y2)/2], color = 'black')
    ax.text((x1+xc)/2, (y1+y2)/2, txte, ha = ha, va = 'center', fontweight = 'bold', backgroundcolor = bc, color = c)
#### 기둥

#### 변형률
y_ep = 0.85*ylim
plt.plot([x, x+w], [y_ep, y_ep], color = 'black')

fix = R.ep_cu;  y_ep_cu = 8;  d = (R.hD - In.dc[0]);  x_gap = 1.;  lw = 2
for PM in [1, 2]:
    if PM == 1: ep_cu = R.ep_cu;  c = 'lightgreen';  ct = 'red'
    if PM == 2: ep_cu = F.ep_cu;  c = 'cyan';        ct = 'magenta'

    if 'one' in selected_row:    # 초기값 (startupFcn)
        if PM == 1: ep_t = -0.002  # 인장(-)
        if PM == 2: ep_t = -0.004
        cc = ep_cu/(abs(ep_t) + ep_cu)*d        
    elif 'one' not in selected_row and 'RC' in PM_Type:
        if 'A ' in selected_row: n = 0
        if 'B ' in selected_row: n = 1
        if 'C ' in selected_row: n = 2
        if 'D ' in selected_row: n = 3
        if 'E ' in selected_row: n = 4
        if 'F ' in selected_row: n = 5
        if 'G ' in selected_row: n = 6
        if PM == 1: ep_t = R.ep_s[n, 1];  cc = R.c[n]
        if PM == 2: ep_t = F.ep_s[n, 1];  cc = F.c[n]        
        txt = r'$\bf \epsilon_{t} = ' + str(abs(round(ep_t,4))) + '$'        
        
    xd = x + w - d*mr;  y_ep_t = y_ep - y_ep_cu*ep_t/fix;  xcc = x + w - cc*mr
    if PM == 1: y_text = y_ep - 0.70*y_ep_cu
    if PM == 2: y_text = y_ep - 0.25*y_ep_cu
    y_ep_c = y_ep - y_ep_cu*ep_cu/fix

    if cc == np.inf:
        plt.plot([xd, x+w], [y_ep_t, y_ep_t], color = c, linewidth = lw)
        plt.plot([xd, xd], [y_ep, y_ep_t], color = c, linewidth = lw)
        plt.plot([x+w, x+w], [y_ep, y_ep_t], color = c, linewidth = lw)
        txt = r'$\bf \epsilon_{t} = \epsilon_{cu} = ' + str(round(ep_cu,4)) + '$'
        ax.text(x+w-d*mr/2, y_text, txt, ha = 'center', va = 'center', fontweight = 'bold', backgroundcolor = c, color = 'black')
    elif cc == -np.inf:
        plt.plot([xd, x+w], [y_ep_t, y_ep_t], color = ct, linewidth = lw)
        plt.plot([xd, xd], [y_ep, y_ep_t], color = ct, linewidth = lw)
        plt.plot([x+w, x+w], [y_ep, y_ep_t], color = ct, linewidth = lw)
        txt = r'$\bf \epsilon_{t} = ' + str(round(ep_t,4)) + '$'
        ax.text(x+w-d*mr/2, y_text, txt, ha = 'center', va = 'center', fontweight = 'bold', backgroundcolor = ct, color = 'black')
    else:
        if ep_t >= 0:  # 모두 압축(+)인 경우
            plt.plot([xd, xd], [y_ep, y_ep_t], color = c, linewidth = lw)
            plt.plot([xd, x+w], [y_ep_t, y_ep_c], color = c, linewidth = lw)
            plt.plot([x+w, x+w], [y_ep, y_ep_c], color = c, linewidth = lw)
            txt = r'$\bf \epsilon_{cu} = ' + str(round(ep_cu,4)) + '$'
            ax.text(x+w+x_gap, y_text, txt, ha = 'left', va = 'center', fontweight = 'bold', backgroundcolor = c, color = 'black')
            txt = r'$\bf \epsilon_{t} = ' + str(round(ep_t,4)) + '$'
            ax.text(xd-x_gap, y_text, txt, ha = 'right', va = 'center', fontweight = 'bold', backgroundcolor = c, color = 'black')
        else:          # 인장(-)이 포함된 경우
            plt.plot([xd, xd], [y_ep, y_ep_t], color = c, linewidth = lw)
            plt.plot([xd, x+w], [y_ep_t, y_ep_c], color = c, linewidth = lw)
            plt.plot([x+w, x+w], [y_ep, y_ep_c], color = c, linewidth = lw)
            txt = r'$\bf \epsilon_{cu} = ' + str(round(ep_cu,4)) + '$'
            ax.text(x+w+x_gap, y_text, txt, ha = 'left', va = 'center', fontweight = 'bold', backgroundcolor = c, color = 'black')

            plt.plot([xd, xd], [y_ep, y_ep_t], color = ct, linewidth = lw)
            plt.plot([xd, xcc], [y_ep_t, y_ep], color = ct, linewidth = lw)            
            txt = r'$\bf \epsilon_{t} = ' + str(round(ep_t,4)) + '$'
            if 'one' in selected_row: txt = r'$\bf \epsilon_{t}$'    # 초기값 (startupFcn)
            ax.text(xd-x_gap, y_text, txt, ha = 'right', va = 'center', fontweight = 'bold', backgroundcolor = ct, color = 'black')
        
    # if PM == 1: txt = r'$\bf RC$'
    # if PM == 2: txt = r'$\bf FRP$'
    # ax.text(0, y_text, txt, ha = 'left', va = 'center', fontweight = 'bold', color = 'black')
#### 변형률
# print(st.session_state.selected_row.index, 'gggg')
container_center.pyplot(fig)

# print(fig.dpi,'dpi')
# print(plt.rcParams['figure.dpi'], plt.rcParams['figure.figsize'])
###? Plot  #############################################################################################################


###? creating a DataFrame  ############################################################################################################
# pd.set_option('display.colheader_justify', 'center')
def color_np_c(value, c1, c2):
    if value < 0: color = c1
    else:         color = c2
    return "color: " + color

container_table.dataframe(df.style.applymap(color_np_c, c1 = 'red', c2 = '', subset = pd.IndexSlice[['e ', 'c ', 'Pn ', 'Pd ', 'e', 'c', 'Pn', 'Pd']])
            .format({'φ': '{:.3f}', 'φ ': '{:.3f}'}, precision = 1, thousands = ',')
            .set_properties(**{'font-size': '150%', 'background-color': '', 'border-color': 'red', 'text-align': 'center'})
            .set_properties(align = "center")
            .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
            # .hide_index()
            # .set_table_styles([{'selector': 'tr:hover', 'props': [('background-color', 'yellow')]}])
            , use_container_width = True)  #,width = 1100)


st.markdown('#### :blue[Datailed Data]')
check = st.checkbox('세부 데이터를 보시려면 체크하세요')
if check:
    index = range(len(R.Ze))
    dataR = pd.DataFrame({'e':R.Ze, 'c':R.Zc, 'Pn':R.ZPn, 'Mn':R.ZMn, 'φ':R.Zphi, 'Pd':R.ZPd, 'Md':R.ZMd, 'ε_t':R.Zep_s[:,1], 'f_t':R.Zfs[:,1], 'Detail':index})
    R = F
    dataF = pd.DataFrame({'e':R.Ze, 'c':R.Zc, 'Pn':R.ZPn, 'Mn':R.ZMn, 'φ':R.Zphi, 'Pd':R.ZPd, 'Md':R.ZMd, 'ε_t':R.Zep_s[:,1], 'f_t':R.Zfs[:,1], 'Detail':index})
    df = pd.merge(dataR, dataF, on = 'Detail', suffixes = (' ',''))
    # df = pd.concat([dataR, dataF], axis = 1)
    # df.index = index

    st.dataframe(df.style.applymap(color_np_c, c1 = 'red', c2 = '', subset = pd.IndexSlice[['e ', 'c ', 'Pn ', 'Pd ', 'e', 'c', 'Pn', 'Pd', 'ε_t', 'ε_t ', 'f_t', 'f_t ']])
                .format({'φ': '{:.3f}', 'φ ': '{:.3f}', 'ε_t ': '{:.4f}', 'ε_t ': '{:.4f}'}, precision = 1, thousands = ',')
                .set_properties(**{'font-size': '150%', 'background-color': '', 'border-color': 'red', 'text-align': 'center'})
                .set_properties(align = "center")
                .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
                # .hide_index()
                # .set_table_styles([{'selector': 'tr:hover', 'props': [('background-color', 'yellow')]}])
                , use_container_width = True, height = 3500)


# # hovering 미지원 streamlit
# df = pd.DataFrame(np.random.randn(10, 4),
#                 columns=['A', 'B', 'C', 'D'])
# df = df.style.set_table_styles(
#     [{'selector': 'tr:hover',
#     'props': [('background-color', 'yellow')]}])
# st.dataframe(df)
# df

# df = pd.DataFrame(k=1:3, delta = (1:3)/2)
# row.names(df) <- c('$\\lambda$', "$\\mu$", "$\\pi$")
# st.dataframe(df)
# df = pd.DataFrame(dataF)
# st.dataframe(df.style.set_precision(1).highlight_max(axis=0, color = 'green'))
# df = pd.DataFrame(
#     np.random.randn(5, 10),
#     columns=('col %d' % i for i in range(10)))

# df = df.style.highlight_max(axis = 0, color = 'red').set_caption('tsfasd').format(precision=2)
# st.dataframe(df.style.format("{:.1f}").highlight_max(axis=0, color = 'green'))

# # st.dataframe(df)
# df.head(10).style.set_properties(**{'background-color': 'black',                                                   
#                                     'color': 'lawngreen',                       
#                                     'border-color': 'white'})

# df.apply(lambda x: x.max()-x.min())
# df.applymap(lambda x: np.nan if x < 0 else x)
# data = data.applymap(lambda x: x*10)
# df.style.format({
#     "A": "{:.2f}",
#     "B": "{:,.5f}",
#     "C": "{:.1f}",
#     "D": "$ {:,.2f}"
# })
# style_dict = dict(A = "{:.2}%", B = "{:.2}%", C = "{:,}원", D = "{:,}$")
# df.style.format(style_dict)
# styles = [dict(selector = "thead th", props = [("font-size", "150%"), 
#                                                ("text-align", "center"), 
#                                                ("background-color", "#6DDBFC")])]
# def color_np_custom(value, c1, c2):
#     if value < 0:
#         color = c1
#     else:
#         color = c2
#     return "color: " + color
# df.style.applymap(color_np_custom, c1 = "#FF0000", c2 = "#0000BB")
# df.style.applymap(color_np,
#                   subset = pd.IndexSlice[[1, 2], ["B", "D"]])  # 스타이 부분 적용
###? creating a DataFrame  #############################################################################################################

