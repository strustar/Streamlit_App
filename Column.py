import os
os.system('cls')
import time
import streamlit as st
# import streamlit_nested_layout  # nested columns
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.patches as patches
# import matplotlib.lines as lines

###! PM_plot function   ###################
fs = 14
def PM_plot(loc):    
    txt1 = r'$P_n-M_n$ Diagram';  txt2 = r'$\phi P_n-\phi M_n$ Diagram'  # txt2 = r'$P_d-M_d$ Diagram'
    txt3 = RC_Code + ' (RC)';     txt4 = FRP_Code + ' (FRP)'
    # st.markdown("<h3 style='text-align: center; color: red;'> Some title</h3>", unsafe_allow_html=True)  # 문자 센터 정렬
    # st.markdown('### :blue[PM Diagram (RC : ' + RC_Code + ')] :sparkle:')
    if 'RC' in PM_Type:
        if 'left' in loc:
            PM_x1 = R.ZMn;  PM_y1 = R.ZPn;  PM_x2 = R.ZMd;  PM_y2 = R.ZPd;  PM_x7 = R.Mn;  PM_y7 = R.Pn;  PM_x8 = R.Md;  PM_y8 = R.Pd            
            c1 = 'red';  c2 = 'green';  ls1 = '--';  ls2 = '-';  lb1 = txt1;  lb2 = txt2;  txt_title = txt3
        elif 'right' in loc:
            PM_x1 = F.ZMn;  PM_y1 = F.ZPn;  PM_x2 = F.ZMd;  PM_y2 = F.ZPd;  PM_x7 = F.Mn;  PM_y7 = F.Pn;  PM_x8 = F.Md;  PM_y8 = F.Pd
            c1 = 'magenta';  c2 = 'cyan';  ls1 = '--';  ls2 = '-';  lb1 = txt1;  lb2 = txt2;  txt_title = txt4
    else:   ## elif 'Pn' in PM_Type:
        if 'left' in loc:
            PM_x1 = R.ZMn;  PM_y1 = R.ZPn;  PM_x2 = F.ZMn;  PM_y2 = F.ZPn;  PM_x7 = R.Mn;  PM_y7 = R.Pn;  PM_x8 = F.Mn;  PM_y8 = F.Pn
            c1 = 'red';  c2 = 'magenta';  ls1 = '--';  ls2 = '--';  lb1 = txt3;  lb2 = txt4;  txt_title = txt1
        elif 'right' in loc:
            PM_x1 = R.ZMd;  PM_y1 = R.ZPd;  PM_x2 = F.ZMd;  PM_y2 = F.ZPd;  PM_x7 = R.Md;  PM_y7 = R.Pd;  PM_x8 = F.Md;  PM_y8 = F.Pd
            c1 = 'green';  c2 = 'cyan';  ls1 = '-';  ls2 = '-';  lb1 = txt3;  lb2 = txt4;  txt_title = txt2

    fig, ax = plt.subplots(layout = 'constrained')  # tight_layout = True
    ax.set_xlabel(r'$\rm M_{n}$ or $\rm \phi M_{n}$ [kN$\cdot$m]', fontdict = {'size': fs})
    ax.set_ylabel(r'$\rm P_{n}$ or $\rm \phi P_{n}$ [kN]', fontdict = {'size': fs})    
    r = 1.15;  xmax = r*np.max(PM_x1);  ymin = 1.25*np.min([np.min(PM_y1), np.min(PM_y2)]);  ymax = r*np.max(PM_y1)
    ax.set(xlim = (0, xmax), ylim = (ymin, ymax))    
    
    for i in [1, 2]:      # x, y ticks
        if i == 1: [mx, mn] = [xmax, 0]
        if i == 2: [mx, mn] = [ymax, ymin]
        n = len(str(round(mx-mn)))        # 숫자의 자리수 확인
        r = 10 if n <= 3 else 10**(n-2)   # 9는 표시될 tick의 수 (8~10)
        s = np.ceil((mx-mn)/9/r)*r
        if i == 1: ax.set_xticks(np.arange(0, xmax, s))
        if i == 2: ax.set_yticks(np.arange(s*(ymin//s), ymax, s))

    [lw1, lw2] = [1.2, 2.]
    ax.plot([0, xmax],[0, 0], linewidth = lw1, color = 'black')     # x축 (y축 = 0)
    ax.set_title(txt_title, pad = 12, backgroundcolor = 'orange', fontdict = {'size': fs+4})
    ax.plot(PM_x1,PM_y1, linewidth = lw2, color = c1, linestyle = ls1, label = lb1)
    ax.plot(PM_x2,PM_y2, linewidth = lw2, color = c2, linestyle = ls2, label = lb2)
    ax.legend(loc = 'upper right', prop = {'size': fs})
    ax.grid(linestyle = '--', linewidth = 0.4)
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])   # 천단위 (,)

    # e_min(B), Zero Tension(C), e_b(D), phi*Pn(max)
    if 'RC' in PM_Type:   #! PM_Type = RC 일때만 Only
        if 'left' in loc:  [x2, y2, c] = [R.Md[2-1], R.Pd[2-1], 'green']
        if 'right' in loc: [x2, y2, c] = [F.Md[2-1], F.Pd[2-1], 'cyan']
        ax.plot([0, x2],[y2, y2], linewidth = lw2, color = c)  # phi*Pn(max)
        txt = r'$\phi P_{n(max)}$ =' + str(round(y2, 1)) + 'kN'
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
            if i == 2-1:  txt = r'$e_{min}$ =' + str(round(val, 1)) + 'mm'
            if i == 3-1:  txt = txt = r'$e_{0}$ =' + str(round(val, 1)) + 'mm'
            if i == 4-1:  txt = r'$e_{b}$ =' + str(round(val, 1)) + 'mm'
            ax.text(0.4*x, 0.4*y, txt, ha = ha, color = c, backgroundcolor = 'yellow')

    # A, B, C, D, E, F, G 점 찍기
    for i in [1, 2]:     # 1 : Pn-Mn,  2 : Pd-Md
        for z1 in range(len(PM_x7)):
            if i == 1:
                [x1, y1] = [PM_x7[z1], PM_y7[z1]]
                if z1 == len(PM_x7) - 1: txt = 'c = 0'
                if z1 == 1-1: txt = r'$\bf A \; (Pure \; Compression)$'   # \;, \quad : 공백 넣기
                if z1 == 2-1: txt = r'$\bf B \; (e_{min}$)'
                if z1 == 3-1: txt = r'$\bf C \; (Zero \; Tension)$'
                if z1 == 4-1: txt = 'D ($e_b$) \n Balance Point'
                if z1 == 5-1: txt = r'$\bf E \; (\epsilon_t = 2.5\epsilon_y)$'
                if z1 == 6-1: txt = r'$\bf F \; (Pure \; Moment)$'
                if z1 == 7-1: txt = r'$\bf G \; (Pure \; Tension)$'
                
                bc = 'cyan' if z1 == 4-1 else 'None'
                [sgnx, ha] = [1, 'left'];  va = 'center'
                sgny = -1 if z1 >= 4-1 else 1
                if 'right' in loc:
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

    # 테이블 연동 플롯 (직선)
    if 'Choose' not in selected_row and 'RC' in PM_Type:
        if 'A ' in selected_row: n = 0
        if 'B ' in selected_row: n = 1
        if 'C ' in selected_row: n = 2
        if 'D ' in selected_row: n = 3
        if 'E ' in selected_row: n = 4
        if 'F ' in selected_row: n = 5
        if 'G ' in selected_row: n = 6
        if 'left' in loc:  [x1, y1] = [R.Mn[n], R.Pn[n]]
        if 'right' in loc: [x1, y1] = [F.Mn[n], F.Pn[n]]
        ax.plot([0, x1],[0, y1], linewidth = 2, color = 'magenta')
    # st.pyplot(fig)
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
# st.set_option(server.runOnSave, True)  # stremlit config show
### * -- Set page config

### * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main setting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # web app 설정
# with st.expander("왼쪽 사이드바(sidebar)를 적당한 크기로 하시고, 화면은 다크모드(어둡게)로 설정하세요. 클릭하세요"):
#     st.write("#### :blue[Edge browser : 설정 >> 브라우저 디스플레이 (다크모드로 변경)] ####")
#     st.write("#### :blue[Chrome browser : 설정 >> 모양 (다크모드로 변경)] ####")

def examples():
    st.session_state.fck = 27
    print(selected_example,'aaa')
    if 'KDS' and 'Rec' in selected_example:        
        st.session_state.fck = 44
        print(st.session_state.fck)
    else:
        st.session_state.fck = 22
        print(st.session_state.fck)

col_left, col_center, col_right = st.columns([1.6, 1, 1.6], gap = "small")
with col_left:
    container_left = st.container()
    col = st.columns([3, 2])
    with col[0]:
        st.markdown('### :blue[Design Code]')
    with col[1]:
        st.markdown('### :blue[Load Case (LC)]')

    col = st.columns([1.5, 1.5, 1, 1])
    with col[0]:        
        RC_Code = st.selectbox('**:green[RC Code]**', ('KCI-2012', 'KDS-2021'))        
    with col[1]:        
        FRP_Code = st.selectbox('**:green[FRP Code]**', ('AASHTO-2018', 'ACI 440.1R-06(15)', 'ACI 440.11-22'))
    with col[2]:
        Pu = st.number_input('**:green[$P_{u}$ [kN]]**', min_value = 10., value = 1500.0, step = 100., format = '%f')
    with col[3]:
        Mu = st.number_input('**:green[$M_{u}$ [kN $\cdot$ m]]**', min_value = 10., value = 100.0, step = 10., format = '%f')

    col = st.columns([3, 2])
    with col[0]:
        st.markdown('### :blue[Column Type]')
        Column_Type = st.radio('Choose one', ('Tied Column', 'Spiral Column'), horizontal = True, label_visibility = 'collapsed')
    with col[1]:
        st.markdown('### :blue[PM Diagram Option]')
        PM_Type = st.radio('PM Type', ('RC vs. FRP', 'Pn-Mn vs. Pd-Md'), horizontal = True, label_visibility = 'collapsed')


with col_right:
    container_right = st.container()
    col = st.columns([3, 2])
    with col[0]:
        st.markdown('### :blue[Material Properties]')
    with col[1]:
        st.markdown('### :blue[Section Type]')
        [b, h, D] = [[], [], []]

    col = st.columns([1, 1, 1, 1, 1])
    with col[0]:
        fck = st.number_input('**:green[$f_{ck}$ [MPa]]**', min_value = 0.1, value = 27.0, step = 1., format = '%f', key = 'fck')
        Ec = st.number_input('**:green[$E_{c}$ [GPa]]**', min_value = 0.1, value = 30.0, step = 1., format = '%f', disabled = True)    
    with col[1]:
        fy = st.number_input('**:green[$f_{y}$ [MPa]]**', min_value = 0.1, value = 400.0, step = 10., format = '%f')
        Es = st.number_input('**:green[$E_{s}$ [GPa]]**', min_value = 0.1, value = 200.0, step = 10., format = '%f')
    with col[2]:
        ffu = st.number_input('**:green[$f_{fu}$ [MPa]]**', min_value = 0.1, value = 1000.0, step = 10., format = '%f')
        Ef = st.number_input('**:green[$E_{f}$ [GPa]]**', min_value = 0.1, value = 100.0, step = 10., format = '%f')
    with col[3]:
        Section_Type = st.radio('Section Type', ('Rectangle', 'Circle'), horizontal = True, label_visibility = 'collapsed')
    with col[4]:
        if "Rectangle" in Section_Type:
            b = st.number_input('**:green[$b$ [mm]]**', min_value = 0.1, value = 400., step = 10., format = '%f')
        else:
            D = st.number_input('**:green[$D$ [mm]]**', min_value = 0.1, value = 500., step = 10., format = '%f')
        if "Rectangle" in Section_Type:
            h = st.number_input('**:green[$h$ [mm]]**', min_value = 0.1, value = 400., step = 10., format = '%f')
        
with col_center:    
    container_center = st.container()

col1, col2, col3, col4 = st.columns([0.6, 2.4, 0.1, 1.])
with col1:
    st.markdown('##### :red[Select One]')
    selected_row = st.selectbox('**:green[Select one]**', ('Choose one below', 'A (Pure Compression)', 'B (Minimum Eccentricity)', 'C (Zero Tension)', 'D (Balance Point)', 'E (ε_t = 2.5*ε_y or 0.8*ε_fu)', 'F (Pure Moment)', 'G (Pure Tension)'), label_visibility = 'collapsed')
    
    st.markdown('##### :red[Examples]')
    selected_example = st.selectbox('**:green[Examples]**', ('Choose one below', 'KDS-2021 Rectangle', 'KDS-2021 Circle'), label_visibility = 'collapsed', on_change = examples)
    # if 'KDS' in selected_example and 'Rec' in selected_example:
    # print(st.session_state['fck'])
    print(selected_example, 'cc')
    # if 'KDS' and 'Rec' in selected_example:
        # st.session_state.fck = 55        
with col2:
    container_table = st.container()
with col4:
    st.markdown('#### :blue[Reinforcement Layer (Rebar & FRP)]')
    Layer = st.radio('The number of layer', ('Layer 1', 'Layer 2', 'Layer 3'), horizontal = True, help = '보강층의 수')
    if 'Layer 1' in Layer: Layer = 1
    elif 'Layer 2' in Layer: Layer = 2
    elif 'Layer 3' in Layer: Layer = 3

    col = st.columns(3)
    [dia, dc, nh, nb, nD] = [[], [], [], [], []]   # dia = []; dc = []; nh = []; nb = []; nD = []
    for i in range(Layer):
        with col[i]:
            dia.append(st.number_input('**:green[dia [mm]]**', min_value = 0.1, value = 19.1, step = 1., format = '%f', help = '보강재 직경'+str(i+1)))
            if i == 0:
                dc.append(st.number_input('**:green[$d_c$ [mm]]**', min_value = 0.1, value = 59.1 + 40*i, step = 1., format = '%f', help = '피복 두께'+str(i+1)))
            elif i > 0:
                dc.append(st.number_input('**:green[$d_c$ [mm]]**', min_value = dc[i-1], value = 59.1 + 40*i, step = 1., format = '%f', help = '피복 두께'+str(i+1)))

            if "Rectangle" in Section_Type:
                nh.append(st.number_input('**:green[$n_h$ [EA]]**', min_value = 2, value = 3, step = 1, format = '%d', help = 'h방향 보강재 개수'+str(i+1)))
                nb.append(st.number_input('**:green[$n_b$ [EA]]**', min_value = 2, value = 3, step = 1, format = '%d', help = 'b방향 보강재 개수'+str(i+1)))
            else:
                nD.append(st.number_input('**:green[$n_D$ [EA]]**', min_value = 2, value = 8, step = 1, format = '%d', help = '원형 단면 총 보강재 개수'+str(i+1)))
[dia, dc, nh, nb, nD] = [np.array(dia), np.array(dc), np.array(nh), np.array(nb), np.array(nD)]

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
label = ['A (Pure Compression)', 'B (Minimum Eccentricity)', 'C (Zero Tension)', 'D (Balance Point)', 'E (ε_t = 2.5ε_y or 0.8ε_fu)', 'F (Pure Moment)', 'G (Pure Tension)']
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
# plt.rcParams['figure.dpi'] = 200  # plt.rcParams['figure.facecolor'] = 'gainsboro' # plt.rcParams['axes.facecolor'] = 'green'
plt.rcParams['font.size'] = fs

fig = PM_plot('left');   container_left.pyplot(fig)
fig = PM_plot('right');  container_right.pyplot(fig)

fig, ax = plt.subplots(figsize = (8, 2.2*8))
# plt.axis('off')
# plt.axis('equal')
ax.set(xlim = (0, 100), ylim = (0, 2.2*100))
plt.plot([0, 50], [0, 50])
# lines.Line2D([1., 2., 30.], [10., 20.],  linewidth = 3.)
ax.add_patch(patches.Rectangle((50, 50), 50, 50, color = 'green'))
ax.add_patch(patches.Circle((20, 30), 10, color = 'blue'))
# plt.plot([50, 100])
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



