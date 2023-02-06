import os
os.system('cls')
import time
import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.patches as patches
# import matplotlib.lines as lines

### * -- Set page config
# emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.set_page_config(page_title = "P-M Diagram", page_icon = ":star2:", layout = "wide",    # centered, wide
# st.set_page_config(page_title = "P-M Diagram", page_icon = ":star2:", layout = "centered",    # centered, wide
    menu_items = {
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    })
### * -- Set page config

### * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sidebar setting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
st.sidebar.markdown('## :blue[Design Code] ##')
[col1, col2] = st.sidebar.columns([1,1], gap = "small")
with col1:
    RC_Code = st.selectbox(':green[RC Code]', ('KCI-2012', 'KDS-2021'))
with col2:
    FRP_Code = st.selectbox(':green[FRP Code]', ('AASHTO-2018', 'ACI 440.1R-06(15)', 'ACI 440.11-22'))

# st.sidebar.markdown('---')
[col1, col2] = st.sidebar.columns(2)
with col1:
    st.markdown('## :blue[Column Type] ##')
    Column_Type = st.radio('Column Type', ('Tied Column', 'Spiral Column'), horizontal = False, label_visibility = 'collapsed')
with col2:
    st.markdown('## :blue[PM Diagram Option] ##')
    PM_Type = st.radio('PM Type', ('RC vs. FRP', 'Pn-Mn vs. Pd-Md'), horizontal = False, label_visibility = 'collapsed')
    
st.sidebar.markdown('## :blue[Material Properties] ##')
[col1, col2, col3] = st.sidebar.columns(3)
with col1:
    fck = st.number_input(':green[$f_{ck}$ [MPa]]', min_value = 0.1, value = 27.0, step = 1., format = '%f')
    Ec = st.number_input(':green[$E_{c}$ [GPa]]', min_value = 0.1, value = 30.0, step = 1., format = '%f', disabled = True)    
with col2:
    fy = st.number_input(':green[$f_{y}$ [MPa]]', min_value = 0.1, value = 400.0, step = 10., format = '%f')
    Es = st.number_input(':green[$E_{s}$ [GPa]]', min_value = 0.1, value = 200.0, step = 10., format = '%f')
with col3:
    ffu = st.number_input(':green[$f_{fu}$ [MPa]]', min_value = 0.1, value = 1000.0, step = 10., format = '%f')
    Ef = st.number_input(':green[$E_{f}$ [GPa]]', min_value = 0.1, value = 100.0, step = 10., format = '%f')

st.sidebar.markdown('## :blue[Section Type] ##')
[col1, col2, col3] = st.sidebar.columns(3)
[b, h, D] = [[], [], []]
with col1:
    Section_Type = st.radio('Section Type', ('Rectangle', 'Circle'), horizontal = True, label_visibility = 'collapsed')
with col2:
    if "Rectangle" in Section_Type:
        b = st.number_input(':green[$b$ [mm]]', min_value = 0.1, value = 400., step = 10., format = '%f')
    else:
        D = st.number_input(':green[$D$ [mm]]', min_value = 0.1, value = 500., step = 10., format = '%f')
with col3:
    if "Rectangle" in Section_Type:
        h = st.number_input(':green[$h$ [mm]]', min_value = 0.1, value = 400., step = 10., format = '%f')

st.sidebar.markdown('## :blue[Reinforcement Layer (Rebar & FRP)] ##')
Layer = st.sidebar.radio('The number of layer', ('Layer 1', 'Layer 2', 'Layer 3'), horizontal = True, help = '보강층의 수')
if 'Layer 1' in Layer: Layer = 1
elif 'Layer 2' in Layer: Layer = 2
elif 'Layer 3' in Layer: Layer = 3

col = st.sidebar.columns(3)
[dia, dc, nh, nb, nD] = [[], [], [], [], []]   # dia = []; dc = []; nh = []; nb = []; nD = []
for i in range(Layer):
    with col[i]:
        dia.append(st.number_input(':green[dia [mm]]', min_value = 0.1, value = 19.1, step = 1., format = '%f', help = '보강재 직경'+str(i+1)))
        if i == 0:
            dc.append(st.number_input(':green[dc [mm]]', min_value = 0.1, value = 59.1 + 40*i, step = 1., format = '%f', help = '피복 두께'+str(i+1)))
        elif i > 0:
            dc.append(st.number_input(':green[dc [mm]]', min_value = dc[i-1], value = 59.1 + 40*i, step = 1., format = '%f', help = '피복 두께'+str(i+1)))

        if "Rectangle" in Section_Type:
            nh.append(st.number_input(':green[nh [EA]]', min_value = 2, value = 3, step = 1, format = '%d', help = 'h방향 보강재 개수'+str(i+1)))
            nb.append(st.number_input(':green[nb [EA]]', min_value = 2, value = 3, step = 1, format = '%d', help = 'b방향 보강재 개수'+str(i+1)))
        else:
            nD.append(st.number_input(':green[nD [EA]]', min_value = 2, value = 8, step = 1, format = '%d', help = '원형 단면 총 보강재 개수'+str(i+1)))
[dia, dc, nh, nb, nD] = [np.array(dia), np.array(dc), np.array(nh), np.array(nb), np.array(nD)]

st.sidebar.markdown('## :blue[Load Case (LC)] ##')
[col1, col2, col3] = st.sidebar.columns(3)
with col1:
    Pu = st.number_input(':green[$P_{u}$ [kN]]', min_value = 10., value = 1500.0, step = 100., format = '%f')
with col2:
    Mu = st.number_input(':green[$M_{u}$ [kN $\cdot$ m]]', min_value = 10., value = 100.0, step = 10., format = '%f')
### * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sidebar setting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# web app 설정
with st.expander("왼쪽 사이드바(sidebar)를 적당한 크기로 하시고, 화면은 다크모드(어둡게)로 설정하세요. 클릭하세요"):
    st.write("#### :blue[Edge browser : 설정 >> 브라우저 디스플레이 (다크모드로 변경)] ####")
    st.write("#### :blue[Chrome browser : 설정 >> 모양 (다크모드로 변경)] ####")

# Input 변수 설정
class In: pass
[In.RC_Code, In.FRP_Code, In.Column_Type, In.Section_Type] = [RC_Code, FRP_Code, Column_Type, Section_Type]
[In.fck, In.fy, In.ffu, In.Ec, In.Es, In.Ef] = [fck, fy, ffu, Ec, Es, Ef]
[In.b, In.h, In.D] = [b, h, D]
[In.Layer, In.dia, In.dc, In.nh, In.nb, In.nD] = [Layer, dia, dc, nh, nb, nD]


###? Data Import   ####################################################
import PM_Cal
# label = ['A (Pure Compression)', r"$b^{2}$", '$\sqrt{x^2+y^2}=1$', r'\sqrt{x^2+y^2}=1', '\alpha', '\u03C6 \u03BC', 'G', 'H']
label = ['A (Pure Compression)', 'B (Minimum Eccentricity)', 'C (Zero Tension)', 'D (Balance Point)', 'E (ε_t = 2.5ε_y or 0.8ε_fu)', 'F (Pure Moment)', 'G (Pure Tension)']
index = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
In.Reinforcement_Type = 'RC';  PM = PM_Cal.Cal(In);  R = PM()
dataR = pd.DataFrame({'e':PM.e, 'c':PM.c, 'Pn':PM.Pn, 'Mn':PM.Mn, 'φ':PM.phi, 'Pd':PM.Pd, 'Md':PM.Md, '':label})
In.Reinforcement_Type = 'FRP';  PM = PM_Cal.Cal(In);  F = PM()
dataF = pd.DataFrame({'e':PM.e, 'c':PM.c, 'Pn':PM.Pn, 'Mn':PM.Mn, 'φ':PM.phi, 'Pd':PM.Pd, 'Md':PM.Md, '':label})
# pd.set_option("display.precision", 1)
df = pd.merge(dataR, dataF, on = '', suffixes = (' ',''))
df.index = index
###? Data Import   ####################################################
# print(dataR.e)
# print(dataR.columns[4])
# print(dataR.index[4])


###? Plot   ####################################################
plt.style.use('default')  # 'dark_background'
plt.style.use('classic')
# plt.style.use("fast")
# plt.style.available
plt.rcParams['figure.figsize'] = (8, 6)  # 13in.*23in. (27in. Monitor 모니터)
# plt.rcParams['figure.dpi'] = 200  # plt.rcParams['figure.facecolor'] = 'gainsboro' # plt.rcParams['axes.facecolor'] = 'green'
plt.rcParams['font.size'] = 14

def PM_plot(loc):    
    if 'RC' in PM_Type:
        if 'left' in loc:
            PM_x1 = R.ZMn;  PM_y1 = R.ZPn;  PM_x2 = R.ZMd;  PM_y2 = R.ZPd;  PM_x7 = R.Mn;  PM_y7 = R.Pn;  PM_x8 = R.Md;  PM_y8 = R.Pd
            st.write('### :blue[PM Diagram (RC : ' + RC_Code + ')] :sparkle:')
            c1 = 'red';  c2 = 'green';  ls1 = '--';  ls2 = '-';  lb1 = r'$\rm P_n-M_n Diagram$';  lb2 = r'$\rm \phi P_n-\phi M_n Diagram$'
        elif 'right' in loc:
            PM_x1 = F.ZMn;  PM_y1 = F.ZPn;  PM_x2 = F.ZMd;  PM_y2 = F.ZPd;  PM_x7 = F.Mn;  PM_y7 = F.Pn;  PM_x8 = F.Md;  PM_y8 = F.Pd
            st.write('### :blue[PM Diagram (FRP : ' + FRP_Code + ')] :sparkle:')
            c1 = 'magenta';  c2 = 'cyan';  ls1 = '--';  ls2 = '-';  lb1 = r'$\rm P_n-M_n Diagram$';  lb2 = r'$\rm \phi P_n-\phi M_n Diagram$'
    else:   ## elif 'Pn' in PM_Type:
        if 'left' in loc:
            PM_x1 = R.ZMn;  PM_y1 = R.ZPn;  PM_x2 = F.ZMn;  PM_y2 = F.ZPn;  PM_x7 = R.Mn;  PM_y7 = R.Pn;  PM_x8 = F.Mn;  PM_y8 = F.Pn
            st.write('### :blue[Pn - Mn Diagram] :sparkle:')
            c1 = 'red';  c2 = 'magenta';  ls1 = '--';  ls2 = '--';  lb1 = RC_Code + ' (RC)';  lb2 = FRP_Code + ' (FRP)'
        elif 'right' in loc:
            PM_x1 = R.ZMd;  PM_y1 = R.ZPd;  PM_x2 = F.ZMd;  PM_y2 = F.ZPd;  PM_x7 = R.Md;  PM_y7 = R.Pd;  PM_x8 = F.Md;  PM_y8 = F.Pd
            st.write('### :blue[Pd - Md Diagram] :sparkle:')
            c1 = 'green';  c2 = 'cyan';  ls1 = '-';  ls2 = '-';  lb1 = RC_Code + ' (RC)';  lb2 = FRP_Code + ' (FRP)'

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\rm M_{n}$ or $\rm \phi M_{n}$ [kN$\cdot$m]', fontdict = {'size': 16})
    ax.set_ylabel(r'$\rm P_{n}$ or $\rm \phi P_{n}$ [kN]', fontdict = {'size': 16})
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
    ax.plot(PM_x1,PM_y1, linewidth = lw2, color = c1, linestyle = ls1, label = lb1)
    ax.plot(PM_x2,PM_y2, linewidth = lw2, color = c2, linestyle = ls2, label = lb2)    
    ax.legend(loc = 'upper right', prop = {'size': 14})
    ax.grid(linestyle = '--', linewidth = 0.2)
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])   # 천단위 (,)

    # e_min(B), Zero Tension(C), e_b(D), phi*Pn(max)
    if 'RC' in PM_Type:   #! PM_Type = RC 일때만 Only
        if 'left' in loc:  [x2, y2, c] = [R.Md[2-1], R.Pd[2-1], 'green']
        if 'right' in loc: [x2, y2, c] = [F.Md[2-1], F.Pd[2-1], 'cyan']
        ax.plot([0, x2],[y2, y2], linewidth = lw2, color = c)  # phi*Pn(max)
        txt = r'$\bf \phi P_{n(max)} =$' + str(round(y2, 1)) + 'kN'
        ax.text(0, 0.95*y2, txt, ha = 'left', va = 'top')
        
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

    print(fig)
    st.pyplot(fig)


col_left, col_center, col_right = st.columns([1.4, 1, 1.4])
with col_left:
    PM_plot('left')
with col_right:
    PM_plot('right')
    
with col_center:
    fig, ax = plt.subplots(figsize = (8, 2.2*8))
    # plt.axis('off')
    # plt.axis('equal')
    ax.set(xlim = (0, 100), ylim = (0, 2.2*100))
    plt.plot([10, 50], [10, 50])
    # lines.Line2D([1., 2., 30.], [10., 20.],  linewidth = 3.)
    ax.add_patch(patches.Rectangle((50, 50), 50, 50, color = 'green'))
    ax.add_patch(patches.Circle((20, 30), 10, color = 'blue'))
    # plt.plot([50, 100])
    print(fig)
    st.pyplot(fig)


# print(fig.dpi,'dpi')
# print(plt.rcParams['figure.dpi'], plt.rcParams['figure.figsize'])
###? Plot   ####################################################


###? creating a DataFrame   ####################################################
# pd.set_option('display.colheader_justify', 'center')
def color_np_c(value, c1, c2):
    if value < 0: color = c1
    else:         color = c2
    return "color: " + color

st.dataframe(df.style.applymap(color_np_c, c1 = 'red', c2 = '', subset = pd.IndexSlice[['e ', 'c ', 'Pn ', 'Pd ', 'e', 'c', 'Pn', 'Pd']])
                .format({'φ': '{:.3f}', 'φ ': '{:.3f}'}, precision = 1, thousands = ',')
                .set_properties(**{'font-size': '150%', 'background-color': '', 'border-color': 'red', 'text-align': 'center'})
                .set_properties(align = "center")
                .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
                # .set_table_styles([{'selector': 'tr:hover', 'props': [('background-color', 'yellow')]}])
                ,width = 1100) #, use_container_width = True)


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
###? creating a DataFrame   ####################################################



