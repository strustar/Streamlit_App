import os
os.system('cls')
# import time
import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.patches as patches
# import matplotlib.lines as lines

# -- Set page config
# emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.set_page_config(page_title = "P-M Diagram", page_icon = ":star2:", layout = "wide",    # centered, wide
# st.set_page_config(page_title = "P-M Diagram", page_icon = ":star2:", layout = "centered",    # centered, wide
    menu_items = {
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    })

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sidebar setting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
st.sidebar.markdown('## :blue[Design Code] ##')
[col1, col2] = st.sidebar.columns([1,1], gap = "small")
with col1:
    RC_Code = st.selectbox(':green[RC Code]', ('KCI-2012', 'KDS-2021'))
with col2:
    FRP_Code = st.selectbox(':green[FRP Code]', ('AASHTO-2018', 'ACI 440.1R-06(15)', 'ACI 440.11-22'))

# st.sidebar.markdown('---')
st.sidebar.markdown('## :blue[Column Type] ##')
Column_Type = st.sidebar.radio('Column Type', ('Tied Column', 'Spiral Column'), horizontal = True, label_visibility = 'collapsed')

st.sidebar.markdown('## :blue[Material Properties] ##')
[col1, col2, col3] = st.sidebar.columns(3)
with col1:
    fck = st.number_input(':green[$f_{ck}$ [MPa]]', min_value = 0.1, value = 28.0, step = 1., format = '%f')
    Ec = st.number_input(':green[$E_{c}$ [GPa]]', min_value = 0.1, value = 30.0, step = 1., format = '%f', disabled = True)    
with col2:
    fy = st.number_input(':green[$f_{y}$ [MPa]]', min_value = 0.1, value = 400.0, step = 10., format = '%f')
    Es = st.number_input(':green[$E_{s}$ [GPa]]', min_value = 0.1, value = 200.0, step = 10., format = '%f')
with col3:
    ffu = st.number_input(':green[$f_{fu}$ [MPa]]', min_value = 0.1, value = 560.0, step = 10., format = '%f')
    Ef = st.number_input(':green[$E_{f}$ [GPa]]', min_value = 0.1, value = 45.0, step = 1., format = '%f')

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
# ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sidebar setting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# web app 설정
with st.expander("왼쪽 사이드바(sidebar)를 적당한 크기로 하시고, 화면은 다크모드(어둡게)로 설정하세요. 클릭하세요"):
    st.write("#### :blue[Edge browser : 설정 >> 브라우저 디스플레이 (다크모드로 변경)] ####")
    st.write("#### :blue[Chrome browser : 설정 >> 모양 (다크모드로 변경)] ####")

# Input, Output 변수 설정
class In:
    pass
class PM:
    pass
# In.RC_Code = RC_Code;  In.FRP_Code = FRP_Code;  In.Column_Type = Column_Type;  In.Section_Type = Section_Type
[In.RC_Code, In.FRP_Code, In.Column_Type, In.Section_Type] = [RC_Code, FRP_Code, Column_Type, Section_Type]
[In.fck, In.fy, In.ffu, In.Ec, In.Es, In.Ef] = [fck, fy, ffu, Ec, Es, Ef]
[In.b, In.h, In.D] = [b, h, D]
[In.Layer, In.dia, In.dc, In.nh, In.nb, In.nD] = [Layer, dia, dc, nh, nb, nD]
import PM_Cal
# print(In.nD, nh)

In.Reinforcement_Type = 'RC'
PM_Cal.Cal(In, PM)
In.Reinforcement_Type = 'FRP'
PM_Cal.Cal(In, PM)
# print(PM.eta, PM.beta1, PM.alpha, PM.phi0)


# [i for i in [1, 2, 3]]
# sample = 3
# True if sample > 2 else False


# Plot
plt.style.use('default')  # 'dark_background'
fx = 6
plt.rcParams['figure.figsize'] = (fx, fx)  # 13in.*23in. (27in. Monitor 모니터)
# plt.rcParams['figure.dpi'] = 200

# plt.rcParams['figure.facecolor'] = 'gainsboro'
# plt.rcParams['axes.facecolor'] = 'green'
# plt.rcParams['font.size'] = 12
# print(plt.style.available)

# col1, col2, col3 = st.columns([1.4, 1, 1.4])
# with col1:
#     fig, ax = plt.subplots() #(figsize = (5, 5))
#     # plt.axis('off')
#     plt.axis('equal')
#     ax.add_patch(patches.Circle((20, 30), 10, color = 'blue'))
#     ax.set(xlim = (0, 100), ylim = (0, 100))
#     # ax.set_xlabel('Performance', fontsize = 12)
#     ax.set_xlabel('Performance')
#     ax.set_ylabel('Performance')

#     xmin, xmax, ymin, ymax = plt.axis()
#     print(fig)
#     st.pyplot(fig)
#     st.write('# Example here #')
# with col3:
#     st.write('# Example here #')
# with col2:
#     fig, ax = plt.subplots(figsize = (fx, 2.2*fx))
#     plt.axis('off')
#     plt.axis('equal')
#     ax.set(xlim = (0, 100), ylim = (0, 2.2*100))
#     plt.plot([10, fck])
#     # lines.Line2D([1., 2., 30.], [10., 20.],  linewidth = 3.)
#     ax.add_patch(patches.Rectangle((50, 50), 50, 50, color = 'green'))
#     ax.add_patch(patches.Circle((20, 30), 10, color = 'blue'))
#     plt.plot([50, fy])
#     print(fig)
#     st.pyplot(fig)


# print(fig.dpi,'dpi')
# print(plt.rcParams['figure.dpi'], plt.rcParams['figure.figsize'])
# # creating a DataFrame
# df = pd.DataFrame(
#     np.random.randn(5, 10),
#     columns=('col %d' % i for i in range(10)))

# # displaying the DataFrame
# dd = df.style.highlight_max(axis = 0, color = 'red').set_caption('테스트 이니').format(precision=2)
# st.dataframe(dd)
# # print(selected_row)


# # 캡션 적용
# st.caption('캡션을 한 번 넣어 봤습니다')

# # 마크다운 문법 지원
# st.markdown('streamlit은 **마크다운 문법을 지원**합니다.')
# # 컬러코드: blue, green, orange, red, violet
# st.markdown("텍스트의 색상을 :green[초록색]으로, 그리고 **:blue[파란색]** 볼트체로 설정할 수 있습니다.")
# st.markdown(":green[$\sqrt{x^2+y^2}=1$] 와 같이 latex 문법의 수식 표현도 가능합니다 :pencil:")

# # LaTex 수식 지원
# st.latex(r'\sqrt{x^2+y^2}=1')

