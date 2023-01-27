# import os
# os.system('cls')
import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -- Set page config
# emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.set_page_config(page_title = "P-M Diagram", page_icon = ":star2:", layout = "wide",    # centered, wide
    menu_items = {
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    })

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sidebar setting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
st.sidebar.markdown('## :blue[Design Code] ##')
col1, col2 = st.sidebar.columns([1,1], gap = "small")
with col1:    
    RC_Code = st.selectbox(':green[RC Code]', ('KCI-2012', 'KDS-2021'))
with col2:
    FRP_Code = st.selectbox(':green[FRP Code]', ('AASHTO-2018', 'ACI 440.1R-06(15)', 'ACI 440.11-22'))

# st.sidebar.markdown('---')
st.sidebar.markdown('## :blue[Column Type] ##')
Column_Type = st.sidebar.radio('Column Type', ('Tied Column', 'Spiral Column'), horizontal = True, label_visibility = 'collapsed')

st.sidebar.markdown('## :blue[Material Properties] ##')
col1, col2, col3 = st.sidebar.columns(3)
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
col1, col2, col3 = st.sidebar.columns(3)
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
if 'Layer 1' in Layer:
    Layer = 1
elif 'Layer 2' in Layer:
    Layer = 2
elif 'Layer 3' in Layer:
    Layer = 3

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    dia1 = st.number_input(':green[dia1 [mm]]', min_value = 0.1, value = 19.1, step = 1., format = '%f', help = '보강재 직경1')
    dc1 = st.number_input(':green[dc1 [mm]]', min_value = 0.1, value = 59.1, step = 1., format = '%f', help = '피복 두께1')
    if "Rectangle" in Section_Type:
        nh1 = st.number_input(':green[nh1 [EA]]', min_value = 1, value = 3, step = 1, format = '%d', help = 'h방향 보강재 개수1')
        nb1 = st.number_input(':green[nb1 [EA]]', min_value = 1, value = 3, step = 1, format = '%d', help = 'b방향 보강재 개수1')
    else:
        nD1 = st.number_input(':green[nD1 [EA]]', min_value = 1, value = 8, step = 1, format = '%d', help = '원형 단면 총 보강재 개수1')
if Layer == 2 or Layer == 3:
    with col2:
        dia2 = st.number_input(':green[dia2 [mm]]', min_value = 0.1, value = 19.1, step = 1., format = '%f', help = '보강재 직경2')
        dc2 = st.number_input(':green[dc2 [mm]]', min_value = 0.1, value = 100., step = 1., format = '%f', help = '피복 두께2')
        if "Rectangle" in Section_Type:
            nh2 = st.number_input(':green[nh2 [EA]]', min_value = 1, value = 3, step = 1, format = '%d', help = 'h방향 보강재 개수2')
            nb2 = st.number_input(':green[nb2 [EA]]', min_value = 1, value = 3, step = 1, format = '%d', help = 'b방향 보강재 개수2')
        else:
            nD2 = st.number_input(':green[nD2 [EA]]', min_value = 1, value = 8, step = 1, format = '%d', help = '원형 단면 총 보강재 개수2')
if Layer == 3:
    with col3:
        dia3 = st.number_input(':green[dia_{3} [mm]]', min_value = 0.1, value = 19.1, step = 1., format = '%f', help = '보강재 직경3')
        dc3 = st.number_input(':green[dc3 [mm]]', min_value = 0.1, value = 140., step = 1., format = '%f', help = '피복 두께3')
        if "Rectangle" in Section_Type:
            nh3 = st.number_input(':green[nh3 [EA]]', min_value = 1, value = 3, step = 1, format = '%d', help = 'h방향 보강재 개수3')
            nb3 = st.number_input(':green[nb3 [EA]]', min_value = 1, value = 3, step = 1, format = '%d', help = 'b방향 보강재 개수3')
        else:
            nD3 = st.number_input(':green[nD3 [EA]]', min_value = 1, value = 8, step = 1, format = '%d', help = '원형 단면 총 보강재 개수3')

st.sidebar.markdown('## :blue[Load Case (LC)] ##')
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    Pu = st.number_input(':green[$P_{u}$ [kN]]', min_value = 10., value = 1500.0, step = 100., format = '%f')
with col2:
    Mu = st.number_input(':green[$M_{u}$ [kN $\cdot$ m]]', min_value = 10., value = 100.0, step = 10., format = '%f')
# ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sidebar setting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# web app 설정
with st.expander("왼쪽 사이드바(sidebar)를 적당한 크기로 하시고, 화면은 다크모드(어둡게)로 설정하세요. 클릭하세요"):
    st.write("#### :blue[Edge browser : 설정 >> 브라우저 디스플레이 (다크모드로 변경)] ####")
    st.write("#### :blue[Chrome browser : 설정 >> 모양 (다크모드로 변경)] ####")

# __name__
dia = dia1
if Layer == 2:
    dia = [dia1, dia2]
if Layer == 3:
    dia = [dia1, dia2, dia3]

class In:
    dia = 0

In1 = In()
# In.dia = dia

a = [dia, RC_Code, b, h]
print(In.dia)

fig, ax = plt.subplots()

fruits = ['apple', 'blueberry', 'cherry', 'orange']
counts = [ffu, 200, 30, 1e2]
if 'KDS' in RC_Code:
    counts = [50, 20, 30, 75]

bar_labels = ['red', 'blue', '_red', 'orange']
bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

ax.set_ylabel('fruit supply')
ax.set_title('Fruit supply by kind and color')
ax.legend(title='Fruit color')


st.pyplot(fig)


d = 4
print(d)


#st.empty()

# st.sidebar.markdown('##### Design **_Code_** :red[colored red] #####')
# st.sidebar.header('Design Code')


# 캡션 적용
st.caption('캡션을 한 번 넣어 봤습니다')


# 마크다운 문법 지원
st.markdown('streamlit은 **마크다운 문법을 지원**합니다.')
# 컬러코드: blue, green, orange, red, violet
st.markdown("텍스트의 색상을 :green[초록색]으로, 그리고 **:blue[파란색]** 볼트체로 설정할 수 있습니다.")
st.markdown(":green[$\sqrt{x^2+y^2}=1$] 와 같이 latex 문법의 수식 표현도 가능합니다 :pencil:")

# LaTex 수식 지원
st.latex(r'\sqrt{x^2+y^2}=1')

