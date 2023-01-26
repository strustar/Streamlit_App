# import os
# os.system('cls')

import streamlit as st
import time

# -- Set page config
# emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.set_page_config(page_title = "P-M Diagram", page_icon = ":star2:", layout = "wide",
    menu_items = {
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    })

# st.snow()
# st.ballons()
color = st.color_picker('Pick A Color', '#00f900')
st.write('The current color is', color)

# Sidebar setting
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
col1, col2 = st.sidebar.columns([1,2])
with col1:
    Section_Type = st.radio('Section Type', ('Rectangle', 'Circle'), horizontal = True, label_visibility = 'collapsed')
with col2:
    if "Rectangle" in Section_Type:
        b = 400.
        b = st.number_input(':green[$b$ [mm]]', min_value = 0.1, value = b, step = 1., format = '%f')
        
    else:
        st.write('tt')





print(fck*0.1)
# st.help(time)

#st.empty()
with st.expander("See explanation"):
    st.write("""
        The chart above shows some numbers I picked for you.
        I rolled actual dice for these, so they're *guaranteed* to
        be random.
    """)
    st.image("https://static.streamlit.io/examples/dice.jpg")

# st.sidebar.markdown('##### Design **_Code_** :red[colored red] #####')
# st.sidebar.header('Design Code')



# 타이틀 적용 예시
st.title('이것은 타이틀 입니다')

# 특수 이모티콘 삽입 예시
# emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.title('스마일 :sunglasses:')

# Header 적용
st.header('헤더를 입력할 수 있어요! :sparkles:')

# Subheader 적용
st.subheader('이것은 subheader 입니다 :star2:')

# 캡션 적용
st.caption('캡션을 한 번 넣어 봤습니다')

# 코드 표시
sample_code = '''
def function():
    print('hello, world')
'''
st.code(sample_code, language="python")

# 일반 텍스트
st.text('일반적인 텍스트를 입력')

# 마크다운 문법 지원
st.markdown('streamlit은 **마크다운 문법을 지원**합니다.')
# 컬러코드: blue, green, orange, red, violet
st.markdown("텍스트의 색상을 :green[초록색]으로, 그리고 **:blue[파란색]** 볼트체로 설정할 수 있습니다.")
st.markdown(":green[$\sqrt{x^2+y^2}=1$] 와 같이 latex 문법의 수식 표현도 가능합니다 :pencil:")

# LaTex 수식 지원
st.latex(r'\sqrt{x^2+y^2}=1')


