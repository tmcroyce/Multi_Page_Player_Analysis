import streamlit as st
import plotly.graph_objs as go

data = [go.Scatter(x=[1, 2, 3], y=[4, 5, 6])]

layout = go.Layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

fig = go.Figure(data=data, layout=layout)





custom_css = """
<style>
[class="user-select-none svg-container"] {
background: linear-gradient(to right, #2c3333, #0e1117);

}
</style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

st.markdown('<div style="background-color: #ADD8E6;">', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.plotly_chart(fig)