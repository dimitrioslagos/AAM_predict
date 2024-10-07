import streamlit as st
import pandas as pd
import math
from pathlib import Path



# Set the title of the Streamlit app
st.title("Short Term Asset Management")

# Define the tabs
tabs = st.tabs(["Historical Data Input","End of Life Curves","Critical Meters Calculation"])
