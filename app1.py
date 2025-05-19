# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import re
# from dotenv import load_dotenv
# from openai import AzureOpenAI
# from langchain_openai import AzureChatOpenAI

# # Load environment variables
# load_dotenv()

# st.set_page_config(page_title="📊 Procurement GPT", layout="wide")
# st.title("🤖 Procurement GPT")


# # Initialize OpenAI client
# def init_openai_client():
#     return AzureChatOpenAI(
#         api_key=os.getenv("AZURE_API_KEY"),
#         api_version=os.getenv("AZURE_API_VERSION"),
#         azure_endpoint=os.getenv("AZURE_ENDPOINT"),
#         azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
#     )

# # Extract Python code from LLM response
# def extract_python_code(text):
#     match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
#     return match.group(1) if match else ""

# # Execute the generated code
# def execute_pandas_code(code, df):
#     local_vars = {"df": df}
#     globals_dict = {
#         "pd": pd,
#         "np": np,
#         "plt": plt,
#     }

#     try:
#         clean_code = extract_python_code(code)
#         exec(clean_code, globals_dict, local_vars)

#         result_vars = [
#             var for var in local_vars.keys()
#             if isinstance(local_vars[var], pd.DataFrame)
#         ]
#         result_df = local_vars[result_vars[-1]] if result_vars else None
#         summary_text = local_vars.get("summary_text", "No summary was generated.")
#         plot_file = None
#         for file in os.listdir():
#             if file.endswith(".png"):
#                 plot_file = file
#                 break

#         return result_df, summary_text, plot_file

#     except Exception as e:
#         return None, f"Error: {str(e)}", None

# # Load data
# df = pd.read_excel("Sample Data_Copilot.xlsx")


# query = st.text_area("💬 Ask a question about the data:", height=100)

# if st.button("Generate Report"):
#     if query:
#         with st.spinner("⏳ Thinking..."):
#             client = init_openai_client()
            
#             df_info = f"""
#             Columns and their types:
#             {df.dtypes.to_string()}

#             Sample data (first few rows):
#             {df.head().to_string()}
#             """

#             prompt = f"""
#             You are a data analyst assistant. You are given a DataFrame with the following structure:

#             {df_info}

#             Your task is to:
#             1. Write clean and reproducible Python code using pandas and matplotlib to answer the following question: *{query}*.
#             2. Assign the final result to a variable named `result_df`.
#             3. Generate a short, natural language summary of the result and assign it to a variable named `summary_text`.
                        
#             4. Create a plot that visualizes the key insights and save it using `plt.savefig('top_vendors_plot.png')`. 
#             - Adjust the figure size dynamically based on the number and length of labels.

#             - Rotate x-axis or y-axis labels if needed.
#             - Use `plt.tight_layout()` or `bbox_inches='tight'` to prevent clipping

#             5. Do not use random sampling or introduce variability in the output.
#             6. Ensure the code is deterministic and produces the same result every time it is run.

#             Output only the Python code in a single code block.
#             """

#             ai_response = client.invoke(prompt)
#             result_df, summary, plot_path = execute_pandas_code(ai_response.content, df)

#             # Output in tabs
#             tab1, tab2, tab3 = st.tabs(["📄 Table", "📝 Summary", "📊 Plot"])

#             with tab1:
#                 if result_df is not None:
#                     st.dataframe(result_df, use_container_width=True)
#                 else:
#                     st.warning("No DataFrame returned.")

#             with tab2:
#                 st.markdown("### Summary")
#                 st.write(summary or "No summary generated.")

#             with tab3:
#                 if plot_path and os.path.exists(plot_path):
#                     st.image(plot_path, use_column_width=True)
#                 else:
#                     st.warning("No plot was generated.")
#     else:
#         st.warning("Please enter a query.")


import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
import plotly.express as px

# Load environment variables
load_dotenv()

# Set page configuration with a white background
st.set_page_config(page_title="📊 Procurement GPT", layout="wide", page_icon="📊", initial_sidebar_state="expanded")

# Update the theme of the Streamlit app
st.markdown(
    """
    <style>
    .css-1d391kg {
        background-color: white;
    }
    .reportview-container {
        background-color: white;
        font-family: "Arial", sans-serif;
        font-size: 18px;
    }
    .css-18e3th9 {
        font-size: 20px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

st.title("🤖 Procurement GPT")

# Initialize OpenAI client
def init_openai_client():
    return AzureChatOpenAI(
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
    )

# Extract Python code from LLM response
def extract_python_code(text):
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    return match.group(1) if match else ""

# Execute the generated code
def execute_pandas_code(code, df):
    local_vars = {"df": df}
    globals_dict = {
        "pd": pd,
        "np": np,
        "px": px
    }

    try:
        clean_code = extract_python_code(code)
        exec(clean_code, globals_dict, local_vars)

        result_vars = [
            var for var in local_vars.keys()
            if isinstance(local_vars[var], pd.DataFrame)
        ]
        result_df = local_vars[result_vars[-1]] if result_vars else None
        summary_text = local_vars.get("summary_text", "No summary was generated.")
        plot_file = None
        for file in os.listdir():
            if file.endswith(".html"):
                plot_file = file
                break

        return result_df, summary_text, plot_file

    except Exception as e:
        return None, f"Error: {str(e)}", None

# Load data
df = pd.read_excel("Sample Data_Copilot.xlsx")

# Layout: Input on top
query = st.text_area("💬 Ask a question about the data:", height=100)

if st.button("Generate Report"):
    if query:
        with st.spinner("⏳ Thinking..."):
            client = init_openai_client()
            
            df_info = f"""
            Columns and their types:
            {df.dtypes.to_string()}

            Sample data (first few rows):
            {df.head().to_string()}
            """

            prompt = f"""
            You are a data analyst assistant. You are given a DataFrame with the following structure:

            {df_info}

            Your task is to:
            1. Write clean and reproducible Python code using pandas and Plotly to answer the following question: *{query}*.
            2. Assign the final result to a variable named `result_df`.
            3. Generate a short, natural language summary of the result and assign it to a variable named `summary_text`.
                        
            4. Create an interactive plot using Plotly that visualizes the key insights. Ensure the plot is interactive, where users can:
            - Hover over data points to see values.
            - Zoom in and out.
            - Use legends to filter the plot.
            - Save the plot as an HTML file using `fig.write_html('interactive_plot.html')`.

            5. Do not use random sampling or introduce variability in the output.
            6. Ensure the code is deterministic and produces the same result every time it is run.

            Output only the Python code in a single code block.
            """

            ai_response = client.invoke(prompt)
            result_df, summary, plot_path = execute_pandas_code(ai_response.content, df)

            # Output in tabs
            tab1, tab2, tab3 = st.tabs(["📄 Table", "📝 Summary", "📊 Plot"])

            with tab1:
                if result_df is not None:
                    st.dataframe(result_df, use_container_width=True)
                else:
                    st.warning("No DataFrame returned.")

            with tab2:
                st.markdown("### Summary")
                st.write(summary or "No summary generated.")

            with tab3:
                if plot_path and os.path.exists(plot_path):
                    st.markdown("### Plot")
                    st.components.v1.html(open(plot_path, 'r', encoding='utf-8').read(), height=600)

                else:
                    st.warning("No plot was generated.")
    else:
        st.warning("Please enter a query.")
