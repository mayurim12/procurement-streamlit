# import streamlit as st
# import pandas as pd
# import numpy as np
# import io
# import os
# import re
# from dotenv import load_dotenv
# from openai import AzureOpenAI
# from langchain_openai import AzureChatOpenAI
# import plotly.express as px

# # Load environment variables
# load_dotenv()

# # Set page configuration with a white background
# st.set_page_config(page_title="📊 Procurement GPT", layout="wide", page_icon="📊", initial_sidebar_state="expanded")

# # Custom style to reduce sidebar width
# st.markdown("""
#     <style>
#     /* Change the sidebar width */
#     section[data-testid="stSidebar"] {
#         width: 200px !important;
#     }

#     /* Push the main content right to avoid overlap */
#     div[data-testid="stSidebarContent"] {
#         padding-right: 10px;
#     }

#     /* Adjust button styles for narrower layout */
#     [data-testid="stSidebar"] button {
#         font-size: 14px;
#         padding: 4px 8px;
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown(
#     """
#     <style>
#         .header-container {
#             display: flex;
#             justify-content: space-between;
#             align-items: center;
#             padding: 10px 20px;
#             background-color: #f9f9f9;
#             border-bottom: 1px solid #ddd;
#         }

#         .logo-container img {
#             height: 40px;
#         }

#         .icon-links a {
#             margin-left: 15px;
#             font-size: 22px;
#             text-decoration: none;
#             color: #333;
#         }

#         .icon-links a:hover {
#             color: #4CAF50;
#         }
#     </style>

#     <div class="header-container">
#         <div class="logo-container">
#             <img src="/assets/i360 Logo.png" alt="Company Logo">
#         </div>
#         <div class="icon-links">
#             <a href="#" title="Settings">⚙️</a>
#             <a href="#" title="User Login">👤</a>
#         </div>
#     </div>
#     """,
#     unsafe_allow_html=True
# )




# # Update the theme of the Streamlit app
# st.markdown(
#     """
#     <style>
#     .css-1d391kg {
#         background-color: white;
#     }
#     .reportview-container {
#         background-color: white;
#         font-family: "Arial", sans-serif;
#         font-size: 18px;
#     }
#     .css-18e3th9 {
#         font-size: 20px;
#     }
#     </style>
#     """, 
#     unsafe_allow_html=True
# )

# st.title("🤖 Procurement Analytics GPT")

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
#         "px": px
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
#             if file.endswith(".html"):
#                 plot_file = file
#                 break

#         return result_df, summary_text, plot_file

#     except Exception as e:
#         return None, f"Error: {str(e)}", None


# def format_summary_text(summary):
#     try:
#         # Try to extract list of dictionaries using ast.literal_eval
#         match = re.search(r'\[.*\]', summary, re.DOTALL)
#         if match:
#             list_str = match.group(0)
#             data = ast.literal_eval(list_str)

#             if isinstance(data, list) and isinstance(data[0], dict):
#                 # Format as Markdown table with improved headers
#                 headers = [h.replace('_', ' ').title() for h in data[0].keys()]
#                 header_row = "| " + " | ".join(headers) + " |"
#                 separator = "| " + " | ".join(["---"] * len(headers)) + " |"
#                 rows = []
#                 for item in data:
#                     row = "| " + " | ".join(
#                         f"{item[h]:,.2f}" if isinstance(item[h], (int, float)) else str(item[h]).upper()
#                         for h in data[0].keys()
#                     ) + " |"
#                     rows.append(row)
#                 table = "\n".join([header_row, separator] + rows)
#                 return f"### Total Spend by Region\n\n{table}"
#     except Exception:
#         pass  # Fall back to basic formatting

#     # Fallback: Clean up plain text and improve the English
#     lines = re.split(r',|\n', summary)
#     cleaned = [line.strip() for line in lines if line.strip()]

#     if len(cleaned) > 1:
#         intro = cleaned[0].rstrip(':') + ":"
#         region_lines = []
#         for line in cleaned[1:]:
#             parts = line.strip().split()
#             if len(parts) == 2:
#                 region, amount = parts
#                 try:
#                     # Try to convert amount to float
#                     amount = float(amount)
#                     region_lines.append(f"- **{region.upper()}**: ${amount:,.2f}")
#                 except ValueError:
#                     # If conversion fails, just add the line as-is
#                     region_lines.append(f"- {line}")
#             else:
#                 region_lines.append(f"- {line}")
#         return f"**{intro}**\n" + "\n".join(region_lines)
#     else:
#         return "\n".join(f"- {line.capitalize()}" for line in cleaned)


# # Load data
# df = pd.read_excel("Sample Data_Copilot.xlsx")

# # --- Sidebar: Session Memory / Chat History ---
# st.sidebar.header("Chat History")

# # Inject CSS to left-align sidebar buttons
# st.sidebar.markdown("""
#     <style>
#     [data-testid="stSidebar"] button {
#         text-align: left !important;
#         justify-content: flex-start !important;
#         width: 80% !important;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if "query_history" not in st.session_state:
#     st.session_state.query_history = []

# # Display history in reverse order (latest on top)
# for i, past_query in enumerate(reversed(st.session_state.query_history[-5:])):
#     if st.sidebar.button(f"{past_query}", key=f"history_{i}"):
#         query = past_query  # Preload into main input

# # Clear history button
# if st.sidebar.button("Clear History"):
#     st.session_state.query_history = []
#     st.experimental_rerun()  # Refresh to reflect the cleared state

# # Layout: Input on top
# query = st.text_area("💬 Hi, I am your AI Assistant! How can I assist you today?", value=query if 'query' in locals() else "", height=130)

# if st.button("Generate Report"):
#     if query:
#         # Save new query to session history
#         if query not in st.session_state.query_history:
#             st.session_state.query_history.append(query)

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
#             1. Write clean and reproducible Python code using pandas and Plotly to answer the following question: *{query}*.
#             2. Assign the final result to a variable named `result_df`.
#             3. Generate a short, natural language summary of the result and assign it to a variable named `summary_text`.
                        
#             4. Create an interactive plot using Plotly that visualizes the key insights. Ensure the plot is interactive, where users can:
#             - Hover over data points to see values.
#             - Zoom in and out.
#             - Use legends to filter the plot.
#             - Save the plot as an HTML file using `fig.write_html('interactive_plot.html')`.

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
#                     # Step 1: Prepare display version (with $ formatting)
#                     display_df = result_df.applymap(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)

#                     # Step 2: UI styling
#                     st.markdown(
#                         """
#                         <style>
#                             .streamlit-table th, .streamlit-table td {
#                                 font-size: 20px;
#                                 padding: 5px;
#                                 text-align: left;
#                             }
#                             .streamlit-table {
#                                 width: 60%;
#                                 margin: auto;
#                             }
#                             body {
#                                 font-size: 18px;
#                             }
#                         </style>
#                         """, unsafe_allow_html=True
#                     )


#                     # Show in Streamlit
#                     st.dataframe(display_df, use_container_width=False, height=300)

#                     # Step 3: Prepare clean export version (no formatted $, no index column)
#                     export_df = result_df.copy()

#                     # 🔍 Remove accidental index column if present
#                     first_col = export_df.columns[0].lower()
#                     if first_col.startswith("unnamed") or first_col == "index":
#                         export_df = export_df.drop(columns=export_df.columns[0])

#                     # Step 4: Excel export function
#                     def to_excel(df):
#                         output = io.BytesIO()
#                         with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#                             df.to_excel(writer, index=False, sheet_name='Data')

#                             workbook  = writer.book
#                             worksheet = writer.sheets['Data']

#                             # Optional currency formatting
#                             currency_format = workbook.add_format({'num_format': '$#,##0.00'})
#                             for i, col in enumerate(df.columns):
#                                 if pd.api.types.is_numeric_dtype(df[col]):
#                                     worksheet.set_column(i, i, 20, currency_format)
#                                 else:
#                                     worksheet.set_column(i, i, 20)
#                         return output.getvalue()

#                     # Step 5: Download button
#                     excel_data = to_excel(export_df)
#                     st.download_button(
#                         label="Download Table as Excel",
#                         data=excel_data,
#                         file_name="result_table.xlsx",
#                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#                     )

#                 else:
#                     st.warning("No DataFrame returned.")



#             with tab2:
#                 st.markdown("### Summary")
#                 formatted_summary = format_summary_text(summary)
#                 st.markdown(formatted_summary or "No summary generated.")



#                 #st.write(summary or "No summary generated.")

#             with tab3:
#                 if plot_path and os.path.exists(plot_path):
#                     st.markdown("### Interactive Plot")
#                     st.components.v1.html(open(plot_path, 'r', encoding='utf-8').read(), height=600)

#                 else:
#                     st.warning("No plot was generated.")
#     else:
#         st.warning("Please enter a query.")


import streamlit as st
import pandas as pd
import numpy as np
import io
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

# Custom style to reduce sidebar width
st.markdown("""
    <style>
    /* Change the sidebar width */
    section[data-testid="stSidebar"] {
        width: 200px !important;
    }

    /* Push the main content right to avoid overlap */
    div[data-testid="stSidebarContent"] {
        padding-right: 10px;
    }

    /* Adjust button styles for narrower layout */
    [data-testid="stSidebar"] button {
        font-size: 14px;
        padding: 4px 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 20px;
            background-color: #f9f9f9;
            border-bottom: 1px solid #ddd;
        }

        .logo-container img {
            height: 40px;
        }

        .icon-links a {
            margin-left: 15px;
            font-size: 22px;
            text-decoration: none;
            color: #333;
        }

        .icon-links a:hover {
            color: #4CAF50;
        }
    </style>

    <div class="header-container">
        <div class="logo-container">
            <img src="./assets/i360.png" alt="Company Logo">
        </div>
        <div class="icon-links">
            <a href="#" title="Settings">⚙️</a>
            <a href="#" title="User Login">👤</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Update the theme of the Streamlit app ---
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

st.title("🤖 Procurement Analytics GPT")

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



def format_summary_text(summary):
    try:
        # Try to extract list of dictionaries using ast.literal_eval
        match = re.search(r'\[.*\]', summary, re.DOTALL)
        if match:
            list_str = match.group(0)
            data = ast.literal_eval(list_str)

            if isinstance(data, list) and isinstance(data[0], dict):
                # Format as Markdown table with improved headers
                headers = [h.replace('_', ' ').title() for h in data[0].keys()]
                header_row = "| " + " | ".join(headers) + " |"
                separator = "| " + " | ".join(["---"] * len(headers)) + " |"
                rows = []
                for item in data:
                    row = "| " + " | ".join(
                        f"{item[h]:,.2f}" if isinstance(item[h], (int, float)) else str(item[h]).upper()
                        for h in data[0].keys()
                    ) + " |"
                    rows.append(row)
                table = "\n".join([header_row, separator] + rows)
                return f"### Total Spend by Region\n\n{table}"
    except Exception:
        pass  # Fall back to basic formatting

    # Fallback: Clean up plain text and improve the English
    lines = re.split(r',|\n', summary)
    cleaned = [line.strip() for line in lines if line.strip()]

    if len(cleaned) > 1:
        intro = cleaned[0].rstrip(':') + ":"
        region_lines = []
        for line in cleaned[1:]:
            parts = line.strip().split()
            if len(parts) == 2:
                region, amount = parts
                try:
                    # Try to convert amount to float
                    amount = float(amount)
                    region_lines.append(f"- **{region.upper()}**: ${amount:,.2f}")
                except ValueError:
                    # If conversion fails, just add the line as-is
                    region_lines.append(f"- {line}")
            else:
                region_lines.append(f"- {line}")
        return f"**{intro}**\n" + "\n".join(region_lines)
    else:
        return "\n".join(f"- {line.capitalize()}" for line in cleaned)


# Load data
df = pd.read_excel("Sample Data_Copilot 1.xlsx")

# --- Sidebar: Session Memory / Chat History ---
st.sidebar.header("Chat History")

# Inject CSS to left-align sidebar buttons
st.sidebar.markdown("""
    <style>
    [data-testid="stSidebar"] button {
        text-align: left !important;
        justify-content: flex-start !important;
        width: 80% !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# Display history in reverse order (latest on top)
for i, past_query in enumerate(reversed(st.session_state.query_history[-5:])):
    if st.sidebar.button(f"{past_query}", key=f"history_{i}"):
        query = past_query  # Preload into main input

# Clear history button
if st.sidebar.button("Clear History"):
    st.session_state.query_history = []
    st.experimental_rerun()  # Refresh to reflect the cleared state

# Layout: Input on top
query = st.text_area("💬 Hi, I am your AI Assistant! How can I assist you today?", value=query if 'query' in locals() else "", height=130)

if st.button("Generate Report"):
    if query:
        # Save new query to session history
        if query not in st.session_state.query_history:
            st.session_state.query_history.append(query)

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
                    # Step 1: Prepare display version (with $ formatting)
                    display_df = result_df.applymap(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)

                    # Show in Streamlit
                    st.dataframe(display_df, use_container_width=False, height=300)

                else:
                    st.warning("No DataFrame returned.")


            with tab2:
                st.markdown("### Summary")
                formatted_summary = format_summary_text(summary)
                st.markdown(formatted_summary or "No summary generated.")

            with tab3:
                if plot_path and os.path.exists(plot_path):
                    st.markdown("### Plot") 
                    # Adjust the plot size here
                    st.components.v1.html(open(plot_path, 'r', encoding='utf-8').read(), height=1000, width=1300)

                else:
                    st.warning("No plot was generated.")
    else:
        st.warning("Please enter a query.")
