import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import re
import json
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import plotly.express as px
import plotly.graph_objects as go
from faiss_similarity_search import get_few_shot_examples 
from typing import List, Tuple, Optional 
import ast

# Load environment variables
load_dotenv()

# Set page configuration with a white background
st.set_page_config(page_title="Procurement GPT", layout="wide", page_icon="assets/i360.png", initial_sidebar_state="expanded")

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

# Render the logo using Streamlit's `st.image`
st.image("assets/i360.png", width=100)

# --- Update the theme of the Streamlit app ---
st.markdown(
    """
    <style>
    .css-1d391kg {
        background-color: #ffffff;
    }
    .reportview-container {
        background-color: #ffffff;
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

st.title("Procurement Analytics GPT")

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
        print("\n---Executing code-----------:\n", clean_code)
        exec(clean_code, globals_dict, local_vars)

        result_vars = [
            var for var in local_vars.keys()
            if isinstance(local_vars[var], pd.DataFrame)
        ]
        result_df = local_vars[result_vars[-1]] if result_vars else None
        return result_df

    except Exception as e:
        return None

def generate_summary_with_pandas_code(
    question: str,
    pandas_code: str,
    df: pd.DataFrame,
    client,
    **kwargs,
) -> str:
    """
    Generate a summary of the resulted DataFrame using the provided pandas code.

    Args:
        question (str): The question that was asked.
        pandas_code (str): The LLM-generated pandas code.
        df (pd.DataFrame): The results of the pandas operation.
        client: The LLM client.
    Returns:
        str: A markdown-formatted summary of the data output.
    """
    system_message = (
        f"You are a helpful data assistant. The user asked the question: '{question}'.\n\n"
        f"The pandas code used to generate the result was:\n{pandas_code}\n\n"
        f"The following is a pandas DataFrame with the results of the query:\n{df.to_markdown()}\n\n"
    )

    user_message = (
        "Provide the summary in a short, concise, and precise manner in markdown format.\n"
        "- Do not include any prefixes like `markdown\\n` or additional metadata in the response.\n"
        "- Ensure the output begins directly with the markdown content.\n"
        "- Use bullet points, DO NOT make tables, but use other appropriate formatting as needed.\n"
        "- Use a title at the top of the summary in **H2** format (e.g., `## Summary`).\n"
        "- Ensure the output is styled for readability in light themes (no black background or dark fonts).\n"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    response = client.invoke(messages)

    return response.content

class MarkdownCodeExtractor:
    """A class to extract code blocks from Markdown content."""

    def __init__(self, content: str):
        """Initialize with markdown content.

        Args:
            content (str): The markdown content to process
        """
        self.content = content

    def extract_code_blocks(self) -> List[Tuple[Optional[str], str]]:
        """Extract all code blocks from the markdown content.

        Returns:
            List[Tuple[Optional[str], str]]: List of tuples containing (language, code)
                where language might be None if not specified
        """
        # Pattern matches both fenced code blocks (```) and indented code blocks
        pattern = r"```(\w*)\n(.*?)\n```|(?:(?:^[ ]{4}.*?\n)+)"
        matches = re.finditer(pattern, self.content, re.MULTILINE | re.DOTALL)

        code_blocks = []
        for match in matches:
            if match.group(1) is not None:  # Fenced code block
                language = match.group(1) or None
                code = match.group(2).strip()
            else:  # Indented code block
                language = None
                # Remove the 4 space indent from each line
                code = "\n".join(line[4:] for line in match.group(0).splitlines())

            code_blocks.append((language, code))

        return code_blocks

    def extract_python_code(self) -> List[str]:
        """Extract only Python code blocks from the markdown content.

        Returns:
            List[str]: List of Python code blocks
        """
        code_blocks = self.extract_code_blocks()
        return [
            code for lang, code in code_blocks if lang in (None, "", "python", "py")
        ] 
        
def create_plotly_figure(code: str, df: pd.DataFrame) -> go.Figure:
    """
    Create and return a Plotly figure from code.
    """
    try:
        local_dict = {"df": df, "px": px, "go": go}
        exec(code, globals(), local_dict)
        fig = local_dict.get("fig")
        return fig
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        return None 

def quick_encode_plot(plot):
    return json.loads(plot.to_json())  

def get_llm_plot_suggestions(question: str, df: pd.DataFrame, client) -> str:
    """
    Get plot suggestions from LLM based on the question and data.
    """
    df_info = f"""
    Columns and their types:
    {df.dtypes.to_string()}
    
    Sample data (first few rows):
    {df.head().to_string()}
    """

    prompt = f"""
    Question: {question}
    DataFrame Info: {df_info}
    
    Based on this data, suggest Plotly code that can help best to visualize the results. Return only the Python code without any explanation.
    Use basic Plotly Express (px) or Plotly Graph Objects (go) functions. The plot must have title and make sure it uses plotly_dark template.
    Do not reinstialize the df variable. use the df variable that is already defined.
    The data is in a pandas DataFrame called 'df'. Under no circumstances, do not use 'fig.show()' in the code, just end the code after generating the fig object.
    
    """
    messages = (
            {
                "role": "system",
                "content": "You are a data visualization expert. Provide only Python code without explanations.",
            },
            {"role": "user", "content": prompt},
    )
    response = client.invoke(messages,temperature=0.7)

    llm_code = response.content
    return llm_code 

def generate_plots(df: pd.DataFrame, question, client):
    """
    Generate both standard and LLM-suggested plots.
    """
    plots = {}

    if question:
        llm_code = get_llm_plot_suggestions(question, df, client=client)
        if "python" in llm_code:
            extractor = MarkdownCodeExtractor(llm_code)
            llm_code = extractor.extract_python_code()[0]

        if llm_code:
            fig = create_plotly_figure(llm_code, df)
            if fig:
                plots["LLM Suggested Plot"] = fig

    return plots
    

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
    st.rerun()  # Refresh to reflect the cleared state

# Layout: Input on top
query = st.text_area("💬 Hi, I am your AI Assistant! How can I assist you today?", value=query if 'query' in locals() else "", height=130)

# Update the logic to use `generate_summary_with_pandas_code` for summary and `get_llm_plot_suggestions` for plot generation.
if st.button("Generate Report"):
    if query:
        examples = get_few_shot_examples(query, num_examples=1)
        # Save new query to session history
        if query not in st.session_state.query_history:
            st.session_state.query_history.append(query)

        with st.spinner("⏳ Generating..."):
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
            1. The data is already loaded as a pandas DataFrame called 'df'. Do Not include any code to create or load the dataframe.
            1. Write clean and reproducible Python code using pandas to answer the following question: *{query}*.
            2. Assign the final result to a variable named `result_df`.
            3. Ensure the final resultant DataFrame is assigned to `result_df` as the last variable in the code block.
            4. Make sure to reset index of the result_df.
            5. Must add "$" sign to the Spend column if it exists in the DataFrame. 
            Output only the Python code in a single code block.
            """
            
            if examples:
                prompt += f"\n\nHere are some similar questions and their corresponding code: {examples}\n"
            print("\n---Prompt is: \n", prompt)
            ai_response = client.invoke(prompt)
            result_df = execute_pandas_code(ai_response.content, df)
            if "Spend" in result_df.columns:
                result_df["Spend"] = result_df["Spend"].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)
            print(result_df.head())
            

            if result_df is not None:
                # Generate summary using `generate_summary_with_pandas_code`
                summary = generate_summary_with_pandas_code(
                    question=query,
                    pandas_code=ai_response.content,
                    df=result_df,
                    client=client
                )

                # Generate plot using `get_llm_plot_suggestions`
                llm_code = get_llm_plot_suggestions(query, result_df, client)
                extractor = MarkdownCodeExtractor(llm_code)
                llm_code = extractor.extract_python_code()[0]
                fig = create_plotly_figure(llm_code, result_df)

                # Output in tabs
                tab1, tab2, tab3 = st.tabs(["📄 Table", "📝 Summary", "📊 Plot"])
                with tab1:
                    # Ensure the "Spend" column is formatted with a dollar sign and converted to a string
                    if "Spend" in result_df.columns:
                        result_df["Spend"] = result_df["Spend"].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)

                    display_df = result_df  # Assign the formatted DataFrame
                    st.dataframe(display_df, use_container_width=False, height=300)

                with tab2:
                    st.markdown(summary or "No summary generated.")

                with tab3:
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No plot was generated.")

            else:
                st.warning("No DataFrame returned.")
    else:
        st.warning("Please enter a query.")
