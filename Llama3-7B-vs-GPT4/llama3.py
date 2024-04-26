from dotenv import load_dotenv
import seaborn as sns
import streamlit as st 
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

load_dotenv()

st.title("Llama-3-70b")
df = sns.load_dataset("titanic")
st.write(df.head(3))

agent = create_pandas_dataframe_agent(
    ChatGroq(model_name="llama3-70b-8192", temperature=0),
    df,
    verbose=True
)

prompt = st.text_input("Enter your prompt:")

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response..."):
            response = agent.invoke(prompt)
            st.write(response["output"])

