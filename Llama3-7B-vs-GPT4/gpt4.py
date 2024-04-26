from dotenv import load_dotenv
import seaborn as sns
import streamlit as st 
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

load_dotenv()

st.title("GPT-4")
df = sns.load_dataset("titanic")
st.write(df.head(3))

agent = create_pandas_dataframe_agent(
    ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0),
    df,
    verbose=True
)

prompt = st.text_input("Enter your prompt:")

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response..."):
            response = agent.invoke(prompt)
            st.write(response["output"])

