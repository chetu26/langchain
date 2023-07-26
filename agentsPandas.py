import os
import pandas as pd
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import AzureOpenAI

class DataQuestionAnswerer:
    def __init__(self, csv_file_path, azure_api_key):
        self.csv_file_path = csv_file_path
        self.azure_api_key = azure_api_key

    def read_data(self):
        self.df = pd.read_csv(self.csv_file_path)

    def setup_openai_environment(self):
        load_dotenv()
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = "2023-05-15"
        openai.api_type = os.environ["OPENAI_API_BASE"]
        os.environ["OPENAI_API_BASE"]
        openai.api_key = self.azure_api_key
        os.environ["OPENAI_API_KEY"]

    def create_language_model(self):
        self.llm = AzureOpenAI(
            openai_api_type="azure",
            deployment_name="GPT3-5",
            model_name="gpt-35-turbo"
        )

    def create_dataframe_agent(self):
        self.agent = create_pandas_dataframe_agent(self.llm, self.df, verbose=True)

    def answer_question(self, question):
        try:
            output = self.agent.run(question)
            return output['final_answer']
        except Exception as e:
            return f"Error: {e}"

csv_file_path = "file.csv"
azure_api_key = "your_azure_api_key_here"

data_answerer = DataQuestionAnswerer(csv_file_path, azure_api_key)

data_answerer.read_data()
data_answerer.setup_openai_environment()
data_answerer.create_language_model()
data_answerer.create_dataframe_agent()

question = "how many rows and columns are there?"
answer = data_answerer.answer_question(question)
print(f"Answer: {answer}")
