import os
from langchain.llms import AzureOpenAI
import openai
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class GPT3_5Model:
    def __init__(self, api_key, deployment_name="GPT3-5", model_name="text-davinci-002",temperature=0.0):
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = "2023-05-15"
        openai.api_type = os.environ["OPENAI_API_TYPE"]
        openai.api_key = api_key
        
        self.llm = AzureOpenAI(
            deployment_name=deployment_name,
            model_name=model_name,
            temperature=temperature
        )

    def get_response(self, prompt):
        return self.llm(prompt)

if __name__ == "__main__":
    api_key = os.environ["OPENAI_API_KEY"]
    gpt_model = GPT3_5Model(api_key)
    
    # Run the LLM
    prompt = "Tell me a joke"
    response = gpt_model.get_response(prompt)
    print(response)
