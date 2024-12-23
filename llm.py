from langchain_community.llms.llamafile import Llamafile
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class OllamaLLMWrapper:
    def __init__(self):
        """
        Initializes the Ollama LLM using LangChain's Ollama integration.

        Args:
            model_name (str): The name of the Ollama model to use.
        """
        self.llm = Llamafile()

    def create_chain(self, template: str):
        """
        Creates an LLMChain using a custom prompt template.

        Args:
            template (str): The template for generating responses.

        Returns:
            LLMChain: A chain configured with the prompt and LLM.
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        return LLMChain(llm=self.llm, prompt=prompt)

    def generate_response(self, context: str, question: str, chain: LLMChain):
        """
        Generates a response using the given chain.

        Args:
            context (str): The contextual information for the question.
            question (str): The user's question.
            chain (LLMChain): The chain for generating responses.

        Returns:
            str: The LLM's response.
        """
        return chain.run(context=context, question=question)
