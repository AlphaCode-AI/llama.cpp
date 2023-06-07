import torch
from langchain.llms.base import LLM
from llama_index import SimpleDirectoryReader, LangchainEmbedding, ListIndex
from llama_index import LLMPredictor, ServiceContext
from transformers import pipeline
from typing import Optional, List, Mapping, Any


# set context window size
context_window = 2048
# set number of output tokens
num_output = 256

# store the pipeline/model outisde of the LLM class to avoid memory issues
model_name = "beomi/KoAlpaca"
pipeline = pipeline("qa", model=model_name, device="cuda:0", model_kwargs={"torch_dtype":torch.bfloat16})

class CustomLLM(LLM):
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        response = pipeline(prompt, max_new_tokens=num_output)[0]["generated_text"]

        # only return newly generated tokens
        return response[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"

# define our LLM
llm_predictor = LLMPredictor(llm=CustomLLM())

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, 
    context_window=context_window, 
    num_output=num_output
)

# Load the your data
documents = SimpleDirectoryReader('./data').load_data()
index = ListIndex.from_documents(documents, service_context=service_context)

# Query and print response
query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")
print(response)
