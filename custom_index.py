from langchain.llms.base import LLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
import argparse
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from llama_index import (
    GPTListIndex,
    QuestionAnswerPrompt,
    PromptHelper,
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    Document,
    SimpleDirectoryReader,
    download_loader,
    set_global_service_context,
    load_index_from_storage
)
from llama_index.llm_predictor import HuggingFaceLLMPredictor
from pathlib import Path
from typing import Optional, List, Mapping, Any


parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", 
    "--model", 
    dest="model", 
    default="beomi/kollama-7b", 
    action="store"
)
parser.add_argument(
    "-p", 
    "--path", 
    dest="path", 
    default="./", 
    action="store")
parser.add_argument(
    "-d", 
    "--device", 
    dest="device", 
    default="cpu", 
    action="store")
parser.add_argument(
    "-q",
    "--query",
    dest="query",
    default="저는 가입 기간은 1년 미만이며 주계약은 가입금액 1000만원, 암진단특약VID는 가입금액 500만원으로 가입하였습니다. 유방암 진단을 받은 경우 지급금액이 얼마인가요?",
    action="store",
)
args = parser.parse_args()

context_window = 2048
num_output = 256



QA_PROMPT_TMPL = (
    "We have provided context information below. \n "
    "{context_str} \n"
    " Given this information, I want you to act as insurance assistance. I’ll write you my insurance info and my diagnosis , and you’ll inform for me which guaranteed amount, info from via my insurance terms and answer this : {query_str}\n"
)

config = AutoConfig.from_pretrained("beomi/kollama-7b")

tokenizer = AutoTokenizer.from_pretrained("beomi/kollama-7b")
model = AutoModelForCausalLM.from_config(config)

hf_predictor = HuggingFaceLLMPredictor(
    max_input_size=2048,
    max_new_tokens=256,
    tokenizer=tokenizer,
    model=model,
    tokenizer_kwargs={"max_length": 2048},
)

embed_model = LangchainEmbedding(
   HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
)

service_context = ServiceContext.from_defaults(
   chunk_size_limit=512, llm_predictor=hf_predictor, embed_model=embed_model
)

set_global_service_context(service_context)

QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

if os.path.exists("./storage"):
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
else:
    CJKPDFReader = download_loader("CJKPDFReader")
    loader = CJKPDFReader()

    documents = None
    for file in os.listdir(args.path):
        if file.endswith(".pdf"):
            d = loader.load_data(file=os.path.join(args.path, file))
            if documents is None:
                documents = d
            else:
                for innerd in d:
                    documents.append(innerd)

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()

print(QA_PROMPT.prompt)
print(args.query)

engine = index.as_query_engine()
response = engine.query(args.query)

print(response)
