from langchain.retrievers import PineconeHybridSearchRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone_text.sparse import SpladeEncoder
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool , initialize_agent
from langchain import SerpAPIWrapper
from langchain.agents import AgentType
# from fastapi.responses import StreamingResponse
from langchain.callbacks.base import CallbackManager
from langchain.tools.human.tool import HumanInputRun
# from pydantic import BaseModel
import os
import pinecone
import openai
import time
# from fastapi import FastAPI
from lcserve import serving

import nltk
import os

@serving(websocket=True)
def get_answers(prompt:str , **kwargs):
    streaming_handler = kwargs.get('streaming_handler')

    #initialize the pinecone index
    pinecone.init(
        api_key="API_KEY",
        environment="ENV"
    )
    index = pinecone.Index("criminal-laws")

    #Getting our encoding models ---> dense + sparse
    embeddings = HuggingFaceEmbeddings(model_name="msmarco-distilbert-base-tas-b")
    splade_encoder = SpladeEncoder()

    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=splade_encoder, index=index)

    #setting up open ai env
    os.environ["OPENAI_API_KEY"] = "API_KEY" 
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = "ORG_ID"
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo" , verbose = True , streaming=True , callback_manager=CallbackManager([streaming_handler]))

    # Creating a langchain document retriver chain
    dnd_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever , callback_manager = CallbackManager([streaming_handler]))

    os.environ["SERPAPI_API_KEY"] = "API_KEY"
    search = SerpAPIWrapper()

    #Adding tools to help the agent
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or when you get no results from criminal-laws tool",
            return_direct=True
        ),
        Tool(
            name="criminal-laws",
            func=dnd_qa.run,
            description="useful for when you need to know something about Criminal law. Input: an objective to know more about a law and it's applications. Output: Correct interpretation of the law. Please be very clear what the objective is!",
            return_direct=True
        ),
    ]
    tools.append(HumanInputRun())

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, return_intermediate_steps=True)
    
    # Returning the thoughts of the agent with api results
    hey = agent(prompt)
    steps = []
    for line in hey["intermediate_steps"][0][0]:
        steps.append(line)
    Final_ans = hey["output"].split(" ")
    newdict = {"Thoughts":steps, "Final Answer :" : Final_ans}
    for line in newdict["Thoughts"]:
        word_split = line.split(" ")
        for word in word_split:
            # print(word, end = " ")
            yield str(str(word) + " ")
            time.sleep(0.2)
    yield b'\nFinal Answer : '
    stringk = newdict["Final Answer :"]
    # stringk = stringk.split(" ")
    for word in stringk:
        # print(word, end=" ")
        yield str(str(word) + ' ')
        time.sleep(0.2)
    return newdict


# @app.get('/{prompt}')
# async def main(prompt : str):
#     return StreamingResponse(get_answers(prompt), media_type='text/event-stream')
