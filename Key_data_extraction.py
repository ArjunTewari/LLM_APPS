import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field


class Car(BaseModel):
    """
    Information about cars
    """
    name : Optional[str] = Field(
        default=None, description="Name of the car"
    )
    Summary: Optional[str] = Field(
        default=None, description="Summary of the car from the given text"
    )

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert data extraction algorithm which extracts the relevant information from the text."),
        ("human", "{text}"),
    ]
)
chain = prompt | llm.with_structured_output(schema=Car)

print(chain.invoke({"text":"The car mahindra thar was first built in 2023 it was a car which appealed mostly to youth and the people who used to go for offroading"}))
