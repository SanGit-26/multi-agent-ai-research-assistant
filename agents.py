from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, srcape_url

import os

from dotenv import load_dotenv

load_dotenv()

#model setup
llm = ChatGoogleGenerativeAI(model = "gemini-3.6-flash")

#1st attempt
def build_search_agent():
    return create_agent(
        model = llm,
        tools = [web_search]
    )

#2nd agents
def build_reader_agent():
    return create_agent(
        model = llm, 
        tools = [srcape_url]
    )

#writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are an expert research writer. write clear, structured and insightful reports."),
    ("human", """write a detailed research report on the topic below.
      Topic: {topic}

      Research Gathered:
      {research}

      Structure the report as:
      -Introduction
      -Key Finiding (minimun 3 well explained points)  
      -Conclusions
      -Sources (list all URLs found in the research)

      be detailed, factual and profesional."""),
])


#writer chain
writer_chain = writer_prompt | llm | StrOutputParser()

#critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a sharp and constructive research critic. be honest and specific."),
    ("human", """Review the reseach report below and evaluate it strictly.
    Report:
    {report}

    Respond in this exact format:
    Score: x/10

    Strengths:
    -...
    -...

    Area to Improve:
    -...
    -...

    one line verdict:
    -..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
