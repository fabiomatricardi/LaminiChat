# For graphic unterface
import streamlit as st
# Internal usage
from time import  sleep
#### IMPORTS FOR AI PIPELINES ###############
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline

#AVATARS
av_us = './man.png'  #"🦖"  #A single emoji, e.g. "🧑‍💻", "🤖", "🦖". Shortcodes are not supported.
av_ass = './lamini.png'

# FUNCTION TO LOG ALL CHAT MESSAGES INTO chathistory.txt
def writehistory(text):
    with open('chathistory.txt', 'a') as f:
        f.write(text)
        f.write('\n')
    f.close()
### INITIALIZING LAMINI MODEL
checkpoint = "./model/"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                                    device_map='auto',
                                                    torch_dtype=torch.float32)
### INITIALIZING PIPELINE CHAIN WITH LANGCHAIN
llm = HuggingFacePipeline.from_model_id(model_id=checkpoint,
                                        task = 'text2text-generation',
                                        model_kwargs={"temperature":0.45,"min_length":30, "max_length":350, "repetition_penalty": 5.0})
from langchain import PromptTemplate, LLMChain
template = """{text}"""
prompt = PromptTemplate(template=template, input_variables=["text"])
chat = LLMChain(prompt=prompt, llm=llm)


st.title("LaMiniGPT ChatBot")
st.subheader("All the power of your local 248M parameter AI model")

repo="MBZUAI/LaMini-Flan-T5-248M"

# Set a default model
if "hf_model" not in st.session_state:
    st.session_state["hf_model"] = "MBZUAI/LaMini-Flan-T5-248M"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"],avatar=av_us):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"],avatar=av_ass):
            st.markdown(message["content"])

# Accept user input
if myprompt := st.chat_input("What can you do for me?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": myprompt})
    # Display user message in chat message container
    with st.chat_message("user", avatar=av_us):
        st.markdown(myprompt)
        usertext = f"user: {myprompt}"
        writehistory(usertext)
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar=av_ass):
        message_placeholder = st.empty()
        full_response = ""
        res  =  chat.run(myprompt)
        response = res.split(" ")
        for r in response:
            full_response = full_response + r + " "
            message_placeholder.markdown(full_response + "▌")
            sleep(0.1)
        message_placeholder.markdown(full_response)
        asstext = f"assistant: {full_response}"
        writehistory(asstext)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
