import validators 
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader


st.page_config(page_title="URL Summarizer", page_icon=":guardsman:", layout="wide")
st.title("URL Summarizer")
st.write("This app summarizes the content of a given URL using Groq LLM.")

# Sidebar
st.sidebar.title("Settings")
groq_api_key=st.sidebar.text_input("Enter your Groq API key:", type="password")

llm = ChatGroq(model_name="gemma2-9b-it", api_key=groq_api_key)

prompt_template ="""Summarize the following content in a concise manner:\n\n{content}"""
prompt = PromptTemplate(template=prompt_template, input_variables=["content"])

if not groq_api_key:
    st.warning("Please enter your Groq API key.")
    st.stop()
    
# Main content
st.header("Enter the URL to summarize:")
url = st.text_input("URL", placeholder="Enter a URL to summarize")

if st.button("Summarize"):
    with st.spinner("Loading..."):
        try:
            # Check if the API key is valid
            if not groq_api_key:
                st.warning("Please enter your Groq API key.")
                st.stop()
            
            # Check if the URL is valid
            if not url:
                st.warning("Please enter a URL.")
                st.stop()
            
            if not validators.url(url):
                st.warning("Please enter a valid URL.")
                st.stop()
            
            # Load the content from the URL
            if "youtube.com" in url:
                loader = YoutubeLoader.from_youtube_url(url)
            else:
                loader = UnstructuredURLLoader(urls=[url])

            docs = loader.load()    

            # Create the summarization chain
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt) ## stuff, map_reduce, refine
            
            # Run the summarization chain
            summary = chain.run(docs)

            # Display the summary
            st.subheader("Summary:")
            st.write(summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")