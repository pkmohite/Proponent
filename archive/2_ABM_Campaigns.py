import os
import pandas as pd
import streamlit as st
import getpass
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel


def product_chatbot_v4():
    # Concatenate the feature names and value propositions at the line level
    features_value_prop = "\n".join([f"{feature}: {value_prop}" for feature, value_prop in zip(selected_recommendations["featureName"], selected_recommendations["valueProposition"])])

    # Load files-db.csv
    files_df = pd.read_csv("assets/files-db.csv")

    # Load content from hyperlinks using WebBaseLoader
    bs_strainer = bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    loader = WebBaseLoader(
        web_paths=files_df["hyperlink"].tolist(),
        bs_kwargs={"parse_only": bs_strainer},
    )
    docs = loader.load()

    # Create embeddings for the text
    embeddings = OpenAIEmbeddings()

    # Create a vector store from the texts and embeddings
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

    # Initialize the chat model
    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("USER_API_KEY"))

    # Define a prompt template
    prompt_template = """
    As a product marketing expert, use the following customer details to provide a highly targeted response:

    Company Name: {customer_name}
    Company Industry: {industry}
    Deal Stage: {deal_stage}
    Deal Amount: {deal_amount}

    Internal Notes: {internal_notes}
    Email History: {history_email}
    Chat History: {history_chat}
    Competitor Battlecard: {competitor_battlecard}
    Case Studies: {case_studies_concat}
    Summary of Contact's Involved in the deal and their needs: {contact_summary}
    Recommended Features: {features_value_prop}

    Focus on addressing the customer's specific needs, pain points, and challenges based on their industry, deal stage, and history. Highlight relevant features, benefits, and case studies that demonstrate how our product can provide value and solve their problems. Tailor your language and tone to match the customer's background and preferences.

    User Query: {query}

    Assistant Response:
    """

    prompt = hub.pull("rlm/rag-prompt").format(
        customer_name=st.session_state.customer_name,
        industry=industry,
        deal_stage=deal_stage,
        deal_amount=deal_amount,
        internal_notes=internal_notes,
        history_email=history_email,
        history_chat=history_chat,
        competitor_battlecard=competitor_battlecard,
        case_studies_concat=case_studies_concat,
        contact_summary=contact_summary,
        features_value_prop=features_value_prop,
        query="{query}"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vectorstore.as_retriever()

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | chat
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    # Get user input
    query = st.text_area("Enter your query:")

    if st.button("Submit"):
        # Call the RunnableParallel chain with the query and customer details
        result = rag_chain_with_source.invoke({"query": query})

        # Display the result
        st.info(result["answer"])


def product_chatbot_v5(customer_name, industry, deal_stage, deal_amount, internal_notes, history_email, history_chat, competitor_battlecard, case_studies_concat, contact_summary):
    # Load files-db.csv
    files_df = pd.read_csv("assets/files-db.csv")

    # Get all hyperlink values from files-db.csv
    hyperlinks = ("https://lilianweng.github.io/posts/2023-06-23-agent/", "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/")

    # Load, chunk and index the contents of the blog.
    bs_strainer = bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    loader = WebBaseLoader(
        web_paths=hyperlinks,
        bs_kwargs={"parse_only": bs_strainer},
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Define a prompt template
    prompt_template = """
    As a product marketing expert, use the following customer details to provide a highly targeted response:

    Company Name: {customer_name}
    Company Industry: {industry}
    Deal Stage: {deal_stage}
    Deal Amount: {deal_amount}

    Internal Notes: {internal_notes}
    Email History: {history_email}
    Chat History: {history_chat}
    Competitor Battlecard: {competitor_battlecard}
    Case Studies: {case_studies_concat}
    Summary of Contact's Involved in the deal and their needs: {contact_summary}
    Recommended Features: {features_value_prop}

    Focus on addressing the customer's specific needs, pain points, and challenges based on their industry, deal stage, and history. Highlight relevant features, benefits, and case studies that demonstrate how our product can provide value and solve their problems. Tailor your language and tone to match the customer's background and preferences.

    User Query: {query}

    Assistant Response:
    """

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")#, prompt=prompt_template)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)


    st.title("Chatbot Test")
    query = st.text_area("Enter your query:")
    if st.button("Submit"):
        result = rag_chain_with_source.invoke(query)
        st.write(result)

    # button for clearing language model cache

# Main
st.title("Product Chatbot")
customer_name = st.text_input("Enter Customer Name:", value="Monday.com")
industry = st.text_input("Enter Industry:", value="Technology")
deal_stage = st.text_input("Enter Deal Stage:", value="Negotiation")
deal_amount = st.text_input("Enter Deal Amount:", value="$100,000")
internal_notes = st.text_area("Enter Internal Notes:", value="Customer is interested in purchasing Monday.com. They have a budget of $100,000 and are looking for a solution that can streamline their project management process.")
history_email = st.text_area("Enter Email History:", value="No email history available. This is the first interaction with the customer.")
history_chat = st.text_area("Enter Chat History:", value="No chat history available. This is the first interaction with the customer.")
competitor_battlecard = st.text_area("Enter Competitor Battlecard:", value="No competitor battlecard available. The customer has not mentioned any specific competitors.")
case_studies_concat = st.text_area("Enter Case Studies:", value="No case studies available. However, we have several success stories from customers in the technology industry who have achieved significant improvements in their project management efficiency using Monday.com.")
contact_summary = st.text_area("Enter Contact Summary:", value="Contact is interested in Monday.com features and pricing. They have expressed specific interest in the task management and collaboration features, as well as integrations with other tools they use.")


product_chatbot_v5(customer_name, industry, deal_stage, deal_amount, internal_notes, history_email, history_chat, competitor_battlecard, case_studies_concat, contact_summary)