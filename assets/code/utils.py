import openai, json, hmac, os, base64, requests, re
import streamlit as st
import pandas as pd
from PyPDF2 import PdfMerger
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from assets.code.element_configs import parquet_schema_log, config_about
import assemblyai as aai
from fpdf import FPDF
from dotenv import load_dotenv
from openai import OpenAI

## Load the environment variables
load_dotenv()

## Authentication Functions

# Function to check if the password is correct
def check_password():
    # Function to check if the password is correct

    def login_form():
        logincont = st.container(border=True)
        logincont.markdown('###')
        left_co, cent_co,last_co = logincont.columns([1,1.5,1])
        with cent_co:
            get_themed_logo()
        
        tab1, tab2 = logincont.tabs(["Log In", "Login as Guest"])
        ## Log In
        with tab1.form("Log In", border=False):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)
        ## Log in as Guest
        with tab2.form("Guest", border=False):
            st.text_input("Email", key="email")
            st.form_submit_button("Log in as Guest", on_click=submit_email)
                

    def password_entered():
        login_credentials = json.loads(os.environ["LOGIN"])
        # Check if the username and password are correct
        if st.session_state["username"] in login_credentials and hmac.compare_digest(
            st.session_state["password"], login_credentials[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    def submit_email():
        # Get access key from environment variable
        access_key = os.getenv('ACCESS_KEY')
        email = st.session_state["email"]

        # Set up the form data
        form_data = {
            'access_key': access_key,
            'email': email,
            'redirect': 'https://web3forms.com/success'
        }

        # Valid email using regex
        email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(email_regex, email):
            st.session_state["email_valid"] = "invalid"
        else:
            # Submit the form
            response = requests.post('https://api.web3forms.com/submit', data=form_data)

            # Check the response
            if response.status_code == 200:
                st.session_state["password_correct"] = True
            else:
                st.session_state["email_valid"] = "error"
            
    
    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    if "email_valid" in st.session_state:
        if st.session_state["email_valid"] == "invalid":
            st.error("😕 Please enter a valid email.")
        elif st.session_state["email_valid"] == "error":    
            st.error("😕 Something went wrong. Please try again later.")
    return False


def verify_password():
    # Return True if the username + password is validated.
    if not st.session_state.get("password_correct", False):
        st.switch_page("Proponent.py")
        
## OpenAI Functions

## AssemblyAI Functions
def transcribe_video(file):
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    transcriber = aai.Transcriber()
    
    transcript = transcriber.transcribe(file)
    # transcript = transcriber.transcribe("./my-local-audio-file.wav")
    
    return transcript.text

# Function to pass the OpenAI key
def pass_openAI_key(api_key=None):
    if "USER_API_KEY" in os.environ:
        openai.api_key = os.getenv("USER_API_KEY")
    else:
        st.sidebar.error("OpenAI API key not found. Please set the API key in the Setting page.")

# Function to get the embedding of a text from the OpenAI API
def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return openai.embeddings.create(input=[text], model=model).data[0].embedding

# Function to generate a summary using the OpenAI API
def create_summary(user_input, customer_name, customer_title, customer_company, model="gpt-3.5-turbo-0125"):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"You are a sales enablement assistant helping to identify key customer needs and pain points. The customer is {customer_name}, {customer_title} from {customer_company}."
            },
            {
                "role": "user",
                "content": f"Please analyze the following text and extract the key customer pain points, needs, and asks. Summarize them concisely, focusing only on the essential information. The summary should be suitable for semantic similarity search against product features and value propositions.\n\nText: {user_input}"
            }
        ],
        temperature=0.5,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    summary = response.choices[0].message.content
    return summary

# Function to generate a customized email using the OpenAI API
def generate_customized_email(recommendations, user_input, customer_name, customer_title, customer_company, history, competitor, model="gpt-3.5-turbo-0125"):

    # Extract the feature names and value propositions from the recommendations DataFrame
    features_str = "\n".join(recommendations["featureName"])
    value_prop_str = "\n".join(recommendations["valueProposition"])
    example_email = f"""Dear {customer_name},

    Thank you for sharing your valuable feedback with us. We understand that you are facing certain challenges in your current workflow and are looking for suitable solutions.

    Based on our analysis, we believe the following features from our product can help you overcome these challenges and improve your productivity:

    * Feature 1: Value Proposition 1
    * Feature 2: Value Proposition 2
    * Feature 3: Value Proposition 3

    These features, when combined, will not only address your specific pain points but also provide a significant advantage over your current solution from {competitor}.

    We would love to demonstrate these features in action and discuss how they can be tailored to your unique requirements. If you're interested, please let us know a suitable time for a quick demo or a discussion.

    Looking forward to hearing from you soon.

    Best regards,
    [Your Name]
    Sales Team"""

    conversation = [
    {
        "role": "system",
        "content": "You are an expert sales communication assistant. Your goal is to craft a personalized, engaging, and concise email under 200 words to follow up with a customer based on their pain points, feature requests, our proposed solutions, their interaction history, and their current competitor solution.",
    },
    {
        "role": "user",
        "content": f"Here is the context of my conversation with the customer {customer_name}, {customer_title} from {customer_company}:\n\n{user_input}\n\nBased on their input, we have identified the following features and value propositions:\n\nFeatures:\n{features_str}\n\nValue Propositions:\n{value_prop_str}\n\nThe customer's interaction history is as follows:\n\n{history}\n\nTheir current competitor solution is:\n\n{competitor}\n\nPlease draft a short follow-up email that:\n1. Thanks the customer for their input and acknowledges their pain points\n2. Highlights all the shortlisted features and their corresponding value propositions in a bullet-point format\n3. Explains how these features collectively address their needs and improve their workflow, while differentiating from their current competitor solution\n4. Ends with a clear call-to-action, inviting them to schedule a demo or discuss further\n\nKeep the email concise, personalized, and focused on the customer's unique situation. Use a friendly yet professional tone.\n\nHere is an example email to guide you:",
    },
    {
        "role": "assistant",
        "content": example_email,
    },
    ]
    # Generate the email body using the OpenAI API
    response = openai.chat.completions.create(
        model=model, messages=conversation, stream=True
    )

    return response

# # Function to generate an EYNK (Everything You Need to Know) content using the OpenAI API
def generate_enyk(recommendations, user_input, customer_name, customer_title, customer_company, history, competitor, model="gpt-3.5-turbo-0125"):
    # Concatenate the feature names and value propositions at the line level
    features_value_prop = "\n".join([f"{feature}: {value_prop}" for feature, value_prop in zip(recommendations["featureName"], recommendations["valueProposition"])])
    
    competitors = pd.read_csv("assets/competitors.csv")
    competitor_data = competitors[competitors["name"] == competitor]
    if not competitor_data.empty:
        battlecard = competitor_data["battlecard"].values[0]
    else:
        battlecard = "No Competitor"
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant designed to output JSON."
        },
        {
            "role": "user",
            "content": f"""
            Generate a JSON response with the following sections, keeping each section under 300 words, using bullet points  and  markdown formatting to bold/italicize key parts of the bullets.

            1. Based on the recent conversation provided, summarize the customer needs expressing the the user_input for {customer_name} with title {customer_title} from {customer_company}.

            User input:
            {user_input}

            2. Reflect on the historical interactions with {customer_name} and summarize their needs and use-cases. Don't overlap with the current needs summary.

            History:
            {history}

            3. Compare the features and benefits of Monday.com's product with the current solution used by {customer_name} at {customer_company}, which is {competitor}. Highlight the advantages in the context of the customer's needs.

            Competitor Battlecard:
            {battlecard}

            4. Explain the benefits the following recommended features for {customer_name} in the context of their specific needs and use-case.

            Recommended Features:
            {features_value_prop}

            Please provide the response in the following JSON format, using bullet points for each section and  markdown formatting to bold/italicize key parts of the bullets.
            {{
                "customer_needs_summary": "- Point 1\\n - Point 2\\n - Point 3.. \\n - Point n",
                "historical_needs_summary": "- Point 1\\n - Point 2\\n - Point 3.. \\n - Point n",
                "competitor_comparison": "- Point 1\\n - Point 2\\n - Point 3.. \\n - Point n",
                "recommended_features_benefits": "- Point 1\\n - Point 2\\n - Point 3.. \\n - Point n"
            }}
            """
        }
    ]
    response = openai.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=messages,
    )
    
    enyk_json = response.choices[0].message.content
    enyk = json.loads(enyk_json)
    
    return enyk

# Function to find the cosine similarity between two vectors
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Function to calculate the similarity scores between the user input and the data
def calculate_similarity_ordered(user_input_embedding):
    
    # Load assets/mf_embeddings.parquet
    mf_data = load_mf_data()
    
    # Calculate the similarity scores
    mf_data["similarity_score"] = mf_data["embedding"].apply(
        lambda x: cosine_similarity(user_input_embedding, x)
    )

    # Sort the data based on the similarity scores
    df = mf_data.sort_values(by="similarity_score", ascending=False)
    
    return df


## Media Processing Functions

# Function to create a PDF deck from the DataFrame
def create_pdf_deck(output_path):

    merger = PdfMerger()

    # Add the start PDF
    merger.append("slides/slide-start.pdf")

    # Add the main content PDF
    merger.append(output_path)

    # Add the end PDF
    merger.append("slides/slide-end.pdf")

    # Write to an output PDF document
    merger.write(output_path)

    merger.close()

    print(f"Combined PDF created: {output_path}")


def create_image_deck(df):
    # Create a list to store the image paths
    image_paths = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Construct the file path
        image_file = row["pdfFile"]
        file_path = os.path.join("slides", image_file)

        # Check if the file exists
        if os.path.exists(file_path):
            # Add the image path to the list
            image_paths.append(file_path)
        else:
            st.error(f"No slides in database. Please upload via Messaging Framework Tab or Contact Support if that doesn't work.")

    # Specify the output file path
    output_path = "downloads/combined_PDF.pdf"

    # Create a new PDF document with 16:9 layout
    pdf = FPDF(orientation="L", format=(285, 510))

    # Add each image to the PDF document
    for image_path in image_paths:
        pdf.add_page()
        pdf.image(image_path, x=-1, y=2, w=510)
    
    # Save the PDF document
    pdf.output(output_path)

    create_pdf_deck(output_path)

    print(f"Combined image PDF created: {output_path}")


def displayPDF(file, column = st):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    # Embedding PDF in HTML
    # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="1000" height="600" type="application/pdf">'
    
    # Method 2 - Using IFrame
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="840" height="520" type="application/pdf"></iframe>'

    # Displaying File
    column.markdown(pdf_display, unsafe_allow_html=True)


## DB Functions

# Function to load the data from the Parquet file
def load_mf_data(file="assets/mf_embeddings.parquet"):
    if not os.path.exists(file):  
        st.warning("No data found. Upload painpoints via CSV.")
        st.stop()
    else:
        mf_data = pd.read_parquet(file)
        # mf_data = pd.read_csv("assets/mf_embeddings.csv")
    
    return mf_data


# Function to update the log in a Parquet file
def update_log_parquet(
    customer_name,
    customer_title,
    customer_company,
    persona_category1,
    persona_category2,
    persona_category3,
    user_input,
    top_7,
):
    # Get the current date and time
    date = pd.Timestamp.now().strftime("%Y-%m-%d")
    time = pd.Timestamp.now().strftime("%H:%M:%S")

    # Extract the pain point IDs from the top 7 recommendations
    painPointIDs = [int(id) for id in top_7["painPointId"].tolist()]

    # Specify the Parquet file path
    parquet_file = "assets/log.parquet"

    # Create a new Arrow table with the variable values
    data = [
        [customer_name],
        [customer_title],
        [customer_company],
        [persona_category1],
        [persona_category2],
        [persona_category3],
        [user_input],
        [painPointIDs],
        [date],
        [time],
    ]
    table = pa.Table.from_arrays(data, schema=parquet_schema_log)

    # Check if the Parquet file exists and append the new data
    if os.path.exists(parquet_file):
        # Read the existing Parquet file
        existing_table = pq.read_table(parquet_file)

        # Append the new data to the existing table
        new_table = pa.concat_tables([existing_table, table])

        # Write the updated table to the Parquet file
        pq.write_table(new_table, parquet_file)
    else:
        # Write the table to a new Parquet file
        pq.write_table(table, parquet_file)


# Function to get the log and the mf embeddings
def get_mf_and_log(log_file = 'assets/log.parquet', mf_file = 'assets/mf_embeddings.parquet'):
   
    # Read the Parquet file into an Arrow Table
    table = pq.read_table(log_file)
    
    # Convert the Arrow Table to a Pandas DataFrame
    df = table.to_pandas()
    
    # Convert the 'paintpoints' array into separate columns
    df = pd.concat([df.drop('paintpoints', axis=1), df['paintpoints'].apply(pd.Series).add_prefix('pp')], axis=1)
    
    # Read the data from assets/mf_embeddings.parquet
    mf_content = pq.read_table(mf_file).to_pandas()

    return df, mf_content


## Auxiliary Functions

# Function to create the .env file if it does not exist 
def create_env_file():
    if not os.path.exists(".env"):
        with open(".env", "w") as file:
            file.write("USER_API_KEY=\n")


def get_themed_logo(width=300):
    # read assets/themes.csv and check if active theme's vibe column is dark or light
    themes_df = pd.read_csv("assets/themes.csv")
    current_theme_index = int(themes_df[themes_df["active"] == "x"].index[0])
    current_theme_values = themes_df.loc[current_theme_index]
    if current_theme_values["vibe"] == "dark":
        st.image("assets/images/logo_full_white.png", width=width)
    else:
        st.image("assets/images/logo_full_black.png", width=width)
    st.markdown('###')


def set_page_config(page_title, page_icon = "assets/images/logo.ico", layout="wide", initial_sidebar_state="expanded"):
    st.set_page_config(
    page_title=page_title, 
    page_icon=page_icon,
    layout=layout, 
    initial_sidebar_state=initial_sidebar_state,
    menu_items={'Get Help': "mailto:prashant@yourproponent.com",
                'About': config_about})



## Archive Functions


def product_chatbot():
    client = OpenAI(api_key=os.getenv("USER_API_KEY"))

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([6, 1])
    prompt = col1.chat_input("Say something")
    if col2.button("Reset Chat"):
        st.session_state.messages = []
        with st.chat_message("assistant"):
            st.markdown("Hello! How can I help you today?")
        st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I help you today?"})

    if prompt :
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

def product_chatbot_v2():
    client = OpenAI(api_key=os.getenv("USER_API_KEY"))
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    col1, col2 = st.columns([6, 1])
    prompt = col1.chat_input("Say something")
    if col2.button("Reset Chat"):
        st.session_state.messages = []
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            # Use ChatGPT API to determine user's choice
            choice_prompt = f"Based on the user's input: '{prompt}', determine which of the following options the user is requesting:\n1. Create Personalized Landing Webpage\n2. Create Personalized Sales Deck\n3. Create Personalized Product Demo\n4. Answer general questions\n\nProvide your response in JSON format with the key 'choice' and the corresponding option number as the value."
            choice_response = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[{"role": "user", "content": choice_prompt}],
            )
            choice_result = choice_response.choices[0].message.content.strip()
            choice_json = json.loads(choice_result)
            choice = int(choice_json["choice"])

            if choice == 1:
                # response = create_personalized_landing_webpage()
                col1, col2, col3 = st.columns([2.5, 2, 1.5])
                col1.markdown("#### Personalized Sales Deck")
                create_image_deck(selected_recommendations)
                with open("downloads/combined_PDF.pdf", "rb") as file:
                    col3.download_button(
                        label="Download PDF Deck",
                        data=file.read(),
                        file_name="customized_deck.pdf",
                        mime="application/pdf",
                    )
                
                if os.path.exists("downloads/combined_PDF.pdf"):
                    displayPDF("downloads/combined_PDF.pdf", st)
                else:
                    st.error("Error generating PDF. Please try again or contact me at prashant@yourproponent.com if this persists.")                

            elif choice == 2:
                # response = create_personalized_sales_deck()
                response = "Sales Deck Creation is not available in this demo deployment. Please download the PDF deck and video for the recommendations."
            elif choice == 3:
                # response = create_personalized_product_demo()
                response = "Product Demo Creation is not available in this demo deployment. Please download the PDF deck and video for the recommendations."
            else:
                
                # Concatenate the feature names and value propositions at the line level
                features_value_prop = "\n".join([f"{feature}: {value_prop}" for feature, value_prop in zip(selected_recommendations["featureName"], selected_recommendations["valueProposition"])])
                # Pass context variables to the general query
                context = f"Chat History: {history_chat}\nEmail History: {history_email}\nRecommended Features: {features_value_prop}\nCompetitor Battlecard: {competitor_battlecard}"
                general_query_prompt = f"{context}\n\nUser Query: {prompt}\n\nAssistant Response:"
                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": "system", "content": general_query_prompt},
                        *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
