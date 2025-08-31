import json
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

intents = json.load(open('intents.json'))
#To view the json file
#print(intents)

tags = []
patterns = []

#To see what are the list of tags and pattern
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

#printing all the pattern tag for the greeting tag
#print(tags)
#print(patterns)

# patterns are the text message. so we have to it with TFIDF_VECTORIZER
# To Extract the feature we have use vector
vector = TfidfVectorizer()
patterns_scaled = vector.fit_transform(patterns)

#Creating Model with maximum iteration of 100000 words
Bot = LogisticRegression(max_iter=100000)

#Based upon the pattern message is has to find out the tags
Bot.fit(patterns_scaled, tags)

#Let's check it is identifying the tags or not 

#input_msg = 'What is your name'
#input_msg = vector.transform([input_msg])
#print(Bot.predict(input_msg))



#Function for the chatbot
def ChatBot(input_msg):
    input_msg = vector.transform([input_msg])
    predict_tag = Bot.predict(input_msg)[0]
    #print(predict_tag)
    for intent in intents['intents']:
        if intent['tag'] == predict_tag:
            response = random.choice(intent['responses'])
            return response


# Take input and check 
#Message = input('Message ChatBOT ')
#result_msg = ChatBot(Message)
#print(result_msg)


#For the web app using streamlit
# Set the title and icon of the app
st.set_page_config(
    page_title="UIT AI ChatBot",          # The title that appears on the browser tab
    page_icon="LogoUIT.png",              # The icon for the app (can be an emoji or an image path)
    # layout="wide"                       # Optional: Change layout to "wide" for a wider display
)

#Centering the Image 
with st.container():
    col1, col2, col3 = st.columns([2, 3, 2])        # This splits the page into 3 parts
    with col2:
        st.image('LogoUIT.png', width=100)          # Image centered in the middle column

st.title(":blue[University Institute of Technology :] ChatBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask ChatBot"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"UIT ChatBot: "+ChatBot(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})