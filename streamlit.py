import streamlit as st
from streamlit_chat import message as st_message
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from PIL import Image
from textblob import TextBlob
st.set_page_config(layout="wide")

image = Image.open('Mental Health (1).png')
st.sidebar.image(image)

image = Image.open('header.png')

st.image(image, caption=' ')
t=0

@st.experimental_singleton
def get_models():
    # it may be necessary for other frameworks to cache the model
    # seems pytorch keeps an internal state of the conversation
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


if "history" not in st.session_state:
    st.session_state.history = []

st.title("Talk To The Bot")


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.freecreatives.com/wp-content/uploads/2016/01/Pink-Abstract-Floral-Background.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 



def generate_answer():
    tokenizer, model = get_models()
    user_message = st.session_state.input_text
    inputs = tokenizer(st.session_state.input_text, return_tensors="pt")
    result = model.generate(**inputs)
    message_bot = tokenizer.decode(
        result[0], skip_special_tokens=True
    )  # .replace("<s>", "").replace("</s>", "")

    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": message_bot, "is_user": False})
 



a=st.text_input("", key="input_text", on_change=generate_answer)



text = str(a)
f=TextBlob(text).sentiment.polarity
t+=f
st.write(t)








for chat in st.session_state.history:
    st_message(**chat)  # unpacking
st.write(t)
