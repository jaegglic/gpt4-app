import os
from typing import List
import json

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
_API_KEY = os.environ.get("OPENAI_API_KEY", "")
_CLIENT = OpenAI(api_key=_API_KEY)

st.title("Text Simplification by GPT-4")


def prompting(text: str) -> List[dict]:
    # Taken from here
    # https://platform.openai.com/docs/guides/chat/introduction
    sys_msg = 'Act as a medical professional who summarizes technical literature for ' \
              'non-medical people in an understandable and simple way. Do not add any ' \
              'additional information and do not comment the reviewed text. The only task you ' \
              'do is to output the rewritten and simplified text.'

    messages = [{'role': 'system', 'content': sys_msg}]
    text = f'Do the above simplification task with the following text: "{text}"'
    messages.append({'role': 'user', 'content': text})
    return messages


def gpt_4(messages: List[dict]) -> str:
    """This function generate a response to the given query using gpt-4."""
    response = _CLIENT.chat.completions.create(
        model="gpt-4",
        messages=messages,
    )
    return response.choices[0].message.content


def main():
    """This function gets the user input, pass it to ChatGPT function and
    displays the response
    """
    # Get user input
    text = st.text_input(label='enter your text...')
    if text:
        messages = prompting(text=text)
        st.markdown('### Raw text')
        st.markdown(text)

        st.markdown("---")
        st.markdown('### Simplified text')

        response = gpt_4(messages=messages)
        st.markdown(response)


if __name__ == '__main__':
    main()
