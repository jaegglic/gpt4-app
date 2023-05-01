import os
from typing import List
import json

import streamlit as st
import openai
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY", "")

st.title("NER by GPT-4")


def initialize_messages():
    # Taken from here
    # https://platform.openai.com/docs/guides/chat/introduction
    sys_msg = 'Identify named entities "MEDICINAL PRODUCT" and "ADVERSE EVENT" from a given text ' \
              'and return it in a JSON format.'
    shots = [
        {
            'text': "Treatment with ipilimumab has been linked to the rare yet serious adverse "
                    "event of serious retinal detachment. The amount of photoreceptor degeneration "
                    "and loss of vision can be minimised by early diagnosis and treatment.",
            'ner': [
                {'T': 'MEDICINAL PRODUCT', 'E': 'ipilimumab'},
                {'T': 'ADVERSE EVENT', 'E': 'serous retinal detachment'},
                {'T': 'ADVERSE EVENT', 'E': 'loss of vision'}],
        },
        {
            'text': "Clindamycin is potentially nephrotoxic. Acute kidney injury including acute "
                    "renal failure has been reported.",
            'ner': [
                {'T': 'MEDICINAL PRODUCT', 'E': 'Clindamycin'},
                {'T': 'ADVERSE EVENT', 'E': 'Acute kidney injury'},
                {'T': 'ADVERSE EVENT', 'E': 'renal failure'}],
        }
    ]

    messages = [{'role': 'system', 'content': sys_msg}]
    for shot in shots:
        messages.append({'role': 'user', 'content': shot['text']})
        messages.append({'role': 'assistant', 'content': json.dumps(shot['ner'])})
    return messages


def gpt_4(messages: List):
    """This function generate a response to the given query using gpt-4."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        api_key=API_KEY,
    )
    msg = {'role': 'assistant', 'content': response.choices[0].message['content']}
    return msg


def main():
    """This function gets the user input, pass it to ChatGPT function and
    displays the response
    """
    messages = initialize_messages()
    # Get user input
    query_text = st.text_input(label='enter your text...')
    if query_text:
        user_query = {'role': 'user', 'content': query_text}
        messages.append(user_query)
        st.markdown(query_text)

        response = gpt_4(messages=messages)
        messages.append(response)
        ner = json.loads(response['content'])
        st.json(ner)


if __name__ == '__main__':
    main()
