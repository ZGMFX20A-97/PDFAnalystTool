import streamlit as st
from langchain.memory import ConversationBufferMemory
from utils import qa_agent
from openai import AuthenticationError

# UIの見出し
st.title("PDF内容解析ツール")

# OpenAI API Keyを入力するためのサイドバー
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Keyを入力してください", type="password")
    st.markdown("[OpenAI API Keyを取得する](https://platform.openai.com/account/)")
st.session_state["apikey"] = openai_api_key

# メモリーが存在しない場合、文脈を覚えるためのメモリーを初期化する
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True, memory_key="chat_history", output_key="answer"
    )

# ファイルアップローダー
file = st.file_uploader("PDFをアップロードする", type="pdf")
# ファイルをアップロードしていない状態で質問はできないようにする
question = st.text_input(
    "PDFの中身について質問する", disabled=not file
)  

# API Keyが入力していない場合
if file and question and not openai_api_key:
    st.info("OpenAI API Keyを入力してください")
    
if file and question and openai_api_key:
    try:
        # 処理中はスピナーと適切な文言とスピナーを提示する
        with st.spinner("AIが考えています、少々お待ちを。。。"):
            response = qa_agent(
                openai_api_key, file, st.session_state["memory"], question
            )
    except AuthenticationError:
        st.info("正しいAPI Keyを入力してください")
        st.stop()
    st.write("### 回答")
    st.write(response["answer"])

    # 会話の履歴を作成し、書き換える
    st.session_state["chat_history"] = response["chat_history"]

# もし会話の履歴がある場合表示できるようにする
if "chat_history" in st.session_state:
    with st.expander("会話の履歴"):
        # 一問一答のため二個づつ履歴を読み取る(ユーザー質問してAI返す)
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i + 1]
            st.write(human_message.content)
            st.write(ai_message.content)
            # 会話と会話の間は罫線で分割する
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()
