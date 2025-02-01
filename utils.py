from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain


# QAエージェントを作成する
def qa_agent(openai_api_key, file, memory, question):
    # モデルを生成する
    model = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)

    # ユーザーがアップロードしたPDFファイルの中身を取得する
    file_content = file.read()

    # AIに渡すTEMP PDFファイルのファイルパスを作る
    temp_file_path = "temp.pdf"

    # 取得したPDFの中身を仮のファイルに書き込む
    with open(temp_file_path, "wb") as f:
        f.write(file_content)

    # langchainのPDFローダーでPDFの中身をロードする
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    """コンテキストが長くならないよう、PDF内のテキストを
    「改行」「句点」「ビックリ符号」「ハテナ符号」「半角・全角読点」「から文」で分割するスプリッターインスタンスを定義"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""],
    )

    # スプリッターを使用して分割したテキストを生成する
    texts = text_splitter.split_documents(docs)

    # ベクトルデータベースと類似性検索ライブライを定義する
    embedding_model = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embedding_model)
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model, retriever=retriever, memory=memory
    )

    """エイジェントのinvoke関数で処理を実行しAIからの返答を受け取る。
    関数に使う会話の記憶とユーザーの質問は外部から受け取る"""
    response = qa.invoke({"chat_history": memory, "question": question})

    return response
