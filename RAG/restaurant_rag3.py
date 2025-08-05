# RAG 기반 대규모 텍스트 검색 구현
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

from RAG.restaurant_rag1 import embeddings

# CharacterTextSplitter : 텍스트를 작은 단위로 나누는데 사용
# OpenAIEmbeddings : 텍스트를 임베딩 벡터로 변환하는 클래스
# TextLoader : 텍스트 파일을 로드하는데 사용
# PromptTemplate : AI모델에 보낼 프롬포트를 템플릿으로 저장
# StrOutParser : AI 모델의 출력을 처리
# RunnablePassThrough : 데이터 흐름에서 특정 값을 그대로 전달
# os, dotenv : 환경 변수와 파일 경로 처리를 위한 표준 라이브러리

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

current_dir = os.path.dirname(os.path.abspath(__file__))
restaurants_text = os.path.join(current_dir, "restaurants.txt")
restaurant_faiss = os.path.join(current_dir, "restaurant-faiss")

# FAISS 인덱스 생성
# 텍스트 데이터를 처리하여 검색 가능한 형태로 변환
def create_faiss_index():
    # TextLoader를 사용하여 txt 파일에서 텍스트를 로드
    loader = TextLoader(os.path.join(current_dir, "restaurants.txt"))
    documents = loader.load()
    # 텍스트를 300자 단위로 나누고, 연속된 청크 사이에 50자의 겹침을 두어 텍스트를 분할
    text_splitter = CharacterTextSplitter(chunk_size = 300, chunk_overlap = 50)
    chunks = text_splitter.create_documents(documents)
    # OpenAI API 를 사용하여 임베딩 생성
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    # Faiss 인덱스를 생성하고 저장
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(restaurant_faiss)
    print("Faiss Index created and saved")

# 이미 생성된 FAISS 인덱스를 로드
# 사용자가 입력한 질문과 관련되 문서 검색 준비 단계
def load_faiss_index():
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    load_db = FAISS.load_local(
        restaurant_faiss, embeddings, allow_dangerous_deserialization=True
    )
    return load_db

# 문서 포메팅과 답변 생성
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 답변 생성, RAG 핵심 부분
def answer_question(db, query):
    # OpenAI 언어 모델 초기화
    llm = OpenAI(api_key=OPENAI_API_KEY)
    # 사용자 정의 프롬포트 템플릿 생성
    prompt_template = """
    당신은 유능한 AI 비서입니다. 주어진 맥락 정보를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해야 합니다.
    
    맥락 : {context}
    
    질문 : {question}
    
    답변을 작성할 때 다음 지침을 따르세요 :
    1. 주어진 맥락 정보에 있는 내용만을 사용하여 답변
    2. 맥락 정보에 없는 내용은 답변에 포함하지 말 것
    3. 질문과 관련이 없는 정보는 제외
    4. 답변은 간결하고 명확하게 작성
    5. 불확실한 경우, "주어진 정보로는 정확한 답변을 드릴 수 없습니다."라고 말할 것
    
    답변 : 
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    qa_chain = (
            {
                "context":db.as_retriever() | format_docs,
                "question":RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
    )
    # 질문에 대한 답변 생성
    result = qa_chain.invoke(query)
    return result
# OpenAI : OpenAI 언어 모델 초기화
# PromptTemplate : 프롬포트 템플릿을 정의하여, AI가 질문에 답변할 때 따를 지침을 제공
# qa_chain : 검색된 문서들과 사용자 질문을 함께 사용해 답변을 생성하는 파이프라인 구성

# 메인 함수 작성
def main():
    # FAISS 인덱스가 없으면 생성
    if not os.path.exists(restaurant_faiss):
        create_faiss_index()

    # FAISS 인덱스 로드
    db = load_faiss_index()
    while True:
        query = input("레스토랑에 대해서 궁금한 점을 물어보세요 (종료하려면 'quit' 입력'): ")

        if query.lower() == 'quit' :
            break

        answer = answer_question(db, query)
        print(f"답변 : {answer}\n")

if __name__ == "__main__":
    main()