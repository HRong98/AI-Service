from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
current_dir = os.path.dirname(os.path.abspath(__file__))

# 쿼리 기반 유사 문서 검색
# main() 비동기 함수 정의
async def main():
    # 환경 변수에서 가져온 OpenAI API 키를 사용하여 OpenAIEmbeddings 클래스 초기화
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    # 지정된 임베딩을 사용하여 로컬에 저장된 FAISS 인덱스 로드
    # allow_dangerous_deserialization=True 옵션은 역직렬화를 허용
    load_db = FAISS.load_local(f'{current_dir}/restaurant-faiss', embeddings, allow_dangerous_deserialization=True)
    # 유사성 검색 준비, allow_dangerous_deserialization=True 는 인덱스를 안전하게 로드하기 위해 필요

    # 검색할 쿼리 문자열 정의
    query = "음식점의 룸 서비스는 어떻게 운영되나요?"
    # `query` 변수는 사용자가 검색하는 질문이나 문장을 담음
    # `k=2` 는 가장 유사한 문서 2개를 반환하도록 지정
    result = load_db.similarity_search(query, k=2)
    # 검색 결과를 출력
    print(result, "\n")

    # 임베딩 벡터로 문서 유사도 검색
    # 쿼리르 임베딩 벡터로 변환
    embedding_vector_query = embeddings.embed_query(query)
    print("Query vector: ", embedding_vector_query, "\n")

    # 벡터 기반의 비동기 검색 수행
    # 임베딩 벡터를 사용하여 비동기 방식으로 유사한 문서 검색
    docs = await load_db.asimilarity_search_by_vector(embedding_vector_query)
    # 거색된 문서 중 첫 문서를 출력
    print(docs[0])

if __name__ == "__main__":
    asyncio.run(main())