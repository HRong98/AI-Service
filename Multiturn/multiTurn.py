from openai import OpenAI
# 환경변수 불러오기
from dotenv import load_dotenv
load_dotenv()
#프론트 엔드 구현
import streamlit as st

client = OpenAI()

st.header("현진건 작가님과의 대화")

# Streamlit Session 상태 저장된 값 유지
# 대화 히스토리 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 지난 대화 히스토리 출력
if 'response_id' in st.session_state:
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.write(message['content'])

# 질문 입력
prompt = st.chat_input("물어보고 싶은 것을 입력하세요!")

if prompt :
    # st.chat_message() = user 전달
    # st.write() = prompt 전달
    # 사용자 질문 출력 및 히스토리 저장
    with st.chat_message('user') :
        st.write(prompt)
    # 메세지의 role과 content를 딕셔너리 형태로 저장
    st.session_state.chat_history.append({'role':'user', 'content':prompt})

    # 지난 대화가 없을 때
    if 'response_id' not in st.session_state:
        # 질문에 대한 답변 생성
        # API 파라미터 instructions
        response = client.responses.create(
            model = "gpt-4o-mini",
            instructions = "당신은 소설 운수 좋은 날을 집필한 현진건 작가님입니다.",
            input = prompt,
            tools = [{
                "type": "file_search",
                "vector_store_ids" : ["벡터 아이디"]
            }]
        )
    # 지난 대화가 있을 때
    else :
        # 로딩 애니메이션 설정
        with st.spinner('Wait for it ...'):
            response = client.responses.create(
                previous_response_id = st.session_state.response_id,
                model = "gpt-4o-mini",
                instructions = "당신은 소설 운수 좋은 날을 집필한 현진건 작가님입니다.",
                input = prompt,
                tools = [{
                    "type": "file_search",
                    "vector_store_ids" : ["벡터 아이디"]
                }]
            )
    # LLM 답변 출력 및 히스토리 저장
    with st.chat_message('assistant'):
        st.write(response.output_text)
    st.session_state.chat_history.append({'role':'assistant', 'content':response.output_text})
    st.session_state.response_id = response.id
