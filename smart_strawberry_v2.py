import os
import psycopg2

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_openai import OpenAIEmbeddings, ChatOpenAI


from langchain_chroma import Chroma


from langchain_core.prompts import ChatPromptTemplate


from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()

class SmartFarmRAG:
    def __init__(self):
        # 1. PDF 로드 및 벡터 DB 구축
        # WSL 경로에 맞춰 수정됨
        pdf_path = "data/docs/농업기술길잡이40_딸기.PDF"
        if not os.path.exists(pdf_path):
            print(f"❌ 파일을 찾을 수 없습니다: {pdf_path}")
            
        loader = PyMuPDFLoader(pdf_path)
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        
        self.vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=OpenAIEmbeddings(),
            persist_directory="./chroma_db"
        )
        
        # 2. LLM 설정
        self.llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)

    def get_latest_sensor_data(self):
        """PostgreSQL에서 Airflow가 수집한 최신 데이터 로드"""
        try:
            conn = psycopg2.connect(
                host="localhost", # Docker가 윈도우에서 실행중이므로 그대로 유지
                database="agriculture",
                user="tae_tae",
                password="smartfarm"
            )
            cur = conn.cursor()
            cur.execute("SELECT forecast_time, pred_in_temp, pred_in_hum FROM strawberry_ai_forecast ORDER BY created_at DESC LIMIT 1")
            row = cur.fetchone()
            cur.close()
            conn.close()
            return row if row else ("N/A", 0.0, 0.0)
        except Exception as e:
            print(f"⚠️ DB 연결 에러: {e}")
            return ("N/A", 0.0, 0.0)

    def ask_advisor(self, user_question):
        ftime, temp, hum = self.get_latest_sensor_data()
        
        # 시스템 프롬프트 정의
        system_template = f"""
        당신은 딸기 스마트팜 전문 AI 어드바이저입니다. 
        아래의 [현재 농장 상태]와 [가이드북 지식]을 바탕으로 관리 처방을 내리세요.
        [현재 농장 상태] 시간: {ftime}, 온도: {temp:.2f}℃, 습도: {hum:.2f}%
        
        [가이드북 지식]
        {{context}}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{input}")
        ])

        # 체인 조립 (최신 방식)
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(self.vector_db.as_retriever(), combine_docs_chain)
        
        # 실행 (invoke 사용)
        response = retrieval_chain.invoke({"input": user_question})
        return response["answer"]

# 실행 예시
if __name__ == "__main__":
    print("🍓 스마트팜 AI 어드바이저를 시작합니다. (종료하려면 '종료' 또는 'exit' 입력)")
    
    # 1. 시스템 초기화
    rag = SmartFarmRAG()
    print("✅ 데이터 로딩 및 분석 완료! 질문을 입력해주세요.\n")

    # 2. 무한 루프를 통해 지속적으로 질문 받기
    while True:
        user_input = input("👤 질문: ").strip()
        
        # 종료 조건 확인
        if user_input.lower() in ['종료', 'exit', 'quit', 'q']:
            print("👋 어드바이저를 종료합니다. 풍년 되세요!")
            break
            
        if not user_input:
            continue

        print("🔍 가이드북 분석 중...")
        
        try:
            # AI에게 질문하고 답변 받기
            answer = rag.ask_advisor(user_input)
            print(f"\n🤖 AI 어드바이저 조언:\n{answer}\n")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")