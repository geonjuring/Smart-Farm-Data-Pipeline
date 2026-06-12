# smart_farm_rag.py
import os
import psycopg2
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class SmartFarmRAG:
    def __init__(self):
        self.pdf_path = "data/docs/농업기술길잡이40_딸기.PDF"
        self.persist_db_dir = "./chroma_db"
        
        # 1. OpenAI 임베딩 및 Vector DB 설정
        self.embeddings = OpenAIEmbeddings()
        
        if os.path.exists(self.persist_db_dir) and os.listdir(self.persist_db_dir):
            print("📦 기존에 구축된 ChromaDB 벡터 기지를 로드합니다...")
            self.vector_db = Chroma(
                persist_directory=self.persist_db_dir, 
                embedding_function=self.embeddings
            )
        else:
            print("🚀 기존 벡터 기지가 없습니다. 딸기 PDF 분석 및 초기 임베딩을 시작합니다...")
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"❌ 딸기 가이드북 PDF를 찾을 수 없습니다: {self.pdf_path}")
                
            loader = PyMuPDFLoader(self.pdf_path)
            data = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            
            self.vector_db = Chroma.from_documents(
                documents=chunks, 
                embedding=self.embeddings,
                persist_directory=self.persist_db_dir
            )
            print("✅ ChromaDB 임베딩 기지 구축 완료!")
        
        # 2. 최신 LLM 및 검색기(Retriever) 레이어 구성
        self.llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 4})

    def get_latest_sensor_data(self):
        """PostgreSQL에서 Airflow가 적재한 최신 AI 예측치 및 컨텍스트 로드"""
        try:
            conn = psycopg2.connect(
                dbname="agriculture", 
                user="tae_tae", 
                password="smartfarm",
                host="localhost",
                port="5432"
            )
            cursor = conn.cursor()
            
            query = """
                SELECT 측정시간, 온도_내부, 상대습도_내부, 내부_VPD
                FROM farm_crop_prediction_results
                ORDER BY 측정시간 DESC
                LIMIT 1;
            """
            cursor.execute(query)
            row = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if row:
                return row
            else:
                return ("N/A", 0.0, 0.0, 0.0, 0)
        except Exception as e:
            print(f"⚠️ [PostgreSQL RDB 연결 실패]: {e}")
            return ("N/A", 0.0, 0.0, 0.0, 0)

    def ask_advisor(self, user_question):
        """최신 LCEL 체인 프레임워크 기반 하이드리브 RAG 연산 가동"""
        from datetime import datetime as dt
        
        
        ftime, temp, hum, vpd = self.get_latest_sensor_data()
        
        
        try:
            real_plantation = dt(2025, 9, 15)
            dat = (dt.now() - real_plantation).days
        except Exception:
            dat = 260 
            
        
        if dat <= 30: crop_phase = 'GROWTH'
        elif dat <= 60: crop_phase = 'FLOWER'
        else: crop_phase = 'HARVEST'
        
        # 시스템 핵심 템플릿 정의
        system_template = f"""
        당신은 농촌진흥청 '농업기술길잡이 딸기' 가이드북 핵심 지식을 마스터한 스마트팜 AI 어드바이저입니다.
        현재 농장 상태와 하우스 예측 수치를 면밀히 분석한 후, 농민의 질문에 대해 실전적이고 정밀한 데이터 처방을 내리십시오.
        
        [실시간 농장 예측 컨텍스트]
        - 예측 대상 시간: {ftime}
        - 정식 후 경과일 (DAT): {dat}일차 (현재 생육 단계: {crop_phase})
        - 1시간 뒤 내부 온도 예측치: {temp:.2f}℃
        - 1시간 뒤 내부 습도 예측치: {hum:.2f}%
        - 1시간 뒤 물리 포차 (VPD): {vpd:.2f} kPa
        
        [가이드북 전문 지식 베이스]
        {{context}}
        
        위 가이드북 지식과 실시간 예측 컨텍스트를 융합하여 농민이 즉시 실행 가능한 솔루션을 제공하세요. 수치 한계선(온습도 임계치)을 언급할 때는 반드시 정확한 수치를 팩트 기반으로 제시하십시오.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{input}")
        ])

        # 내부 문서들을 조인하는 헬퍼 함수
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 🌟 최신 LCEL 파이프라인 사슬 조립 (연산자 선언식)
        rag_chain = (
            {"context": self.retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # 최종 인보크 실행
        answer = rag_chain.invoke(user_question)
        return answer