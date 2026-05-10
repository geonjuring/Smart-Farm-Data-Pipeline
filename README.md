🍓 Smart Farm Data Pipeline & AI Forecast System
본 프로젝트는 과거 농가 환경 데이터와 실시간 기상청 예보 데이터를 결합하여, 딸기 비닐하우스 내부의 온습도를 예측하고 최적의 재배 가이드를 제공하는 지능형 데이터 파이프라인 구축 프로젝트입니다.

🚀 Key Achievements (현재 진행 상황)
다중 변수 회귀 모델 구축: 외부 기온, 풍속, 강우 데이터를 결합한 Multiple Linear Regression 모델 구현.

데이터 파이프라인 자동화: Apache Airflow를 통한 데이터 수집, 학습, 예측 및 DB 적재 전 과정 자동화.

모델 성능 지표: MAE 3.17도 / R2 Score 0.61 달성 (성능 최적화 진행 중).

🛠 Tech Stack
Languages: Python (Scikit-learn, Pandas)

Workflow: Apache Airflow

Database: PostgreSQL

Infrastructure: Docker, WSL2

AI Models: Linear Regression (Next: RAG with Gemini)

📋 System Architecture
Data Ingestion: 기상청 단기예보 API 및 과거 센서 데이터(CSV) 수집.

ML Pipeline: 다중 변수를 활용한 내부 환경 예측 모델 학습 및 검증.

Data Storage: PostgreSQL에 실시간 예측 수치 및 성능 로그 저장.

Knowledge Base (In-progress): RAG를 활용한 딸기 재배 지식 상담 서비스 구축 중.

📈 Future Roadmap
[x] 다중 변수 기반 AI 예측 모델 최적화

[ ] ChromaDB 연동 및 PDF 재배 지식 임베딩 (RAG)

[ ] Gradio를 활용한 지능형 스마트팜 대시보드 시각화