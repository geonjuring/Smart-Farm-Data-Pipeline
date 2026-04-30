# Smart-Farm-Data-Pipeline 🌿

> **Airflow와 Docker를 활용한 스마트팜 실시간 기상 데이터 수집 및 자동화 인프라 구축**

## 1. 프로젝트 개요
스마트팜 지능형 관리 시스템 구축의 1단계로, 기상청 공공데이터 API를 활용하여 외부 환경 데이터를 실시간으로 수집하고 PostgreSQL 데이터베이스에 자동 적재하는 ETL 파이프라인을 구축했습니다.

## 2. 기술 스택 (Tech Stack)
* **Orchestration**: Apache Airflow
* **Database**: PostgreSQL
* **Infrastructure**: Docker, Docker Compose
* **Language**: Python (Requests, PostgresHook, dotenv)

## 3. 핵심 기능 및 문제 해결 (Troubleshooting)
* **API 데이터 파이프라인 자동화**: 매시간 정각(`@hourly`) 기상 실황 데이터를 수집하도록 스케줄링 구현.
* **데이터 정합성 해결**: API 응답 필드명 오류(`obsValue` -> `obsrValue`)를 로그 분석을 통해 발견하고 수정하여 데이터 유실 방지.
* **보안 관리**: `.env` 파일과 환경 변수를 활용하여 API 키 및 DB 자격 증명을 코드에서 분리하여 보안성 강화.
* **인프라 표준화**: Docker Compose를 통해 개발 환경에 구애받지 않는 독립적인 데이터 수집 환경 구축.

## 4. 실행 결과
* **Airflow DAG**: 모든 작업이 성공적으로 스케줄링됨을 확인.
* **Database**: PostgreSQL 테이블에 실시간 기온, 습도 등 8종의 기상 데이터가 정상 적재됨을 확인.

<img width="1583" height="591" alt="Image" src="https://github.com/user-attachments/assets/7d670cc8-f524-4c1a-849a-db7d400230b9" />

<img width="1001" height="322" alt="Image" src="https://github.com/user-attachments/assets/c31788c3-cbc2-42d7-9765-56bd993a795f" />
