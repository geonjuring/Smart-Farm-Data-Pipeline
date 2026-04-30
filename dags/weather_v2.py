from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv

load_dotenv()


def collect_and_save():
    
    api_key = os.getenv("DATA_GO_KR_API_KEY")
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
    
    
    params = {
        'serviceKey': api_key, 
        'pageNo': '1', 
        'numOfRows': '10', 
        'dataType': 'JSON',
        'base_date': (datetime.now() - timedelta(hours=1)).strftime('%Y%m%d'),
        'base_time': (datetime.now() - timedelta(hours=1)).strftime('%H00'),
        #'base_date': '20260429',  # 어제 날짜로 고정
        #'base_time': '2300',      # 밤 11시로 고정
        'nx': '63', 
        'ny': '61'
    }
    
    try:
        response = requests.get(url, params=params).json()
        items = response['response']['body']['items']['item']
        
        # 2. DB 연결 (우리가 설정한 postgres_default 사용)
        pg_hook = PostgresHook(postgres_conn_id='postgres_default')
        
        # 3. 테이블 자동 생성 (정석적인 방법)
        create_sql = """
        CREATE TABLE IF NOT EXISTS weather_data (
            id SERIAL PRIMARY KEY,
            base_date TEXT,
            base_time TEXT,
            category TEXT,
            obs_value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        pg_hook.run(create_sql)
        
        # 4. 데이터 삽입
        for item in items:
            insert_sql = "INSERT INTO weather_data (base_date, base_time, category, obs_value) VALUES (%s, %s, %s, %s)"
            # 'obsValue'를 기상청 표준 명칭인 'obsrValue'로 수정합니다.
            pg_hook.run(insert_sql, parameters=(item['baseDate'], item['baseTime'], item['category'], item['obsrValue']))
        
        print("✅ 데이터 수집 및 DB 저장 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

with DAG(
    'smart_farm_weather_v2',
    start_date=datetime(2026, 4, 30),
    schedule_interval='@hourly',
    catchup=False
) as dag:
    task = PythonOperator(task_id='save_to_db', python_callable=collect_and_save)