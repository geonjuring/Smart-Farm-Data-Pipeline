import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from airflow import DAG
from airflow.operators.python import PythonOperator
import psycopg2

# 기본 설정 (DAG Arguments)
default_args = {
    'owner': 'geon_tae',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1), # 2026년 가동
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def train_and_evaluate_task():
    print("========= [Airflow Task] 스마트팜 ML 파이프라인 팩트 기반 극한 고도화 =========")
    
    # 1. 데이터 로드
    try:
        df = pd.read_csv('/opt/airflow/dags/data/2022_env.CSV', encoding='cp949')
    except FileNotFoundError:
        df = pd.read_csv('2022_env.CSV', encoding='cp949')

    print(f"-> 원본 데이터 로드 완료 (총 {len(df):,}건)")

    # 2. '딸기' 품목 최우선 필터링
    strawberry_df = df[df['품목'] == '딸기'].copy()
    print(f"-> 딸기 품목 필터링 완료 ({len(strawberry_df):,}건)")

    # 3. 시계열 정렬
    strawberry_df['측정시간'] = pd.to_datetime(strawberry_df['측정시간'])
    strawberry_df = strawberry_df.sort_values('측정시간')

    # 4. 확인된 정확한 팩트 컬럼명 반영하여 결측치 선형 보간 정제
    core_columns = [
        '온도_외부', '일사량_외부', '강우감지', '풍속_외부', 
        '누적일사량_외부', '잔존이산화탄소(CO2)', '온도_내부', '상대습도_내부'
    ]
    for col in core_columns:
        strawberry_df[col] = strawberry_df[col].interpolate(method='linear').ffill().bfill()
    print(f"-> [성공] 확인된 팩트 컬럼 포함 8대 핵심 변수 선형 보간 완료!")

    # 5. [시계열 파생 변수] 1시간 전 내부습도(Lag Feature) 생성
    strawberry_df['상대습도_내부_Lag1'] = strawberry_df['상대습도_내부'].shift(1).bfill()
    print(f"-> 시계열 관성 피처(상대습도_내부_Lag1) 생성 완료!")

    # 6. 독립 변수(X) 및 종속 변수(y) 최종 확정
    features = [
        '온도_외부', '일사량_외부', '강우감지', '풍속_외부', 
        '누적일사량_외부', '잔존이산화탄소(CO2)', '상대습도_내부_Lag1'
    ]
    
    X = strawberry_df[features]
    y_temp = strawberry_df['온도_내부']
    y_hum = strawberry_df['상대습도_내부']

    # 7. 데이터셋 분할
    X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
    X_train_h, X_test_h, y_hum_train, y_hum_test = train_test_split(X, y_hum, test_size=0.2, random_state=42)

    # 8. 자원 대형 오버헤드 방지용 다이어트 RandomForest 학습 기동
    print("-> 🚀 확장 피처 레이어 기반 최적화 머신러닝 학습 시작...")
    model_temp = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=1).fit(X_train, y_temp_train)
    model_hum = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=1).fit(X_train_h, y_hum_train)

    # 9. 모델 검증 지표 계산
    pred_temp = model_temp.predict(X_test)
    pred_hum = model_hum.predict(X_test_h)

    mae_temp = mean_absolute_error(y_temp_test, pred_temp)
    r2_temp = r2_score(y_temp_test, pred_temp)
    mae_hum = mean_absolute_error(y_hum_test, pred_hum)
    r2_hum = r2_score(y_hum_test, pred_hum)

    print("\n=======================================================")
    print(f"🌡️ 극한 고도화 내부 온도 예측 R2: {r2_temp:.4f} | MAE: {mae_temp:.4f}")
    print(f"💧 극한 고도화 내부 습도 예측 R2: {r2_hum:.4f} | MAE: {mae_hum:.4f}")
    print("=======================================================\n")

    # 10. PostgreSQL에 최종 성과 로그 적재
    try:
        conn = psycopg2.connect(
            dbname="agriculture",       
            user="tae_tae",             
            password="smartfarm",       
            host="host.docker.internal",
            port="5432"
        )
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance_logs (
                id SERIAL PRIMARY KEY,
                run_time TIMESTAMP,
                version_info VARCHAR(50),
                temp_mae FLOAT,
                temp_r2 FLOAT,
                hum_mae FLOAT,
                hum_r2 FLOAT
            );
        """)
        cursor.execute("""
            INSERT INTO model_performance_logs (run_time, version_info, temp_mae, temp_r2, hum_mae, hum_r2)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (datetime.now(), "v5_extreme_fact", mae_temp, r2_temp, mae_hum, r2_hum))
        conn.commit()
        cursor.close()
        conn.close()
        print("-> [성공] 도커 컴포즈 프로필 매핑을 통해 PostgreSQL 로그 적재 완벽 완료!")
    except Exception as e:
        print(f"🚨 DB 적재 실패: {e}")

# Airflow DAG 정의
with DAG(
    'smart_farm_strawberry_v5',
    default_args=default_args,
    description='',
    schedule_interval='0 7 * * *',
    catchup=False
) as dag:

    ml_training_task = PythonOperator(
        task_id='strawberry_ml_training_and_eval',
        python_callable=train_and_evaluate_task
    )

    ml_training_task