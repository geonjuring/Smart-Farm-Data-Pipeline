from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import psycopg2

# 기본 설정 (DAG Arguments)
default_args = {
    'owner': 'geon_tae',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1), # 2026년 기준 가동
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def train_and_evaluate_task():
    print("========= [Airflow Task] 스마트팜 ML 파이프라인 고도화 가동 =========")
    
    # 1. 원본 데이터 로드 (Airflow 가동 서버 내의 CSV 경로 확인 필요)
    try:
        df = pd.read_csv('/opt/airflow/dags/data/2022_env.CSV', encoding='cp949')
        print(f"-> 원본 데이터 로드 완료 (총 {len(df):,}건)")
    except FileNotFoundError:
        # 만약 경로가 다르면 로컬 경로로 재시도
        try:
            df = pd.read_csv('2022_env.CSV', encoding='cp949')
            print(f"-> 로컬 경로에서 데이터 로드 완료 (총 {len(df):,}건)")
        except FileNotFoundError:
            print("🚨 에러: '2022_env.CSV' 파일을 찾을 수 없습니다.")
            raise FileNotFoundError

    # 2. 교수님 피드백 반영: Pandas를 활용하여 '딸기' 품목만 최우선 필터링
    strawberry_df = df[df['품목'] == '딸기'].copy()
    print(f"-> 딸기 품목 필터링 완료 ({len(strawberry_df):,}건)")

    # 3. 시계열 정렬: 선형 보간을 위해 측정시간 순서대로 정렬
    strawberry_df['측정시간'] = pd.to_datetime(strawberry_df['측정시간'])
    strawberry_df = strawberry_df.sort_values('측정시간')

    # 4. 시계열 선형 보간법(Linear Interpolation) 적용하여 결측치 완벽 정제
    core_columns = ['온도_외부', '일사량_외부', '강우감지', '풍속_외부', '온도_내부', '상대습도_내부']
    for col in core_columns:
        strawberry_df[col] = strawberry_df[col].interpolate(method='linear').ffill().bfill()
    print(f"-> 결측치 선형 보간 완료. 최종 학습 데이터: {len(strawberry_df):,}건")

    # 5. 독립 변수(X) 및 종속 변수(y) 설정 (풍속 포함 4개 변수 확정)
    X = strawberry_df[['온도_외부', '일사량_외부', '강우감지', '풍속_외부']]
    y_temp = strawberry_df['온도_내부']
    y_hum = strawberry_df['상대습도_내부']

    # 6. 데이터셋 분할 및 학습
    X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
    X_train_h, X_test_h, y_hum_train, y_hum_test = train_test_split(X, y_hum, test_size=0.2, random_state=42)

    model_temp = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42).fit(X_train, y_temp_train)
    model_hum = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42).fit(X_train_h, y_hum_train)

    # 7. 모델 검증 지표 계산
    pred_temp = model_temp.predict(X_test)
    pred_hum = model_hum.predict(X_test_h)

    mae_temp = mean_absolute_error(y_temp_test, pred_temp)
    r2_temp = r2_score(y_temp_test, pred_temp)
    mae_hum = mean_absolute_error(y_hum_test, pred_hum)
    r2_hum = r2_score(y_hum_test, pred_hum)

    print(f"🌡️ 온도 예측 R2: {r2_temp:.4f} | MAE: {mae_temp:.4f}")
    print(f"💧 습도 예측 R2: {r2_hum:.4f} | MAE: {mae_hum:.4f}")

    # 8. PostgreSQL 데이터베이스에 로그 기록 저장
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="password", # 건태님의 실제 DB 패스워드로 설정 필요
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
        """, (datetime.now(), "v5", mae_temp, r2_temp, mae_hum, r2_hum))
        conn.commit()
        cursor.close()
        conn.close()
        print("-> [성공] PostgreSQL에 고도화 성과 로그 적재 완료!")
    except Exception as e:
        print(f"🚨 DB 적재 실패: {e}")

# Airflow DAG 정의
with DAG(
    'smart_farm_strawberry_v5',
    default_args=default_args,
    description='교수님 피드백 반영: 딸기 필터링 및 시계열 선형 보간 기반의 온습도 예측 파이프라인',
    schedule_interval='0 7 * * *',  # 매일 아침 7시 자동 가동
    catchup=False
) as dag:

    # 머신러닝 전처리 및 학습 태스크 정의
    ml_training_task = PythonOperator(
        task_id='strawberry_ml_training_and_eval',
        python_callable=train_and_evaluate_task
    )

    # 태스크 가동
    ml_training_task