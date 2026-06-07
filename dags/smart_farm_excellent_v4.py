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
    'start_date': datetime(2026, 1, 1), 
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def calculate_vpd(temp, humidity):
    """
    [물리 공식 레이어] 내부 온도(°C)와 상대습도(%)를 받아 
    Tetens 공식을 기반으로 포차(VPD, kPa 단위)를 연산
    """
    svp = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
    avp = svp * (humidity / 100.0)
    vpd = svp - avp
    return vpd

def train_and_evaluate_task():
    print("========= [Airflow Task] 스마트팜 ML 파이프라인 3주차 인프라-도메인 융합 가동 =========")
    
    # 1. 데이터 로드
    try:
        df = pd.read_csv('/opt/airflow/dags/data/2022_env.CSV', encoding='cp949')
    except FileNotFoundError:
        df = pd.read_csv('2022_env.CSV', encoding='cp949')

    print(f"-> 원본 데이터 로드 완료 (총 {len(df):,}건)")

    # 2. '딸기' 품목 최우선 필터링
    strawberry_df = df[df['품목'] == '딸기'].copy()
    print(f"-> 딸기 품목 필터링 완료 ({len(strawberry_df):,}건)")

    # 3. 시계열 정렬 및 날짜 처리
    strawberry_df['측정시간'] = pd.to_datetime(strawberry_df['측정시간'])
    strawberry_df = strawberry_df.sort_values('측정시간')
    
    # 조인(Join) 키로 사용할 날짜(Date) 컬럼 임시 추출
    strawberry_df['target_date'] = strawberry_df['측정시간'].dt.date

    # 4. 핵심 변수 결측치 선형 보간 정제
    core_columns = [
        '온도_외부', '일사량_외부', '강우감지', '풍속_외부', 
        '누적일사량_외부', '잔존이산화탄소(CO2)', '온도_내부', '상대습도_내부'
    ]
    for col in core_columns:
        strawberry_df[col] = strawberry_df[col].interpolate(method='linear').ffill().bfill()
    print(f"-> [성공] 8대 핵심 변수 선형 보간 정제 완료!")

    # 5. [PostgreSQL RDB 데이터 동적 융합 및 피처 엔지니어링]
    try:
        conn = psycopg2.connect(
            dbname="agriculture", user="tae_tae", password="smartfarm",
            host="host.docker.internal", port="5432"
        )
        
        # 5-1. 일출몰 데이터 긁어오기
        sun_df = pd.read_sql("SELECT target_date, sunrise_time, sunset_time FROM sun_time_context", conn)
        sun_df['target_date'] = pd.to_datetime(sun_df['target_date']).dt.date
        
        # 5-2. 딸기 농가 프로필 긁어오기
        crop_df = pd.read_sql("SELECT farm_id, plantation_date, current_growth_stage FROM farm_crop_context WHERE farm_id='FRM_ST_01'", conn)
        
        conn.close()
        print("-> [성공] PostgreSQL 백엔드 마스터 데이터셋 Read 성공!")

        # 5-3. Pandas 데이터프레임 조인(Merge)
        strawberry_df = pd.merge(strawberry_df, sun_df, on='target_date', how='left')
        
        # 크롭 컨텍스트는 단일 농가 정보이므로 브로드캐스팅 결합
        for col in ['plantation_date', 'current_growth_stage']:
            strawberry_df[col] = crop_df[col].values[0]
            
        print("-> [성공] RDB 컨텍스트 데이터 파이프라인 융합 완료!")

        # 5-4. 파생 변수 연산 (낮/밤 판별 및 DAT 계산)
        strawberry_df['plantation_date'] = pd.to_datetime(strawberry_df['plantation_date'])
        
        # DAT (정식 후 경과일) 계산
        strawberry_df['DAT'] = (strawberry_df['측정시간'] - strawberry_df['plantation_date']).dt.days
        
        # Is_Daytime (낮 여부 부울 플래그) 계산
        current_time = strawberry_df['측정시간'].dt.time
        strawberry_df['Is_Daytime'] = (current_time >= strawberry_df['sunrise_time']) & (current_time <= strawberry_df['sunset_time'])
        strawberry_df['Is_Daytime'] = strawberry_df['Is_Daytime'].astype(int)
        
    except Exception as e:
        print(f"🚨 RDB 연동 및 파생변수 생성 실패 (기본값 대체): {e}")
        strawberry_df['DAT'] = 260  # 수확기 평균 DAT 베이스라인
        strawberry_df['Is_Daytime'] = 1

    # 6. [물리 공식 및 시계열 멀티 시차 확장]
    # 물리 포차(VPD) 파생 변수 레이어 투하
    strawberry_df['내부_VPD'] = calculate_vpd(strawberry_df['온도_내부'], strawberry_df['상대습도_내부'])
    
    # 멀티 시차 변수 확장 (Lag1, Lag2, Lag3)
    strawberry_df['상대습도_내부_Lag1'] = strawberry_df['상대습도_내부'].shift(1).bfill()
    strawberry_df['상대습도_내부_Lag2'] = strawberry_df['상대습도_내부'].shift(2).bfill()
    strawberry_df['상대습도_내부_Lag3'] = strawberry_df['상대습도_내부'].shift(3).bfill()
    print(f"-> 시계열 멀티 시차(Lag1~3) 및 물리 포차(VPD) 파생 변수 레이어 탑재 완료!")

    # 🌟 [위치 조정] 독립 변수(X) 및 종속 변수(y) 컬럼 지정을 먼저 수행합니다.
    features = [
        '온도_외부', '일사량_외부', '강우감지', '풍속_외부', 
        '누적일사량_외부', '잔존이산화탄소(CO2)', 
        '상대습도_내부_Lag1', '상대습도_내부_Lag2', '상대습도_내부_Lag3', 
        '내부_VPD', 'DAT', 'Is_Daytime' 
    ]

    # 7. [수정] 월별 블록 기반 시계열 분할 (학습-테스트 연관성 확보 및 데이터 누수 차단)
    
    # 2022년 9월 15일 정식 이후의 정상 작기 데이터만 타겟팅
    actual_crop_df = strawberry_df[strawberry_df['측정시간'] >= pd.to_datetime('2022-09-15')].copy()
    actual_crop_df['day'] = actual_crop_df['측정시간'].dt.day
    
    # 연관성 유지를 위해 매월 일자별로 8:2 분할
    train_df = actual_crop_df[actual_crop_df['day'] <= 24]
    test_df = actual_crop_df[actual_crop_df['day'] > 24]

    print(f"-> [블록 시계열 분할 완료] 학습셋(1~24일): {len(train_df):,}건 | 테스트셋(25일~말일): {len(test_df):,}건")

    # 독립 변수(X) 및 종속 변수(y) 추출
    X_train = train_df[features]
    y_temp_train = train_df['온도_내부']
    y_hum_train = train_df['상대습도_내부']

    X_test = test_df[features]
    y_temp_test = test_df['온도_내부']
    y_hum_test = test_df['상대습도_내부']

    # 8. 자원 최적화 RandomForest 학습 기동 (중복 제거 및 변수명 정상화)
    print("-> 🚀 과적합 방지 시계열 검증 레이어 기반 학습 시작...")
    model_temp = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=1).fit(X_train, y_temp_train)
    model_hum = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=1).fit(X_train, y_hum_train)

    # 9. 모델 검증 지표 계산 (존재하지 않는 _h 접미사 완전 제거)
    pred_temp = model_temp.predict(X_test)
    pred_hum = model_hum.predict(X_test)

    # 10. 스코어 연산
    mae_temp = mean_absolute_error(y_temp_test, pred_temp)
    r2_temp = r2_score(y_temp_test, pred_temp)
    mae_hum = mean_absolute_error(y_hum_test, pred_hum)
    r2_hum = r2_score(y_hum_test, pred_hum)
    print("\n=======================================================")
    print(f"🏆 [3주차 최종] 내부 온도 예측 R2: {r2_temp:.4f} | MAE: {mae_temp:.4f}")
    print(f"🏆 [3주차 최종] 내부 습도 예측 R2: {r2_hum:.4f} | MAE: {mae_hum:.4f}")
    print("=======================================================\n")

    # 11. PostgreSQL에 최종 성과 로그 적재
    try:
        conn = psycopg2.connect(
            dbname="agriculture", user="tae_tae", password="smartfarm",
            host="host.docker.internal", port="5432"
        )
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO model_performance_logs (run_time, version_info, temp_mae, temp_r2, hum_mae, hum_r2)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (datetime.now(), "v6_ultimate_infrastructure", mae_temp, r2_temp, mae_hum, r2_hum))
        conn.commit()
        cursor.close()
        conn.close()
        print("-> [성공] 3주차 고도화 성과 데이터베이스 최종 적재 완료!")
    except Exception as e:
        print(f"🚨 성과 로그 DB 적재 실패: {e}")

# Airflow DAG 정의
with DAG(
    'smart_farm_strawberry_v6',
    default_args=default_args,
    description='3주차 인프라 조인 및 물리 공식 융합 파이프라인',
    schedule_interval='0 7 * * *',
    catchup=False
) as dag:

    ml_training_task = PythonOperator(
        task_id='strawberry_ml_training_and_eval',
        python_callable=train_and_evaluate_task
    )