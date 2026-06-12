# dags/smart_farm_strawberry_v6.py
import os
import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from airflow import DAG
    from airflow.providers.standard.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    print("ℹ️ [인프라 알림] 외부 가상환경 런타임 감지 (Airflow 모듈 스킵)")
    AIRFLOW_AVAILABLE = False
import psycopg2

# 기본 설정 (DAG Arguments)
default_args = {
    'owner': 'geon_tae',
    'depends_on_past': False,
    'start_date': dt(2026, 1, 1), 
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def calculate_vpd(temp, humidity):
    """[물리 공식 레이어] 내부 온도와 상대습도를 받아 Tetens 공식을 기반으로 물리 포차(VPD) 연산"""
    svp = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
    avp = svp * (humidity / 100.0)
    return svp - avp

def train_and_evaluate_task():
    print("========= [Airflow Task] 스마트팜 ML 파이프라인 3주차 인프라-도메인 융합 가동 =========")
    
    current_file_dir = os.path.dirname(os.path.abspath(__file__))  # dags 폴더 위치 추출
    absolute_csv_path = os.path.join(current_file_dir, "data", "2022_env.CSV")
    
    possible_paths = [
        absolute_csv_path,
        "dags/data/2022_env.CSV",
        "data/2022_env.CSV",
        "/opt/airflow/dags/data/2022_env.CSV"
    ]
    
    csv_path = None
    for p in possible_paths:
        if os.path.exists(p):
            csv_path = p
            break
            
    if not csv_path:
        raise FileNotFoundError("❌ 원천 데이터셋(2022_env.CSV)의 물리적 경로를 절대 찾을 수 없습니다.")
        
    df = pd.read_csv(csv_path, encoding='cp949')
    print(f"-> 원본 데이터 로드 완료 (총 {len(df):,}건 | 경로: {csv_path})")

    # 2. '딸기' 품목 필터링
    strawberry_df = df[df['품목'] == '딸기'].copy()
    print(f"-> 딸기 품목 필터링 완료 ({len(strawberry_df):,}건)")

    # 3. 시계열 정렬 및 날짜 처리
    strawberry_df['측정시간'] = pd.to_datetime(strawberry_df['측정시간'])
    strawberry_df = strawberry_df.sort_values('측정시간')
    strawberry_df['target_date'] = strawberry_df['측정시간'].dt.date

    # 4. 핵심 변수 결측치 선형 보간 정제
    core_columns = [
        '온도_외부', '일사량_외부', '강우감지', '풍속_외부', 
        '누적일사량_외부', '잔존이산화탄소(CO2)', '온도_내부', '상대습도_내부'
    ]
    for col in core_columns:
        strawberry_df[col] = strawberry_df[col].interpolate(method='linear').ffill().bfill()
    print("-> [성공] 8대 핵심 변수 선형 보간 정제 완료!")

    # 5. PostgreSQL RDB 데이터 동적 융합 및 피처 엔지니어링
    try:
        db_host = "host.docker.internal" if AIRFLOW_AVAILABLE else "localhost"
        conn = psycopg2.connect(
            dbname="agriculture", user="tae_tae", password="smartfarm",
            host=db_host, port="5432"
        )
        
        sun_df = pd.read_sql("SELECT target_date, sunrise_time, sunset_time FROM sun_time_context", conn)
        sun_df['target_date'] = pd.to_datetime(sun_df['target_date']).dt.date
        crop_df = pd.read_sql("SELECT farm_id, plantation_date, current_growth_stage FROM farm_crop_context WHERE farm_id='FRM_ST_01'", conn)
        conn.close()
        print("-> [성공] PostgreSQL 백엔드 마스터 데이터셋 Read 성공!")

        strawberry_df = pd.merge(strawberry_df, sun_df, on='target_date', how='left')
        
        
        db_plantation_date = pd.to_datetime(crop_df['plantation_date'].values[0])
        
        
        strawberry_df['virtual_plantation'] = pd.to_datetime(f"{strawberry_df['측정시간'].dt.year.min()}-09-15")
        strawberry_df['DAT'] = (strawberry_df['측정시간'] - strawberry_df['virtual_plantation']).dt.days
        
        for col in ['current_growth_stage']:
            strawberry_df[col] = crop_df[col].values[0]
            
        print("-> [성공] RDB 컨텍스트 데이터 파이프라인 융합 완료!")
        
        current_time = strawberry_df['측정시간'].dt.time
        strawberry_df['Is_Daytime'] = (current_time >= strawberry_df['sunrise_time']) & (current_time <= strawberry_df['sunset_time'])
        strawberry_df['Is_Daytime'] = strawberry_df['Is_Daytime'].astype(int)
        
    except Exception as e:
        print(f"🚨 RDB 연동 실패 (기본값 대체): {e}")
        strawberry_df['virtual_plantation'] = pd.to_datetime("2022-09-15")
        strawberry_df['DAT'] = (strawberry_df['측정시간'] - strawberry_df['virtual_plantation']).dt.days
        strawberry_df['Is_Daytime'] = 1

    # 6. 물리 공식 및 시계열 멀티 시차 확장
    strawberry_df['내부_VPD'] = calculate_vpd(strawberry_df['온도_내부'], strawberry_df['상대습도_내부'])
    strawberry_df['상대습도_내부_Lag1'] = strawberry_df['상대습도_내부'].shift(1).bfill()
    strawberry_df['상대습도_내부_Lag2'] = strawberry_df['상대습도_내부'].shift(2).bfill()
    strawberry_df['상대습도_내부_Lag3'] = strawberry_df['상대습도_내부'].shift(3).bfill()

    features = [
        '온도_외부', '일사량_외부', '강우감지', '풍속_외부', 
        '누적일사량_외부', '잔존이산화탄소(CO2)', 
        '상대습도_내부_Lag1', '상대습도_내부_Lag2', '상대습도_내부_Lag3', 
        '내부_VPD', 'DAT', 'Is_Daytime' 
    ]

    # 7. 월별 블록 기반 시계열 분할
    start_filter_date = strawberry_df['virtual_plantation'].iloc[0]
    actual_crop_df = strawberry_df[strawberry_df['측정시간'] >= start_filter_date].copy()
    actual_crop_df['day'] = actual_crop_df['측정시간'].dt.day
    
    train_df = actual_crop_df[actual_crop_df['day'] <= 24]
    test_df = actual_crop_df[actual_crop_df['day'] > 24]
    print(f"-> [블록 시계열 분할 완료] 학습셋: {len(train_df):,}건 | 테스트셋: {len(test_df):,}건")

    X_train = train_df[features]
    y_temp_train = train_df['온도_내부']
    y_hum_train = train_df['상대습도_내부']

    X_test = test_df[features]
    y_temp_test = test_df['온도_내부']
    y_hum_test = test_df['상대습도_내부']

    # 8. 자원 최적화 RandomForest
    print("-> 🚀 과적합 방지 시계열 검증 레이어 기반 학습 시작...")
    model_temp = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=1).fit(X_train, y_temp_train)
    model_hum = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=1).fit(X_train, y_hum_train)

    # 9. 모델 검증 지표 계산
    pred_temp = model_temp.predict(X_test)
    pred_hum = model_hum.predict(X_test)
    pred_vpd = calculate_vpd(pred_temp, pred_hum)

    # 10. 스코어 연산 로그 출력
    mae_temp = mean_absolute_error(y_temp_test, pred_temp)
    r2_temp = r2_score(y_temp_test, pred_temp)
    mae_hum = mean_absolute_error(y_hum_test, pred_hum)
    r2_hum = r2_score(y_hum_test, pred_hum)
    print(f"🏆 [성공] 내부 온도 예측 R2: {r2_temp:.4f} | 내부 습도 예측 R2: {r2_hum:.4f}")

    # 11. 실시간 이력 적재
    try:
        db_host = "host.docker.internal" if AIRFLOW_AVAILABLE else "localhost"
        conn = psycopg2.connect(
            dbname="agriculture", user="tae_tae", password="smartfarm",
            host=db_host, port="5432"
        )
        cursor = conn.cursor()
        
        base_real_time = dt.now()
        meas_time = base_real_time.strftime("%Y-%m-%d %H:%M:%S")
        
        current_hour = base_real_time.hour
        target_idx = current_hour % len(pred_temp) 
        
        temp = float(pred_temp[target_idx])  
        hum = float(pred_hum[target_idx])    
        vpd = float(pred_vpd[target_idx])    
        
        
        
        # [수정] DAT 컬럼 및 %s 바인딩 제거
        query = """
            INSERT INTO farm_crop_prediction_results (측정시간, 온도_내부, 상대습도_내부, 내부_VPD)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (측정시간) 
            DO UPDATE SET 
                온도_내부 = EXCLUDED.온도_내부,
                상대습도_내부 = EXCLUDED.상대습도_내부,
                내부_VPD = EXCLUDED.내부_VPD;
        """
        # [수정] 튜플 파라미터에서 dat 변수 제거
        cursor.execute(query, (meas_time, temp, hum, vpd))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        
    except Exception as e:
        print(f"❌ DB 예측 이력 적재 중 에러 발생: {e}")

    return pred_temp, pred_hum, pred_vpd

# 신규 오퍼레이터 함수
def generate_farm_notification():
    try:
        db_host = "host.docker.internal" if AIRFLOW_AVAILABLE else "localhost"
        conn = psycopg2.connect(
            dbname="agriculture", user="tae_tae", password="smartfarm",
            host=db_host, port="5432"
        )
        cursor = conn.cursor()
        
        # 1. 최신 예측 기후 데이터 로드 (DAT 제외)
        select_pred_query = """
            SELECT 측정시간, 온도_내부, 상대습도_내부, 내부_VPD, Is_Daytime
            FROM farm_crop_prediction_results
            ORDER BY 측정시간 DESC
            LIMIT 1;
        """
        cursor.execute(select_pred_query)
        pred_data = cursor.fetchone()
        
        if not pred_data:
            print("⚠️ [경고] 제어 규칙을 적용할 최근 AI 예측 데이터가 존재하지 않습니다.")
            cursor.close()
            conn.close()
            return
            
        # [수정] 5개 인자만 언패킹
        pred_time, pred_temp, pred_hum, pred_vpd, is_daytime = pred_data
        print(f"-> [AI 최신 예측치 로드] 시간: {pred_time} | 온도: {pred_temp}°C | 습도: {pred_hum}%")

        # [수정] 마스터 테이블 정보로부터 진짜 DAT 직접 연산하여 단계 분기
        cursor.execute("SELECT plantation_date FROM farm_crop_context WHERE farm_id='FRM_ST_01';")
        meta_date = cursor.fetchone()
        if meta_date:
            real_plantation = pd.to_datetime(meta_date[0])
            dat = (pd.to_datetime(pred_time) - real_plantation).days
        else:
            dat = 260 # 매칭 실패시 기본 수확기 처리 가이드

        if dat <= 30: crop_phase = 'GROWTH'
        elif dat <= 60: crop_phase = 'FLOWER'
        else: crop_phase = 'HARVEST'

        # 2. 제어 규칙 대조 및 알림 로그 적재 (이후 로직 동일)
        select_rules_query = """
            SELECT crop_phase, is_daytime, min_temp, max_temp, min_humidity, max_humidity, min_vpd, max_vpd, status_code, guide_message 
            FROM farm_control_rules;
        """
        cursor.execute(select_rules_query)
        rules = cursor.fetchall()

        final_status = 'STABLE'
        final_message = '1시간 뒤 딸기 생육 환경이 안정적인 상태로 유지될 예정입니다. 현재 제어 상태를 보존하십시오.'
        db_daytime_bool = True if is_daytime == 1 else False

        for rule in rules:
            r_phase, r_daytime, r_min_t, r_max_t, r_min_h, r_max_h, r_min_v, r_max_v, r_status, r_msg = rule
            if r_phase != 'ALL' and r_phase != crop_phase: continue
            if r_daytime != db_daytime_bool: continue
                
            is_matched = False
            if r_min_h and float(pred_hum) >= float(r_min_h): is_matched = True  
            if r_max_v and float(pred_vpd) <= float(r_max_v): is_matched = True  
            if r_min_t and float(pred_temp) < float(r_min_t): is_matched = True  
            if r_max_t and float(pred_temp) > float(r_max_t): is_matched = True  

            if is_matched:
                final_status = r_status
                final_message = r_msg
                break

        insert_log_query = """
            INSERT INTO farm_notification_logs 
            (predicted_target_time, pred_temperature, pred_humidity, pred_vpd, control_status, guide_message, is_read)
            VALUES (%s, %s, %s, %s, %s, %s, FALSE);
        """
        cursor.execute(insert_log_query, (pred_time, float(pred_temp), float(pred_hum), float(pred_vpd), final_status, final_message))
        conn.commit()
        
        print(f"🏆 제어 판정 완료: [{final_status}] -> DB 로그 최종 적재 완료!")
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ [에러] 알림 파이프라인 연동 중 인프라 오작동 발생: {e}")
        raise e

if __name__ == "__main__":
    train_and_evaluate_task()

# Airflow DAG 정의
if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id='smart_farm_llm_pipeline_v6',
        default_args=default_args,
        schedule='0 7 * * *',
        catchup=False
    ) as dag:
        
        task_ml = PythonOperator(
            task_id='ml_prediction_and_rdb_load',
            python_callable=train_and_evaluate_task
        )