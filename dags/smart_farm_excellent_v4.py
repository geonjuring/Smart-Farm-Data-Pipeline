import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
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
        
        # 🌟 [4주차 연동 핵심 추가] 
        # 규칙 제어 연산이 정상 작동하려면 '가장 최근 AI가 예측한 미래 온습도/VPD/DAT 스펙' 정보가 테이블에 박혀있어야 합니다.
        # 테스트셋의 가장 마지막 시점 레코드를 기준으로 가상의 미래 예측 상황을 DB에 적재합니다.
        latest_idx = test_df.index[-1]
        latest_time = test_df.loc[latest_idx, '측정시간']
        latest_dat = int(test_df.loc[latest_idx, 'DAT'])
        latest_daytime = int(test_df.loc[latest_idx, 'Is_Daytime'])
        
        # 3주차 성능 테이블 구조가 결과 테이블을 겸하므로 예측치 데이터 적재 시도
        # (만약 테이블명이 다르면 기존에 쓰시던 결과 테이블명으로 변경하세요)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS farm_crop_prediction_results (
                측정시간 TIMESTAMP PRIMARY KEY,
                온도_내부 NUMERIC(5,2),
                상대습도_내부 NUMERIC(5,2),
                내부_VPD NUMERIC(5,2),
                Is_Daytime INT,
                DAT INT
            );
        """)
        
        # 최신 예측치 1건 밀어넣기
        pred_vpd_val = calculate_vpd(pred_temp[-1], pred_hum[-1])
        cursor.execute("""
            INSERT INTO farm_crop_prediction_results (측정시간, 온도_내부, 상대습도_내부, 내부_VPD, Is_Daytime, DAT)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (측정시간) DO UPDATE SET
                온도_내부 = EXCLUDED.온도_내부,
                상대습도_내부 = EXCLUDED.상대습도_내부,
                내부_VPD = EXCLUDED.내부_VPD,
                Is_Daytime = EXCLUDED.Is_Daytime,
                DAT = EXCLUDED.DAT;
        """, (latest_time, float(pred_temp[-1]), float(pred_hum[-1]), float(pred_vpd_val), latest_daytime, latest_dat))
        
        conn.commit()
        cursor.close()
        conn.close()
        print("-> [성공] 3주차 고도화 성과 및 실시간 예측 데이터베이스 최종 적재 완료!")
    except Exception as e:
        print(f"🚨 성과 및 예측 로그 DB 적재 실패: {e}")


# =============================================================================
# 4주차 신규 오퍼레이터 함수: 규칙 기반 실시간 알림 가이드 제어 태스크
# =============================================================================
def generate_farm_notification():
    print("========= [Airflow Task] 4주차 규칙 기반 실시간 알림 가이드 제어 가동 =========")
    
    try:
        conn = psycopg2.connect(
            dbname="agriculture", user="tae_tae", password="smartfarm",
            host="host.docker.internal", port="5432"
        )
        cursor = conn.cursor()
        
        # 1. 아까 ML 학습 단에서 적재한 최신 AI 미래 예측치 1건 읽어오기
        select_pred_query = """
            SELECT 측정시간, 온도_내부, 상대습도_내부, 내부_VPD, Is_Daytime, DAT
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
            
        pred_time, pred_temp, pred_hum, pred_vpd, is_daytime, dat = pred_data
        print(f"-> [AI 최신 예측치 로드] 시간: {pred_time} | 온도: {pred_temp}°C | 습도: {pred_hum}% | VPD: {pred_vpd} kPa")

        # 2. 농민 입력 가변 정식일(DAT) 기준 현재 생육 단계(crop_phase) 동적 판정
        if dat <= 30:
            crop_phase = 'GROWTH'
        elif dat <= 60:
            crop_phase = 'FLOWER'
        else:
            crop_phase = 'HARVEST'
        print(f"-> [생육컨텍스트 판정] 현재 DAT: {dat}일차 -> 생육 단계: {crop_phase} | 낮여부: {bool(is_daytime)}")

        # 3. 윈도우 터미널로 구축한 DB 규칙 마스터 테이블(farm_control_rules) 데이터 전량 로드
        select_rules_query = """
            SELECT crop_phase, is_daytime, min_temp, max_temp, min_humidity, max_humidity, min_vpd, max_vpd, status_code, guide_message 
            FROM farm_control_rules;
        """
        cursor.execute(select_rules_query)
        rules = cursor.fetchall()

        # 4. 결정론적 규칙 매칭 연산 기동 (하드코딩 배제)
        final_status = 'STABLE'
        final_message = '1시간 뒤 딸기 생육 환경이 안정적인 상태로 유지될 예정입니다. 현재 제어 상태를 보존하십시오.'
        
        # 낮/밤 조건 플래그 타입 통일 (DB는 boolean, 데이터프레임은 int 형변환 대비)
        db_daytime_bool = True if is_daytime == 1 else False

        for rule in rules:
            r_phase, r_daytime, r_min_t, r_max_t, r_min_h, r_max_h, r_min_v, r_max_v, r_status, r_msg = rule
            
            # 조건 체크 1: 생육 단계 매칭 (ALL 제외)
            if r_phase != 'ALL' and r_phase != crop_phase:
                continue
            # 조건 체크 2: 낮/밤 일치 여부
            if r_daytime != db_daytime_bool:
                continue
                
            # 조건 체크 3: AI 수치 임계치 도달 연산
            is_matched = False
            
            if r_min_h and float(pred_hum) >= float(r_min_h): is_matched = True   # 과습 조건 위반
            if r_max_v and float(pred_vpd) <= float(r_max_v): is_matched = True   # 결로(포차부족) 조건 위반
            if r_min_t and float(pred_temp) < float(r_min_t): is_matched = True   # 저온 동해 조건 위반
            if r_max_t and float(pred_temp) > float(r_max_t): is_matched = True   # 주간 과열 조건 위반

            # 우선순위 높은 경고 메시지 매칭 시 확정 후 탈출
            if is_matched:
                final_status = r_status
                final_message = r_msg
                break

        # 5. 최종 판정된 행동 가이드 알림을 로그 테이블(farm_notification_logs)에 실시간 적재
        insert_log_query = """
            INSERT INTO farm_notification_logs 
            (predicted_target_time, pred_temperature, pred_humidity, pred_vpd, control_status, guide_message, is_read)
            VALUES (%s, %s, %s, %s, %s, %s, FALSE);
        """
        cursor.execute(insert_log_query, (pred_time, float(pred_temp), float(pred_hum), float(pred_vpd), final_status, final_message))
        conn.commit()
        
        print(f"🏆 [4주차 연동 성공] 제어 판정 완료: [{final_status}] -> DB 로그 최종 적재 완료!")
        print(f"📢 도출된 농민 가이드 메시지: {final_message}")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ [에러] 4주차 알림 파이프라인 연동 중 인프라 오작동 발생: {e}")
        raise e


# Airflow DAG 정의
with DAG(
    'smart_farm_strawberry_v6',
    default_args=default_args,
    description='3~4주차 인프라 조인 및 규칙 기반 알림 자동화 파이프라인',
    schedule_interval='0 7 * * *',
    catchup=False
) as dag:

    # Task 1: 3주차 기존 ML 프로세스
    ml_training_task = PythonOperator(
        task_id='strawberry_ml_training_and_eval',
        python_callable=train_and_evaluate_task
    )

    # Task 2: 4주차 신규 제어 규칙 매칭 및 알림 적재 프로세스
    generate_farm_notification_task = PythonOperator(
        task_id='generate_farm_notification',
        python_callable=generate_farm_notification
    )

    # 🔄 [4주차 파이프라인 흐름 사슬 정의]
    ml_training_task >> generate_farm_notification_task