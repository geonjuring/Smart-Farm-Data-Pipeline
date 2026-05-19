import os
import requests
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from dotenv import load_dotenv

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

def get_env_path():
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, '..', '.env')

# [설정] 건태님의 경로를 반영한 컨테이너 내부 경로
CSV_PATH = '/opt/airflow/dags/data/2022_env.csv'

def get_external_weather():
    """기상청 단기예보 API: TMP(기온), WSD(풍속), PCP(강수량) 수집"""
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    api_key = os.getenv("DATA_GO_KR_API_KEY")
    base_date = datetime.now().strftime('%Y%m%d')
    
    params = {
        'serviceKey': api_key, 'pageNo': '1', 'numOfRows': '1000', 'dataType': 'JSON',
        'base_date': base_date, 'base_time': '0500', 'nx': '95', 'ny': '77'
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        items = response.json().get('response', {}).get('body', {}).get('items', {}).get('item', [])
        
        forecast_dict = {}
        for item in items:
            time = item['fcstTime']
            if time not in forecast_dict:
                forecast_dict[time] = {'time': time}
            
            # 다중 변수 매핑
            if item['category'] == 'TMP': forecast_dict[time]['temp'] = float(item['fcstValue'])
            if item['category'] == 'WSD': forecast_dict[time]['wind'] = float(item['fcstValue'])
            if item['category'] == 'PCP': 
                val = item['fcstValue']
                # 강수량을 '강우감지' 바이너리 데이터(0 또는 1)로 변환
                forecast_dict[time]['rain'] = 1.0 if val not in ['강수없음', '0'] else 0.0
            # 💡 [추가] 하늘상태와 외부 상대습도 데이터도 같이 추출
            if item['category'] == 'SKY': forecast_dict[time]['sky'] = float(item['fcstValue'])
            if item['category'] == 'REH': forecast_dict[time]['out_hum'] = float(item['fcstValue'])

        # 💡 변수가 6개로 늘어났으므로 조건을 len(v) >= 6 으로 변경
        return sorted([v for v in forecast_dict.values() if len(v) >= 6], key=lambda x: x['time'])[:24]
    except Exception as e:
        print(f"❌ 기상청 API 수집 실패: {e}")
        return []

def run_strawberry_forecast():
    """다중 회귀 모델 학습, 검증 및 예측 데이터 DB 적재"""
    # 💡 '하늘상태'와 '상대습도_외부' 컬럼을 CSV에서 추가로 로드
    df = pd.read_csv(CSV_PATH, encoding='cp949', 
                 usecols=['품목', '온도_외부', '풍속_외부', '강우감지', '일사량_외부', '온도_내부', '상대습도_내부'])
    strawberry_df = df[df['품목'] == '딸기'].dropna()
    
    # 💡 머신러닝 모델이 학습할 독립 변수를 5개로 확장
    features = ['온도_외부', '풍속_외부', '강우감지', '일사량_외부']
    X = strawberry_df[features]
    y_temp = strawberry_df['온도_내부']
    y_hum = strawberry_df['상대습도_내부']
    
    X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
    X_train_h, X_test_h, y_hum_train, y_hum_test = train_test_split(X, y_hum, test_size=0.2, random_state=42)

    # 3. 모델 학습 및 MAE 검증
    model_temp = LinearRegression().fit(X_train, y_temp_train)
    model_hum = LinearRegression().fit(X_train_h, y_hum_train)
    
    temp_pred_test = model_temp.predict(X_test)
    mae_temp = mean_absolute_error(y_temp_test, temp_pred_test)
    r2_temp = r2_score(y_temp_test, temp_pred_test)

    print(f"📊 [모델 검증 결과] MAE(평균 절대 오차): {mae_temp:.4f}도")
    print(f"📊 [모델 검증 결과] R2 Score(결정계수): {r2_temp:.4f}")

    # 4. 실시간 예보 기반 미래 데이터 예측
    weather_forecasts = get_external_weather()
    if not weather_forecasts: return

    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # 테이블 생성
    pg_hook.run("""
        CREATE TABLE IF NOT EXISTS strawberry_ai_forecast (
            id SERIAL PRIMARY KEY,
            forecast_time TEXT,
            pred_in_temp FLOAT,
            pred_in_hum FLOAT,
            kma_out_temp FLOAT,
            /* 💡 신규 컬럼 2개 주입 */
            kma_sky_status FLOAT,
            kma_out_hum FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 💡 2022_env.csv 분석 기반 시간별 평균 일사량 베이스라인 프로파일
    hourly_solar_base = {
        0: 1.4, 1: 1.4, 2: 1.4, 3: 1.4, 4: 1.4, 5: 1.5,
        6: 5.8, 7: 34.0, 8: 110.6, 9: 227.1, 10: 335.3, 11: 422.2,
        12: 461.7, 13: 455.1, 14: 390.0, 15: 324.6, 16: 220.0, 17: 102.9,
        18: 29.1, 19: 4.2, 20: 1.3, 21: 1.3, 22: 1.3, 23: 1.0
    }

    for f in weather_forecasts:
        # 💡 기상청 하늘상태(SKY) 코드값과 예측 시간대를 조합하여 실제 일사량(W/m²) 추정치 계산
        hour_val = int(f['time']) // 100
        base_solar = hourly_solar_base.get(hour_val, 0.0)
        
        if f['sky'] == 1.0:      # 맑음
            est_solar = base_solar * 1.2
        elif f['sky'] == 3.0:    # 구름많음
            est_solar = base_solar * 0.7
        else:                    # 4: 흐림
            est_solar = base_solar * 0.4
            
        # 4개 확장 피처 순서대로 예측용 input 생성
        input_df = pd.DataFrame([[f['temp'], f['wind'], f['rain'], est_solar]], columns=features)
        
        pred_t = model_temp.predict(input_df)[0]
        pred_h = model_hum.predict(input_df)[0]
        
        # ====================================================================================
        # [PostgreSQL 데이터 적재 파라미터 상세 설명]
        # ====================================================================================
        # 1. forecast_time (예보 대상 시간 | ex: "0900", "1500")
        #    - 원천: 기상청 단기예보 API (fcstTime)
        #    - 역할: RAG 시스템에서 주간(환기 제어) 및 야간(보온 제어)의 시간적 맥락을 판단하는 기준
        #
        # 2. pred_in_temp (예측 내부 기온 | ex: 20.80)
        #    - 원천: 다중 회귀 모델(Linear Regression) 온도 예측치
        #    - 역할: RAG 핵심 지표. 가이드북의 적정 재배 온도와 비교하여 환기창 제어 및 보온 처방 결정
        #
        # 3. pred_in_hum (예측 내부 습도 | ex: 66.60)
        #    - 원천: 다중 회귀 모델(Linear Regression) 습도 예측치
        #    - 역할: 딸기 다습 환경 경고. 잿빛곰팡이병 등 습도 관련 병해충 선제적 방제 조언 트리거
        #
        # 4. kma_out_temp (기상청 예보 외부 기온 | ex: 15.4)
        #    - 원천: 기상청 단기예보 API (TMP)
        #    - 역할: 외부 날씨 모니터링 및 사용자 대화창(UI) 내 현재 외부 기상 상황 브리핑용 데이터
        #
        # 5. kma_solar_rad (추정 외부 일사량 | W/m², ex: 461.7)
        #    - 원천: 기상청 하늘상태(SKY) 데이터(기상청 단기예보 API) + 2022_env.csv 시간별 평균 일사량 통계 융합 로직
        #    - 역할: 하우스 내부 온도 폭등의 핵심 원인인 '햇빛(온실효과)' 맥락을 AI 어드바이저에게 제공
        # ====================================================================================
        pg_hook.run("""
            INSERT INTO strawberry_ai_forecast (forecast_time, pred_in_temp, pred_in_hum, kma_out_temp, kma_solar_rad)
            VALUES (%s, %s, %s, %s, %s)
        """, parameters=(f['time'], float(pred_t), float(pred_h), f['temp'], float(est_solar)))
    
    print("✅ 내일의 딸기 농가 고도화 예측 및 DB 적재 성공!")

# DAG 정의
with DAG(
    dag_id='strawberry_ai_forecast_final',
    start_date=datetime(2026, 5, 1),
    schedule_interval='0 7 * * *',
    catchup=False
) as dag:

    execute_forecast = PythonOperator(
        task_id='execute_forecast_logic',
        python_callable=run_strawberry_forecast
    )