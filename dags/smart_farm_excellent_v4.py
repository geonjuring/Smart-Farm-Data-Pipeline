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

        return sorted([v for v in forecast_dict.values() if len(v) >= 4], key=lambda x: x['time'])[:24]
    except Exception as e:
        print(f"❌ 기상청 API 수집 실패: {e}")
        return []

def run_strawberry_forecast():
    """다중 회귀 모델 학습, 검증 및 예측 데이터 DB 적재"""
    # 1. 데이터 로드 및 전처리 (다중 변수 포함)
    df = pd.read_csv(CSV_PATH, encoding='cp949', 
                     usecols=['품목', '온도_외부', '풍속_외부', '강우감지', '온도_내부', '상대습도_내부'])
    strawberry_df = df[df['품목'] == '딸기'].dropna()
    
    # 2. 다중 변수 설정 및 데이터 분할
    features = ['온도_외부', '풍속_외부', '강우감지']
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    for f in weather_forecasts:
        # 'Feature Names' 경고 방지를 위해 DataFrame 형태로 입력
        input_df = pd.DataFrame([[f['temp'], f['wind'], f['rain']]], columns=features)
        
        pred_t = model_temp.predict(input_df)[0]
        pred_h = model_hum.predict(input_df)[0]
        
        pg_hook.run("""
            INSERT INTO strawberry_ai_forecast (forecast_time, pred_in_temp, pred_in_hum, kma_out_temp)
            VALUES (%s, %s, %s, %s)
        """, parameters=(f['time'], float(pred_t), float(pred_h), f['temp']))
    
    print("✅ 내일의 딸기 농가 다중 변수 예측 및 DB 적재 성공!")

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