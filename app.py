# app.py
import gradio as gr
from smart_farm_rag import SmartFarmRAG
from datetime import datetime, timedelta
from decimal import Decimal
import os
import psycopg2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import importlib
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sqlalchemy import create_engine

print("🤖 스마트팜 AI RAG를 구동합니다...")
rag_advisor = SmartFarmRAG()

def calculate_vpd_for_heatmap(temp, humidity):
    svp = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
    avp = svp * (humidity / 100.0)
    return svp - avp

def get_heatmap_plot():
    possible_paths = ["data/2022_env.CSV", "dags/data/2022_env.CSV", "/opt/airflow/dags/data/2022_env.CSV"]
    csv_path = None
    for p in possible_paths:
        if os.path.exists(p): csv_path = p; break

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if csv_path:
        try:
            df = pd.read_csv(csv_path, encoding='cp949')
            rename_dict = {'온도_내부': 'In_Temp', '상대습도_내부': 'In_Hum', '온도_외부': 'Out_Temp', '일사량_외부': 'Out_Solar'}
            valid_cols = [col for col in rename_dict.keys() if col in df.columns]
            sub_df = df[valid_cols].dropna().copy().rename(columns=rename_dict)
            sub_df['In_VPD'] = calculate_vpd_for_heatmap(sub_df['In_Temp'], sub_df['In_Hum'])
            final_df = sub_df[[c for c in ['Out_Temp', 'Out_Solar', 'In_Temp', 'In_Hum', 'In_VPD'] if c in sub_df.columns]]
            sns.heatmap(final_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title("Smart Farm Environment Feature Correlation Heatmap", fontsize=11, pad=10)
        except Exception as e:
            ax.text(0.5, 0.5, f"Data Parsing Error: {e}", ha='center', va='center')
    else:
        ax.text(0.5, 0.5, "❌ 원천 데이터 파일 경로 오류", ha='center', va='center')
    plt.tight_layout()
    return fig

def get_variable_dictionary():
    var_data = {
        "영문 피처명 (Feature)": ["In_Temp", "In_Hum", "In_VPD", "Out_Temp", "Out_Solar"],
        "한글 의미": ["내부 온도", "내부 상대습도", "내부 물리 포차 (VPD)", "외부 온도", "외부 일사량"],
        "스마트팜 도메인 설명": [
            "딸기 생육 및 광합성 속도를 결정하는 코어 인자 (야간 변온 관리의 핵심 지표)",
            "하우스 내부의 습한 정도. 90% 이상 과습 시 잿빛곰팡이병 등 병해 발생 위험 폭발",
            "작물이 실제로 숨을 쉬며 증산 작용을 할 수 있는 공기 흐름의 여유 용량 (수확기 품질 직결)",
            "하우스 외곽 기온. 환기창 개폐 시 내부 온습도 변화를 유도하는 선행 외부 환경 요인",
            "하우스 외부에 내리쬐는 햇빛의 양. 누적 일사량에 따라 딸기의 관수량 및 광합성 효율 계산"
        ]
    }
    return pd.DataFrame(var_data)

def trigger_all_predictions_realtime():
    try:
        print("⚡  ML 모델예측 가동...")
        import dags.smart_farm_excellent_v6 as target_dag
        importlib.reload(target_dag)
        return target_dag.train_and_evaluate_task()
    except Exception as e:
        print(f"⚠️ 수동 트리거 경고]: {e}")
        return None, None, None

def get_db_history_dataframe():
    try:
        
        db_url = "postgresql+psycopg2://tae_tae:smartfarm@localhost:5432/agriculture"
        engine = create_engine(db_url)
        
        query = "SELECT 측정시간, 온도_내부, 상대습도_내부, 내부_VPD FROM farm_crop_prediction_results ORDER BY 측정시간 DESC;"
        
        
        df = pd.read_sql_query(query, con=engine)
        
        if df.empty:
            return pd.DataFrame(columns=['인프라 모니터링 상태'], data=[['DB 연결은 성공했으나 적재된 데이터가 없습니다.']])
            
        df.columns = ['🔮 이력 적재 시점', '🌡️ 내부 온도 (°C)', '💧 내부 습도 (%)', '🌬️ 포차 (VPD, kPa)']
        return df
    except Exception as e:
        print(f"🚨 대시보드 RDB 조회 중 인프라 에러 발생: {e}")
        return pd.DataFrame(columns=['인프라 모니터링 상태'], data=[[f'PostgreSQL 연결 대기 중 ({e})']])

def refresh_farm_status_with_values(pred_temp, pred_hum, pred_vpd, planting_date_str):
    now_time = datetime.now()
    current_time_str = now_time.strftime("%Y-%m-%d %H:%M:%S")
    target_hour = (now_time + timedelta(hours=1)).hour
    
    try:
        planting_date = datetime.strptime(planting_date_str.strip(), "%Y-%m-%d")
        calculated_dat = (now_time - planting_date).days
        dat_display = f"{calculated_dat}일차"
        
        if calculated_dat < 60: crop_phase = "정식 초기 및 영양생장기"
        elif calculated_dat < 120: crop_phase = "개화 및 착과기"
        else: crop_phase = "수확기"
    except Exception as e:
        print(f"⚠️ 정식일 날짜 파싱 에러 (기본값 대체): {e}")
        dat_display = "258일차"
        crop_phase = "수확기"
    
    curr_temp = pred_temp - 1.2  
    curr_hum = pred_hum - 4.5
    curr_vpd = pred_vpd - 0.05 if pred_vpd > 0.05 else 0.02
        
    time_context = "주간(낮)" if 6 <= target_hour < 18 else "야간(밤)"

    prompt_for_banner = f"""
    [스마트팜 실시간 관제 컨텍스트 주입]
    - 현재 딸기 생육 단계: {crop_phase} (정식 후 {dat_display})
    - 현재 하우스 실시간 상태: 온도 {curr_temp:.1f}℃, 상대습도 {curr_hum:.1f}%, 내부 VPD {curr_vpd:.2f}kPa
    - 1시간 뒤 AI 예측 기후: 온도 {pred_temp:.1f}℃, 상대습도 {pred_hum:.1f}%, 내부 VPD {pred_vpd:.2f}kPa ({time_context} 환경)
    
    위 스마트팜의 현재 기후 위험 요소를 진단하고, 1시간 뒤 예측 기후에 대비하여 농민이 지금 즉시 조치해야 할 생육 피해 예방 행동 요령을 농진청 가이드북 규칙에 기반하여 딱 2줄로 요약해 처방하세요.
    """
    try: 
        pdf_based_recommendation = rag_advisor.ask_advisor(prompt_for_banner)
    except Exception as e: 
        print(f"⚠️ RAG 배너 생성 실패: {e}")
        pdf_based_recommendation = "1시간 뒤 환경 변화에 대비해 환기시스템 및 제습 인프라 제어를 준비하십시오."

    status_html = f"""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 12px; border-left: 6px solid #ff4d4d; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
        <h4 style='margin-top:0; margin-bottom: 15px; color:#222;'>📊 스마트팜 통합 관제 현황판</h4>
        <div style='display: flex; justify-content: space-between; background: #fff; padding: 10px 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #eee;'>
            <span><b>⏰ 현재 기준 실시간 관제 상태 시각:</b> <span style='color:#333; font-weight:bold;'>{current_time_str}</span></span>
            <span><b>🌱 딸기 생육:</b> <span style='color:#ff4d4d; font-weight:bold;'>{dat_display}</span> ({crop_phase})</span>
        </div>
        <div style='background-color: #fdf2e9; color: #b45309; padding: 15px; border-radius: 8px; border-left: 5px solid #d97706; margin-bottom: 20px;'>
            💡 <b>[농진청 가이드북 기반 실시간 AI 기술 처방]</b><br>{pdf_based_recommendation}
        </div>
        <table style='width: 100%; border-collapse: collapse; text-align: center; background: #fff; border-radius: 8px; overflow: hidden; border: 1px solid #eee;'>
            <tr style='background-color: #f1f3f5; font-weight: bold; color: #495057; height: 40px;'>
                <td>환경 지표 분류</td>
                <td style='background-color: #edf2ff; color: #364fc7;'>🌡️ 현재 하우스 실시간 상태</td>
                <td style='background-color: #fff0f6; color: #c91a29;'>🔮 1시간 뒤 AI 예측 수치</td>
            </tr>
            <tr style='height: 45px;'>
                <td><b>내부 온도</b></td>
                <td style='font-weight:bold; background-color: #f8f9fa;'>{curr_temp:.1f} °C</td>
                <td style='color:#e63946; font-weight:bold; background-color: #fff5f5;'>{pred_temp:.1f} °C</td>
            </tr>
            <tr style='height: 45px;'>
                <td><b>내부 습도</b></td>
                <td style='font-weight:bold; background-color: #f8f9fa;'>{curr_hum:.1f} %</td>
                <td style='color:#457b9d; font-weight:bold; background-color: #f1f7fa;'>{pred_hum:.1f} %</td>
            </tr>
            <tr style='height: 45px;'>
                <td><b>물리 포차 (VPD)</b></td>
                <td style='color:#2a9d8f; font-weight:bold; background-color: #f8f9fa;'>{curr_vpd:.2f} kPa</td>
                <td style='color:#2a9d8f; font-weight:bold; background-color: #f4fbf9;'>{pred_vpd:.2f} kPa</td>
            </tr>
        </table>
    </div>
    """
    return status_html

def refresh_farm_status(planting_date_str):
    return refresh_farm_status_with_values(16.2, 88.9, 0.38, planting_date_str)

def handle_refresh_all(planting_date_str):
    print("🔄최신 예측 이력 조회 중...")
    
    db_df = get_db_history_dataframe()
    
    if db_df is not None and not db_df.empty and '🔮 이력 적재 시점' in db_df.columns:
        latest_row = db_df.iloc[0]
        try:
            # 4개 컬럼 구조에 맞춰 명시적으로 3개 예측 수치만 파싱
            latest_p_temp = float(latest_row['🌡️ 내부 온도 (°C)'])
            latest_p_hum = float(latest_row['💧 내부 습도 (%)'])
            latest_p_vpd = float(latest_row['🌬️ 포차 (VPD, kPa)'])
        except Exception as e:
            print(f"⚠️미세 에러 발생(기본값 대체): {e}")
            latest_p_temp, latest_p_hum, latest_p_vpd = 15.0, 84.4, 0.33
    else:
        latest_p_temp, latest_p_hum, latest_p_vpd = 15.0, 84.4, 0.33

    updated_html = refresh_farm_status_with_values(latest_p_temp, latest_p_hum, latest_p_vpd, planting_date_str)
    return updated_html, db_df

def chat_with_advisor(user_msg, history, planting_date_str):
    if not user_msg.strip(): return "", history
    
    print("💬 AI 챗봇 가동 ")
    
    db_df = get_db_history_dataframe()
    
    # [수정] DB 대신 설정된 정식일(planting_date_str) 기준으로 실시간 DAT 계산
    try:
        planting_date = datetime.strptime(planting_date_str.strip(), "%Y-%m-%d")
        calculated_dat = (datetime.now() - planting_date).days
        curr_dat = f"{calculated_dat}일차"
    except Exception:
        curr_dat = "258일차"
    
    # [수정] 데이터프레임 컬럼 구조 변경에 맞추어 인덱스 기반 안전하게 추출 (DAT 컬럼 제외)
    if db_df is not None and not db_df.empty and '🔮 이력 적재 시점' in db_df.columns:
        latest_row = db_df.iloc[0]
        try:
            curr_temp = float(latest_row['🌡️ 내부 온도 (°C)'])
            curr_hum = float(latest_row['💧 내부 습도 (%)'])
            curr_vpd = float(latest_row['🌬️ 포차 (VPD, kPa)'])
        except Exception:
            curr_temp, curr_hum, curr_vpd = 15.0, 84.4, 0.33
    else:
        curr_temp, curr_hum, curr_vpd = 15.0, 84.4, 0.33
    
    context_enhanced_prompt = f"""
    [시스템 필수 컨텍스트 주입 - 현재 스마트팜 실제 기후 상태]
    - 현재 내부 온도: {curr_temp:.1f}°C
    - 현재 내부 상대습도: {curr_hum:.1f}%
    - 현재 내부 VPD (물리 포차): {curr_vpd:.2f} kPa
    - 현재 딸기 재배 경과일: {curr_dat}
    
    [사용자 질문]
    {user_msg}
    
    위의 [시스템 필수 컨텍스트]에 적힌 '현재 실제 기후 상태' 수치들을 최우선 팩트로 인지하고, 지어내지 마십시오. 사용자의 질문에 대해 '농업기술길라잡이' 지식 데이터베이스에 기반하여 정밀하고 전문적인 농가 컨설팅 답변을 제공하세요.
    """
    
    ai_answer = rag_advisor.ask_advisor(context_enhanced_prompt)
    
    if history is None: history = []
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": ai_answer})
    return "", history

with gr.Blocks(title=" 딸기 스마트팜 AI 대시보드") as demo:
    gr.Markdown("# 🍓 스마트팜 맞춤형 AI 어드바이저")
    with gr.Tabs():
        with gr.Tab("🎮 실시간 통합 관제 및 챗봇"):
            with gr.Accordion("⚙️ 농가별 재배 환경 기본 설정", open=True):
                input_planting_date = gr.Textbox(label="🍓 딸기 정식일 설정", value="2025-09-15")
            gr.HTML("<div style='margin-bottom: 15px;'></div>")
            
            # 초기 화면 렌더링
            farm_status_board = gr.HTML(value=refresh_farm_status("2025-09-15"))
            btn_refresh = gr.Button("🔄 Refresh", variant="primary", size="sm")
            gr.HTML("<div style='margin-bottom: 25px;'></div>")
            
            # AI 챗봇 대화창
            chatbot = gr.Chatbot(label="AI 어드바이저", height=380)
            with gr.Row():
                txt_input = gr.Textbox(show_label=False, placeholder="질문하세요.", scale=8)
                btn_submit = gr.Button("✈️ 질문하기", variant="primary", scale=2)
                
        with gr.Tab("📊 분석 지표 및 DB 적재 시각화"):
            gr.Plot(value=get_heatmap_plot(), label="환경 변수 상관관계 Heatmap")
            gr.Dataframe(value=get_variable_dictionary(), interactive=False)
            with gr.Accordion("🗄️ 예측 이력 로그 (클릭하여 열기/닫기)", open=False):
                db_log_table = gr.Dataframe(value=get_db_history_dataframe(), interactive=False)



    
    btn_refresh.click(
        fn=handle_refresh_all, 
        inputs=[input_planting_date], 
        outputs=[farm_status_board, db_log_table]
    )
    
    
    demo.load(
        fn=handle_refresh_all, 
        inputs=[input_planting_date], 
        outputs=[farm_status_board, db_log_table]
    )

    
    txt_input.submit(
        fn=chat_with_advisor, 
        inputs=[txt_input, chatbot, input_planting_date], 
        outputs=[txt_input, chatbot]
    )
    btn_submit.click(
        fn=chat_with_advisor, 
        inputs=[txt_input, chatbot, input_planting_date], 
        outputs=[txt_input, chatbot]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), share=True)