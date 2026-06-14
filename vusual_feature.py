
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


data_path = "dags/data/2022_env.CSV"

if not os.path.exists(data_path):
    print(f"❌ 원천 데이터 파일이 해당 경로에 없습니다: {data_path}")
    print("💡 3주차 데이터셋 파일명을 확인하거나 경로를 맞춰주세요.")
else:
    # 한글 깨짐 방지 설정 (리눅스 환경 폰트 고려 대체 설정)
    plt.rcParams['axes.unicode_minus'] = False
    
    # 데이터 읽기 및 필요한 환경 변수 컬럼만 필터링
    df = pd.read_csv(data_path)
    
    # 분석에 사용할 핵심 수치형 변수들 선택 (프로젝트 스펙에 맞게 조정 가능)
    # 예시: 내부온도, 내부습도, VPD, 외부온도 등
    target_cols = [col for col in df.columns if '온도' in col or '습도' in col or 'VPD' in col or '포차' in col]
    
    if not target_cols:
        # 적절한 컬럼이 없을 경우 수치형 컬럼 상위 5개 추출
        target_cols = df.select_dtypes(include=['float64', 'int64']).columns[:5].tolist()

    sub_df = df[target_cols].dropna()

    # 2. 상관관계 행렬 계산
    corr_matrix = sub_df.corr()

    # 3. 히트맵 그리기
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True,          # 격자 안에 수치 표시
        cmap='coolwarm',     # 고온/저온, 다습/건조에 어울리는 컬러맵
        fmt=".2f",           # 소수점 둘째 자리까지
        linewidths=0.5,      # 칸 사이 간격
        cbar=True            # 우측 컬러바 표시
    )
    plt.title("Smart Farm Environment Feature Correlation Heatmap", fontsize=14, pad=20)
    plt.tight_layout()

    # 4. 이미지 파일로 저장 (PPT 및 GitHub 리드미 첨부용)
    output_path = "feature_importance_heatmap.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ 변수별 상관관계 히트맵 이미지 저장 완료! -> {output_path}")