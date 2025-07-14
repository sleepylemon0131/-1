import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress

# --- 1. 데이터 생성 (가상 데이터) ---
# 실제 데이터 사용 시 이 부분은 공공 데이터 로드 코드로 대체됩니다.
@st.cache_data
def generate_data():
    np.random.seed(42) # 재현성을 위한 시드 설정
    regions = ['서울', '경기', '부산', '대구', '인천', '광주', '대전', '울산', '세종', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
    num_data_points = len(regions) * 50 # 각 지역당 50개의 데이터 포인트

    data = {
        '지역': np.random.choice(regions, num_data_points),
        '소득(만원/월)': np.random.normal(loc=300, scale=80, size=num_data_points).astype(int),
        '자산(만원)': np.random.normal(loc=50000, scale=20000, size=num_data_points).astype(int),
        '학업성취도(점)': np.random.normal(loc=70, scale=15, size=num_data_points).astype(int)
    }
    df = pd.DataFrame(data)

    # 지역별 소득 및 학업성취도에 편차 부여 (격차를 더 잘 보이게 하기 위함)
    df.loc[df['지역'].isin(['서울', '경기']), '소득(만원/월)'] = np.random.normal(loc=400, scale=70, size=len(df[df['지역'].isin(['서울', '경기'])])).astype(int)
    df.loc[df['지역'].isin(['전남', '경북', '강원']), '소득(만원/월)'] = np.random.normal(loc=200, scale=50, size=len(df[df['지역'].isin(['전남', '경북', '강원'])])).astype(int)
    df.loc[df['지역'].isin(['서울', '경기']), '학업성취도(점)'] = np.random.normal(loc=85, scale=10, size=len(df[df['지역'].isin(['서울', '경기'])])).astype(int)
    df.loc[df['지역'].isin(['전남', '경북', '강원']), '학업성취도(점)'] = np.random.normal(loc=60, scale=10, size=len(df[df['지역'].isin(['전남', '경북', '강원'])])).astype(int)

    # 소득 및 자산이 음수가 되지 않도록 조정
    df['소득(만원/월)'] = df['소득(만원/월)'].apply(lambda x: max(100, x))
    df['자산(만원)'] = df['자산(만원)'].apply(lambda x: max(1000, x))
    df['학업성취도(점)'] = df['학업성취도(점)'].apply(lambda x: min(100, max(0, x))) # 0-100점 사이로 조정

    return df

df = generate_data()

# --- 2. 지니 계수 및 소득 5분위 배율 계산 함수 ---
def calculate_gini(income_series):
    income_series = income_series.sort_values().reset_index(drop=True)
    n = len(income_series)
    if n == 0:
        return 0
    numerator = 2 * sum((i + 1) * income_series.iloc[i] for i in range(n))
    denominator = n * sum(income_series)
    if denominator == 0:
        return 0
    return (numerator / denominator) - ((n + 1) / n)

def calculate_quintile_ratio(income_series):
    income_series = income_series.sort_values().reset_index(drop=True)
    n = len(income_series)
    if n == 0:
        return 0
    # 하위 20%
    lower_20_idx = int(n * 0.2)
    sum_lower_20 = income_series.iloc[:lower_20_idx].sum()
    # 상위 20%
    upper_20_idx = int(n * 0.8)
    sum_upper_20 = income_series.iloc[upper_20_idx:].sum()

    if sum_lower_20 == 0:
        return float('inf') # 분모가 0이면 무한대 (불평등 매우 심함)
    return sum_upper_20 / sum_lower_20

# --- Streamlit 앱 구성 ---
st.set_page_config(layout="wide")
st.title("지속 가능한 발전을 위한 사회 불평등 분석 대시보드")
st.markdown("""
이 대시보드는 **교육 격차**, **소득 불평등 지표**, 그리고 **소득/자산 데이터**를 분석하여 사회적 불평등 현상을 시각적으로 보여줍니다.
지속 가능한 발전 목표(SDGs) 중 **양질의 교육(SDG 4)** 및 **불평등 감소(SDG 10)**와 관련된 인사이트를 제공하고자 합니다.
""")

st.sidebar.header("분석 옵션")
selected_region = st.sidebar.selectbox('지역 선택 (전체 또는 특정 지역)', ['전체'] + list(df['지역'].unique()))

if selected_region != '전체':
    filtered_df = df[df['지역'] == selected_region]
    st.sidebar.info(f"**{selected_region}** 데이터가 선택되었습니다.")
else:
    filtered_df = df
    st.sidebar.info("**전체 지역** 데이터가 선택되었습니다.")

st.sidebar.markdown("---")
st.sidebar.subheader("데이터 필터링")
min_income, max_income = st.sidebar.slider(
    "소득 범위 (만원/월)",
    int(df['소득(만원/월)'].min()), int(df['소득(만원/월)'].max()),
    (int(df['소득(만원/월)'].min()), int(df['소득(만원/월)'].max()))
)
filtered_df = filtered_df[(filtered_df['소득(만원/월)'] >= min_income) & (filtered_df['소득(만원/월)'] <= max_income)]


# --- 3. 교육 격차 시각화 ---
st.header("1. 지역별 교육 격차 분석")
st.markdown("각 지역의 평균 학업 성취도를 통해 교육 불균형을 시각화합니다.")
region_avg_score = filtered_df.groupby('지역')['학업성취도(점)'].mean().reset_index()
region_avg_score = region_avg_score.sort_values(by='학업성취도(점)', ascending=False)

fig1 = px.bar(region_avg_score, x='지역', y='학업성취도(점)',
              title='지역별 평균 학업 성취도',
              labels={'학업성취도(점)': '평균 학업 성취도 (점)', '지역': '지역'},
              color='학업성취도(점)',
              color_continuous_scale=px.colors.sequential.Viridis)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# --- 4. 소득 불평등 지표 시각화 ---
st.header("2. 소득 불평등 지표")
st.markdown("지니 계수와 소득 5분위 배율을 통해 소득 분배의 불평등 정도를 파악합니다.")

if not filtered_df.empty:
    gini_val = calculate_gini(filtered_df['소득(만원/월)'])
    quintile_ratio_val = calculate_quintile_ratio(filtered_df['소득(만원/월)'])

    col_gini, col_quintile = st.columns(2)
    with col_gini:
        st.metric(label="**지니 계수 (Gini Coefficient)**", value=f"{gini_val:.3f}")
        st.info("지니 계수는 0(완전 평등)부터 1(완전 불평등) 사이의 값입니다. 값이 높을수록 소득 불평등이 심합니다.")
    with col_quintile:
        st.metric(label="**소득 5분위 배율**", value=f"{quintile_ratio_val:.2f}배")
        st.info("소득 5분위 배율은 상위 20% 소득 합을 하위 20% 소득 합으로 나눈 값입니다. 값이 높을수록 소득 불균형이 심합니다.")

    st.subheader("소득 분포 히스토그램")
    fig2 = px.histogram(filtered_df, x='소득(만원/월)', nbins=50,
                        title='소득 분포 히스토그램',
                        labels={'소득(만원/월)': '월 소득 (만원)', 'count': '인구수'},
                        color_discrete_sequence=['#1f77b4'])
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.warning("선택된 필터에 해당하는 데이터가 없습니다. 필터를 조정해주세요.")

st.markdown("---")

# --- 5. 소득, 자산 데이터 분석 및 관계 시각화 ---
st.header("3. 소득 및 자산 데이터 분석")
st.markdown("소득과 학업 성취도, 자산과 학업 성취도 간의 관계를 탐색합니다.")

col_scatter1, col_scatter2 = st.columns(2)

with col_scatter1:
    st.subheader("소득 vs. 학업 성취도")
    fig3 = px.scatter(filtered_df, x='소득(만원/월)', y='학업성취도(점)',
                      hover_data=['지역'],
                      title='소득과 학업 성취도 관계',
                      labels={'소득(만원/월)': '월 소득 (만원)', '학업성취도(점)': '학업 성취도 (점)'},
                      color='지역' if selected_region == '전체' else None)

    # 추세선 추가
    if not filtered_df.empty:
        slope, intercept, r_value, p_value, std_err = linregress(filtered_df['소득(만원/월)'], filtered_df['학업성취도(점)'])
        x_range = np.array([filtered_df['소득(만원/월)'].min(), filtered_df['소득(만원/월)'].max()])
        fig3.add_trace(go.Scatter(x=x_range, y=slope * x_range + intercept,
                                  mode='lines', name=f'추세선 (R²={r_value**2:.2f})',
                                  line=dict(color='red', dash='dash')))

    st.plotly_chart(fig3, use_container_width=True)

with col_scatter2:
    st.subheader("자산 vs. 학업 성취도")
    fig4 = px.scatter(filtered_df, x='자산(만원)', y='학업성취도(점)',
                      hover_data=['지역'],
                      title='자산과 학업 성취도 관계',
                      labels={'자산(만원)': '자산 (만원)', '학업성취도(점)': '학업 성취도 (점)'},
                      color='지역' if selected_region == '전체' else None)
    # 추세선 추가
    if not filtered_df.empty:
        slope_asset, intercept_asset, r_value_asset, p_value_asset, std_err_asset = linregress(filtered_df['자산(만원)'], filtered_df['학업성취도(점)'])
        x_range_asset = np.array([filtered_df['자산(만원)'].min(), filtered_df['자산(만원)'].max()])
        fig4.add_trace(go.Scatter(x=x_range_asset, y=slope_asset * x_range_asset + intercept_asset,
                                  mode='lines', name=f'추세선 (R²={r_value_asset**2:.2f})',
                                  line=dict(color='red', dash='dash')))

    st.plotly_chart(fig4, use_container_width=True)

st.subheader("데이터 개요")
st.dataframe(filtered_df.describe())

st.markdown("---")
st.markdown("""
### **결론 및 시사점**
이 대시보드는 가상의 데이터를 통해 **지역 간 교육 및 소득 불균형**이 존재함을 보여줍니다.
* **교육 격차:** 특정 지역의 학업 성취도가 높은 경향을 보이며, 이는 지역별 교육 자원의 불균형이나 사회 경제적 배경의 차이와 관련될 수 있습니다.
* **소득 불평등:** 계산된 지니 계수와 소득 5분위 배율은 소득 분배의 불평등 정도를 나타냅니다. 이 지표들이 높다면 소득 양극화 문제가 심각함을 시사합니다.
* **소득/자산과 교육의 관계:** 소득이나 자산이 높을수록 학업 성취도 또한 높아지는 경향이 있다면, 이는 사회 경제적 배경이 교육 기회 및 성과에 영향을 미칠 수 있음을 의미합니다.

**지속 가능한 발전을 위해서는** 이러한 불평등을 해소하기 위한 정책적 노력이 중요합니다. 예를 들어, 교육 자원 재분배, 취약 계층을 위한 교육 지원 확대, 소득 재분배 정책 강화 등을 고려할 수 있습니다.
""")

st.info("이 분석은 가상 데이터에 기반하며, 실제 데이터를 통해 더 정확하고 의미 있는 인사이트를 얻을 수 있습니다.")
