import streamlit as st
import pandas as pd
import os
from datetime import time, datetime
import plotly.express as px
import json
import numpy as np
import re

# --------------------------------------------------------------------------
# Streamlit 페이지 기본 설정
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="LPI TEAM 피킹 작업 성과 분석 대시보드",
    page_icon="📊",
    layout="wide"
)

DAYS_ORDER = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일']

# --------------------------------------------------------------------------
# 설정 파일 처리 및 데이터 처리 함수
# --------------------------------------------------------------------------
CONFIG_FILE = "config.json"

def save_config(config_data):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config_data, f)

def load_config():
    default_config = {'minute_threshold': 30, 'picking_count_threshold': 0}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                default_config.update(config)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return default_config

def convert_time_to_seconds(t):
    if isinstance(t, (time, datetime)):
        return t.hour * 3600 + t.minute * 60 + t.second
    return np.nan

@st.cache_data
def load_and_process_data(uploaded_files):
    if not uploaded_files:
        return pd.DataFrame(), pd.DataFrame()

    valid_data_list, all_workers_list = [], []
    for uploaded_file in uploaded_files:
        try:
            date_str = os.path.basename(uploaded_file.name).replace('피킹바코드입력-', '').split('.')[0]
            pickup_date = pd.to_datetime(date_str, format='%Y%m%d')
            
            if pickup_date.weekday() == 6: continue

            df = pd.read_excel(uploaded_file, sheet_name='작업자현황', header=2, usecols=['작업자명', '피킹횟수', '1회평균분'],
                               dtype={'작업자명': str, '피킹횟수': str, '1회평균분': str})
            df.dropna(subset=['작업자명'], inplace=True)
            df = df[df['작업자명'].str.strip() != '']
            if df.empty: continue
            df['날짜'] = pickup_date
            all_workers_list.append(df[['날짜', '작업자명']])
            hangul_pattern = re.compile(r'[가-힣]+')
            df = df[df['작업자명'].apply(lambda x: hangul_pattern.search(str(x)) is not None)]
            df['피킹횟수'] = pd.to_numeric(df['피킹횟수'], errors='coerce')
            df.dropna(subset=['피킹횟수'], inplace=True)
            temp_time = pd.to_datetime(df['1회평균분'], errors='coerce').dt.time
            df['유효시간'] = temp_time
            df.dropna(subset=['유효시간'], inplace=True)
            if not df.empty: valid_data_list.append(df)
        except Exception:
            # [수정] 개별 파일 오류 메시지를 표시하지 않고 그냥 건너뜀
            continue
            
    if not valid_data_list: return pd.DataFrame(), pd.DataFrame()

    valid_master_df = pd.concat(valid_data_list, ignore_index=True)
    all_workers_df = pd.concat(all_workers_list, ignore_index=True).drop_duplicates()
    valid_master_df['피킹횟수'] = valid_master_df['피킹횟수'].astype(int)
    valid_master_df['소요시간(초)'] = valid_master_df['유효시간'].apply(convert_time_to_seconds)
    valid_master_df['평균소요시간(분)'] = valid_master_df['소요시간(초)'] / 60.0
    valid_master_df['연도'] = valid_master_df['날짜'].dt.year
    valid_master_df['월'] = valid_master_df['날짜'].dt.month
    valid_master_df['일'] = valid_master_df['날짜'].dt.day
    valid_master_df['연월'] = valid_master_df['날짜'].dt.to_period('M').astype(str)
    days_map = {i: day for i, day in enumerate(DAYS_ORDER)}
    valid_master_df['요일'] = valid_master_df['날짜'].dt.weekday.map(days_map)
    valid_master_df['요일'] = pd.Categorical(valid_master_df['요일'], categories=DAYS_ORDER, ordered=True)
    final_cols = ['날짜', '연도', '월', '일', '연월', '요일', '작업자명', '피킹횟수', '평균소요시간(분)']
    return valid_master_df[final_cols], all_workers_df

# --------------------------------------------------------------------------
# Streamlit 앱 UI
# --------------------------------------------------------------------------
st.title("LPI TEAM 피킹 작업 성과 분석 대시보드")

st.header("1. 분석 설정")

config = load_config()

uploaded_files = st.file_uploader(
    "분석할 엑셀 파일을 업로드하세요.",
    type=['xlsx', 'xlsm'],
    accept_multiple_files=True
)

col1, col2 = st.columns(2)
with col1:
    minute_threshold = st.number_input('평균 소요시간 제외 기준 (분):', min_value=0, value=config['minute_threshold'])
with col2:
    picking_count_threshold = st.number_input('일일 피킹횟수 제외 기준 (건):', min_value=0, value=config['picking_count_threshold'])

base_data, all_workers = pd.DataFrame(), pd.DataFrame()
if uploaded_files:
    base_data, all_workers = load_and_process_data(uploaded_files)
    if not base_data.empty:
        base_data = base_data[base_data['평균소요시간(분)'] <= minute_threshold].copy()
        base_data = base_data[base_data['피킹횟수'] >= picking_count_threshold].copy()

filtered_data = base_data.copy()
filtered_all_workers = all_workers.copy()

if not base_data.empty:
    # [수정] 파일 수와 기간만 요약해서 표시
    successful_file_count = base_data['날짜'].nunique()
    start_date = base_data['날짜'].min().strftime('%Y-%m-%d')
    end_date = base_data['날짜'].max().strftime('%Y-%m-%d')
    st.info(f"총 {successful_file_count}개 파일의 데이터를 성공적으로 불러왔습니다. (기간: {start_date} ~ {end_date})")
    
    st.subheader("기간 필터")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        filter_type = st.selectbox("필터 종류", ["전체", "연도별", "월별", "일별", "요일별"])
    
    if filter_type == "연도별":
        with filter_col2:
            years = sorted(base_data['연도'].unique(), reverse=True)
            selected_year = st.selectbox("연도 선택", years)
        filtered_data = base_data[base_data['연도'] == selected_year]
        filtered_all_workers = all_workers[all_workers['날짜'].dt.year == selected_year]
    elif filter_type == "월별":
        with filter_col2:
            years = sorted(base_data['연도'].unique(), reverse=True)
            selected_year = st.selectbox("연도 선택", years)
        with filter_col3:
            months = sorted(base_data[base_data['연도'] == selected_year]['월'].unique())
            selected_month = st.selectbox("월 선택", months)
        filtered_data = base_data[(base_data['연도'] == selected_year) & (base_data['월'] == selected_month)]
        filtered_all_workers = all_workers[(all_workers['날짜'].dt.year == selected_year) & (all_workers['날짜'].dt.month == selected_month)]
    elif filter_type == "일별":
        with filter_col2:
            selected_date = st.date_input("날짜 선택", base_data['날짜'].max())
        filtered_data = base_data[base_data['날짜'] == pd.to_datetime(selected_date)]
        filtered_all_workers = all_workers[all_workers['날짜'] == pd.to_datetime(selected_date)]
    elif filter_type == "요일별":
        with filter_col2:
            days_of_week_map = {'월요일': 0, '화요일': 1, '수요일': 2, '목요일': 3, '금요일': 4, '토요일': 5}
            selected_days_kr = st.multiselect("요일 선택", options=days_of_week_map.keys(), default=list(days_of_week_map.keys()))
        if selected_days_kr:
            selected_weekdays = [days_of_week_map[day] for day in selected_days_kr]
            filtered_data = base_data[base_data['날짜'].dt.weekday.isin(selected_weekdays)]
            filtered_all_workers = all_workers[all_workers['날짜'].dt.weekday.isin(selected_weekdays)]
        else:
            filtered_data, filtered_all_workers = pd.DataFrame(), pd.DataFrame()

if st.button('분석 시작', type="primary"):
    if filtered_data.empty and filtered_all_workers.empty:
        st.warning("분석할 데이터가 없습니다. 파일을 먼저 업로드해주세요.")
    else:
        current_config = {'minute_threshold': minute_threshold, 'picking_count_threshold': picking_count_threshold}
        save_config(current_config)
        
        total_picks = filtered_data['피킹횟수'].sum()
        st.success(f"분석 완료! (기간 내 총 피킹 횟수: {int(total_picks):,} 건)")

        st.header("2. 종합 분석 결과")
        avg_time_minutes = (filtered_data['평균소요시간(분)'] * filtered_data['피킹횟수']).sum() / total_picks if total_picks > 0 else 0
        daily_avg_workers = filtered_data.groupby('날짜')['작업자명'].nunique().mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("기간 내 총 피킹 횟수", f"{int(total_picks):,} 건") 
        col2.metric("평균 소요시간 (1회당)", f"{round(avg_time_minutes)} 분" if avg_time_minutes > 0 else "N/A")
        col3.metric("일 평균 작업자 수", f"{daily_avg_workers:.1f} 명" if pd.notna(daily_avg_workers) else "N/A")
        
        st.markdown("---")
        st.header("3. 상세 분석")
        tabs = st.tabs(["작업자별 분석", "기간별 추이 분석", "요일별 분석", "상세 데이터 보기"])
        
        with tabs[0]:
            st.subheader("작업자별 성과 요약")
            worker_analysis = filtered_data.groupby('작업자명').agg(
                평균소요시간_분=pd.NamedAgg(column='평균소요시간(분)', aggfunc=lambda x: (x * filtered_data.loc[x.index, '피킹횟수']).sum() / filtered_data.loc[x.index, '피킹횟수'].sum()),
                총_피킹횟수=('피킹횟수', 'sum'),
                작업일수=('날짜', 'nunique')
            ).reset_index()

            all_period_workers = pd.DataFrame(filtered_all_workers['작업자명'].unique(), columns=['작업자명'])
            final_worker_analysis = pd.merge(all_period_workers, worker_analysis, on='작업자명', how='left').fillna(0)
            
            final_worker_analysis['시간순위'] = final_worker_analysis['평균소요시간_분'].rank(method='min', ascending=True).where(final_worker_analysis['총_피킹횟수'] > 0, 0)
            final_worker_analysis['횟수순위'] = final_worker_analysis['총_피킹횟수'].rank(method='min', ascending=False).where(final_worker_analysis['총_피킹횟수'] > 0, 0)
            final_worker_analysis['작업일수'] = final_worker_analysis['작업일수'].astype(int)
            final_worker_analysis['일평균_피킹횟수'] = (final_worker_analysis['총_피킹횟수'] / final_worker_analysis['작업일수']).where(final_worker_analysis['작업일수'] > 0, 0).round(1)
            final_worker_analysis['일평균순위'] = final_worker_analysis['일평균_피킹횟수'].rank(method='min', ascending=False).where(final_worker_analysis['총_피킹횟수'] > 0, 0)

            final_worker_analysis[['총_피킹횟수', '시간순위', '횟수순위', '일평균순위']] = final_worker_analysis[['총_피킹횟수', '시간순위', '횟수순위', '일평균순위']].astype(int)
            final_worker_analysis['평균소요시간_분'] = final_worker_analysis['평균소요시간_분'].round().astype(int)

            display_cols = ['작업자명', '평균소요시간_분', '시간순위', '총_피킹횟수', '횟수순위', '작업일수', '일평균_피킹횟수', '일평균순위']
            st.dataframe(final_worker_analysis[display_cols].sort_values(by='시간순위'), hide_index=True)

            fig_avg_time = px.bar(final_worker_analysis[final_worker_analysis['총_피킹횟수']>0].sort_values(by='평균소요시간_분'), x='작업자명', y='평균소요시간_분', title='작업자별 평균 피킹 소요시간 (분)', text_auto=True)
            st.plotly_chart(fig_avg_time, use_container_width=True)
            fig_pick_count = px.bar(final_worker_analysis.sort_values(by='총_피킹횟수', ascending=False), x='작업자명', y='총_피킹횟수', title='작업자별 기간 내 총 피킹 횟수', text_auto=True)
            st.plotly_chart(fig_pick_count, use_container_width=True)
            fig_daily_avg = px.bar(final_worker_analysis.sort_values(by='일평균_피킹횟수', ascending=False), x='작업자명', y='일평균_피킹횟수', title='작업자별 일평균 피킹 횟수', text_auto='.1f')
            st.plotly_chart(fig_daily_avg, use_container_width=True)

        with tabs[1]:
            st.subheader("기간별 성과 추이")
            if not filtered_data.empty:
                group_by_period = filtered_data['날짜'].dt.to_period('M' if filter_type in ["전체", "연도별"] else 'D')
                trend_analysis = filtered_data.groupby(group_by_period).apply(lambda x: (x['평균소요시간(분)'] * x['피킹횟수']).sum() / x['피킹횟수'].sum() if x['피킹횟수'].sum() > 0 else 0).reset_index(name='평균소요시간_분')
                trend_analysis.rename(columns={'날짜': '기간'}, inplace=True)
                trend_analysis['기간'] = trend_analysis['기간'].astype(str)
                trend_analysis['평균소요시간_분'] = trend_analysis['평균소요시간_분'].round().astype(int)
                fig_trend_time = px.line(trend_analysis, x='기간', y='평균소요시간_분', title='기간별 평균 피킹 소요시간 추이', markers=True)
                st.plotly_chart(fig_trend_time, use_container_width=True)
            else:
                st.info("기간 내 유효한 피킹 작업 데이터가 없어 추이 분석을 표시할 수 없습니다.")
        
        with tabs[2]:
            st.subheader("요일별 성과 분석")
            if not filtered_data.empty:
                dow_analysis = filtered_data.groupby('요일', observed=False).apply(lambda x: pd.Series({
                    '총_피킹횟수': x['피킹횟수'].sum(),
                    '평균소요시간_분': (x['평균소요시간(분)'] * x['피킹횟수']).sum() / x['피킹횟수'].sum() if x['피킹횟수'].sum() > 0 else 0,
                    '작업일수': x['날짜'].nunique()
                })).reset_index()
                dow_analysis['일평균_피킹횟수'] = (dow_analysis['총_피킹횟수'] / dow_analysis['작업일수']).fillna(0).round(1)
                dow_analysis['평균소요시간_분'] = dow_analysis['평균소요시간_분'].round().astype(int)
                dow_analysis['시간순위'] = dow_analysis['평균소요시간_분'].rank(method='min', ascending=True).astype(int)
                dow_analysis['횟수순위'] = dow_analysis['총_피킹횟수'].rank(method='min', ascending=False).astype(int)
                
                dow_display_cols = ['요일', '평균소요시간_분', '시간순위', '총_피킹횟수', '횟수순위', '작업일수', '일평균_피킹횟수']
                st.dataframe(dow_analysis[dow_display_cols].sort_values(by='요일', key=lambda x: x.map({day: i for i, day in enumerate(DAYS_ORDER)})), hide_index=True)
                
                fig_dow_time = px.bar(dow_analysis, x='요일', y='평균소요시간_분', title='요일별 평균 피킹 소요시간 (분)', text_auto=True)
                st.plotly_chart(fig_dow_time, use_container_width=True)
                fig_dow_count = px.bar(dow_analysis, x='요일', y='총_피킹횟수', title='요일별 총 피킹 횟수', text_auto=True)
                st.plotly_chart(fig_dow_count, use_container_width=True)
            else:
                st.info("기간 내 유효한 피킹 작업 데이터가 없어 요일별 분석을 표시할 수 없습니다.")

        with tabs[3]:
            st.subheader("상세 데이터 보기")
            display_df = filtered_data.sort_values(by=['날짜', '작업자명'], ascending=[False, True]).copy()
            display_df['시간순위'] = display_df['평균소요시간(분)'].rank(method='min', ascending=True).astype(int)
            display_df['횟수순위'] = display_df['피킹횟수'].rank(method='min', ascending=False).astype(int)
            display_df['평균소요시간(분)'] = display_df['평균소요시간(분)'].round().astype(int)
            
            detail_cols = ['날짜', '요일', '작업자명', '피킹횟수', '횟수순위', '평균소요시간(분)', '시간순위']
            st.dataframe(display_df[detail_cols], hide_index=True)
            
# --------------------------------------------------------------------------
# 개발자 서명 (화면 우측 하단 고정)
# --------------------------------------------------------------------------
st.markdown("""<style>.footer{position:fixed;right:10px;bottom:10px;width:auto;background-color:transparent;color:#888;text-align:right;padding:10px;font-size:14px;}</style><div class="footer">by suhyuk (twodoong@gmail.com)</div>""", unsafe_allow_html=True)