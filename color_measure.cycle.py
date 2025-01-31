import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("패턴 감지 및 편차 분석 V1.2 (멀티 파일 지원)")

# 여러 파일 업로드
uploaded_files = st.file_uploader("CSV 파일을 업로드하세요 (여러 개 선택 가능)", type="csv", accept_multiple_files=True)

# 모든 파일의 CoV 데이터를 저장할 리스트
all_cov_data = []

if uploaded_files:
    # 각 파일별로 분석 수행
    for uploaded_file in uploaded_files:
        st.write(f"## 파일: {uploaded_file.name}")
        df = pd.read_csv(uploaded_file)
        st.write("데이터 미리보기:")
        st.dataframe(df)

        # X축과 Y축 기본값 설정
        default_x_col = df.columns[0] if len(df.columns) > 0 else None
        default_y_col = df.columns[1] if len(df.columns) > 1 else None

        # X축과 Y축 선택
        x_col = st.selectbox("X축 선택", df.columns, index=df.columns.get_loc(default_x_col) if default_x_col else 0, key=f"x_{uploaded_file.name}")
        y_col = st.selectbox("Y축 선택", df.columns, index=df.columns.get_loc(default_y_col) if default_y_col else 1, key=f"y_{uploaded_file.name}")

        # 패턴 감지를 위한 설정
        start_x = st.number_input(
            "패턴 감지 시작 X값", 
            min_value=float(df[x_col].min()), 
            max_value=float(df[x_col].max()), 
            value=float(df[x_col].min()), 
            key=f"start_x_{uploaded_file.name}"
        )

        # 종료 X값의 기본값: 두 번째 열(Y축)에서 데이터가 있는 마지막 행의 X값
        valid_data = df.dropna(subset=[y_col])  # Y축 데이터가 있는 행만 필터링
        end_x_default = valid_data[x_col].iloc[-1]  # 마지막 행의 X값

        end_x = st.number_input(
            "패턴 감지 종료 X값", 
            min_value=start_x, 
            max_value=float(df[x_col].max()), 
            value=float(end_x_default),  # 기본값으로 마지막 행의 X값 사용
            key=f"end_x_{uploaded_file.name}"
        )
        
        initial_value = st.number_input("초기값 설정", min_value=float(df[y_col].min()), max_value=float(df[y_col].max()), value=min(float(df[y_col].max()), 2000.0), key=f"initial_{uploaded_file.name}")
        initial_value_range = st.slider("초기값 범위 (±)", 50, 200, 100, key=f"range_{uploaded_file.name}")
        threshold = st.number_input("Y 값 변화 임계값", min_value=0.0, value=100.0, key=f"threshold_{uploaded_file.name}")

        # 선택한 X축 범위 내 데이터 필터링
        filtered_df = df[(df[x_col] >= start_x) & (df[x_col] <= end_x)].copy()

        # 패턴 감지
        patterns = []
        temp_pattern = []
        for i in range(len(filtered_df) - 1):
            curr_value = filtered_df[y_col].iloc[i]
            next_value = filtered_df[y_col].iloc[i + 1]

            if len(temp_pattern) == 0 and (initial_value - initial_value_range) <= curr_value <= (initial_value + initial_value_range):
                temp_pattern.append(i)  # 패턴 시작

            elif len(temp_pattern) == 1 and abs(next_value - curr_value) >= threshold:
                temp_pattern.append(i)  # 패턴 종료
                patterns.append(temp_pattern)
                temp_pattern = []

        if temp_pattern:
            temp_pattern.append(len(filtered_df) - 1)
            patterns.append(temp_pattern)

        st.write(f"감지된 패턴 개수: {len(patterns)}")

        # Y축 데이터의 최대값 및 최소값 출력
        y_max = float(df[y_col].max())
        y_min = float(df[y_col].min())

        st.write(f"📊 **Y축 최대값:** {y_max}, **최소값:** {y_min}")

        if patterns:
            # 패턴 데이터프레임 생성
            pattern_values = [{
                "Select": False,  # 체크박스 기본값 (선택 여부)
                "Start Index": start_idx,
                "Start Value": filtered_df[y_col].iloc[start_idx],
                "End Index": end_idx,
                "End Value": filtered_df[y_col].iloc[end_idx],
                "Del Adc": filtered_df[y_col].iloc[start_idx] - filtered_df[y_col].iloc[end_idx]
            } for start_idx, end_idx in patterns]

            pattern_table = pd.DataFrame(pattern_values)

            # "모든 패턴 선택 / 해제" 체크박스
            select_all = st.checkbox("모든 패턴 선택 / 해제", key=f"select_all_{uploaded_file.name}")

            # 패턴 데이터프레임 생성
            pattern_values = [{
                "Select": select_all,  # 사용자가 전체 선택하면 True, 아니면 False
                "Start Index": start_idx,
                "Start Value": filtered_df[y_col].iloc[start_idx],
                "End Index": end_idx,
                "End Value": filtered_df[y_col].iloc[end_idx],
                "Del Adc": filtered_df[y_col].iloc[start_idx] - filtered_df[y_col].iloc[end_idx]
            } for start_idx, end_idx in patterns]

            pattern_table = pd.DataFrame(pattern_values)

            # 사용자가 개별 선택할 수 있도록 데이터 테이블 표시 (행 추가 방지)
            edited_table = st.data_editor(
                pattern_table, 
                use_container_width=True, 
                disabled=["Start Index", "Start Value", "End Index", "End Value", "Del Adc"],  # 편집 방지
                key=f"table_{uploaded_file.name}"
            )

            # 사용자가 선택한 행 필터링
            selected_rows = edited_table[edited_table["Select"]]

            # 선택된 데이터 출력
            if not selected_rows.empty:
                st.write("✅ **선택된 패턴 데이터:**")
                st.dataframe(selected_rows)

                # 선택된 패턴들의 Del Adc 값들의 통계량 계산
                del_adc_mean = selected_rows["Del Adc"].mean()
                del_adc_std = selected_rows["Del Adc"].std()
                del_adc_cv = (del_adc_std / del_adc_mean) * 100 if del_adc_mean != 0 else 0  # ZeroDivision 방지

                # 통계값 출력
                st.write("📊 **선택된 패턴의 Del Adc(Start-End) 통계량**")
                st.write(f"- 평균 (Mean): {del_adc_mean:.2f}")
                st.write(f"- 표준편차 (Std Dev): {del_adc_std:.2f}")
                st.write(f"- 표준편차율 (Coefficient of Variation): {del_adc_cv:.2f} %")

                # 선택된 패턴만 그래프 출력
                fig, ax = plt.subplots()
                for _, row in selected_rows.iterrows():
                    start_idx, end_idx = int(row["Start Index"]), int(row["End Index"])
                    pattern_data = filtered_df.iloc[start_idx:end_idx + 1]
                    ax.plot(pattern_data[x_col], pattern_data[y_col], label=f"Pattern {start_idx}")

                # 파일명을 그래프 제목에 추가
                ax.set_title(f"Selected Patterns Visualization - {uploaded_file.name}")  # 파일명 추가
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                ax.grid(True, linestyle="--", alpha=0.5, color="gray")
                st.pyplot(fig)

                # 여러 개의 X축 (Normalized Time) 선택
                normalized_options = np.round(np.linspace(0, 1, num=20), 2).tolist()

                # 기본값: 선택 가능한 모든 항목
                default_values = normalized_options  # 모든 항목을 기본값으로 설정

                x_norm_inputs = st.multiselect(
                    "Normalized Time 선택 (여러 개 가능)", 
                    options=normalized_options,
                    default=default_values,  # 모든 항목이 기본값으로 선택됨
                    key=f"x_norm_{uploaded_file.name}"
                )

                # 선택된 X축 (Normalized Time)에서 Y값들의 최소, 최대, 평균, 표준편차율 계산
                if not selected_rows.empty and x_norm_inputs:
                    y_value_list = []

                    for x_norm in x_norm_inputs:
                        y_values_at_x = {}

                        for _, row in selected_rows.iterrows():
                            start_idx, end_idx = int(row["Start Index"]), int(row["End Index"])
                            pattern_data = filtered_df.iloc[start_idx:end_idx + 1]

                            # X축 정규화
                            norm_x = np.linspace(0, 1, len(pattern_data))

                            # 가장 가까운 인덱스를 찾음
                            nearest_idx = np.abs(norm_x - x_norm).argmin()
                            y_value = pattern_data[y_col].iloc[nearest_idx] - pattern_data[y_col].iloc[0]  # Deviation from start

                            y_values_at_x[f"Pattern {start_idx}"] = y_value

                        # 통계 계산
                        y_values = list(y_values_at_x.values())
                        y_max = max(y_values)
                        y_min = min(y_values)
                        y_mean = np.mean(y_values)  # 평균
                        y_std = np.std(y_values)    # 표준편차
                        y_cv = abs((y_std / y_mean) * 100) if y_mean != 0 else 0  # 표준편차율(CoV, 절대값)

                        y_value_list.append({
                            "Normalized Time": x_norm,
                            **y_values_at_x,
                            "Y값 최대": y_max,
                            "Y값 최소": y_min,
                            "최대-최소 차이": y_max - y_min,
                            "평균": round(y_mean, 2),
                            "표준편차율 (%)": round(y_cv, 2)
                        })

                    # 📋 결과 테이블 생성
                    y_value_table = pd.DataFrame(y_value_list)


                    # 파일별 CoV 데이터 저장
                    all_cov_data.append({
                        "file_name": uploaded_file.name,
                        "cov_data": y_value_table[["Normalized Time", "표준편차율 (%)"]]
                    })

                
                    st.write("📋 **선택된 Normalized Time에서의 Y값 (Deviation from Start Value) 비교**")
                    st.dataframe(y_value_table)

                    # 📊 표준편차율(CoV) 산포도 그래프
                    fig_cov, ax_cov = plt.subplots()
                    ax_cov.scatter(y_value_table["Normalized Time"], y_value_table["표준편차율 (%)"], color='b', alpha=0.7)

                    ax_cov.set_title("Standard Deviation Coefficient (CoV) Over Normalized Time")
                    ax_cov.set_xlabel("Normalized Time")
                    ax_cov.set_ylabel("Coefficient of Variation (%)")
                    ax_cov.grid(True, linestyle="--", alpha=0.5, color="gray")
                    st.pyplot(fig_cov)

                    # 선택된 패턴만 그래프 출력 (Normalized Time 기준 수직선 추가)
                    fig, ax = plt.subplots()
                    
                    for _, row in selected_rows.iterrows():
                        start_idx, end_idx = int(row["Start Index"]), int(row["End Index"])
                        pattern_data = filtered_df.iloc[start_idx:end_idx + 1]

                        # X축 정규화
                        norm_x = np.linspace(0, 1, len(pattern_data))
                        norm_y = pattern_data[y_col] - pattern_data[y_col].iloc[0]  # Deviation from start

                        ax.plot(norm_x, norm_y, label=f"Pattern {start_idx}")

                    # 선택된 Normalized Time에 대해 수직선 추가
                    for x_norm in x_norm_inputs:
                        ax.axvline(x=x_norm, color='r', linestyle='--', alpha=0.7, label=f"T={x_norm}")

                    ax.set_title(f"Overlapping Pattern Comparison - {uploaded_file.name}")  # 파일명 추가
                    ax.set_xlabel("Normalized Time")
                    ax.set_ylabel("Deviation from Start Value")
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                    ax.grid(True, linestyle="--", alpha=0.5, color="gray")
                    st.pyplot(fig)



                # 모든 파일의 CoV 데이터를 중첩해서 비교하는 그래프 (가장 하단에 1번만 표시)
                if all_cov_data:
                    st.write("## 모든 파일의 표준편차율(CoV) 비교")

                    fig_combined, ax_combined = plt.subplots()
                    for file_data in all_cov_data:
                        file_name = file_data["file_name"]
                        cov_data = file_data["cov_data"]
                        ax_combined.scatter(
                            cov_data["Normalized Time"], 
                            cov_data["표준편차율 (%)"], 
                            label=file_name, 
                            alpha=0.7
                        )

                    ax_combined.set_title("Combined Standard Deviation Coefficient (CoV) Over Normalized Time")
                    ax_combined.set_xlabel("Normalized Time")
                    ax_combined.set_ylabel("Coefficient of Variation (%)")
                    ax_combined.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                    ax_combined.grid(True, linestyle="--", alpha=0.5, color="gray")
                    st.pyplot(fig_combined)

                
