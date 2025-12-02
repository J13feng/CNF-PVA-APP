import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
import traceback
import io

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 加载资源 (v3.0 全能版)
# ==========================================
@st.cache_resource
def load_resources():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 路径
    models_path = os.path.join(current_dir, 'models_pack.pkl')  # 加载模型包
    scaler_path = os.path.join(current_dir, 'scaler.pkl')
    cols_path = os.path.join(current_dir, 'feature_cols.pkl')
    data_path = os.path.join(current_dir, 'train_data.pkl')

    # 加载
    models_pack = joblib.load(models_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(cols_path)
    train_data = joblib.load(data_path)

    return models_pack, scaler, feature_cols, train_data


try:
    models_pack, scaler, feature_cols, train_data = load_resources()
    # 解包模型
    model_cls = models_pack['svm_cls']
    model_tensile = models_pack['rf_tensile']
    model_elong = models_pack['rf_elong']
    model_trans = models_pack['rf_trans']

    X_train_df = train_data['X_df']
except FileNotFoundError as e:
    st.error(f"❌ 缺少文件: {e}")
    st.info("请先运行 Train_SVM.py (v3.0) 生成 models_pack.pkl")
    st.stop()


# ==========================================
# 辅助函数：连接 Google Sheets
# ==========================================
def get_gsheet_client():
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_key_path = os.path.join(current_dir, 'key.json')

    if os.path.exists(json_key_path):
        creds = ServiceAccountCredentials.from_json_keyfile_name(json_key_path, scope)
    elif "gcp_service_account" in st.secrets:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    else:
        return None
    return gspread.authorize(creds)


def add_data_to_gsheet(data_row):
    try:
        cleaned_row = []
        for item in data_row:
            if hasattr(item, "item"): item = item.item()
            cleaned_row.append(item)

        client = get_gsheet_client()
        if not client:
            st.error("未找到密钥")
            return False

        sheet_id = "1CQ6VoA24v6KNoVOSDoKmM4_1Lv35eC20oxBTJ8opMKw".strip()
        sheet = client.open_by_key(sheet_id).sheet1
        sheet.append_row(cleaned_row)
        return True
    except Exception as e:
        st.error(f"错误: {e}")
        return False


# ==========================================
# 侧边栏：数据仪表盘 (图标已修复)
# ==========================================
with st.sidebar:
    # 修复图标：使用 Emoji 或者 Streamlit 原生 Logo，最稳
    st.header("🧪 实验室看板")

    if st.button("🔄 刷新数据"):
        try:
            client = get_gsheet_client()
            if client:
                sheet_id = "1CQ6VoA24v6KNoVOSDoKmM4_1Lv35eC20oxBTJ8opMKw".strip()
                sheet = client.open_by_key(sheet_id).sheet1
                records = sheet.get_all_records()
                df_online = pd.DataFrame(records)

                st.metric("📊 累计数据", f"{len(df_online)} 条")

                # 尝试找最大强度
                # 模糊匹配表头
                ts_cols = [c for c in df_online.columns if 'Tensile' in c or '拉伸' in c]
                if ts_cols:
                    max_val = pd.to_numeric(df_online[ts_cols[0]], errors='coerce').max()
                    st.metric("🏆 最高强度记录", f"{max_val} MPa")
            else:
                st.warning("连接失败")
        except:
            st.error("连接超时")

    st.markdown("---")
    st.caption("Developed by FengYan Group")

# ==========================================
# 主页面
# ==========================================
st.title("PVA/CNF 复合薄膜智能协作平台 v3.0")

tab1, tab2, tab3 = st.tabs(["🚀 全能预测", "📝 数据录入", "📊 深度分析"])

# ==========================================
# Tab 1: 预测 (分类 + 3回归)
# ==========================================
with tab1:
    st.subheader("1. 单点全指标预测")

    col1, col2 = st.columns(2)
    with col1:
        cnf_content = st.number_input("CNF 含量 (%)", 0.0, 100.0, 0.5, format="%.3f", key="p_cnf")
        pva_conc = st.number_input("CNF/PVA 浓度 (%)", 0.0, 100.0, 10.0, key="p_conc")
        num_layer = st.number_input("刮涂层数", 1, 50, 1, key="p_layer")
        ts_val = st.number_input("强度 Ts (MPa)", 0.0, 300.0, 25.0, key="p_ts")
    with col2:
        angle1 = st.number_input("角度 Angle1", 0.0, 180.0, 0.0, key="p_ang1")
        angle2 = st.number_input("角度 Angle2", 0.0, 180.0, 0.0, key="p_ang2")
        thickness = st.number_input("厚度 (mm)", 0.0, 5.0, 0.1, key="p_thick")
        tempo = st.number_input("Tempo 参数", 0.0, 100.0, 0.0, key="p_tempo")

    craft_option = st.selectbox("工艺", ("刮涂", "拉伸", "无"), key="p_craft")

    if st.button("🚀 立即计算", type="primary"):
        # 1. 组装
        input_data = {
            'CNF_content': cnf_content, 'CNF/PVA_conc': pva_conc, 'NumofLayer': num_layer,
            'Angle1': angle1, 'Angle2': angle2, 'Thickness': thickness,
            'Ts': ts_val, 'Tempo': tempo,
        }
        input_data[f"Craft_{craft_option}"] = 1
        arr = np.zeros(len(feature_cols))
        for i, col in enumerate(feature_cols): arr[i] = input_data.get(col, 0)

        # 2. 预测
        X_in = scaler.transform(arr.reshape(1, -1))

        # 跑 4 个模型
        p_cls = model_cls.predict(X_in)[0]
        p_ten = model_tensile.predict(X_in)[0]
        p_elo = model_elong.predict(X_in)[0]
        p_tra = model_trans.predict(X_in)[0]

        # 3. 展示
        st.success("✅ 计算完成")

        # 第一行：难易度
        res_map = {0: "难 (Hard)", 1: "中 (Medium)", 2: "易 (Easy)"}
        color = "red" if p_cls == 0 else "orange" if p_cls == 1 else "green"
        st.markdown(f"**工艺难度:** :{color}[{res_map[p_cls]}]")

        # 第二行：3个性能指标
        m1, m2, m3 = st.columns(3)
        m1.metric("预估强度 (Tensile)", f"{p_ten:.1f} MPa")
        m2.metric("预估伸长率 (Elongation)", f"{p_elo:.1f} %")
        m3.metric("预估透光率 (Trans.)", f"{p_tra:.1f} %")

    st.divider()

    # === 批量预测 ===
    st.subheader("2. 批量预测 (支持所有指标)")
    uploaded_file = st.file_uploader("上传 Excel 文件", type=["xlsx"])

    if uploaded_file:
        if st.button("开始批量计算"):
            try:
                df_upload = pd.read_excel(uploaded_file)

                # 预处理
                if 'Craft' in df_upload.columns:
                    df_upload = pd.get_dummies(df_upload, columns=['Craft'], prefix='Craft')

                df_input = pd.DataFrame(0, index=df_upload.index, columns=feature_cols)
                for col in feature_cols:
                    if col in df_upload.columns: df_input[col] = df_upload[col]

                X_batch = scaler.transform(df_input.values)

                # 批量预测所有指标
                p_cls = model_cls.predict(X_batch)
                p_ten = model_tensile.predict(X_batch)
                p_elo = model_elong.predict(X_batch)
                p_tra = model_trans.predict(X_batch)

                # 写入结果
                res_map = {0: "难", 1: "中", 2: "易"}
                df_upload['预测_工艺难度'] = [res_map[x] for x in p_cls]
                df_upload['预测_拉伸强度'] = p_ten
                df_upload['预测_伸长率'] = p_elo
                df_upload['预测_透光率'] = p_tra

                st.dataframe(df_upload.head())

                # 下载
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_upload.to_excel(writer, index=False)
                output.seek(0)
                st.download_button("📥 下载完整结果.xlsx", data=output, file_name="Full_Prediction.xlsx")

            except Exception as e:
                st.error(f"出错: {e}")

# ==========================================
# Tab 2: 录入 (保持不变)
# ==========================================
with tab2:
    st.header("🔬 实验数据反馈")
    with st.form("entry_form"):
        c1, c2 = st.columns(2)
        with c1:
            e_cnf = st.number_input("CNF 含量 (%)", step=0.01)
            e_conc = st.number_input("CNF/PVA 浓度 (%)", step=0.1)
            e_layer = st.number_input("刮涂层数", min_value=1, step=1)
            e_ts = st.number_input("强度 Ts (MPa)", step=1.0)
        with c2:
            e_ang1 = st.number_input("角度 Angle1")
            e_ang2 = st.number_input("角度 Angle2")
            e_thick = st.number_input("厚度 (mm)", step=0.01)
            e_tempo = st.number_input("Tempo 参数")
        e_craft = st.selectbox("所用工艺", ("刮涂", "拉伸", "无"))
        st.divider()
        c3, c4, c5 = st.columns(3)
        with c3:
            e_tensile = st.number_input("成品拉伸强度 (MPa)", step=0.1)
        with c4:
            e_elongation = st.number_input("断裂伸长率 (%)", step=0.1)
        with c5:
            e_transmittance = st.number_input("透光率 (%)", step=0.1)
        st.divider()
        e_result = st.selectbox("🧪 综合评价 (可去向性)", ("难", "中", "易"))

        if st.form_submit_button("🚀 提交"):
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = [e_cnf, e_conc, e_layer, e_ang1, e_ang2, e_thick, e_ts, e_tempo, e_craft, e_result, e_tensile,
                   e_elongation, e_transmittance, ts]
            if add_data_to_gsheet(row):
                st.success("✅ 写入成功")

# ==========================================
# Tab 3: 模型分析 (升级版：支持回归分析)
# ==========================================
with tab3:
    st.header("📊 模型深度分析")

    # 1. 选择要分析的模型
    analysis_target = st.selectbox(
        "🔎 选择分析目标",
        ("工艺难易度 (分类)", "拉伸强度 (回归)", "断裂伸长率 (回归)", "透光率 (回归)")
    )

    # 根据选择，确定要用的模型
    if "工艺" in analysis_target:
        target_model = model_cls
        model_type = "classification"
    elif "拉伸" in analysis_target:
        target_model = model_tensile
        model_type = "regression"
    elif "伸长" in analysis_target:
        target_model = model_elong
        model_type = "regression"
    else:
        target_model = model_trans
        model_type = "regression"

    col_a, col_b = st.columns(2)

    # --- 相关性矩阵 (通用) ---
    # --- 相关性矩阵 ---
    with col_a:
        st.subheader("特征相关性")
        if st.checkbox("显示热力图", value=True):
            # 取前8列数值特征
            numeric_df = X_train_df.iloc[:, :8]

            # 1. 计算相关系数
            corr = numeric_df.corr()

            # ==========================================
            # 🛠️ 修复空白问题：将 NaN 填充为 0
            # ==========================================
            # 如果某列数据完全一样（方差为0），相关性计算会得到 NaN
            # 我们把它填为 0，代表“无相关性”
            corr = corr.fillna(0)

            # 2. 画图
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('配方参数相关性')

            # 3. 显示
            st.pyplot(fig)

            # 4. 智能提示
            # 检查一下是不是真的有方差为0的列，提示用户
            # std_dev = numeric_df.std()
            # constant_cols = std_dev[std_dev == 0].index.tolist()
            # if constant_cols:
            #     st.warning(
            #         f"⚠️ 注意：以下特征在所有数据中数值完全相同，因此无法计算相关性（显示为0.00）：\n {constant_cols}")

    # --- SHAP 分析 (根据模型类型自动调整) ---
    with col_b:
        st.subheader("🧬 SHAP 关键因子分析")

        if st.button(f"分析: {analysis_target}"):
            X_subset = X_train_df.iloc[:20, :]

            with st.spinner('正在计算特征重要性...'):
                try:
                    plt.clf()

                    if model_type == "classification":
                        # SVM 分类模型使用 KernelExplainer (较慢)
                        X_summary = shap.kmeans(X_train_df, 5)


                        def predict_fn(x):
                            if isinstance(x, pd.DataFrame): x = x.values
                            x_scaled = scaler.transform(x)
                            return target_model.predict_proba(x_scaled)


                        explainer = shap.KernelExplainer(predict_fn, X_summary)
                        shap_vals = explainer.shap_values(X_subset)
                        # 取 "易" (Index 2)
                        if isinstance(shap_vals, list):
                            sv = shap_vals[2]
                        else:
                            sv = shap_vals
                        if len(sv.shape) == 3: sv = sv.sum(axis=2)

                    else:
                        # 随机森林回归模型使用 TreeExplainer (超级快！)
                        # 注意：TreeExplainer 不需要标准化后的数据，直接用原始数据即可(如果是树模型)
                        # 但因为我们训练时 fit 进去的是 X_scaled，所以这里还是用 X_scaled 比较严谨
                        X_subset_scaled = scaler.transform(X_subset.values)
                        explainer = shap.TreeExplainer(target_model)
                        sv = explainer.shap_values(X_subset_scaled)

                    # 绘图
                    shap.summary_plot(sv, X_subset, max_display=10, show=False)
                    st.pyplot(plt.gcf(), clear_figure=True)
                    st.success("✅ 分析完成")

                    if model_type == "regression":
                        st.info("💡 回归模型解读：红色点在右边 = 该特征数值越大，预测结果(如强度)越高。")

                except Exception as e:
                    st.error(f"分析失败: {e}")
                    st.code(traceback.format_exc())