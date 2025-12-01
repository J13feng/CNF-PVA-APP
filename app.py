import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 加载资源
# ==========================================
@st.cache_resource
def load_resources():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 路径拼接
    model_path = os.path.join(current_dir, 'svm_model.pkl')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')
    cols_path = os.path.join(current_dir, 'feature_cols.pkl')
    data_path = os.path.join(current_dir, 'train_data.pkl')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(cols_path)
    train_data = joblib.load(data_path)

    return model, scaler, feature_cols, train_data


try:
    model, scaler, feature_cols, train_data = load_resources()
    X_train_df = train_data['X_df']
    y_train = train_data['y']
except FileNotFoundError as e:
    st.error(f"❌ 缺少文件: {e}")
    st.stop()


# ==========================================
# 辅助函数：连接 Google Sheets
# ==========================================
def add_data_to_gsheet(data_row):
    try:
        # 1. 获取 Secrets 里的配置
        # 注意：Streamlit 会把 secrets.toml 里的内容自动转换成 st.secrets 对象
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

        # 从 secrets 创建凭证
        # create_scoped 需要一个字典，我们直接用 st.secrets 里的内容
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)

        # 2. 连接 Google Sheets
        client = gspread.authorize(creds)

        # 3. 打开具体的表格 (请把这里改成你的表格名字！)
        sheet = client.open("PVA_Experiment_Data").sheet1

        # 4. 追加一行数据
        sheet.append_row(data_row)
        return True
    except Exception as e:
        st.error(f"云端写入失败: {e}")
        return False

# ==========================================
# 2. 页面整体布局 (Tabs)
# ==========================================
st.title("🧪 PVA/CNF 复合薄膜智能协作平台")

# 创建三个选项卡
tab1, tab2, tab3 = st.tabs(["🚀 配方预测", "📝 实验数据录入", "📊 模型分析"])

# ==========================================
# Tab 1: 配方预测 (原来的功能)
# ==========================================
with tab1:
    st.header("新配方性能预测")

    col1, col2 = st.columns(2)
    with col1:
        # 为了方便复用，我们把输入控件定义好
        cnf_content = st.number_input("CNF 含量 (%)", value=0.5, format="%.3f", key="p_cnf")
        pva_conc = st.number_input("CNF/PVA 浓度 (%)", value=10.0, key="p_conc")
        num_layer = st.number_input("刮涂层数", value=1, key="p_layer")
        temp = st.number_input("温度 Ts (℃)", value=25.0, key="p_temp")
    with col2:
        angle1 = st.number_input("角度 Angle1", value=0.0, key="p_ang1")
        angle2 = st.number_input("角度 Angle2", value=0.0, key="p_ang2")
        thickness = st.number_input("厚度 (mm)", value=0.1, key="p_thick")
        tempo = st.number_input("Tempo 参数", value=0.0, key="p_tempo")

    craft_option = st.selectbox("工艺", ("刮涂", "拉伸", "无"), key="p_craft")

    if st.button("开始预测", type="primary"):
        # 组装数据
        input_data = {
            'CNF_content': cnf_content, 'CNF/PVA_conc': pva_conc, 'NumofLayer': num_layer,
            'Angle1': angle1, 'Angle2': angle2, 'Thickness': thickness, 'Ts': temp, 'Tempo': tempo,
        }
        input_data[f"Craft_{craft_option}"] = 1

        arr = np.zeros(len(feature_cols))
        for i, col in enumerate(feature_cols):
            arr[i] = input_data.get(col, 0)

        pred = model.predict(scaler.transform(arr.reshape(1, -1)))[0]
        res_map = {0: "难 (Hard)", 1: "中 (Medium)", 2: "易 (Easy)"}

        color = "red" if pred == 0 else "orange" if pred == 1 else "green"
        st.markdown(f"### 🎯 预测结果: :{color}[{res_map[pred]}]")

# ==========================================
# Tab 2: 数据录入 (Google Sheets 版)
# ==========================================
with tab2:
    st.header("🔬 实验数据反馈 (云端同步版)")
    st.markdown("数据将直接写入 **Google Sheets**，永久保存。")

    with st.form("entry_form"):
        # ... (输入框代码和之前一样，省略重复部分) ...
        # 假设这里是 cnf_content, pva_conc 等输入框...
        # 为节省篇幅，这里用变量名代替

        # 模拟几个输入框
        c1, c2 = st.columns(2)
        with c1:
            e_cnf = st.number_input("CNF 含量 (%)", step=0.01)
            e_conc = st.number_input("CNF/PVA 浓度 (%)", step=0.1)
            e_layer = st.number_input("刮涂层数", min_value=1, step=1)
            e_temp = st.number_input("温度 Ts (℃)", step=1.0)
        with c2:
            e_ang1 = st.number_input("角度 Angle1")
            e_ang2 = st.number_input("角度 Angle2")
            e_thick = st.number_input("厚度 (mm)", step=0.01)
            e_tempo = st.number_input("Tempo 参数")

        e_craft = st.selectbox("所用工艺", ("刮涂", "拉伸", "无"))
        st.divider()
        e_result = st.selectbox("🧪 实验结果", ("难", "中", "易"))

        submitted = st.form_submit_button("🚀 提交到云数据库")

        if submitted:
            # 1. 准备数据列表 (顺序必须和 Google Sheet 表头一致)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            row_data = [
                e_cnf, e_conc, e_layer, e_ang1, e_ang2, e_thick, e_temp, e_tempo,
                e_craft, e_result, timestamp
            ]

            # 2. 调用函数写入
            with st.spinner("正在连接 Google Cloud..."):
                success = add_data_to_gsheet(row_data)

            if success:
                st.success(f"✅ 成功！数据已写入云端表格。时间: {timestamp}")
                st.balloons()  # 放个气球庆祝一下
            else:
                st.error("❌ 写入失败，请检查网络或联系管理员。")

# ==========================================
# Tab 3: 模型分析 (原来的 Analysis)
# ==========================================
with tab3:
    st.header("📊 模型评估与分析")

    # ... (这里放你之前 Analysis 部分的代码，为了节省篇幅我简写了) ...
    # ... 请把你之前写的 R2, Correlation, SHAP 三个按钮的代码完整复制到这里 ...

    col_a, col_b, col_c = st.columns(3)
    # --- 功能 1: 混淆矩阵 (评估准确度) ---
    with col_a:
        if st.button("📈 查看模型精度"):
            st.subheader("模型性能评估 (混淆矩阵)")

            # 1. 用模型预测所有训练数据
            X_scaled_all = scaler.transform(X_train_df.values)
            y_pred_all = model.predict(X_scaled_all)

            # 2. 画混淆矩阵
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_train, y_pred_all)

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['难', '中', '易'],
                        yticklabels=['难', '中', '易'])
            plt.xlabel('预测值')
            plt.ylabel('真实值')
            plt.title('模型混淆矩阵')

            # 3. 计算准确率
            acc = np.mean(y_pred_all == y_train)
            st.metric("当前模型准确率 (Accuracy)", f"{acc * 100:.1f}%")

            # 4. 显示图表
            st.pyplot(fig)
            st.info("💡 解读：对角线上的数字越大越好。如果'难'被预测成'中'，则第一行第二列会有数字。")

    # --- 功能 2: 特征相关性 (数据分析) ---
    with col_b:
        if st.button("🔗 特征相关性矩阵"):
            st.subheader("特征相关性热力图")

            # 计算相关系数
            # 为了不让图太乱，只选前 8 个数值特征
            numeric_df = X_train_df.iloc[:, :8]
            corr = numeric_df.corr()

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('配方参数相关性')

            st.pyplot(fig)
            st.info("💡 解读：红色越深代表正相关（一起变大），蓝色越深代表负相关（一个大一个小）。")
    # --- 功能 3: SHAP 分析 (修正蜂群图版) ---
    with col_c:
        if st.button("🧬 SHAP 特征贡献"):
            st.subheader("工艺参数影响分析")

            # 1. 锁定前 20 条数据
            # ⚠️ 确保这里的 X_train_df 包含所有的 8 个数值特征 + Craft 工艺特征
            X_shap_subset = X_train_df.iloc[:20, :]

            with st.spinner('正在计算 SHAP 值...'):
                try:
                    # 2. 准备背景数据
                    X_summary = shap.kmeans(X_train_df, 5)


                    # 3. 定义预测函数
                    def predict_fn(x):
                        if isinstance(x, pd.DataFrame):
                            x = x.values
                        x_scaled = scaler.transform(x)
                        return model.predict_proba(x_scaled)


                    # 4. 计算 SHAP 值 (KernelExplainer)
                    explainer = shap.KernelExplainer(predict_fn, X_summary)
                    shap_values_raw = explainer.shap_values(X_shap_subset)

                    # ==========================================
                    # 🛑 关键修复：确保提取正确的维度
                    # ==========================================

                    # 针对 SVM (SVC probability=True)，shap_values_raw 是一个 list
                    # list[0] = 类别0 (难) 的 SHAP 值
                    # list[1] = 类别1 (中) 的 SHAP 值
                    # list[2] = 类别2 (易) 的 SHAP 值

                    if isinstance(shap_values_raw, list):
                        # 我们只看 "易" (Index=2)
                        shap_vals = shap_values_raw[2]
                    else:
                        # 如果是二分类，有时候只返回一个 array
                        shap_vals = shap_values_raw

                    # 再次检查：shap_vals 必须是 (20, n_features) 的 2维数组
                    # 如果它变成了 3维，就会画出你刚才那个错误的图
                    if len(shap_vals.shape) == 3:
                        st.warning("检测到交互值，正在降维...")
                        shap_vals = shap_vals.sum(axis=2)  # 强制压平

                    # ==========================================
                    # 📊 绘图：标准的蜂群图 (Beeswarm)
                    # ==========================================
                    plt.clf()  # 清空画布

                    # max_display=10: 只显示最重要的前 10 个特征
                    shap.summary_plot(
                        shap_vals,
                        X_shap_subset,
                        max_display=10,
                        show=False
                    )

                    # 抓取图片
                    fig_shap = plt.gcf()
                    plt.tight_layout()

                    st.pyplot(fig_shap, clear_figure=True)

                    st.success("✅ 分析完成！")
                    st.markdown("""
                    **如何看这张图（蜂群图）：**
                    1.  **左侧文字**：特征按重要性**从上到下**排列（最上面的最重要）。
                    2.  **横轴 (SHAP Value)**：
                        *   **右侧 (>0)**：促进成膜（容易）。
                        *   **左侧 (<0)**：阻碍成膜（难/中）。
                    3.  **颜色**：
                        *   🔴 **红色**：数值大（如高温、高浓度）。
                        *   🔵 **蓝色**：数值小（如低温、低浓度）。
                    """)

                except Exception as e:
                    st.error(f"分析失败: {e}")
                    st.write("调试信息 - SHAP 数据类型:", type(shap_values_raw))