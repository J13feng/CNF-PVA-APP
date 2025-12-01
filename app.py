import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# 支持中文绘图 (防止乱码)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 加载资源
# ==========================================
@st.cache_resource
def load_resources():
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    # 加载训练数据用于分析
    train_data = joblib.load('train_data.pkl')
    return model, scaler, feature_cols, train_data


try:
    model, scaler, feature_cols, train_data = load_resources()
    X_train_df = train_data['X_df']  # 获取训练数据的 DataFrame
    y_train = train_data['y']  # 获取训练标签
except FileNotFoundError:
    st.error("❌ 缺少文件！请重新运行 train_final.py")
    st.stop()

# ==========================================
# 2. 界面设计 - 预测区 (左侧/上方)
# ==========================================
st.title("🧪 PVA/CNF 复合薄膜")

with st.expander("🛠️ 配方预测 (点击展开/收起)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        cnf_content = st.number_input("CNF 含量", value=0.5, format="%.3f")
        pva_conc = st.number_input("CNF/PVA 浓度", value=10.0)
        num_layer = st.number_input("刮涂层数", value=1)
        temp = st.number_input("强度 (Ts)", value=25.0)
    with col2:
        angle1 = st.number_input("角度 Angle1", value=0.0)
        angle2 = st.number_input("角度 Angle2", value=0.0)
        thickness = st.number_input("厚度", value=0.1)
        tempo = st.number_input("速率 (Tempo)", value=0.0)

    craft_option = st.selectbox("工艺", ("刮涂", "拉伸", "无"))

    if st.button("🚀 开始预测", type="primary"):
        # ... (预测逻辑与之前相同，省略重复代码，重点看下面分析部分) ...
        # 简写预测流程
        input_data = {
            'CNF_content': cnf_content, 'CNF/PVA_conc': pva_conc, 'NumofLayer': num_layer,
            'Angle1': angle1, 'Angle2': angle2, 'Thickness': thickness, 'Ts': temp, 'Tempo': tempo,
        }
        target_craft = f"Craft_{craft_option}"
        input_data[target_craft] = 1

        arr = np.zeros(len(feature_cols))
        for i, col in enumerate(feature_cols):
            arr[i] = input_data.get(col, 0)

        pred = model.predict(scaler.transform(arr.reshape(1, -1)))[0]
        res_map = {0: "难 (Hard)", 1: "中 (Medium)", 2: "易 (Easy)"}

        # 显示结果
        color = "red" if pred == 0 else "orange" if pred == 1 else "green"
        st.markdown(f"### 预测结果 (可操作性): :{color}[{res_map[pred]}]")

# ==========================================
# 3. 界面设计 - 分析区 (重点新增)
# ==========================================
st.divider()
st.header("📊 模型可视分析 (Model Analytics)")
st.caption("基于 40 条历史数据生成的统计图表")

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