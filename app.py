import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import time

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
# Tab 2: 数据录入 (新增功能)
# ==========================================
with tab2:
    st.header("🔬 实验数据反馈")
    st.markdown("请在此处录入您的真实实验数据，这将帮助模型变得更准确！")

    # 使用 Form 表单，防止每填一个数就刷新一次
    with st.form("entry_form"):
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
        # 重点：这是目标列（真实结果）
        e_result = st.selectbox("🧪 实验结果 (可去向性)", ("难 (Hard)", "中 (Medium)", "易 (Easy)"))

        submitted = st.form_submit_button("提交数据")

        if submitted:
            # 1. 整理数据
            new_record = {
                'CNF_content': e_cnf, 'CNF/PVA_conc': e_conc, 'NumofLayer': e_layer,
                'Angle1': e_ang1, 'Angle2': e_ang2, 'Thickness': e_thick, 'Ts': e_temp, 'Tempo': e_tempo,
                'Craft': e_craft,
                'Orientability': e_result.split(" ")[0]  # 只取"难/中/易"
            }

            # 2. 转换为 DataFrame
            df_new = pd.DataFrame([new_record])

            # 3. 显示成功信息
            st.success("✅ 数据已暂存！请点击下方按钮下载 Excel，然后发送给管理员。")
            st.dataframe(df_new)

            # 4. 提供下载按钮 (因为云端无法永久保存，只能下载)
            # 将 DataFrame 转为 CSV
            csv = df_new.to_csv(index=False).encode('utf-8-sig')

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            st.download_button(
                label="📥 下载这条数据 (CSV)",
                data=csv,
                file_name=f"exp_data_{timestamp}.csv",
                mime="text/csv"
            )

            st.info("💡 提示：由于云端安全限制，数据无法直接写入服务器。请下载后汇总。")

# ==========================================
# Tab 3: 模型分析 (原来的 Analysis)
# ==========================================
with tab3:
    st.header("📊 模型评估与分析")

    # ... (这里放你之前 Analysis 部分的代码，为了节省篇幅我简写了) ...
    # ... 请把你之前写的 R2, Correlation, SHAP 三个按钮的代码完整复制到这里 ...

    col_a, col_b, col_c = st.columns(3)
    # 示例：SHAP 按钮 (请替换为你之前那个修好的版本)
    with col_c:
        if st.button("🧬 SHAP 特征贡献", key="shap_btn"):  # key防止冲突
            # ... 你的 SHAP 代码 ...
            st.info("请将之前的 SHAP 代码粘贴回这里")