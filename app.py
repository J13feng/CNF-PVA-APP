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
import traceback
import io
import bcrypt
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher

# ==========================================
# 0. 全局配置
# ==========================================
# st.set_page_config(page_title="PVA/CNF 智能平台", layout="wide")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ⚠️⚠️⚠️ 请填入你的表格 ID
# 1. 新建的用户表 (PVA_Users)
USER_DB_ID = "1f7opzdipkTKe0SiJxI729pWxO7AIGOa6fVfQR5DP_tg".strip()

# 2. 原来的实验数据表（CNF/PVA）
DATA_DB_ID = "1CQ6VoA24v6KNoVOSDoKmM4_1Lv35eC20oxBTJ8opMKw".strip()


# ==========================================
# 1. 核心工具：连接 Google Cloud
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


# ==========================================
# 🛠️ 管理员工具：全量更新 Google Sheet（管理员删除修改数据后传回Google sheets）
# ==========================================
def update_gsheet_from_df(sheet_id, df):
    """
    危险操作：将 DataFrame 的内容完全覆盖写入 Google Sheet
    """
    try:
        client = get_gsheet_client()
        sheet = client.open_by_key(sheet_id).sheet1

        # 1. 清空旧数据
        sheet.clear()

        # 2. 准备新数据 (表头 + 内容)
        # 将 DataFrame 转换为 list of lists
        # 注意：需要把 numpy 类型转为 python 原生类型，否则报错
        data = [df.columns.values.tolist()] + df.astype(str).values.tolist()

        # 3. 写入新数据
        sheet.update(data)
        return True
    except Exception as e:
        st.error(f"保存失败: {e}")
        return False


# ==========================================
# 2. 用户管理模块 (登录/注册)
# ==========================================
def load_users_from_gsheet():
    """从云端加载用户名单"""
    try:
        client = get_gsheet_client()
        if not client: return {}
        try:
            sheet = client.open_by_key(USER_DB_ID).sheet1
        except:
            return {}  # ID未填或连接失败

        records = sheet.get_all_records()
        user_dict = {}
        for row in records:
            u_name = str(row['username'])
            if u_name:
                user_dict[u_name] = {
                    'name': row['name'],
                    'password': row['password'],
                    'email': row['email']
                }
        return user_dict
    except Exception:
        return {}


def register_user_to_gsheet(username, name, password, email):
    """注册新用户"""
    try:
        client = get_gsheet_client()
        sheet = client.open_by_key(USER_DB_ID).sheet1

        try:
            existing_users = sheet.col_values(1)
        except:
            existing_users = []

        if username in existing_users:
            return False, "用户名已存在"

        # ==========================================
        # 🛠️ 修复：使用 bcrypt 直接加密，绕过 Hasher 报错
        # ==========================================
        # 1. 加密
        hashed_bytes = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        # 2. 转为字符串 (以便存入 Google Sheets)
        hashed_pw = hashed_bytes.decode('utf-8')
        # ==========================================

        # 写入
        sheet.append_row([username, name, hashed_pw, email])
        return True, "注册成功"
    except Exception as e:
        return False, f"注册失败: {e}"

# ==========================================
# 3. 登录入口系统 (修正版)
# ==========================================
def login_system():
    # 1. 基础配置读取
    if "credentials" not in st.secrets:
        st.error("请先配置 secrets.toml 中的 [credentials]")
        st.stop()

    config = st.secrets["credentials"]

    # 2. 加载用户名单
    cloud_users = load_users_from_gsheet()
    all_credentials = {'usernames': {}}
    if 'usernames' in config:
        for username, user_data in config['usernames'].items():
            all_credentials['usernames'][username] = dict(user_data)
    all_credentials['usernames'].update(cloud_users)

    # 3. 初始化认证器
    authenticator = stauth.Authenticate(
        all_credentials,
        config['cookie_name'],
        config['cookie_key'],
        config['cookie_expiry_days']
    )

    # ========================================================
    # ⚡ 关键修改 1: 登录状态检测
    # 如果 session_state 里已经是 True (说明已登录)，直接返回，不画界面！
    # ========================================================
    if st.session_state.get("authentication_status"):
        return authenticator, st.session_state["name"], st.session_state["username"]

    # ========================================================
    # ⚡ 关键修改 2: 界面美化
    # 只有【未登录】时，才会执行下面的代码，画出登录框
    # ========================================================

    # A. 标题区 (不再受 columns 限制，全宽居中，强制不换行)
    st.markdown("""
        <h1 style='text-align: center; white-space: nowrap; font-size: 36px; margin-bottom: 30px;'>
            🧪 CNF/PVA 复合薄膜智能协作平台
        </h1>
    """, unsafe_allow_html=True)

    # B. 登录/注册区 (使用居中布局)
    col1, col2, col3 = st.columns([1, 2, 1])  # 中间列占 50% 宽度
    with col2:
        tab_login, tab_reg = st.tabs(["🔑 登录", "📝 注册"])

        # --- 登录 Tab ---
        with tab_login:
            # 渲染登录框
            authenticator.login(location='main')

            # 处理登录失败/未登录的提示
            if st.session_state.get("authentication_status") is False:
                st.error('❌ 账号或密码错误')
            elif st.session_state.get("authentication_status") is None:
                st.info('请输入账号密码登录')

        # --- 注册 Tab ---
        with tab_reg:
            with st.form("reg"):
                new_user = st.text_input("用户名 (英文ID)", max_chars=20)
                new_name = st.text_input("真实姓名")
                new_email = st.text_input("邮箱 (选填)")
                p1 = st.text_input("密码", type="password")
                p2 = st.text_input("确认密码", type="password")

                if st.form_submit_button("立即注册"):
                    if p1 != p2:
                        st.error("两次密码不一致")
                    elif len(p1) < 4:
                        st.error("密码太短")
                    elif not new_user or not new_name:
                        st.error("请填写完整信息")
                    else:
                        with st.spinner("正在创建账号..."):
                            success, msg = register_user_to_gsheet(new_user, new_name, p1, new_email)
                            if success:
                                st.success("✅ 注册成功！请切换到“登录”页进行登录。")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(msg)

    # 未登录状态返回空
    return None, None, None


# ==========================================
# 👑 管理员后台 (Admin Dashboard)
# ==========================================
def admin_dashboard(username, authenticator):
    # 如果已经在主程序入口设置了 set_page_config，这里就不用写了，否则会报错
    # st.set_page_config(page_title="管理员后台", layout="wide")

    # --- 侧边栏 ---
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/admin-settings-male.png", width=80)
        st.title("👨‍💼 管理员控制台")
        st.write(f"当前身份: **{username}**")
        st.info("💡 提示：在表格中双击可修改，选中行点击左侧垃圾桶可删除。修改后务必点击“保存更改”。")
        st.divider()
        authenticator.logout('退出登录', 'sidebar')

    st.title("🎛️ 系统数据管理中心")

    tab_users, tab_data = st.tabs(["👥 用户名单管理", "🧪 实验数据库管理"])

    # ==========================
    # Tab 1: 用户管理 (可编辑)
    # ==========================
    with tab_users:
        st.subheader("已注册用户列表")

        # 1. 读取数据
        client = get_gsheet_client()
        if client:
            try:
                sheet = client.open_by_key(USER_DB_ID).sheet1
                records = sheet.get_all_records()
                df_users = pd.DataFrame(records)

                # 2. 显示编辑器
                # 这里的 num_rows="dynamic" 允许添加和删除行
                edited_users = st.data_editor(
                    df_users,
                    num_rows="dynamic",
                    use_container_width=True,
                    key="editor_users"
                )

                # 3. 保存按钮
                col_save, col_info = st.columns([1, 4])
                with col_save:
                    if st.button("💾 保存用户表更改", type="primary"):
                        with st.spinner("正在同步到云端..."):
                            # ⚠️ 注意：如果在这里新增用户，密码是明文的，不会自动加密
                            # 建议仅在这里做“删除”和“修改姓名/邮箱”的操作
                            if update_gsheet_from_df(USER_DB_ID, edited_users):
                                st.success("✅ 用户表已更新！")
                                time.sleep(1)
                                st.rerun()
                with col_info:
                    st.warning("⚠️ 注意：在此处修改密码无效（不会加密）。请仅用于修改姓名/邮箱或删除违规账号。")

            except Exception as e:
                st.error(f"读取失败: {e}")

    # ==========================
    # Tab 2: 实验数据管理 (可编辑)
    # ==========================================
    with tab_data:
        st.subheader("实验数据清洗 (清洗/修正/删除)")

        # 1. 读取数据
        if client:
            try:
                sheet_data = client.open_by_key(DATA_DB_ID).sheet1
                records_data = sheet_data.get_all_records()
                df_data = pd.DataFrame(records_data)

                # 2. 显示编辑器
                edited_data = st.data_editor(
                    df_data,
                    num_rows="dynamic",
                    use_container_width=True,
                    key="editor_data",
                    column_config={
                        "Submit_Time": st.column_config.TextColumn("提交时间", disabled=True)  # 禁止改时间，防止乱
                    }
                )

                # 3. 操作区
                st.divider()
                c1, c2, c3 = st.columns([1, 2, 2])

                with c1:
                    if st.button("💾 保存数据表更改", type="primary"):
                        with st.spinner("正在覆盖云端数据..."):
                            if update_gsheet_from_df(DATA_DB_ID, edited_data):
                                st.success("✅ 数据库已同步！")
                                time.sleep(1)
                                st.rerun()

                with c2:
                    # 下载功能 (备份)
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        # 下载的是你编辑过后的数据
                        edited_data.to_excel(writer, index=False)
                    output.seek(0)
                    st.download_button(
                        label="📥 导出当前数据 (Excel)",
                        data=output,
                        file_name=f"Data_Backup_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
                    )

                with c3:
                    st.info(f"当前行数: {len(edited_data)}")

            except Exception as e:
                st.error(f"读取失败: {e}")

# ==========================================
# 3. 主程序逻辑 (用户界面)
# ==========================================
def main_application(username, user_real_name, authenticator):
    # ---------------------------------------------------------
    # 🎨 顶部导航栏 UI 修复
    # ---------------------------------------------------------
    # 1. 调整比例：给标题留更多空间 (8.5 vs 1.5)
    # 这样右边的按钮只占很小一块，标题就有足够的地方伸展了
    col_header, col_user = st.columns([8, 2])

    with col_header:
        # 2. 使用 HTML 强制不换行 (white-space: nowrap)
        # 并手动调整一下字体大小和边距，让它看起来更紧凑
        st.markdown(f"""
            <div style="white-space: nowrap; padding-top: 10px;">
                <h1 style="display: inline; font-size: 32px; margin: 0;">
                    🧪 CNF/PVA 复合薄膜智能协作平台
                </h1>
                <span style="color: gray; font-size: 16px; margin-left: 15px;">
                    👋 欢迎回来，<b>{user_real_name}</b>
                </span>
            </div>
        """, unsafe_allow_html=True)

    with col_user:
        # 3. 用户按钮
        # 使用空行或者 margin 来微调垂直对齐，让按钮和文字看起来在一条线上
        st.write("")  # 占位，把按钮往下顶一点点
        with st.popover(f"👤 账号管理", use_container_width=True):
            st.markdown(f"**当前用户**: `{username}`")
            st.divider()
            authenticator.logout('退出登录', 'main')

    st.divider()  # 加一条分割线，更美观

    # --- 资源加载 ---
    @st.cache_resource
    def load_resources(last_updated):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pack_path = os.path.join(current_dir, 'models_pack.pkl')
        scaler_path = os.path.join(current_dir, 'scaler.pkl')
        cols_path = os.path.join(current_dir, 'feature_cols.pkl')
        data_path = os.path.join(current_dir, 'train_data.pkl')

        print(f"Loading models... (TS: {last_updated})")
        models_pack = joblib.load(pack_path)
        scaler = joblib.load(scaler_path)
        feature_cols = joblib.load(cols_path)
        train_data = joblib.load(data_path)
        return models_pack, scaler, feature_cols, train_data

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pack_path = os.path.join(current_dir, 'models_pack.pkl')
        if not os.path.exists(pack_path): raise FileNotFoundError

        last_time = os.path.getmtime(pack_path)
        models_pack, scaler, feature_cols, train_data = load_resources(last_time)

        model_cls = models_pack['svm_cls']
        model_tensile = models_pack['rf_tensile']
        model_elong = models_pack['rf_elong']
        model_trans = models_pack['rf_trans']
        X_train_df = train_data['X_df']
        y_train = train_data['y_cls']  # 对应 v3.0 键名

    except Exception as e:
        st.error(f"系统初始化失败: {e}")
        st.stop()

    # --- 数据写入函数 (写到 DATA 表) ---
    def add_experiment_data(data_row):
        try:
            cleaned = []
            for x in data_row:
                if hasattr(x, "item"): x = x.item()
                cleaned.append(x)

            client = get_gsheet_client()
            sheet = client.open_by_key(DATA_DB_ID).sheet1
            sheet.append_row(cleaned)
            return True
        except Exception as e:
            st.error(f"写入失败: {e}")
            return False

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("📊 实验室看板")
        if st.button("🔄 刷新数据"):
            try:
                client = get_gsheet_client()
                if client:
                    sheet = client.open_by_key(DATA_DB_ID).sheet1
                    df_online = pd.DataFrame(sheet.get_all_records())
                    st.metric("累计数据", f"{len(df_online)} 条")
                    ts_cols = [c for c in df_online.columns if 'Tensile' in c or '拉伸' in c]
                    if ts_cols:
                        max_val = pd.to_numeric(df_online[ts_cols[0]], errors='coerce').max()
                        st.metric("🏆 最高强度", f"{max_val} MPa")
            except:
                st.warning("暂无数据或连接失败")
        st.markdown("---")

    # --- 主功能区 ---
    tab1, tab2, tab3 = st.tabs(["🚀 全能预测", "📝 数据录入", "📊 深度分析"])

    # Tab 1: 预测
    with tab1:
        st.subheader("1. 单点全指标预测")
        c1, c2 = st.columns(2)
        with c1:
            cnf_content = st.number_input("CNF 含量 (%)", 0.0, 100.0, 0.5, format="%.3f", key="p_cnf")
            pva_conc = st.number_input("CNF/PVA 浓度 (%)", 0.0, 100.0, 10.0, key="p_conc")
            num_layer = st.number_input("刮涂层数", 1, 50, 1, key="p_layer")
            ts_val = st.number_input("温度 Ts (℃)", 0.0, 300.0, 25.0, key="p_ts")
        with c2:
            angle1 = st.number_input("角度 Angle1", 0.0, 180.0, 0.0, key="p_ang1")
            angle2 = st.number_input("角度 Angle2", 0.0, 180.0, 0.0, key="p_ang2")
            thickness = st.number_input("厚度 (mm)", 0.0, 5.0, 0.1, key="p_thick")
            tempo = st.number_input("Tempo 参数", 0.0, 100.0, 0.0, key="p_tempo")

        craft_option = st.selectbox("工艺", ("刮涂", "拉伸", "无"), key="p_craft")

        if st.button("🚀 立即计算", type="primary"):
            input_data = {'CNF_content': cnf_content, 'CNF/PVA_conc': pva_conc, 'NumofLayer': num_layer,
                          'Angle1': angle1, 'Angle2': angle2, 'Thickness': thickness, 'Ts': ts_val, 'Tempo': tempo}
            input_data[f"Craft_{craft_option}"] = 1
            arr = np.zeros(len(feature_cols))
            for i, col in enumerate(feature_cols): arr[i] = input_data.get(col, 0)

            X_in = scaler.transform(arr.reshape(1, -1))

            p_cls = model_cls.predict(X_in)[0]
            p_ten = model_tensile.predict(X_in)[0]
            p_elo = model_elong.predict(X_in)[0]
            p_tra = model_trans.predict(X_in)[0]

            st.success("✅ 计算完成")
            res_map = {0: "难 (Hard)", 1: "中 (Medium)", 2: "易 (Easy)"}
            color = "red" if p_cls == 0 else "orange" if p_cls == 1 else "green"
            st.markdown(f"**工艺难度:** :{color}[{res_map[p_cls]}]")

            m1, m2, m3 = st.columns(3)
            m1.metric("预估强度", f"{p_ten:.1f} MPa")
            m2.metric("预估伸长率", f"{p_elo:.1f} %")
            m3.metric("预估透光率", f"{p_tra:.1f} %")

        st.divider()
        st.subheader("2. 批量预测 (Excel上传)")
        uploaded_file = st.file_uploader("上传 Excel 文件", type=["xlsx"])
        if uploaded_file:
            if st.button("开始批量计算"):
                try:
                    df_upload = pd.read_excel(uploaded_file)
                    if 'Craft' in df_upload.columns:
                        df_upload = pd.get_dummies(df_upload, columns=['Craft'], prefix='Craft')

                    df_input = pd.DataFrame(0, index=df_upload.index, columns=feature_cols)
                    for col in feature_cols:
                        if col in df_upload.columns: df_input[col] = df_upload[col]

                    X_batch = scaler.transform(df_input.values)

                    p_cls = model_cls.predict(X_batch)
                    p_ten = model_tensile.predict(X_batch)
                    p_elo = model_elong.predict(X_batch)
                    p_tra = model_trans.predict(X_batch)

                    res_map = {0: "难", 1: "中", 2: "易"}
                    df_upload['预测_工艺难度'] = [res_map[x] for x in p_cls]
                    df_upload['预测_强度'] = p_ten
                    df_upload['预测_伸长率'] = p_elo
                    df_upload['预测_透光率'] = p_tra

                    st.dataframe(df_upload.head())

                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_upload.to_excel(writer, index=False)
                    output.seek(0)
                    st.download_button("📥 下载完整结果.xlsx", data=output, file_name="Batch_Prediction.xlsx")
                except Exception as e:
                    st.error(f"出错: {e}")

    # Tab 2: 录入
    with tab2:
        st.header("🔬 实验数据反馈")
        st.caption(f"当前录入人: **{user_real_name}** (系统将自动署名)")

        with st.form("entry_form"):
            c1, c2 = st.columns(2)
            with c1:
                e_cnf = st.number_input("CNF 含量 (%)", step=0.01)
                e_conc = st.number_input("CNF/PVA 浓度 (%)", step=0.1)
                e_layer = st.number_input("刮涂层数", min_value=1, step=1)
                e_ts = st.number_input("温度 Ts (℃)", step=1.0)
            with c2:
                e_ang1 = st.number_input("角度 Angle1")
                e_ang2 = st.number_input("角度 Angle2")
                e_thick = st.number_input("厚度 (mm)", step=0.01)
                e_tempo = st.number_input("Tempo 参数")
            e_craft = st.selectbox("所用工艺", ("刮涂", "拉伸", "无"))
            st.divider()
            c3, c4, c5 = st.columns(3)
            with c3:
                e_tensile = st.number_input("拉伸强度 (MPa)", step=0.1)
            with c4:
                e_elongation = st.number_input("断裂伸长率 (%)", step=0.1)
            with c5:
                e_transmittance = st.number_input("透光率 (%)", step=0.1)
            st.divider()
            e_result = st.selectbox("🧪 综合评价", ("难", "中", "易"))

            if st.form_submit_button("🚀 提交"):
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # 列表最后加入 user_real_name
                row = [e_cnf, e_conc, e_layer, e_ang1, e_ang2, e_thick, e_ts, e_tempo,
                       e_craft, e_result, e_tensile, e_elongation, e_transmittance,
                       ts, user_real_name]
                if add_experiment_data(row):
                    st.success(f"✅ 录入成功！")
                    st.balloons()

    # Tab 3: 分析
    with tab3:
        st.header("📊 模型深度分析 & 评估")

        # 1. 选择分析目标
        analysis_target = st.selectbox(
            "🔎 选择分析目标",
            ("工艺难易度 (分类)", "拉伸强度 (回归)", "断裂伸长率 (回归)", "透光率 (回归)")
        )

        # 2. 准备数据和模型
        # 注意：必须使用标准化后的数据进行预测
        X_raw = X_train_df.values
        X_scaled = scaler.transform(X_raw)

        if "工艺" in analysis_target:
            target_model = model_cls
            model_type = "classification"
            y_true = train_data['y_cls']
            y_pred = target_model.predict(X_scaled)
        elif "拉伸" in analysis_target:
            target_model = model_tensile
            model_type = "regression"
            y_true = train_data['y_tensile']
            y_pred = target_model.predict(X_scaled)
        elif "伸长" in analysis_target:
            target_model = model_elong
            model_type = "regression"
            y_true = train_data['y_elong']
            y_pred = target_model.predict(X_scaled)
        else:
            target_model = model_trans
            model_type = "regression"
            y_true = train_data['y_trans']
            y_pred = target_model.predict(X_scaled)

        # 3. 三列布局
        col_a, col_b, col_c = st.columns(3)

        # --- 第一列：特征相关性 ---
        with col_a:
            st.subheader("1. 特征相关性")
            if st.checkbox("显示热力图", value=True, key="chk_corr"):
                df_analysis = X_train_df.iloc[:, :8].copy()
                # 拼接当前选中的目标列
                df_analysis['当前目标'] = y_true

                corr = df_analysis.corr().fillna(0)

                # 创建画布
                fig, ax = plt.subplots(figsize=(6, 5))

                # 绘制热力图
                sns.heatmap(
                    corr,
                    annot=True,  # <--- 关键修正：改成 True 就会显示数字了！
                    fmt=".2f",  # <--- 保留两位小数，防止数字太长挤在一起
                    cmap='coolwarm',
                    vmin=-1, vmax=1,
                    center=0,
                    square=True,  # 让格子变成正方形，更好看
                    cbar_kws={"shrink": .5},  # 颜色条缩短一点，不占位置
                    ax=ax # 绑定到 ax
                )

                # --- 🎨 【关键修改】设置字体大小 ---
                # labelsize=10: 设置刻度文字(如 CNF_content)的大小
                # rotation=45: 横轴文字倾斜 45 度
                ax.tick_params(axis='x', labelsize=14, rotation=45) # ax.tick_params 修改刻度大小、颜色等， ax.set_xlabel 修改x轴标题大小、颜色等
                ax.tick_params(axis='y', labelsize=14, rotation=0)

                # 设置标题字体
                ax.set_title('相关性热力图', fontsize=14, fontweight='bold', pad=15)

                # plt.title('相关性热力图', fontsize=12)
                # 让横轴标签倾斜，防止重叠
                plt.xticks(rotation=45, ha='right')

                st.pyplot(fig)
                st.caption("提示：颜色越深/数字越大，相关性越强。")

        # --- 第二列：模型拟合效果 (新增核心功能) ---
        with col_b:
            st.subheader("2. 模型拟合效果")

            if model_type == "regression":
                # === 回归模型 ===
                from sklearn.metrics import r2_score, mean_squared_error

                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                # --- 1. 指标显示区 (紧凑型布局) ---
                # 使用 HTML 减少默认边距，让数字显示得更紧凑，不占太多高度
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f"<div style='text-align: center;'>"
                        f"<div style='font-size: 14px; font-weight: bold; color: #2E7D32;'>R² = {r2:.3f}</div>"
                        #f"<div style='font-size: 24px; font-weight: bold; color: #2E7D32;'>{r2:.3f}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with c2:
                    st.markdown(
                        f"<div style='text-align: center;'>"
                        f"<div style='font-size: 14px; font-weight: bold; color: #2E7D32;'>RMSE = {rmse:.2f}</div>"
                        #f"<div style='font-size: 24px; font-weight: bold; color: #D32F2F;'>{rmse:.2f}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                # 加一点垂直间距
                st.write("")

                # --- 2. 绘图区 ---
                # figsize设为(6, 5.5)，配合上面的文字，高度刚好能和左右两边对齐
                fig, ax = plt.subplots(figsize=(6, 5))

                # 绘制散点
                sns.scatterplot(x=y_true, y=y_pred, color="#4CAF50", alpha=0.7, s=80, ax=ax)

                # 绘制对角线
                min_val = min(min(y_true), min(y_pred))
                max_val = max(max(y_true), max(y_pred))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')

                ax.set_xlabel("真实值 (Actual)", fontsize=14, fontweight='bold')
                ax.set_ylabel("预测值 (Predicted)", fontsize=14, fontweight='bold')

                ax.tick_params(axis='x', labelsize=14, rotation=45)  # ax.tick_params 修改刻度大小、颜色等， ax.set_xlabel 修改x轴标题大小、颜色等
                ax.tick_params(axis='y', labelsize=14, rotation=0)

                # 标题里不再放特殊字符，防止出现方块
                ax.set_title("回归预测拟合图")

                ax.legend(loc='lower right')
                ax.grid(True, linestyle='--', alpha=0.5)

                st.pyplot(fig)

            else:
                # === 分类模型 ===
                from sklearn.metrics import confusion_matrix, accuracy_score

                acc = accuracy_score(y_true, y_pred)

                # 同样使用紧凑布局显示准确率
                st.markdown(
                    f"<div style='text-align: center; margin-bottom: 10px;'>"
                    f"<span style='font-size: 14px; font-weight: bold; color: #1976D2;'>分类准确率 (Accuracy) = {acc * 100:.1f}% </span>"
                    # f"<span style='font-size: 24px; font-weight: bold; color: #1976D2;'>{acc * 100:.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5))

                # 绘制热力图
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['难', '中', '易'],
                            yticklabels=['难', '中', '易'], cbar=False, ax=ax,
                            annot_kws={
                                "size": 24,  # 字号变大 (根据格子大小可微调)
                                "weight": "bold",  # 字体加粗
                                # "color": "white" # ⚠️ 建议注释掉这行，让它自动变色。如果不注释，浅色格子的数字会看不见。
                                }
                            )

                # --- 🎨 【关键修改】设置轴字体 ---
                # 1. 设置轴标题 (Label) 的大小和粗细
                ax.set_xlabel('预测类别 (Predicted)', fontsize=14, fontweight='bold')
                ax.set_ylabel('真实类别 (Actual)', fontsize=14, fontweight='bold')

                # 2. 设置刻度标签 (Tick: 难/中/易) 的大小
                ax.tick_params(axis='both', labelsize=12)

                # 3. 设置图表标题
                ax.set_title('混淆矩阵', fontsize=16, pad=20)


                # ax.set_xlabel('预测类别')
                # ax.set_ylabel('真实类别')
                # ax.set_title('混淆矩阵')

                st.pyplot(fig)

        # --- 第三列：SHAP 归因分析 ---
        with col_c:
            st.subheader("3. 关键因子分析")
            if st.button("🧬 计算 SHAP", key="btn_shap"):
                X_subset = X_train_df.iloc[:20, :]

                with st.spinner('正在解构模型逻辑...'):
                    try:
                        plt.clf()
                        if model_type == "classification":
                            # 分类模型 (SVM)
                            X_summary = shap.kmeans(X_train_df, 5)

                            def predict_fn(x):
                                if isinstance(x, pd.DataFrame): x = x.values
                                x_scaled = scaler.transform(x)
                                return target_model.predict_proba(x_scaled)

                            explainer = shap.KernelExplainer(predict_fn, X_summary)
                            shap_vals = explainer.shap_values(X_subset)
                            if isinstance(shap_vals, list):
                                sv = shap_vals[2]  # "易"
                            else:
                                sv = shap_vals
                            if len(sv.shape) == 3: sv = sv.sum(axis=2)
                        else:
                            # 回归模型 (RF) - 使用 TreeExplainer 超快
                            # 注意：RF 训练时用了 scaler，所以解释时也要传入 scaled 数据
                            X_subset_scaled = scaler.transform(X_subset.values)
                            explainer = shap.TreeExplainer(target_model)
                            sv = explainer.shap_values(X_subset_scaled)

                        # 绘图
                        shap.summary_plot(sv, X_subset, max_display=10, show=False)
                        st.pyplot(plt.gcf(), clear_figure=True)

                        if model_type == "regression":
                            st.success("✅ 分析完成")
                            st.markdown("**解读：**\n🔴 红点在右 = 该特征数值越大，预测值(如强度)越高。")
                        else:
                            st.success("✅ 分析完成 (针对'易')")
                            st.markdown("**解读：**\n🔴 红点在右 = 该特征数值越大，越容易成功。")

                    except Exception as e:
                        st.error(f"分析失败: {e}")

# ==========================================
# 🚀 启动入口 (逻辑分流)
# ==========================================
if __name__ == '__main__':
    # 1. 登录验证
    authenticator, name, username = login_system()

    # 2. 判断登录状态
    if st.session_state.get("authentication_status"):

        # === 🚦 分流路口 ===
        if username == 'admin':
            # 如果是管理员，进入后台
            admin_dashboard(username, authenticator)
        else:
            # 如果是普通用户，进入原来的系统
            main_application(username, name, authenticator)