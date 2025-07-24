import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Project 5", layout="wide")

st.sidebar.title("--***Navigation***--")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Homepage",
        "💱 Currency Converter",
        "📚 Supervised",
        "🧩 Unsupervised"
    ]
)

if page == "🏠 Homepage":
    st.title("✨---***Homepage***---✨")
    st.header("📚 Machine Learning Notes :-")
    st.markdown("---")
    st.subheader("🤖 What is Machine Learning?")
    st.markdown("""
    **Machine Learning (ML)** is a subset of *Artificial Intelligence (AI)* that enables computers to:
    - 🧠 **Learn patterns from data**
    - 🔮 **Make decisions or predictions** without being explicitly programmed
    
    **Types of Machine Learning:**
    - **Supervised:** Building models from input-output data pairs
    - **Unsupervised:** Learning structure from unlabeled data
    - **Reinforcement:** Improving actions based on feedback
    """)
    st.markdown("---")

    st.subheader("🤖 **Types of Machine Learning** :-")
    Model_type = st.selectbox('***Select Your Type***',('None','Supervised', 'Unsupervised', 'Reinforcement'))
    if Model_type == 'Supervised':
        st.header("📘***Supervised Learning***📘")
        st.subheader("🔹**Definition** :- ")
        st.write("Supervised Learning is a type of ML where the algorithm learns from labeled data, meaning both input (X) and output (Y) are provided. The model maps inputs to outputs using classification or regression.")
        st.markdown("---")
        st.subheader("🔹**Techniques** :- ")
        st.markdown("1. **Classification** :- Predicting discrete classes (e.g., spam/not spam)")
        st.markdown("2. **Regression** :- Predicting continuous values (e.g., house prices)")
        st.markdown("---")
        st.subheader("🔹**Algorithm** :- ")
        st.markdown("1. **Linear Regression** :- Finding best-fit line between X and Y")
        st.markdown("2. **Logistic Regression** :- Predicting binary outcomes (e.g., yes/no)")
        st.markdown ("3. **Decision Trees** :- Making decisions based on feature values")
        st.markdown ("4. **Random Forest** :- Combining multiple decision trees")
        st.markdown ("5. **K-Nearest Neighbors** :- Classifying based on nearest neighbors")
    elif Model_type == 'Unsupervised':
        st.header("📙***Unsupervised Learning***📙")
        st.subheader("🔸**Definition** :- ")
        st.write("Unsupervised Learning involves training a model on unlabeled data, where no output variable is provided. The goal is to find hidden patterns or groupings in data..")
        st.markdown("---")
        st.subheader("🔸**Techniques** :- ")
        st.markdown("1. **Clustering** :- Grouping similar data points together (e.g., customer segmentation)")
        st.markdown("---")
        st.subheader("🔸**Algorithm** :- ")
        st.markdown("1. **K-Means Clustering** :- Partitioning data into K distinct clusters")
        st.markdown("2. **DBSCAN** :- Density-based clustering")
    elif Model_type == 'Reinforcement':
        st.header("📒***Reinforcement Learning***📒")
        st.subheader("🔸**Definition** :- ")
        st.write("Reinforcement Learning involves training an agent to make decisions in an environment by rewarding desired behaviors and penalizing undesired ones.")
        st.markdown("---")
        st.subheader("🔸**Techniques** :- ")
        st.markdown("1. **Q-Learning** :- Learning optimal actions using a Q-value table")
        st.markdown("2. **SARSA**(State-Action-Reward-State-Action):- Learning from experience with immediate rewards")
        st.markdown("---")

    st.subheader("🛠️**Types of Algorithms** :-")
    Algo_type = st.selectbox('***Select Your Algorithm***',('None','Linear Regression', 'Logistic Regression', 'Random Forest', 'K-Nearest Neighbors (KNN)', 'Decision Trees', 'KMeans Clustering'))
    if Algo_type == 'Linear Regression':
        st.header("📈***Linear Regression***📈")
        st.write("A supervised algorithm used for predicting continuous output using a best-fit line (y = mx + c).")
        st.markdown("---")
        st.subheader("◽**Graph** :- ")
        st.image(r"C:\Users\Chinmay\OneDrive\Desktop\Internship\Python\Projects\StreamML-Studio\1.webp")  
        st.markdown("---")
        st.subheader("◽**Python Code** :- ")
        st.code('''
        from sklearn.linear_model import LinearRegression

        # Create model
        model = LinearRegression()

        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        ''')
    elif Algo_type == 'Logistic Regression':
        st.header("✏️***Logistic Regression***✏️")
        st.write("Used for classification problems where output is binary (0 or 1). It uses a sigmoid function to predict probabilities between 0 and 1.")
        st.markdown("---")
        st.subheader("🔸**Graph** :- ")
        st.image(r"C:\Users\Chinmay\OneDrive\Desktop\Internship\Python\Projects\StreamML-Studio\2.png") 
        st.markdown("---")
        st.subheader("🔸**Python Code** :- ")
        st.code('''
        from sklearn.linear_model import LogisticRegression

        # Create model
        model = LogisticRegression()

        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        ''')
    elif Algo_type == 'Decision Trees':
        st.header("🌳***Decision Tree***🌳")
        st.write("A tree-based supervised learning algorithm used for both classification and regression. It splits the dataset based on feature values to make predictions")
        st.markdown("---")
        st.subheader("🔰**Graph** :- ")
        st.image(r"C:\Users\Chinmay\OneDrive\Desktop\Internship\Python\Projects\StreamML-Studio\3.webp") 
        st.markdown("---")
        st.subheader("🔰**Python Code** :- ")
        st.code("""from sklearn.tree import DecisionTreeClassifier
        from sklearn.tree import plot_tree
        import matplotlib.pyplot as plt

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        plot_tree(model)
        plt.show()
        """)
    elif Algo_type == 'K-Nearest Neighbors (KNN)':
        st.header("***🔍K-Nearest Neighbors (KNN)🔍***")
        st.write("A classification algorithm that assigns a label based on the majority class of the k nearest points in feature space.")
        st.markdown("---")
    elif Algo_type == 'Random Forest':
        st.header("🌳***Random Forest***🌳")
        st.write("An ensemble method that builds multiple Decision Trees and combines their outputs. Used for both classification and regression tasks.")
        st.markdown("---")
    elif Algo_type == 'KMeans Clustering':
        st.header("🎯***KMeans Clustering***🎯")
        st.write(" A clustering algorithm that partitions data into k clusters by assigning points to the nearest cluster centroid and updating centroids iteratively.")
        st.markdown("---")

elif page == "💱 Currency Converter": 
    st.title("💱✨ Currency Converter ✨💱")
    st.markdown("---")
    
    currencies = ["🇮🇳 INR", "$ USD", "🇦🇪 Dirams", "🇪🇺 Euro", "🇬🇧 Pounds"]
    
    st.subheader("📝 **Input** :")
    input_currency = st.selectbox("🌐 Select Your Currency Type", currencies, key='input_currency_selectbox')
    amount = st.number_input("💰 Enter Your Amount", min_value=0.0, format="%f")
    
    st.markdown("---")
    
    st.subheader("🔄 **Output** :")
    output_currency = st.selectbox("🌐 Select Output Currency Type", currencies, key='output_currency_selectbox')
    
    currency_map = {"🇮🇳 INR": "INR", "$ USD": "USD", "🇦🇪 Dirams": "Dirams", "🇪🇺 Euro": "EUR", "🇬🇧 Pounds": "GBP"}
    in_cur = currency_map[input_currency]
    out_cur = currency_map[output_currency]
    
    rates = {
        ("INR", "USD"): 1/83,
        ("INR", "Dirams"): 1/22.6,
        ("INR", "EUR"): 1/89.5,
        ("INR", "GBP"): 1/105,
        ("USD", "INR"): 83,
        ("USD", "Dirams"): 3.67,
        ("USD", "EUR"): 0.92,
        ("USD", "GBP"): 0.79,
        ("Dirams", "INR"): 22.6,
        ("Dirams", "USD"): 1/3.67,
        ("Dirams", "EUR"): 0.25,
        ("Dirams", "GBP"): 0.21,
        ("EUR", "INR"): 89.5,
        ("EUR", "USD"): 1.09,
        ("EUR", "Dirams"): 4,
        ("EUR", "GBP"): 0.86,
        ("GBP", "INR"): 105,
        ("GBP", "USD"): 1.27,
        ("GBP", "Dirams"): 4.76,
        ("GBP", "EUR"): 1.16,
    }
    
    if in_cur == out_cur:
        converted = amount
    else:
        converted = amount * rates.get((in_cur, out_cur), 1)
    
    st.text_input("💸 Converted Amount", value=str(round(converted, 4)), disabled=True)
    st.markdown("---")
    st.markdown("> 🌟 *Easily convert between INR, USD, Dirams, Euro, and Pounds!* 🌟")

elif page == "📚 Supervised":
    st.title("✨***Supervised Learning***✨")
    st.subheader("Example Dataset (Salary_Dataset.csv)")
    
    try:
        df = pd.read_csv(r"C:\Users\Chinmay\OneDrive\Desktop\Internship\Python\Projects\StreamML-Studio\Salary_dataset.csv")
        for col in df.columns:
            if col.startswith('Unnamed'):
                df = df.rename(columns={col: 'Id_No'})
        st.write(df.head(10))
    except FileNotFoundError:
        st.error("Dataset file not found. Please update the file path or upload a dataset.")
        st.stop()
    
    st.markdown("---")
    st.header("🧮***Data Analysis***🧮")
    st.subheader("📊 Describe")
    st.write(df.describe())
    st.subheader("ℹ️ Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
    st.subheader("🏷️ Columns")
    st.write(df.columns)
    st.markdown("---")

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error

    x = df[[col for col in df.columns if 'Experience' in col or 'experience' in col][0]].to_numpy().reshape(-1, 1)
    y = df[[col for col in df.columns if 'Salary' in col or 'salary' in col][0]].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'KNN': KNeighborsRegressor()
    }

    mse_scores = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores[name] = mse

    fig, ax = plt.subplots()
    ax.bar(list(mse_scores.keys()), list(mse_scores.values()), color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2'])
    ax.set_ylabel('MSE')
    ax.set_title('Mean Squared Error of Different Models')
    st.pyplot(fig)
    st.markdown("---")

    st.header("***Pickling*** 🥒")
    st.write("You can use the best model (lowest MSE) or upload your own model and data for prediction.")

    best_model_name = min(mse_scores, key=lambda k: mse_scores[k])
    st.write(f"Best model based on MSE: **{best_model_name}** (MSE: {mse_scores[best_model_name]:.2f})")
    best_model = models[best_model_name]

    import pickle as pkl

    uploaded_pkl = st.file_uploader("Upload your own Pickle model for prediction (optional)", type=["pkl"], key='pickle_upload')
    if uploaded_pkl is not None:
        user = pkl.load(uploaded_pkl)
        st.success("Custom model loaded!")
    else:
        user = uploaded_pkl
        st.info("Using current Best model for prediction.")
    
    if st.button("Save Best Model as Pickle"):
        with open("best_model.pkl", "wb") as f:
            pkl.dump(best_model, f)
        st.success("Best model saved as best_model.pkl!")

    st.markdown("---")
    st.header("🔮 Make a Prediction")
    exp_col_selected = [col for col in df.columns if 'Experience' in col or 'experience' in col][0]
    sal_col_selected = [col for col in df.columns if 'Salary' in col or 'salary' in col][0]
    user_exp = st.number_input(f"Enter value for prediction (from column: {exp_col_selected})", min_value=float(df[exp_col_selected].min()), max_value=float(df[exp_col_selected].max()), format="%f", key='predict_exp_input')
    predict_btn = st.button("Predict", key='predict_btn')
    if predict_btn:
        pred_salary = best_model.predict(np.array([[user_exp]]))[0]
        st.success(f"Predicted Value: {pred_salary:.2f} (from column: {sal_col_selected})")

elif page == "🧩 Unsupervised":
    st.title("✨***Unsupervised Learning***✨")
    st.subheader("Example Dataset (Mall_Customers.csv)")
    
    try:
        df_unsup = pd.read_csv(r"C:\Users\Chinmay\OneDrive\Desktop\Internship\Python\Projects\StreamML-Studio\Mall_Customers.csv")
        st.write(df_unsup.head(10))
    except FileNotFoundError:
        st.error("Dataset file not found. Please update the file path or upload a dataset.")
        st.stop()
    
    st.markdown("---")
    st.header("🧮***Data Analysis***🧮")
    st.subheader("📊 Describe")
    st.write(df_unsup.describe())
    st.subheader("ℹ️ Info")
    buffer = io.StringIO()
    df_unsup.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
    st.subheader("🏷️ Columns")
    st.write(df_unsup.columns)
    st.markdown("---")

    st.header("KMeans Clustering & Elbow Method 🟣")
    from sklearn.cluster import KMeans

    st.subheader("🔢 Select Features for Clustering")
    feature_options = list(df_unsup.select_dtypes(include=[np.number]).columns)
    selected_features = st.multiselect("Select features (at least 2 recommended)", feature_options, default=feature_options[:2])

    if len(selected_features) >= 1:
        X = df_unsup[selected_features].to_numpy()
    
        wcss = []
        max_k = st.slider("Select max number of clusters to test (k)", min_value=3, max_value=15, value=10)
    
        for i in range(1, max_k+1):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
    
        fig, ax = plt.subplots(figsize=(10, 6)) 
        ax.plot(range(1, max_k+1), wcss, marker='o')
        ax.set_title('Elbow Method for Optimal k')
        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel('WCSS')
    
        st.pyplot(fig)
    else:
        st.warning("Please select at least one feature for clustering.")

    st.markdown("---")
    st.header("***Pickling (KMeans)*** 🥒")
    st.write("You can save the KMeans model for the selected k as a pickle file, or upload your own KMeans pickle for cluster prediction.")

    n_clusters = st.number_input("Select number of clusters for pickling (k)", min_value=1, max_value=20, value=3, step=1, key='kmeans_pickle_k')
    kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans_model.fit(df_unsup[selected_features].to_numpy())

    import pickle as pkl
    if st.button("Save KMeans Model as Pickle"):
        with open("kmeans_model.pkl", "wb") as f:
            pkl.dump(kmeans_model, f)
        st.success("KMeans model saved as kmeans_model.pkl!")

    uploaded_pkl = st.file_uploader("Upload your own KMeans Pickle model for cluster prediction (optional)", type=["pkl"], key='kmeans_pkl_upload')
    if uploaded_pkl is not None:
        user_kmeans = pkl.load(uploaded_pkl)
        st.success("Custom KMeans model loaded!")
    else:
        user_kmeans = kmeans_model
        st.info("Using current KMeans model for prediction.")

    st.markdown("---")
    st.header("🔮 Predict Cluster Assignment")
    input_vals = []
    for feat in selected_features:
        val = st.number_input(f"Enter value for {feat}", min_value=float(df_unsup[feat].min()), max_value=float(df_unsup[feat].max()), format="%f", key=f'input_{feat}')
        input_vals.append(val)
    predict_btn = st.button("Predict Cluster", key='predict_cluster_btn')
    if predict_btn:
        cluster = user_kmeans.predict([input_vals])[0]
        st.success(f"Predicted Cluster: {cluster}")