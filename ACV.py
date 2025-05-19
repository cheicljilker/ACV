#code here to be copied into your CS code streamlit app
# (Don't run in colab)

print ('salut')
## STEP 1 : Libraries and initialisation
import streamlit as st
import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import  Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import plotly.express as px


# STEP 2 : Streamlit configuration
st.set_page_config(page_title="ACV Bâtiments E+C-", layout="wide")
st.title("Analyse du Cycle de Vie – Base E+C-")
st.markdown("Ce tableau de bord interactif guide les étudiants à travers chaque étape du notebook original.")


# STEP 3 : Data loading
st.title("Step 3 - Importation de la base E+C-")
uploaded_file = st.file_uploader("Importez le fichier Excel E+C- (feuille 'batiments')", type=["xlsx"])
df_raw = pd.read_excel(uploaded_file, sheet_name='batiments', header=[0,1])

#dealing with multi-index
df_raw.columns = df_raw.columns.droplevel(0)
st.dataframe(df_raw.head(3))
st.subheader("Statistiques descriptives")
st.write(df_raw.describe())

# STEP 4 : Initial scatter plot
st.title("step 4 - Nuage de points")
st.write("Tous les bâtiments sont représentés dans le diagramme ci-dessous. ")
fig = px.scatter(df_raw, x=df_raw['id_batiment'], y=df_raw['eges'])
st.plotly_chart(fig, use_container_width=True)

# STEP 4b : BASIC data cleaning
st.subheader("step 4b - Nettoyage de base des données")
df_raw = df_raw.drop((df_raw[df_raw["eges"]>2500]).index)
df_raw= df_raw.drop((df_raw[df_raw["eges"]<500]).index)
st.dataframe(df_raw)
fig = px.scatter(df_raw, x=df_raw['id_batiment'], y=df_raw['eges'])
st.plotly_chart(fig, use_container_width=True)

# STEP 7: Categorical Correlation Analysis
st.title("STEP 7: Categorical Correlations")
#Generation dummies to be able to carry out correlation with categorical features
st.markdown('Creating dummies for categorical features')
df_d= pd.get_dummies(df_features_in)
st.dataframe(df_d.head(10))

st.markdown("features with correlation to HIGH eges:")
st.write(df_d.corrwith(df_raw["eges"]).sort_values(ascending=False)[:5])
st.markdown("features with correlation to LOW eges:")
st.write(df_d.corrwith(df_raw["eges"]).sort_values()[:5])

# STEP 8: Modeling Preparation : Smaller dataframe with limited number of features
st.title("STEP 8: Modeling Preparation")
""## Creating dataframe with limited number of features"""

# 8a.  select target
st.subheader("Selection des variables a étudier")
target_name = "eges"
target_default = "eges"
target_name = st.selectbox(label="Objectif empreinte carbone :", options = features_target_col, index=features_target_col.index(target_default) )
target_col = [target_name ]

#8b. select features
features_default =[
    'type_plancher',
    'materiau_principal',
    "vecteur_energie_principal_ecs",
    "vecteur_energie_principal_ch",
    "nb_niv_surface",
    ]
features_col= st.multiselect(label="Characteristique du bâtiment", options = sorted(features_input_col), default=features_default )

#8c. Create modeling DataFrame
st.markdown("DataFrame de modélisation - réduit")
df = pd.DataFrame(df_raw, columns=target_col + features_col)
st.dataframe(df.head(5))
st.write('Nombre de batiments : ', df.shape[0])


# STEP 9: Outlier Treatment & Encoding
st.title("STEP 9: Outlier Treatment & Encoding")
original_len = len(df)

# STEP9a - Dealing with Nan - Imputs format"""
df = df.fillna(df.mode(numeric_only=True).iloc[0])   #Replacing NaN of numeric columns with  the most frequent value

#Step 9b - Removing outliners (Categorical)
categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(df)
st.markdown(f"categorical columns: {categorical_columns}")
df[categorical_columns] = df[categorical_columns].astype(str)

outliner_threshold = st.slider("Seuil regroupenent des variables avec peu d'occurences (divers) (%)=", 0, 10, value=4, step=1) / 100
feature_studied = features_col[2]
for feature in categorical_columns:
  df[feature] = df[feature].mask(df[feature].map(df[feature].value_counts(normalize=True)) < outliner_threshold, 'Other')
df[[feature_studied , target_name]].groupby([feature_studied], as_index=False).mean().round().sort_values(by=target_name, ascending=False)

# STEP 9c - Removing outliners (Numerical)
numeric_columns_selector = selector(dtype_include=np.number)
numerical_features = numeric_columns_selector(df)
st.markdown(f"Numerical columns: {numerical_features}")

FilteringQuantile  = st.slider("Seuil elimination valeurs aberrantes en dehors distribution (%)=",1, 10, value=0, step=1) /100
for column in numerical_features:
    q_low = df[column].quantile(FilteringQuantile)
    q_hi  = df[column].quantile(1-FilteringQuantile)
    df = df[(df[column] < q_hi) & (df[column] > q_low)]

final_len = len(df)
st.markdown(final_len)
st.markdown(f"number of rows removed = {original_len-final_len}")


#step 10 - correlation plots
st.title("STEP 10: Correlation plots")

ord_enc = OrdinalEncoder()
categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(df)

#step10a - pair plot with categorical features
st.write(categorical_columns)
for col in features_col:
    df[col+"_enc"] = ord_enc.fit_transform(df[[col]])
fig2 = sns.pairplot(df, hue=feature_studied, height=3)     # (diag kws; required for binary 0/1 inputs) ", diag_kws={'bw': 0.2}
st.pyplot(fig2, use_container_width=True)

# step10b Categorical plots (swarm and violin)
feature_studied = features_col[0]
hue_col = features_col[2]
fig3 = sns.catplot(x=feature_studied, y=target_name, hue=hue_col, kind="swarm", data=df, height = 10)  #'Type de plancher' "Matériau principal"
st.pyplot(fig3, use_container_width=True)
fig4 = sns.catplot(x=feature_studied, y=target_name,  kind="violin", data=df, height = 8, aspect=2)  #'Type de plancher' "Matériau principal"
st.pyplot(fig4, use_container_width=True)



#STEP 11: Machine learning model
st.title("STEP 11: Basic machine learning model  - Sklearn")
# preprocessing of data (label encoder, scaling)
featuresCol = features_col
target_col = target_col #
data = pd.DataFrame(df, columns=featuresCol+target_col)

st.markdown("data")
st.dataframe(data.head())

#Encoding of categorical datas
X_df = data.drop(target_col, axis=1)
encoder = OneHotEncoder(handle_unknown='ignore')   #or OrdinalEncoder()
data_encoded = encoder.fit_transform(X_df)
data_encoded[:5]
X = data_encoded

#scaling of y
Y_df = pd.DataFrame(data[target_col])
scaler = MinMaxScaler()
Y_df[target_col] = scaler.fit_transform(Y_df[target_col])
Y = np.array(Y_df[target_col])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

#Fitting simple Ridge model and calculting score
reg = Ridge()
reg.fit(X_train, Y_train)
accuracy = reg.score(X_test, Y_test)
st.write("score Ridge =",accuracy)



#step 12 - prediction of combination with low and high carbon footpring
st.title("STEP 12: Prediction of combination with low and high carbon footprint")

data_pred = df[featuresCol+target_col]
X_df_test = df[featuresCol].drop_duplicates()
X_df_test.reset_index(drop=True, inplace=True)
X_pred = encoder.transform(X_df_test)
Y_pred = reg.predict(X_pred).reshape(-1,1)
Y_pred = scaler.inverse_transform(Y_pred)

df_prediction = X_df_test.copy()
df_prediction["eges [KgCO2/m²] Prediction"] = pd.DataFrame(Y_pred).round()
df_prediction["eges [KgCO2/m²] Prediction"] = df_prediction["eges [KgCO2/m²] Prediction"]
df_prediction = df_prediction.sort_values(by=["eges [KgCO2/m²] Prediction"])
df_prediction.reset_index(drop=True, inplace=True)

st.markdown("Prediction of eges (LOW)")
st.dataframe(df_prediction.head(10))
st.markdown("Prediction of eges (HIGH)")
st.dataframe(df_prediction.tail(10))



