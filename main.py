import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
import os

data = pd.read_csv('C:/Users/User/Downloads/fashion_data.csv')

english_columns = {
    'Product Name': 'Наименование_продукта',
    'Price': 'Цена',
    'Brand': 'Бренд',
    'Category': 'Категория',
    'Description': 'Описание',
    'Rating': 'Рейтинг',
    'Style Attributes': 'Атрибуты_стиля',
    'Size': 'Размер',
    'Color': 'Цвет',
    'Purchase History': 'История_покупок',
    'Average Age Of Clients': 'Средний_возраст_клиентов',
    'Fashion Magazines': 'Модные_журналы',
    'Fashion Influencers': 'Модные_инфлюенсеры',
    'Season': 'Сезон',
    'Time Period Highest Purchase': 'Период_наибольшего_количества_покупок',
    'Customer Reviews': 'Отзывы_клиентов',
    'Social Media Comments': 'Комментарии_в_социальных_сетях',
    'Feedback': 'Обратная_связь'
}


data = data.rename(columns= english_columns)
data.drop('Наименование_продукта', axis=1, inplace=True)

print(data.head())

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Разделение данных на обучающую и тестовую выборки
X = data.drop(['Рейтинг'], axis=1)
y = data['Рейтинг']

# Преобразование категориальных данных
encoder = TargetEncoder()
X_cat = X.select_dtypes(include='object').astype(str)
encoder.fit(X_cat, y)
X_cat_encoded = encoder.transform(X_cat)
X_cat_encoded = pd.DataFrame(X_cat_encoded, columns=encoder.get_feature_names_out(X_cat.columns))

# Сохранение имен столбцов для последующего использования
column_names = X.columns.tolist()
categorical_columns = X_cat.columns.tolist()
numerical_columns = X.select_dtypes(include='number').columns.tolist()

# Нормализация числовых данных
scaler = StandardScaler()
X_num = X.select_dtypes(include='number')
X_num_scaled = scaler.fit_transform(X_num)
X_num_scaled = pd.DataFrame(X_num_scaled, columns=numerical_columns)

# Объединение нормализованных числовых данных и преобразованных категориальных данных
X = pd.concat([X_num_scaled, X_cat_encoded], axis=1)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Определение архитектуры нейросети
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
history = model.fit(X_train, y_train, epochs=27, validation_split=0.2)

# Оценка качества модели
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'R2: {r2}')

# Создание приложения Flask
app = Flask(__name__)


# Определение маршрута для главной страницы
@app.route('/')
def home():
    return render_template('index.html', data=data)


# Определение маршрута для обработки данных
@app.route('/predict', methods=['POST'])
def predict():
    # Получение данных из формы
    form_data = {col: request.form.get(col) for col in column_names}

    # Преобразование данных в DataFrame
    df = pd.DataFrame(form_data, index=[0])

    # Преобразование категориальных данных
    df_cat = df[categorical_columns].astype(str)
    df_cat_encoded = encoder.transform(df_cat)
    df_cat_encoded = pd.DataFrame(df_cat_encoded, columns=encoder.get_feature_names_out(df_cat.columns))

    # Нормализация числовых данных
    df_num = df[numerical_columns]
    df_num_scaled = scaler.transform(df_num)
    df_num_scaled = pd.DataFrame(df_num_scaled, columns=numerical_columns)

    # Объединение нормализованных числовых данных и преобразованных категориальных данных
    df_processed = pd.concat([df_num_scaled, df_cat_encoded], axis=1)

    # Предсказание с помощью модели
    prediction = model.predict(df_processed)
    prediction_clipped = np.clip(prediction, 0, 5)

    # Передача данных в шаблон
    return render_template('index.html', data=data, prediction=prediction_clipped[0][0])


if __name__ == '__main__':
    app.run(debug=False)