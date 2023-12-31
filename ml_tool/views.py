from django.shortcuts import render
import pandas as pd
from django.core.paginator import Paginator
import csv
import time
import datetime
from django.utils import timezone
import os
from django.core.files.storage import FileSystemStorage
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from django.shortcuts import redirect
import json
from sklearn.preprocessing import StandardScaler
from .models import TrainedModelHistory


def index(request):
    return render(request, 'index.html')

def guide(request):
    return render(request, 'guide.html')

def prepare_data(request):
    uploaded_files = []

    if request.method == 'POST':
        if 'uploaded_file' in request.FILES:
            uploaded_file = request.FILES['uploaded_file']
            fs = FileSystemStorage(location='ml_tool/datasets/')
            filename = fs.save(uploaded_file.name, uploaded_file)
            uploaded_files.append(filename)
    
    # เช็คว่ามีไฟล์อัปโหลดที่เก็บใน session ไหม
    if 'uploaded_files' in request.session:
        uploaded_files.extend(request.session['uploaded_files'])
    
    # เก็บรายชื่อไฟล์อัปโหลดใน session
    request.session['uploaded_files'] = uploaded_files
            
    context = {
        'uploaded_files': uploaded_files,
    }

    if request.method == 'POST' and 'delete_file' in request.POST:
        filename_to_delete = request.POST['delete_file']
        
        if filename_to_delete in uploaded_files:
            uploaded_files.remove(filename_to_delete)
            fs = FileSystemStorage(location='ml_tool/datasets/')
            fs.delete(filename_to_delete)
            if 'uploaded_files' in request.session:
                request.session['uploaded_files'] = uploaded_files

    return render(request, 'prepare_data.html', context)





#this
def get_uploaded_dataset(file_name):
    file_path = os.path.join('ml_tool/datasets/', file_name)
    dataset = pd.read_csv(file_path)
    return dataset

def show_dataset(request):
    dataset = None
    selected_dataset = None
    
    if request.method == 'POST':
        selected_dataset = request.POST.get('dataset')
        if selected_dataset == 'iris':
            dataset = import_iris_dataset()
        elif selected_dataset == 'titanic':
            dataset = import_titanic_dataset()
        elif selected_dataset == 'winequality-white':
            dataset = import_winequality_dataset()
        elif selected_dataset == 'data':
            dataset = import_Possitive_and_Nagative_dataset()
    
    context = {'dataset': dataset,
               'selected_dataset': selected_dataset}
    return render(request, 'show_dataset.html', context)


def show_uploaded_dataset(request):
    uploaded_file_name = None
    dataset = None

    if request.method == 'POST':
        uploaded_file_name = request.POST.get('uploaded_file_name')
        if uploaded_file_name:
            file_path = f'ml_tool/datasets/{uploaded_file_name}'
            dataset = import_dataset(file_path)

    context = {
        'uploaded_file_name': uploaded_file_name,
        'dataset': dataset
    }

    return render(request, 'show_uploaded_dataset.html', context)


def import_iris_dataset():
    file_path = 'ml_tool/datasets/iris.data.csv'
    dataset = pd.read_csv(file_path, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
    dataset['dataset_name'] = 'iris'
    
    return dataset


def import_titanic_dataset():
    file_path = 'ml_tool/datasets/titanic.csv'
    dataset = pd.read_csv(file_path, names=['pclass', 'survived', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body', 'home_dest'])
    dataset['dataset_name'] = 'titanic'
    
    return dataset

def import_winequality_dataset():
    file_path = 'ml_tool/datasets/winequality-white.csv'
    dataset = pd.read_csv(file_path, names=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"], sep=';')
    dataset['dataset_name'] = 'winequality-white'
    
    return dataset

def import_Possitive_and_Nagative_dataset():
    file_path = 'ml_tool/datasets/data.csv'
    dataset = pd.read_csv(file_path, names=['text', 'lebel'])
    dataset['dataset_name'] = 'Possitive_and_Nagative'
    
    return dataset

def import_dataset(file_path):
    dataset = []
    
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            dataset.append(row)
    
    return dataset


label_encoder = LabelEncoder()
def train_model(request):
    global clf, feature_columns, label_encoder
    uploaded_files = request.session.get('uploaded_files', set())


    if request.method == 'POST':
        dataset_name = request.POST.get('dataset')
        model = request.POST.get('model')

        if dataset_name in uploaded_files:
            dataset = get_uploaded_dataset(dataset_name)
            target_column = dataset.columns[0]
            X = dataset.drop(target_column, axis=1)
            y = dataset[target_column]


        elif dataset_name == 'iris':
            file_path = 'ml_tool/datasets/iris.data.csv'
            column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
            dataset = pd.read_csv(file_path, names=column_names)
            X = dataset.drop('species', axis=1)
            y = dataset['species']
            
        elif dataset_name == 'winequality-white':
            file_path = 'ml_tool/datasets/winequality-white.csv'
            column_names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
            dataset = pd.read_csv(file_path, sep=';', names=column_names)
            dataset = dataset.applymap(lambda x: x.split(',')[0] if isinstance(x, str) else x)
    
            X = dataset.drop('quality', axis=1)
            y = dataset['quality']

        elif dataset_name == 'Possitive_and_Nagative':
            file_path = 'ml_tool/datasets/data.csv'
            column_names = ['text', 'lebel']
            dataset = pd.read_csv(file_path, names=column_names)
            X = dataset.drop('lebel', axis=1)
            y = dataset['lebel']

        elif dataset_name == 'titanic':
            file_path = 'ml_tool/datasets/titanic.csv'
            column_names = ['pclass', 'survived', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body', 'home_dest']
            dataset = pd.read_csv(file_path, names=column_names)
            
            X = dataset.drop('survived', axis=1)
            y = dataset['survived']

            
        if model == 'logistic_regression':
            clf = LogisticRegression()
        elif model == 'decision_tree':
            clf = DecisionTreeClassifier()
        elif model == 'naive_bayes':
            clf = MultinomialNB()
        if model == 'random_forest':
            clf = RandomForestClassifier()

        # show_result_button = request.POST.get('show_result_button')

        # แปลงเป็นตัวเลข
        
        for column in X.columns:
            X[column] = label_encoder.fit_transform(X[column])
        y = label_encoder.fit_transform(y)

        scaling_input = float(request.POST.get('scaling', 0))
        scaling = 1 - (scaling_input/100)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=scaling, random_state=42)


        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        timestamp = timezone.now()

        trained_model = TrainedModelHistory.objects.create(dataset=dataset_name, model=model, accuracy=accuracy, timestamp=timestamp, train_scale=scaling_input)


        y_pred = label_encoder.inverse_transform(y_pred)
        y_test = label_encoder.inverse_transform(y_test)
        
        
        result = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
        feature_columns = X.columns.tolist()

        for column in feature_columns:
            result[column] = X_test[column].tolist()
        result = result.sort_index()
        
        
    
        return render(request, 'train_result.html', {'accuracy': accuracy, 'result': result.to_html()})
    
    train_history = TrainedModelHistory.objects.all()

    return render(request, 'train.html',{'uploaded_files': uploaded_files,'train_history': train_history})

def delete_history(request, history_id):
    history = TrainedModelHistory.objects.get(id=history_id)
    history.delete()
    return redirect('train')


def delete_all_history(request):
    TrainedModelHistory.objects.all().delete()
    return redirect('train')

def predict_model(request):
    global feature_columns, label_encoder

    if request.method == 'POST':
        input_features = {}
        for column in feature_columns:
            feature_value = request.POST.get(column)
            input_features[column] = feature_value

        is_numeric_data = all(value.isdigit() for value in input_features.values())

        if not is_numeric_data:
            for column in feature_columns:
                if input_features[column] not in label_encoder.classes_:
                    input_features[column] = 0
                else:
                    input_features[column] = label_encoder.transform([input_features[column]])[0]

        input_data = pd.DataFrame([input_features])
        if not input_data.empty:
            predicted_label = label_encoder.inverse_transform(clf.predict(input_data))[0]
        else:
            predicted_label = None

        return render(request, 'predict.html', {'feature_columns': feature_columns, 'predicted_label': predicted_label})

    return render(request, 'predict.html', {'feature_columns': feature_columns})


def predict_modell(request):
    global feature_columns, label_encoder
    
    if request.method == 'POST':
        input_features = {}
        for column in feature_columns:
            feature_value = request.POST.get(column)
            input_features[column] = feature_value

        input_data = pd.DataFrame([input_features])
        if not input_data.empty:
            predicted_label = label_encoder.inverse_transform(clf.predict(input_data))[0]
        else:
            predicted_label = None

        return render(request, 'predict.html', {'feature_columns': feature_columns, 'predicted_label': predicted_label})

    return render(request, 'predict.html', {'feature_columns': feature_columns})


# บรรทัดหลังจากนี้ไม่เกี่ยว ลองทำแยกออกมา

def machine_learning_demo(request):
    file_path = 'ml_tool/datasets/iris.data.csv'
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    dataset = pd.read_csv(file_path, names=column_names)

    X = dataset.drop('species', axis=1)
    y = dataset['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = (y_pred == y_test).mean()

    data = {
        'accuracy': accuracy,
        'predictions': zip(X_test.values, y_pred),
        'true_species': y_test
    }

    return render(request, 'machine_learning_demo.html', data)



def train_and_predict(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text')

        # data = {'text': ['I love this movie', 'This movie is great', 'I hate this movie', 'This movie is terrible'],
        #         'label': ['positive', 'positive', 'negative', 'negative']}
        data = {'text': ['ฉันชอบหนังเรื่องนี้', 'หนังเรื่องนี้สุดยอด', 'เธอเก่งมาก', 'รักนะ', 'เธอดีที่สุด', 'ชอบเธอ', 
                         'ฉันเกลียดหนังเรื่องนี้', 'หนังเรื่องนี้มันแย่มาก', 'ห่วยมาก', 'เกลียดเธอ', 'เธอแย่ที่สุด', 'น่าเกลียด'
                         'I love this movie', 'This movie is great', 'I hate this movie', 'This movie is terrible'
                         ],
                'label': ['positive', 'positive', 'positive', 'positive', 'positive', 'positive', 
                          'negative', 'negative', 'negative', 'negative', 'negative', 'negative'
                          'positive', 'positive', 'negative', 'negative'
                          ]}
        
        df = pd.DataFrame(data)

        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.1, random_state=42)

        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        clf = MultinomialNB()
        clf.fit(X_train_vec, y_train)

        input_vec = vectorizer.transform([input_text])
        predicted_label = clf.predict(input_vec)[0]

        return render(request, 'train_and_predict.html', {'input_text': input_text, 'predicted_label': predicted_label})

    return render(request, 'train_and_predict.html')
