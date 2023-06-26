from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from django.core.paginator import Paginator


def index(request):
    return render(request, 'index.html')


def prepare_data(request):
    return render(request, 'prepare_data.html')


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

    # paginator = Paginator(dataset, 10)

    # page_number = request.GET.get('page')
    # page_obj = paginator.get_page(page_number)

    context = {'dataset': dataset,
               'selected_dataset': selected_dataset}
    return render(request, 'show_dataset.html', context)


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


def train_model(request):
    global clf, feature_columns
    if request.method == 'POST':
        dataset = request.POST.get('dataset')
        model = request.POST.get('model')
        
        if dataset == 'iris':
            file_path = 'ml_tool/datasets/iris.data.csv'
            column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
            dataset = pd.read_csv(file_path, names=column_names)
            X = dataset.drop('species', axis=1)
            y = dataset['species']
        elif dataset == 'winequality-white':
            file_path = 'ml_tool/datasets/winequality-white.csv'
            column_names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
            dataset = pd.read_csv(file_path, sep=';', names=column_names)
            dataset = dataset.applymap(lambda x: x.split(',')[0] if isinstance(x, str) else x)
    
            X = dataset.drop('quality', axis=1)
            y = dataset['quality']

        # elif dataset == 'titanic':
        #     file_path = 'ml_tool/datasets/titanic.csv'
        #     column_names = ['pclass', 'survived', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body', 'home_dest']
        #     dataset = pd.read_csv(file_path, names=column_names)
            
        #     X = dataset.drop('survived', axis=1)
        #     y = dataset['survived']
            
        if model == 'logistic_regression':
            clf = LogisticRegression()
        elif model == 'decision_tree':
            clf = DecisionTreeClassifier()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        result = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
        feature_columns = X.columns.tolist()

        for column in feature_columns:
            result[column] = X_test[column].tolist()
        result = result.sort_index()
        
        return render(request, 'train_result.html', {'accuracy': accuracy, 'result': result.to_html()})
    
    return render(request, 'train.html')

def predict_model(request):
    global feature_columns
    if request.method == 'POST':
        input_features = {}
        for column in feature_columns:
            feature_value = request.POST.get(column)
            input_features[column] = feature_value

        input_data = pd.DataFrame([input_features])
        if not input_data.empty:
            predicted_label = clf.predict(input_data)[0]
        else:
            predicted_label = None

        return render(request, 'predict.html', {'feature_columns': feature_columns, 'predicted_label': predicted_label})

    return render(request, 'predict.html', {'feature_columns': feature_columns})



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