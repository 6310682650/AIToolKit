from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



def index(request):
    return render(request, 'index.html')


def prepare_data(request):
    return render(request, 'prepare_data.html')


def show_dataset(request):
    dataset = None
    
    if request.method == 'POST':
        selected_dataset = request.POST.get('dataset')
        if selected_dataset == 'iris':
            dataset = import_iris_dataset()
        elif selected_dataset == 'titanic':
            dataset = import_titanic_dataset()
    
    context = {'dataset': dataset, 'selected_dataset': selected_dataset}
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


def train_model(request):
    if request.method == 'POST':
        dataset = request.POST.get('dataset')
        model = request.POST.get('model')
        
        if dataset == 'iris':
            file_path = 'ml_tool/datasets/iris.data.csv'
            column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
            dataset = pd.read_csv(file_path, names=column_names)
            X = dataset.drop('species', axis=1)
            y = dataset['species']
        elif dataset == 'titanic':
            file_path = 'ml_tool/datasets/titanic.csv'
            # column_names = ['pclass', 'survived', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body', 'home_dest']
            column_names = ['pclass', 'survived', 'sex', 'age', 'fare']
            dataset = pd.read_csv(file_path, names=column_names)

            X = dataset.drop('survived', axis=1)
            y = dataset['survived']
            
        if model == 'logistic_regression':
            clf = LogisticRegression()
        elif model == 'decision_tree':
            clf = DecisionTreeClassifier()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return render(request, 'train_result.html', {'accuracy': accuracy})
    
    return render(request, 'train.html')


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