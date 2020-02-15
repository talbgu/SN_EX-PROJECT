from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import random

def generate_edges(edges):

    edges_index = edges[['from', 'to']]
    edges_dict = dict()

    for index, row in edges_index.iterrows():
        key = str(row['from']) + '_' + str(row['to'])
        edges_dict[key] = True

    from_set = list(set(edges['from'].values))
    to_set = list(set(edges['to'].values))

    for i in range(len(edges)):
        from_index = random.randint(0, len(from_set) - 1)
        to_index = random.randint(0, len(to_set) - 1)
        key = str(from_set[from_index]) + '_' + str(to_set[to_index])
        while(key in edges_dict.keys()):
            from_index = random.randint(0, len(from_set) - 1)
            to_index = random.randint(0, len(to_set) - 1)
            key = str(from_set[from_index]) + '_' + str(to_set[to_index])
        edges = edges.append({'from': from_index, 'to': to_index, 'true': 0}, ignore_index=True)

    return edges


def normalize_column(nodes, column_name):
    scaler = MinMaxScaler()
    scaler.fit(nodes[column_name].values)
    nodes[column_name] = scaler.transform(nodes[column_name].values)
    return nodes

def normalize_data(data, columns):
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data


def preprocess(edges_path, nodes_path):
    global X_train, y_train, X_test, y_test
    edges = pd.DataFrame.from_csv(edges_path, index_col=None)
    nodes = pd.DataFrame.from_csv(nodes_path, index_col=None)
    nodes = normalize_data(nodes, ['days', 'views'])
    edges['true'] = 1
    # edges = edges.head(1000)
    print(len(edges))
    edges = generate_edges(edges)
    print(len(edges))
    edges = pd.merge(left=edges, right=nodes, left_on='from', right_on='new_id')
    edges = pd.merge(left=edges, right=nodes, left_on='to', right_on='new_id', suffixes=('_from', '_to'))
    edges = edges[['from', 'to', 'days_from', 'mature_from', 'views_from', 'partner_from', 'days_to',
                   'mature_to', 'views_to', 'partner_to', 'true']]
    print(edges)
    edges = edges.values
    train, test = train_test_split(edges, test_size=0.2)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    y_train = y_train.astype(int)
    X_test = test[:, :-1]
    y_test = test[:, -1]
    y_test = y_test.astype(int)
    print(len(train))
    print(len(test))
    return X_train, y_train, X_test, y_test

edges_path = 'ENGB_edges.csv'
nodes_path = 'ENGB_target.csv'
X_train, y_train, X_test, y_test = preprocess(edges_path, nodes_path)

model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))

edges_path = 'DE_edges.csv'
nodes_path = 'DE_target.csv'
X_train, y_train, X_test, y_test = preprocess(edges_path, nodes_path)

predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))