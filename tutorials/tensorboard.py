import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

rawfile = r'/Users/arminhadzalic/Projects/tensorboard/csv/data.csv'

def PreProcessing(rawfile):
    df = pd.read_csv(rawfile, delimiter=';')
    df.shape

    # Drop columns
    dropped_df = df.drop(columns={
        'Feature X1', 
        'Feature X3',
        'Feature X7',
        'Feature X8',
        'Feature X11',
        'Feature X12'
    })
    dropped_df.shape

    # Drop NaN rows
    dropped_df.isnull().sum()
    dropped_df = dropped_df.dropna()
    dropped_df.isnull().sum()
    dropped_df.shape

    # Convert and define features
    df_converted = dropped_df.apply(lambda s: s.map({k:i for i,k in enumerate(s.unique())}))

    # Features
    df_convertedfeatures = df_converted.drop(columns={'Result', 'Test Status'})
    features = df_convertedfeatures.columns
    print(features)
    df_convertedfeatures.shape
    # separating out the features
    x = df_converted.loc[:, features].values

    # Target
    df_convertedtarget = df_converted['Result'].to_frame()
    target = df_convertedtarget.columns
    print(target)
    df_convertedtarget.shape
    # separating out the target
    #y = df_converted.loc[:, ['Result']].values
    y = df_converted.loc[:, target].values

    return x, y, features

def ExecuteRandomForestClassifier(x, y, features):
    print('Create Random Forest')
    from sklearn.ensemble import RandomForestClassifier

    randomforest = RandomForestClassifier(random_state=42, n_jobs=10)
    model = randomforest.fit(x, y.ravel())

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [features[i] for i in indices]

    print('Show feature importances')
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(features.shape[0]), importances[indices])
    plt.xticks(range(features.shape[0]), names, rotation=90)
    plt.show()

def ExecuteDecisionTreeClassifier(x, y, features):
    print('Create DecisionTree Classifier')

    # Visualize 
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    from IPython.display import Image
    import pydotplus

    decisiontree = DecisionTreeClassifier(random_state=0)
    model = decisiontree.fit(x, y)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [features[i] for i in indices]

    dot_data = tree.export_graphviz(decisiontree,
                                    out_file=None,
                                    feature_names=names,
                                    class_names='result')

    graph = pydotplus.graph_from_dot_data(dot_data)
    type(graph)

    #svg = Image(graph.create_svg())

    print('Write SVG to disk...')
    graph.write_svg('DecisionTree.svg')
    print('Write SVG to disk done')

def ExecuteNeuralNetwork(x, y):
    print('Running Neural Network')
    import tensorflow as tf
    from tensorflow.keras.callbacks import TensorBoard
    import datetime

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    modelActivation = 'tanh' #relu, tanh
    compileOptimizer = 'adam' #adam, sgd
    loss = '' #sparse_categorical_crossentropy, mean_squared_error

    model = tf. keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(10,)),
        tf.keras.layers.Dense(128, activation=modelActivation),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='sigmoid') #sigmoid, softmax
    ])

    print('Running Compile...')
    model.compile(optimizer=compileOptimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
    print('Running compile done')

    log_dir = '../logger/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + compileOptimizer + '-' + modelActivation
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    print('Processing Epochs...')
    model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test) ,callbacks=[tensorboard_callback])
    print('Processing Epochs done')

    #print('Evaluate results...')
    #model.evaluate(x_test, y_test)

x, y, features = PreProcessing(rawfile)
#ExecuteRandomForestClassifier(x, y, features)
#ExecuteDecisionTreeClassifier(x, y, features)
ExecuteNeuralNetwork(x, y)

# Start Tensorboard to check curves
# python  python "/Users/arminhadzalic/opt/anaconda3/envs/tensorboard/lib/python3.8/site-packages/tensorboard/main.py" --logdir "/Users/arminhadzalic/Projects/tensorboard/logger"
