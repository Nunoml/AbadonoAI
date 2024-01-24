import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import tensorflow_decision_forests as tfdf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras_tuner.tuners import RandomSearch
import keras_tuner as kt

def oversample_dataframe(dataset):
    minority = 1
    minority_index = np.where(dataset['Abandono'].values == minority)[0]
    majority_index = np.where(dataset['Abandono'].values != minority)[0]

    minority_data = dataset.iloc[minority_index]
    majority_data = dataset.iloc[majority_index]

    oversample_by = 3

    minority_data_oversampled = pd.concat([minority_data] * oversample_by, ignore_index=True)
    oversampled_data = pd.concat([minority_data_oversampled, majority_data])

    oversampled_data = oversampled_data.sample(frac=1, random_state=0)

    return oversampled_data

def pd_dataframe_to_tf_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop('Abandono')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    #ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

def build_model(hp):
    model = tfdf.keras.RandomForestModel(
        task=tfdf.keras.Task.CLASSIFICATION,
        num_trees = hp.Int('num_trees',min_value=50, max_value=500, step = 50),
        max_depth=hp.Int('max_depth', min_value=5, max_value=20, step=1)
    )
    model.compile(metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()], loss=tf.keras.losses.BinaryCrossentropy())
    return model

raw_dataset = pd.read_excel("./Dados_ML.xlsx", na_values="?", sheet_name="Sheet1", index_col=0)
#5411 samples, 33 colunas, 32 features
dataset = raw_dataset.copy()
dataset['idadeReal_Categoria'] = pd.cut(dataset['idadeReal'], bins=[0, 18, 23, 65, 999], labels=[0, 1, 2, 3])
dataset.columns = dataset.columns.str.replace(" ", "_")
dataset.columns = dataset.columns.str.replace("á", "a")
# Ou tentar corrigir os dados ou simplesmente tirar
dataset = dataset.dropna()
values_to_ignore = ['cd_hab_ant','cd_cur_hab_ant', 'cd_inst_hab_ant', 'cd_tip_est_sec', '1geracao']

for x in values_to_ignore:
    dataset = dataset.drop(labels=x, axis=1)

training_data = dataset.sample(frac=0.8, random_state=0)
twentypercent = dataset.drop(training_data.index)
testing_data = twentypercent.sample(frac=0.5, random_state=0)
validation_data = twentypercent.drop(testing_data.index)

print(training_data.describe().transpose())

#training_data = oversample_dataframe(training_data)

train_ds = pd_dataframe_to_tf_dataset(training_data)
test_ds = pd_dataframe_to_tf_dataset(testing_data)
val_ds = pd_dataframe_to_tf_dataset(validation_data)

train_ds = train_ds.batch(200)
test_ds = test_ds.batch(200)
val_ds = val_ds.batch(200)

tuner = RandomSearch(
    build_model,
    objective=[kt.Objective("val_binary_accuracy", direction="max"), kt.Objective('val_precision', direction='max'),kt.Objective('val_recall', direction="max")],
    max_trials=5,
    directory='tuner_logs',
    project_name='random_forest_tuning'
)

tuner.search(train_ds, validation_data=val_ds, epochs=1)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = build_model(best_hps)

model.fit(train_ds, validation_data=val_ds)

predictions = model.predict(test_ds)

true_labels = np.array(testing_data['Abandono'])

binary_predictions = (predictions > 0.5).astype(int)


true_labels_tensor = tf.convert_to_tensor(true_labels, dtype=tf.float32)
binary_predictions_tensor = tf.convert_to_tensor(binary_predictions, dtype=tf.float32)

precision = tf.metrics.Precision(name='precision', thresholds=0.5)(true_labels_tensor, binary_predictions_tensor)
recall = tf.metrics.Recall(name='recall', thresholds=0.5)(true_labels_tensor, binary_predictions_tensor)
f1 = 2 * (precision * recall) / (precision + recall)
accuracy = tf.metrics.BinaryAccuracy(name='accuracy')(true_labels_tensor, binary_predictions_tensor)
cm = tf.math.confusion_matrix(true_labels_tensor, binary_predictions_tensor, num_classes=2)

cm_np = cm.numpy()

# Visualizar
plt.figure(figsize=(8, 6))
sns.heatmap(cm_np, annot=True, fmt="d", cmap="Blues", xticklabels=['Não Abandonou', 'Abandonou'], yticklabels=['Não Abandonou', 'Abandonou'])
plt.xlabel("Estimativa")
plt.ylabel("Verdade")
plt.title("Confusion Matrix")
plt.show()

# Verificar dados
print(f'Precision: {precision.numpy():.2f}')
print(f'Recall: {recall.numpy():.2f}')
print(f'F1-Score: {f1.numpy():.2f}')
print(f'Accuracy: {accuracy.numpy() * 100:.2f}%')
