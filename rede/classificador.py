import os
import random
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from google.colab import drive
drive.mount('/content/drive')

import os
import shutil
import random
from pathlib import Path

# =========================
# 1. DIVISÃO DOS DADOS
# =========================
original_data_dir = '/content/drive/MyDrive/archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'
output_base = '/content/drive/MyDrive/archive/BreaKHis_split'

# Proporções de divisão
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Garantir que as proporções somam 1.0
assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "As proporções devem somar 1.0"

# Semente para reprodutibilidade
random.seed(42)

# Caminhar por todas as subpastas
for root, dirs, files in os.walk(original_data_dir):
    if files:
        # Identificar o caminho da classe relativa à pasta original
        class_path = Path(root).relative_to(original_data_dir)

        # Filtrar arquivos de imagem
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(image_files)

        # Calcular os limites de corte
        total = len(image_files)
        train_end = int(train_split * total)
        val_end = train_end + int(val_split * total)

        # Dividir os arquivos
        splits = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end],
            'test': image_files[val_end:]
        }

        # Copiar os arquivos para as pastas correspondentes
        for split, split_files in splits.items():
            split_dir = os.path.join(output_base, split, str(class_path))
            os.makedirs(split_dir, exist_ok=True)
            for fname in split_files:
                src = os.path.join(root, fname)
                dst = os.path.join(split_dir, fname)
                shutil.copy2(src, dst)

print("✅ Divisão dos dados concluída.")

# =========================
# 2. PREPARAÇÃO DOS DADOS
# =========================
img_width, img_height = 150, 150
batch_size = 32

train_data_dir = os.path.join(output_base, 'train')
val_data_dir = os.path.join(output_base, 'val')
test_data_dir = os.path.join(output_base, 'test')

print(train_data_dir)
print(val_data_dir)
print(test_data_dir)

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# =========================
# 3. MODELO
# =========================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# =========================
# 4. CALLBACKS
# =========================
checkpoint_path = '/content/drive/MyDrive/breast_cancer_best_model.h5'

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True
)

# =========================
# 5. TREINAMENTO
# =========================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stopping, model_checkpoint]
)

# =========================
# 6. AVALIAÇÃO E MÉTRICAS
# =========================
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Matriz de Confusão - Conjunto de Teste")
plt.tight_layout()

conf_matrix_path = '/content/drive/MyDrive/matriz_confusao.png'
plt.savefig(conf_matrix_path)
print(f"✅ Matriz de confusão salva em: {conf_matrix_path}")

# Relatório de classificação
report = classification_report(y_true, y_pred, target_names=class_labels)
report_path = '/content/drive/MyDrive/relatorio_classificacao.txt'
with open(report_path, 'w') as f:
    f.write("Relatório de Classificação - Conjunto de Teste\n\n")
    f.write(report)
print(f"✅ Relatório salvo em: {report_path}")

# =========================
# 7. CURVAS DE TREINAMENTO
# =========================
plt.figure(figsize=(12, 5))

# Acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia por Época')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

# Perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda por Época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

# Salvar gráfico
plot_path = '/content/drive/MyDrive/curvas_treinamento.png'
plt.tight_layout()
plt.savefig(plot_path)
print(f"✅ Curvas de treinamento salvas em: {plot_path}")

