from ultralytics import YOLO
import os

# Definir o caminho para o arquivo data.yaml
data_yaml_path = "./data.yaml"

# Verificar se o arquivo data.yaml existe
if not os.path.exists(data_yaml_path):
    print(f"Erro: O arquivo data.yaml não foi encontrado em {data_yaml_path}")
    print("Certifique-se de que o dataset foi descompactado corretamente.")
    exit()

# Carregar um modelo YOLOv8n pré-treinado (para fine-tuning)
model = YOLO("./runs/detect/yolov8n_emergency_vehicles2/weights/last.pt")

# Treinar o modelo
# data: caminho para o arquivo data.yaml
# epochs: número de épocas de treinamento (ajuste conforme necessário)
# imgsz: tamanho da imagem de entrada (640 é um valor comum para YOLOv8)
# name: nome da pasta onde os resultados do treinamento serão salvos
results = model.train(data=data_yaml_path, epochs=15, imgsz=640, name="yolov8n_emergency_vehicles3")

print("Treinamento concluído! Os resultados foram salvos em runs/detect/yolov8n_emergency_vehicles3")

# Opcional: Validar o modelo após o treinamento
print("Iniciando validação do modelo treinado...")
metrics = model.val()  # Avalia o modelo no conjunto de validação

print("Métricas de validação:")
print(f"Map50-95: {metrics.box.map}")
print(f"Map50: {metrics.box.map50}")
print(f"Map75: {metrics.box.map75}")
print(f"Classes: {metrics.names}")

# O modelo treinado (best.pt) estará disponível em runs/detect/yolov8n_emergency_vehicles/weights/best.pt

