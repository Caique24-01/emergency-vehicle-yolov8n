# 🚨 Emergency Vehicle Detection — YOLOv8n

Este projeto implementa um sistema de **detecção de veículos de emergência** (ambulâncias, viaturas policiais e caminhões de bombeiro) utilizando o modelo **YOLOv8n (Ultralytics)**.  
O repositório contém scripts de **inferência** em imagem e vídeo, além de um **pipeline de treinamento** supervisionado com dataset anotado no formato YOLO.

---

## 📁 Estrutura do Projeto

```
emergency-vehicle-yolov8n/
│
├── detect_image.py
├── detect_video.py
├── train_emergency_vehicles.py
├── data.yaml
├── requirements.txt
├── yolov8n.pt
├── runs/
│   ├── detect/
│   │   └── emergency-vehicles/
│   │       ├── weights/
│   │       │   ├── best.pt
│   │       │   └── last.pt
│   │       ├── results.png
│   │       ├── confusion_matrix.png
│   │       ├── PR_curve.png
│   │       ├── F1_curve.png
│   │       └── val_predictions/
│   └── predict/
│       └── (saídas de inferência)
└── ...
```

---

## ⚙️ Dependências

- **Python** 3.9 a 3.11  
- **Pip** atualizado (`python -m pip install --upgrade pip`)
- Bibliotecas principais:
  - `ultralytics`
  - `torch`
  - `torchvision`
  - `opencv-python`
  - `numpy`
  - `PyYAML`

Instale tudo automaticamente:
```bash
pip install -r requirements.txt
```

> 💡 Caso ocorra erro de hash com o pacote `ultralytics`, utilize:
> ```bash
> pip install "ultralytics>=8.0.0"
> ```

---

## 🧠 Sobre o Modelo YOLOv8n

O **YOLOv8n** é a versão *nano* da família YOLOv8 — extremamente leve e otimizada para dispositivos com recursos limitados.  
Neste projeto, ele foi utilizado com **fine-tuning** a partir de pesos pré-treinados no dataset **COCO**, e re-treinado em um dataset específico de veículos de emergência.

**Características principais:**
- Arquitetura *anchor-free*, com camadas de detecção otimizadas
- Suporte a **transfer learning** via `yolov8n.pt`
- Métricas principais: *Precision*, *Recall*, *mAP@50* e *mAP@50-95*

---

## 🧩 Estrutura do Dataset (`data.yaml`)

O arquivo `data.yaml` define o caminho do dataset e as classes utilizadas:

```yaml
train: ./train/images
val: ./valid/images
test: ./test/images

nc: 3
names: [ambulance, police, firetruck]
```

Formato esperado das pastas:

```
dataset/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

---

## 🚀 Como Executar

### 🔹 1. Clonar o projeto e preparar o ambiente
```bash
git clone https://github.com/Caique24-01/emergency-vehicle-yolov8n.git
cd emergency-vehicle-yolov8n
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

### 🔹 2. Inferência em imagem
```bash
python detect_image.py caminho/para/imagem.jpg
```

---

### 🔹 3. Inferência em vídeo ou webcam
```bash
# Vídeo local
python detect_video.py caminho/para/video.mp4

```

---

## 🧪 Treinamento do Modelo

O treinamento é realizado com o script `train_emergency_vehicles.py`:

```bash
python train_emergency_vehicles.py ^
  --data data.yaml ^
  --epochs 100 ^
  --imgsz 640 ^
  --batch 16 ^
  --project runs/detect ^
  --name emergency-vehicles ^
  --device 0
```

### Parâmetros principais:
- `--data`: arquivo YAML com o dataset
- `--epochs`: número de épocas de treinamento
- `--imgsz`: tamanho das imagens de entrada
- `--batch`: tamanho do batch
- `--device`: GPU (0) ou CPU (`cpu`)

### Validação:
```bash
yolo val model=runs/detect/emergency-vehicles/weights/best.pt data=data.yaml imgsz=640
```

---

## 📈 Métricas e Resultados

Durante o treinamento, o YOLOv8 gera automaticamente os seguintes gráficos dentro da pasta `runs/detect/...`:

| Arquivo | Descrição |
|----------|------------|
| `results.png` | Evolução de *Loss*, *Precision*, *Recall* e *mAP* por época |
| `confusion_matrix.png` | Matriz de confusão por classe |
| `PR_curve.png` | Curva de Precisão x Recall |
| `F1_curve.png` | Curva de F1-score |
| `weights/best.pt` | Melhor checkpoint salvo |
| `weights/last.pt` | Último checkpoint salvo |

Esses gráficos ajudam a identificar:
- Convergência do modelo (redução de loss)
- Overfitting (divergência entre treino e validação)
- Acurácia geral (mAP)

---

## 📊 Metodologia de Treinamento

1. **Preparação do dataset**
   - Anotações no formato YOLO (`class cx cy w h`)
   - Divisão em *train*, *val* e *test*

2. **Fine-tuning**
   - Pesos iniciais: `yolov8n.pt`
   - Ajuste fino para classes específicas (ambulância, polícia, bombeiro)

3. **Validação**
   - Avaliação por *mAP@50*, *Precision*, *Recall* e *F1-score*

4. **Testes Reais**
   - Execução em vídeos reais para avaliar robustez e generalização

---

## 🧩 Estrutura da Pasta `runs/`

Após o treinamento e testes, o YOLOv8 cria automaticamente subpastas em `runs/`:

```
runs/
 ├── detect/
 │    └── emergency-vehicles/
 │         ├── weights/
 │         │    ├── best.pt
 │         │    └── last.pt
 │         ├── results.png
 │         ├── PR_curve.png
 │         ├── confusion_matrix.png
 │         └── val_predictions/
 └── predict/
      └── (saídas de inferência: imagens/vídeos com bounding boxes)
```

---


## 📜 Licença

O código segue a licença do repositório.  
Verifique a licença do dataset utilizado para treinamento.

---

## 📚 Referências

- [Ultralytics YOLOv8 — Documentação Oficial](https://docs.ultralytics.com/)
- [YOLOv8: Object Detection Task Overview](https://docs.ultralytics.com/tasks/detect/)
- [Ultralytics GitHub Repository](https://github.com/ultralytics/ultralytics)

---

### 👨‍💻 Desenvolvido por:
**Caique Azevedo Coelho Leme**  
Projeto acadêmico — *Detecção de Veículos de Emergência (YOLOv8n)*
