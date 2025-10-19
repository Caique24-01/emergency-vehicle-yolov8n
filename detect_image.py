import sys
import os
import cv2
from ultralytics import YOLO

# Caminho para o modelo treinado
MODEL_PATH = "./runs/detect/yolov8n_emergency_vehicles3/weights/best.pt"

def detect_emergency_vehicle_in_image(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Erro: O modelo treinado não foi encontrado em {MODEL_PATH}")
        print("Certifique-se de que o treinamento foi concluído e o caminho está correto.")
        return

    if not os.path.exists(image_path):
        print(f"Erro: A imagem não foi encontrada em {image_path}")
        return

    model = YOLO(MODEL_PATH)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Erro: Não foi possível carregar a imagem: {image_path}")
        return

    print(f"Processando imagem: {image_path}")
    results = model(frame)

    output_image = frame.copy()
    veiculo_detectado = False
    for r in results:
        output_image = r.plot()  # Desenha as caixas delimitadoras e rótulos
        if len(r.boxes) > 0:
            veiculo_detectado = True
            print("Veículo de emergência detectado: True")
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = box.conf[0]
                print(f"  - Classe: {label}, Confiança: {conf:.2f}, BBox: {box.xyxy[0].tolist()}")
        else:
            print("Veículo de emergência detectado: False")

    # Salvar a imagem de saída
    output_filename = "detected_" + os.path.basename(image_path)
    output_path = os.path.join(os.path.dirname(image_path), output_filename)
    cv2.imwrite(output_path, output_image)
    print(f"Imagem com detecções salva em: {output_path}")
    
    # cv2.imshow("YOLOv8 Inference", output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python detect_image.py <caminho_para_imagem>")
        sys.exit(1)

    image_file = sys.argv[1]
    detect_emergency_vehicle_in_image(image_file)

