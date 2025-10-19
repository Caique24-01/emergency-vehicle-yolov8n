import sys
import os
import cv2
from ultralytics import YOLO

# Caminho para o modelo treinado
MODEL_PATH = "./runs/detect/yolov8n_emergency_vehicles3/weights/best.pt"

def detect_emergency_vehicle_in_video(video_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Erro: O modelo treinado não foi encontrado em {MODEL_PATH}")
        print("Certifique-se de que o treinamento foi concluído e o caminho está correto.")
        return

    if not os.path.exists(video_path):
        print(f"Erro: O vídeo não foi encontrado em {video_path}")
        return

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo: {video_path}")
        return

    # Obter propriedades do vídeo original
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v') # Codec para .mp4

    # Definir o caminho de saída para o vídeo
    output_filename = "detected_" + os.path.basename(video_path)
    output_path = os.path.join(os.path.dirname(video_path), output_filename)
    out = cv2.VideoWriter(output_path, codec, fps, (frame_width, frame_height))

    print(f"Processando vídeo: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Fim do vídeo ou frame inválido.")
            break

        results = model(frame, stream=True) # stream=True para processar frames em tempo real

        output_frame = frame.copy()
        veiculo_detectado = False
        for r in results:
            output_frame = r.plot()  # Desenha as caixas delimitadoras e rótulos
            if len(r.boxes) > 0:
                veiculo_detectado = True
                # print("Veículo de emergência detectado: True")
                # for box in r.boxes:
                #     cls = int(box.cls[0])
                #     label = model.names[cls]
                #     conf = box.conf[0]
                #     print(f"  - Classe: {label}, Confiança: {conf:.2f}, BBox: {box.xyxy[0].tolist()}")
            # else:
                # print("Veículo de emergência detectado: False")
        
        out.write(output_frame)

        # Opcional: mostrar o vídeo em tempo real (pode ser lento)
        # cv2.imshow("YOLOv8 Inference", output_frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == 27:  # ESC para sair
        #     print("Encerrando inferência de vídeo.")
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Vídeo com detecções salvo em: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python detect_video.py <caminho_para_video>")
        sys.exit(1)

    video_file = sys.argv[1]
    detect_emergency_vehicle_in_video(video_file)

