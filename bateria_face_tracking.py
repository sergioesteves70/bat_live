import os
import random
import urllib.request
import cv2
import numpy as np
import sounddevice as sd

intensidade_audio = 0.0
LIMIAR_GLITCH = 0.40  # Sensibilidade para ativar o glitch


def audio_callback(indata, frames, time, status):
    global intensidade_audio
    rms = np.sqrt(np.mean(indata**2))
    intensidade_audio = np.clip(rms * 25, 0.0, 1.0)


def carregar_detector_rosto():
    xml_name = "haarcascade_frontalface_default.xml"

    caminhos = [
        xml_name,
        os.path.join(getattr(cv2, "data", None).haarcascades, xml_name)
        if hasattr(cv2, "data")
        else None,
    ]

    xml_path = next(
        (p for p in caminhos if p and os.path.exists(p)), None
    )

    if not xml_path:
        print("A descarregar modelo de deteção de rosto...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            urllib.request.urlretrieve(url, xml_name)
            xml_path = xml_name
            print("Download concluído com sucesso!")
        except Exception as e:
            print(f"Erro no download do modelo: {e}")

    detector = cv2.CascadeClassifier(xml_path)
    return detector


def iniciar_camera_face_tracking():
    global intensidade_audio

    # 1. Carregar a imagem da caveira
    caminho_skull = "skull.jpg"
    if not os.path.exists(caminho_skull):
        print(f"Erro: Ficheiro '{caminho_skull}' não encontrado na pasta!")
        return

    img_skull = cv2.imread(caminho_skull)
    if img_skull is None:
        print("Erro ao ler a imagem da caveira.")
        return

    # 2. Inicializar o Detetor de Rosto
    detector_rosto = carregar_detector_rosto()

    # 3. Ligar Microfone
    try:
        stream = sd.InputStream(callback=audio_callback, channels=1)
        stream.start()
        print("Microfone ligado!")
    except Exception as e:
        print(f"Erro no microfone: {e}")
        return

    # 4. Ligar Câmara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a câmara web.")
        stream.stop()
        return

    print("Face Tracking Ativo! Pressione 'q' na janela do vídeo para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_cam, w_cam, _ = frame.shape
        intensidade = intensidade_audio
        final_frame = frame.copy()

        # Deteção do Rosto
        rostos = []
        if detector_rosto and not detector_rosto.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rostos = detector_rosto.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

        # Só aplica o efeito SE encontrar pelo menos um rosto
        if len(rostos) > 0:
            rostos = sorted(rostos, key=lambda r: r[2] * r[3], reverse=True)
            x, y, w, h = rostos[0]

            if w > 20 and h > 20:
                skull_rosto = cv2.resize(
                    img_skull, (w, h), interpolation=cv2.INTER_LINEAR
                )
                face_roi = frame[y : y + h, x : x + w]

                # --- EFEITO GLITCH APLICADO APENAS AO ROSTO ---
                if intensidade > LIMIAR_GLITCH:
                    alfa = np.clip((intensidade - LIMIAR_GLITCH) * 1.8, 0.2, 0.75)
                    glitch_roi = cv2.addWeighted(
                        face_roi, 1.0 - alfa, skull_rosto, alfa, 0
                    )

                    arr_roi = glitch_roi.astype(np.float32)
                    arr_skull = skull_rosto.astype(np.float32)

                    num_fatias = random.randint(8, 20)
                    for _ in range(num_fatias):
                        fy1 = random.randint(0, max(1, h - 20))
                        fh = random.randint(5, 20)
                        fy2 = min(fy1 + fh, h)

                        deslocamento_x = random.randint(-20, 20)
                        fatia_skull = arr_skull[fy1:fy2, :, :]

                        M = np.float32([[1, 0, deslocamento_x], [0, 1, 0]])
                        fatia_shifted = cv2.warpAffine(
                            fatia_skull,
                            M,
                            (w, fy2 - fy1),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_REPLICATE,
                        )

                        if random.random() < 0.5:
                            arr_roi[fy1:fy2, :, 2] = fatia_shifted[:, :, 2]
                        else:
                            arr_roi[fy1:fy2, :, 0] = fatia_shifted[:, :, 0]

                    arr_roi = arr_roi * (1.0 + (intensidade - LIMIAR_GLITCH) * 0.6)
                    final_frame[y : y + h, x : x + w] = np.clip(
                        arr_roi, 0, 255
                    ).astype(np.uint8)
                else:
                    arr_roi = face_roi.astype(np.float32) * (
                        1.0 + intensidade * 0.2
                    )
                    final_frame[y : y + h, x : x + w] = np.clip(
                        arr_roi, 0, 255
                    ).astype(np.uint8)

        cv2.imshow("Face Tracking Glitch (Pressione 'q')", final_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    stream.stop()


if __name__ == "__main__":
    iniciar_camera_face_tracking()