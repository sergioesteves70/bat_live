import cv2
import numpy as np
import sounddevice as sd

intensidade_audio = 0.0

def audio_callback(indata, frames, time, status):
    global intensidade_audio
    rms = np.sqrt(np.mean(indata**2))
    # Multiplicador de ganho do som
    intensidade_audio = np.clip(rms * 20, 0.0, 1.0)

def iniciar_camera_live():
    global intensidade_audio

    # Tenta ligar o microfone com a taxa nativa do sistema
    try:
        stream = sd.InputStream(callback=audio_callback, channels=1)
        stream.start()
        print("Microfone detetado e ligado!")
    except Exception as e:
        print(f"\n[ERRO DE ÁUDIO] Não foi possível ligar o microfone: {e}")
        print("Verifique se tem um microfone ligado e com permissão no Windows.\n")
        return

    # Abre a câmara web (índice 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível aceder à câmara web.")
        stream.stop()
        return

    print("Câmara ligada! Pressione a tecla 'q' na janela do vídeo para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        intensidade = intensidade_audio

        # --- EFEITO 1: Zoom Dinâmico ---
        fator_zoom = 1.0 + (intensidade * 0.15)
        novo_w, novo_h = int(w * fator_zoom), int(h * fator_zoom)

        frame_resized = cv2.resize(frame, (novo_w, novo_h), interpolation=cv2.INTER_LINEAR)

        left = (novo_w - w) // 2
        top = (novo_h - h) // 2
        frame_cropped = frame_resized[top:top+h, left:left+w]

        arr = frame_cropped.astype(np.float32)

        # --- EFEITO 2: Flash de Luz (Brilho) ---
        brilho = 1.0 + (intensidade * 0.6)
        arr = arr * brilho

        # --- EFEITO 3: Flash Néon (Canais BGR) ---
        if intensidade > 0.2:
            arr[:, :, 2] += intensidade * 50  # Canal Vermelho
            arr[:, :, 0] += intensidade * 30  # Canal Azul

        frame_final = np.clip(arr, 0, 255).astype(np.uint8)

        cv2.imshow("Visualizador em Direto (Pressione 'q' para sair)", frame_final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    stream.stop()

if __name__ == "__main__":
    iniciar_camera_live()