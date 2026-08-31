import cv2
import numpy as np
import sounddevice as sd
import os
import random

# Variável global para o volume
intensidade_audio = 0.0
# Limiar para ativar o efeito de glitch (0.0 a 1.0)
LIMIAR_GLITCH = 0.5 

def audio_callback(indata, frames, time, status):
    global intensidade_audio
    rms = np.sqrt(np.mean(indata**2))
    # Ganho do microfone
    intensidade_audio = np.clip(rms * 25, 0.0, 1.0)

def iniciar_camera_glitch():
    global intensidade_audio
    
    # --- 1. Carregar e Preparar a Imagem da Caveira ---
    caminho_skull = "skull.png"
    if not os.path.exists(caminho_skull):
        print(f"Erro: Ficheiro '{caminho_skull}' não encontrado na pasta!")
        return
    
    # Carrega a caveira em modo de cor (BGR)
    img_skull = cv2.imread(caminho_skull)
    if img_skull is None:
        print("Erro: Não foi possível ler a imagem da caveira.")
        return

    # --- 2. Iniciar Dispositivos (Som e Câmara) ---
    try:
        # Tenta ligar o microfone nativo
        stream = sd.InputStream(callback=audio_callback, channels=1)
        stream.start()
        print("Microfone ligado!")
    except Exception as e:
        print(f"[ERRO DE ÁUDIO] Certifique-se de que o microfone tem permissão no Windows: {e}")
        return

    # Abre a câmara web (índice 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível aceder à câmara web.")
        stream.stop()
        return

    # Obtém as dimensões da câmara para redimensionar a caveira apenas uma vez
    ret_test, frame_test = cap.read()
    if not ret_test:
        print("Erro ao ler da câmara.")
        cap.release()
        stream.stop()
        return
    
    h_cam, w_cam, _ = frame_test.shape
    # Redimensiona a caveira para o tamanho exato da câmara
    img_skull_resized = cv2.resize(img_skull, (w_cam, h_cam), interpolation=cv2.INTER_LINEAR)

    print(f"Câmara ligada ({w_cam}x{h_cam})! Efeito de Glitch com '{caminho_skull}' ativado nas batidas fortes.")
    print("Pressione 'q' na janela de vídeo para sair.")

    # --- 3. Loop Principal em Tempo Real ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        intensidade = intensidade_audio
        final_frame = frame.copy()

        # --- SE O SOM ULTRAPASSAR O LIMIAR: Aplicar Glitch ---
        if intensidade > LIMIAR_GLITCH:
            
            # 1. Mistura Base (Blending) proporcional à intensidade
            # Quanto mais forte o som, mais visível é a caveira por trás do glitch
            alfa = np.clip((intensidade - LIMIAR_GLITCH) * 1.5, 0.1, 0.6)
            final_frame = cv2.addWeighted(frame, 1.0 - alfa, img_skull_resized, alfa, 0)
            
            # Converte para float para cálculos sem erro de overflow
            arr_frame = final_frame.astype(np.float32)
            arr_skull = img_skull_resized.astype(np.float32)

            # 2. Glitch de Fatias Horizontais e Mistura de Canais
            # Criar várias fatias horizontais aleatórias
            num_fatias = random.randint(10, 30)
            
            for _ in range(num_fatias):
                y1 = random.randint(0, h_cam - 50)
                h_fatia = random.randint(10, 50)
                y2 = y1 + h_fatia
                if y2 > h_cam: y2 = h_cam
                
                # Deslocamento horizontal aleatório para a fatia
                deslocamento_x = random.randint(-50, 50)
                
                # Copia a fatia correspondente da caveira
                fatia_skull = arr_skull[y1:y2, :, :]
                
                # Desloca a fatia horizontalmente
                M = np.float32([[1, 0, deslocamento_x], [0, 1, 0]])
                fatia_shifted = cv2.warpAffine(fatia_skull, M, (w_cam, h_fatia), 
                                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
                
                # 3. Mistura Distorcida de Cores na Fatia
                # Nas fatias do glitch, misturamos canais diferentes (ex: R da caveira no G da cara)
                # OpenCV usa BGR [0, 1, 2]
                
                tipo_glitch = random.random()
                if tipo_glitch < 0.4:
                    # Substitui Vermelho (R) da cara pelo Vermelho da caveira deslocada
                    arr_frame[y1:y2, :, 2] = fatia_shifted[:, :, 2]
                elif tipo_glitch < 0.7:
                    # Substitui Verde (G) da cara pelo Azul (B) da caveira deslocada
                    arr_frame[y1:y2, :, 1] = fatia_shifted[:, :, 0]
                else:
                    # Adiciona luz néon misturada
                    arr_frame[y1:y2, :, 0] += fatia_shifted[:, :, 2] * 0.5 # B <- R skull
                    arr_frame[y1:y2, :, 2] += fatia_shifted[:, :, 0] * 0.5 # R <- B skull

            # 4. Flash de Brilho Geral no pico do Glitch
            brilho = 1.0 + (intensidade - LIMIAR_GLITCH) * 0.5
            arr_frame = arr_frame * brilho

            # Converte de volta para uint8
            final_frame = np.clip(arr_frame, 0, 255).astype(np.uint8)

        # --- SE O SOM FOR FRACO: Apenas Zoom e Brilho Normal ---
        else:
            h, w, _ = frame.shape
            
            # Efeito base de zoom proporcional à intensidade (mesmo abaixo do limiar)
            fator_zoom = 1.0 + (intensidade * 0.10)
            novo_w, novo_h = int(w * fator_zoom), int(h * fator_zoom)
            frame_resized = cv2.resize(frame, (novo_w, novo_h), interpolation=cv2.INTER_LINEAR)
            
            left, top = (novo_w - w) // 2, (novo_h - h) // 2
            frame_cropped = frame_resized[top:top+h, left:left+w]
            
            # Brilho normal
            arr = frame_cropped.astype(np.float32)
            brilho = 1.0 + (intensidade * 0.3)
            arr = arr * brilho
            final_frame = np.clip(arr, 0, 255).astype(np.uint8)

        # Exibe o resultado
        cv2.imshow("Glitch Bateria Live (Pressione 'q' para sair)", final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    stream.stop()

if __name__ == "__main__":
    iniciar_camera_glitch()