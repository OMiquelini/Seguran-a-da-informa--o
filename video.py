import cv2
import os
import glob

def extract_frames(video_path, output_folder):
    # Extrair o nome da pessoa do nome do vídeo
    person_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    success, image = cap.read()
    count = 0

    while success:
        frame_name = f"{person_name}_{count}.png"
        output_path = os.path.join(output_folder, frame_name)

        # Certifique-se de que o diretório de saída exista
        os.makedirs(output_folder, exist_ok=True)

        cv2.imwrite(output_path, image)
        success, image = cap.read()
        print(f'Saved {frame_name}')
        count += 1

    cap.release()

def process_all_videos(videos_dir, output_folder):
    # Lista todos os arquivos de vídeo no diretório
    video_files = glob.glob(os.path.join(videos_dir, '*.mp4'))

    # Itera sobre todos os vídeos no diretório
    for video_file in video_files:
        # Chama a função para extrair os frames para cada vídeo
        print("extraindo vídeo")
        extract_frames(video_file, output_folder)

# Substitua o caminho do diretório de vídeos e da pasta de saída conforme necessário
videos_dir = r'C:\Users\Matheus Miquelini\Desktop\projeto\videos'
output_folder = r'C:\Users\Matheus Miquelini\Desktop\projeto\imagens'

# Chama a função para processar todos os vídeos no diretório
process_all_videos(videos_dir, output_folder)
