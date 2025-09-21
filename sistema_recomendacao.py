import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# --- Etapa de Carregamento e Pré-processamento ---
# Carrega o modelo ResNet50 para gerar o embedding da imagem de entrada
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Define as transformações para as imagens
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_embedding(image_path):
    """Gera o embedding para uma única imagem."""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            embedding = model(image_tensor).squeeze().numpy()
        return embedding
    except Exception as e:
        print(f"Erro ao gerar embedding para a imagem {image_path}: {e}")
        return None

# --- Lógica de Similaridade e Recomendação ---
def find_similar_images(query_image_path, embeddings, file_paths, num_recommendations=5):
    """Encontra e retorna as imagens mais similares."""
    print(f"Buscando imagens similares a: {query_image_path}")
    
    # 1. Gera o embedding da imagem de entrada
    query_embedding = get_embedding(query_image_path)
    
    if query_embedding is None:
        return None

    # 2. Calcula a similaridade de cosseno entre a imagem de entrada e todas as imagens do dataset
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    # 3. Encontra os índices das imagens mais similares
    # A ordenação é decrescente, e o [1:] remove a própria imagem de entrada da lista
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Pega os caminhos das imagens recomendadas
    recommended_indices = sorted_indices[1:num_recommendations + 1]
    recommended_images = file_paths[recommended_indices]
    
    return recommended_images, similarities[recommended_indices]

# --- Construção da Interface ---
def show_recommendations(query_image_path, recommended_images, num_recommendations=5):
    """Exibe a imagem de entrada e as recomendações."""
    fig, axes = plt.subplots(1, num_recommendations + 1, figsize=(20, 10))
    fig.suptitle('Sistema de Recomendação por Similaridade de Imagens', fontsize=16)

    # Mostra a imagem de entrada
    axes[0].imshow(Image.open(query_image_path).convert('RGB'))
    axes[0].set_title('Imagem de Entrada')
    axes[0].axis('off')

    # Mostra as imagens recomendadas
    for i, img_path in enumerate(recommended_images):
        axes[i + 1].imshow(Image.open(img_path).convert('RGB'))
        axes[i + 1].set_title(f'Recomendação {i + 1}')
        axes[i + 1].axis('off')
    
    plt.show()

# --- Lógica Principal de Execução ---
if __name__ == '__main__':
    print("Carregando embeddings e caminhos de arquivos...")
    try:
        embeddings = np.load('embeddings.npy')
        file_paths = np.load('file_paths.npy')
    except FileNotFoundError:
        print("Erro: Os arquivos 'embeddings.npy' ou 'file_paths.npy' não foram encontrados. Certifique-se de que a Etapa 2 foi executada com sucesso.")
    else:
        # **Ajuste para ler a imagem de uma pasta de input**
        INPUT_DIR = 'input_images'
        
        # Encontra a primeira imagem na pasta de input
        input_images = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not input_images:
            print("Nenhuma imagem encontrada na pasta 'input_images'. Por favor, adicione uma imagem para a busca.")
        else:
            # Pega o caminho da primeira imagem encontrada
            QUERY_IMAGE_PATH = input_images[0]
            
            # Encontra as recomendações
            recommended_images, _ = find_similar_images(QUERY_IMAGE_PATH, embeddings, file_paths)
            
            if recommended_images is not None:
                # Exibe os resultados
                show_recommendations(QUERY_IMAGE_PATH, recommended_images)
                
                print("\n---")
                print(f"Sistema de Recomendação concluído. Exibindo resultados para a imagem: {QUERY_IMAGE_PATH}")