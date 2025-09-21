import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

# 1. Carregamento do Modelo Pré-treinado (ResNet50)
print("1. Carregando o modelo pré-treinado ResNet50...")

# Carrega o modelo ResNet50 com pesos pré-treinados
model = models.resnet50(pretrained=True)

# Remove a camada de classificação final, mantendo apenas a rede de extração de features
# O "[0:8]" seleciona as primeiras 8 camadas, que correspondem ao extrator de características
model = torch.nn.Sequential(*list(model.children())[:-1])

# Coloca o modelo em modo de avaliação. Isso desativa funções como dropout,
# garantindo que o modelo se comporte de forma consistente para inferência.
model.eval()

# Define as transformações necessárias para as imagens antes de passá-las pelo modelo
# O ResNet50 foi treinado com imagens de 224x224 pixels e com normalização específica
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Diretório onde as imagens coletadas na Etapa 1 estão salvas
DATASET_DIR = 'dataset_produtos'
embeddings = []
file_paths = []

print("2. Gerando embeddings para cada imagem no dataset...")

# Loop por todas as subpastas e arquivos no diretório de dados
for root, dirs, files in os.walk(DATASET_DIR):
    for filename in files:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, filename)
            
            try:
                # Carrega a imagem e aplica as transformações
                image = Image.open(image_path).convert('RGB')
                image_tensor = transform(image)
                
                # Adiciona uma dimensão extra para o batch (o modelo espera um batch)
                image_tensor = image_tensor.unsqueeze(0)
                
                # Gera o embedding da imagem usando o modelo
                with torch.no_grad(): # Desativa o cálculo de gradientes para otimizar a inferência
                    embedding_tensor = model(image_tensor)
                
                # Converte o tensor para um array NumPy e o salva
                embedding_np = embedding_tensor.squeeze().numpy()
                embeddings.append(embedding_np)
                file_paths.append(image_path)
                
            except Exception as e:
                print(f"Erro ao processar a imagem {image_path}: {e}")

# 3. Salvando os embeddings e os caminhos dos arquivos
print("3. Salvando os embeddings gerados...")
embeddings_array = np.array(embeddings)
file_paths_array = np.array(file_paths)

np.save('embeddings.npy', embeddings_array)
np.save('file_paths.npy', file_paths_array)

print("---")
print(f"Etapa 2: Pré-processamento e Extração de Embeddings concluída com sucesso!")
print(f"Total de {len(embeddings)} embeddings gerados e salvos em 'embeddings.npy'.")