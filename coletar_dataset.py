import os
from icrawler.builtin import GoogleImageCrawler

# Define as categorias de produtos que você deseja coletar
# Adicione ou remova categorias conforme a necessidade do seu projeto
CATEGORIES = ['relogio de pulso', 'camiseta', 'sapato', 'bicicleta']

# Diretório base para salvar o dataset
DATASET_DIR = 'dataset_produtos'
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Loop para cada categoria de produto
for category in CATEGORIES:
    print(f"Iniciando a coleta de imagens para a categoria: {category}...")
    
    # Cria a pasta para a categoria atual dentro do diretório principal
    category_dir = os.path.join(DATASET_DIR, category)
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)
        
    # Inicializa o crawler do Google Images
    # O "keyword" é o termo de busca para a categoria atual
    # O "storage" define onde as imagens serão salvas
    google_crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=4,
        storage={'root_dir': category_dir}
    )
    
    # Inicia a busca e o download das imagens
    # O "max_num" define o número de imagens que vai baixar para cada categoria
    google_crawler.crawl(keyword=category, max_num=200)
    
    print(f"Coleta de imagens para '{category}' concluída. {len(os.listdir(category_dir))} imagens salvas.\n")

print("---")
print("Etapa 1: Coleta e Organização do Dataset concluída com sucesso!")
print(f"Verifique a pasta '{DATASET_DIR}' para ver as imagens organizadas por categoria.")