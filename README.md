# Sistema de Recomendação por Similaridade de Imagens

## Descrição do Projeto

Este projeto implementa um sistema de recomendação por similaridade visual, capaz de encontrar produtos relacionados com base em suas características físicas como cor, formato e textura, e não em dados textuais como preço ou marca.

O sistema utiliza técnicas de **Deep Learning** para extrair as características de cada imagem, convertendo-as em vetores numéricos (embeddings). Em seguida, ele calcula a similaridade entre as imagens para recomendar os itens mais parecidos.

## Tecnologias e Ferramentas

* **Python:** Linguagem de programação principal.
* **PyTorch & TorchVision:** Para carregar um modelo pré-treinado (ResNet50) e extrair os embeddings das imagens.
* **Numpy:** Para manipulação eficiente de vetores numéricos.
* **scikit-learn:** Para o cálculo da similaridade de cosseno.
* **Matplotlib & Pillow:** Para a visualização dos resultados.
* **icrawler:** Para a coleta automatizada de imagens.

## Estrutura do Projeto
├── dataset_produtos/         # Imagens baixadas e organizadas por categoria (criado automáticamente)

├── input_images/             # Imagem de entrada para a recomendação

├── .gitignore                # Arquivos a serem ignorados pelo Git

├── requirements.txt          # Dependências do projeto

├── coletar_dataset.py        # Script para a coleta automatizada de imagens

├── extrair_embeddings.py     # Script para extrair embeddings das imagens

└── sistema_recomendacao.py   # Script principal do sistema de recomendação

## Como Executar o Projeto

Siga os passos abaixo para rodar o sistema em sua máquina:

1.  **Clone o Repositório:**
    ```bash
    git clone (https://github.com/andrewnav/recomend_img_digitais.git)
    cd recomend_img_digitais
    ```

2.  **Crie o Ambiente Virtual e Instale as Dependências:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate   # Para Windows
    source venv/bin/activate  # Para macOS/Linux
    pip install -r requirements.txt
    ```

3.  **Execute as Etapas do Projeto:**
    * **Passo 1: Coleta de Imagens**
        Execute o script para baixar e organizar as imagens em `dataset_produtos/`.
        ```bash
        python coletar_dataset.py
        ```
    * **Passo 2: Extração de Embeddings**
        Rode este script para gerar os arquivos `embeddings.npy` e `file_paths.npy`.
        ```bash
        python extrair_embeddings.py
        ```
    * **Passo 3: Sistema de Recomendação**
        Coloque uma imagem na pasta `input_images/` e execute o script para ver as recomendações.
        ```bash
        python sistema_recomendacao.py
        ```

## Autor

* Andrew Navarro - https://www.linkedin.com/in/navarroandrew/

## Licença

Este projeto está sob a licença MIT.