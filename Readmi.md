# Human Activity Recognition (HAR) Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Jupyter](https://img.shields.io/badge/Tools-Jupyter%20Notebook-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-green.svg)

Este projeto implementa uma pipeline completa de Machine Learning para o reconhecimento de atividades humanas (HAR) utilizando o dataset **FORTH-TRACE**. O sistema processa dados de sensores inerciais (Acelerómetro, Giroscópio e Magnetómetro) para classificar 7 atividades distintas.

## Funcionalidades

O projeto aborda o ciclo completo de Ciência de Dados:

1.  **Pré-processamento:** Limpeza de dados, tratamento de *outliers* e segmentação temporal.
2.  **Engenharia de Features:**
    * Extração de features estatísticas (manuais) no domínio do tempo e frequência.
    * Extração de **Embeddings** utilizando modelos de Deep Learning pré-treinados.
3.  **Data Augmentation:** Implementação de técnicas como SMOTE para balanceamento de classes.
4.  **Seleção de Features:** Redução de dimensionalidade com **PCA** e seleção com **ReliefF** e **Fisher Score**.
5.  **Classificação:** Implementação e validação de modelos **KNN** (K-Nearest Neighbors).

## Tecnologias Utilizadas

* **Linguagem:** Python
* **Análise de Dados:** NumPy, Pandas, SciPy
* **Machine Learning:** Scikit-learn (sklearn)
* **Deep Learning:** PyTorch (para extração de embeddings)
* **Visualização:** Matplotlib, Seaborn

## Análise de Resultados

Foram realizados testes rigorosos utilizando duas estratégias de validação: *Within-Subject* (Intra-sujeito) e *Between-Subjects* (Inter-sujeito).

### Desafios Encontrados (Generalização)
No cenário mais desafiante (*Between-Subjects*), onde o modelo é testado em participantes nunca vistos durante o treino, obteve-se uma accuracy de aproximadamente **58%**.

A análise da **Matriz de Confusão** revela padrões comportamentais importantes:
* **Estático vs Dinâmico:** O modelo distingue com excelente precisão atividades estáticas (ex: *Sitting/Standing* - Ativ. 1).
* **Sobreposição Biomecânica:** Existe uma forte confusão entre *Walking* (Ativ. 2) e *Standing* (Ativ. 3), bem como entre *Running* (Ativ. 4) e *Stairs* (Ativ. 5).

**Conclusão Técnica:** A assinatura biomecânica de atividades dinâmicas varia significativamente entre indivíduos. Sem dados de calibração do próprio utilizador (fine-tuning), o modelo captura a intensidade do movimento, mas perde as nuances cinemáticas específicas de cada pessoa.

## Como Executar

1.  Clone este repositório:
    ```bash
    git clone [https://github.com/AndreFonsecaRamos/ECAC](https://github.com/TEU-USER/NOME-DO-REPO.git)
    ```
2.  Instale as dependências:
    ```bash
    pip install numpy pandas scipy scikit-learn matplotlib seaborn torch
    ```
3.  Coloque a pasta do dataset `FORTH_TRACE_DATASET-master` na raiz do projeto.
4.  Execute o notebook principal:
    ```bash
    jupyter notebook main_activity_recognition.ipynb
    ```

## 👨‍💻 Autores

* **André Ramos**
* **Rodrigo Oliveira**

---
*Projeto desenvolvido no âmbito da Unidade Curricular de Extração de Conhecimento e Aprendizagem Computacional (ECAC).*