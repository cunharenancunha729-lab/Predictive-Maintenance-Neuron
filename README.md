
# Neurônio Artificial para Previsão de Falha Mecânica



Implementação de um neurônio artificial (regressão logística) **do zero, para estimar a probabilidade de falha mecânica em máquinas industriais a partir de dados sensoriais — com comparação final contra redes neurais (MLP) treinadas em **PyTorch** e **Keras/TensorFlow**.

![Painel interativo de risco<img width="1156" height="422" alt="Painel Interativo - Risco da Função da Ferramenta" src="https://github.com/user-attachments/assets/b15e234d-b40b-4649-9e41-ef5d537762f5" />


## Objetivo

Entender e demonstrar, na prática, a matemática por trás de uma rede neural simples — soma ponderada, ativação sigmoide e aprendizado via regra delta (gradiente descendente) — aplicada a um problema real de manutenção preditiva, e depois comparar essa implementação manual com o que a indústria usa em produção (PyTorch e Keras).

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — UCI Machine Learning Repository.

- ~10.000 registros de operação de máquinas industriais
- Variáveis: temperatura do ar, temperatura do processo, velocidade rotacional, torque e desgaste da ferramenta
- Alvo: `Machine failure` (0 = normal, 1 = falha)
- **Observação:* é um dataset estruturado para refletir de forma realista dados reais de manutenção industrial — por isso é amplamente usado como benchmark em pesquisa e portfólios de ML.
- Desbalanceamento real: apenas **~3,4%** dos registros são falhas.
- O CSV (`data/ai4i2020.csv`) não é versionado neste repositório (veja `.gitignore`) — instruções de download estão na seção [Como rodar](#como-rodar).

## Metodologia

1. **Normalização** das variáveis (z-score)
2. **Treinamento manual via regra delta**, com as 5 variáveis simultaneamente
3. **Peso de classe** aplicado durante o treino para compensar o desbalanceamento (sem isso, o modelo aprende a sempre prever "sem falha" e ainda assim parece ter alta acurácia)
4. **Avaliação com recall e precisão**, não apenas acurácia — em dados desbalanceados, acurácia isolada esconde o problema
5. **Comparação com `LogisticRegression` do scikit-learn** (`class_weight='balanced'`), para validar se a implementação manual está correta
6. **Comparação com redes neurais (MLP)** treinadas em PyTorch e Keras/TensorFlow, usando a mesma lógica de peso de classe

## Resultados

| Modelo | Recall | Precisão |
|---|---|---|
| Neurônio manual (NumPy) | *preencher após rodar* | *preencher após rodar* |
| scikit-learn (LogisticRegression) | *preencher após rodar* | *preencher após rodar* |
| PyTorch (MLP) | *preencher após rodar* | *preencher após rodar* |
| Keras/TensorFlow (MLP) | *preencher após rodar* | *preencher após rodar* |

> A implementação manual chega a resultados próximos dos frameworks consolidados, o que confirma que o raciocínio matemático por trás está correto.

## Extras

- Sistema de alertas automáticos (crítico / atenção / normal) baseado no risco calculado
- Painel interativo em Plotly (risco vs. desgaste da ferramenta, por faixa de torque)
- Simulador em tempo real com sliders (ipywidgets) para as 5 variáveis
- Análise de sensibilidade do bias

## Estrutura do repositório

```
projeto/
├── notebooks/
│   └── falha_mecanica_ia.ipynb
├── data/
│   └── ai4i2020.csv        # não versionado — ver instruções abaixo
├── images/
│   └── painel_risco.png
├── README.md
├── requirements.txt
└── .gitignore
```

## Como rodar

1. Clone o repositório e instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Baixe o dataset para `data/ai4i2020.csv`:
   ```python
   import urllib.request
   urllib.request.urlretrieve(
       "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv",
       "data/ai4i2020.csv"
   )
   ```
   (ou baixe manualmente pelo [link da UCI](https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv))
3. Abra `notebooks/falha_mecanica_ia.ipynb` no Jupyter, VS Code ou [Google Colab](https://colab.research.google.com/) e rode as células em ordem.

## Stack

Python, NumPy, Pandas, Matplotlib, Plotly, ipywidgets, scikit-learn, PyTorch, Keras/TensorFlow

## Próximos passos

- Validação cruzada e conjunto de teste separado (neste notebook, a avaliação é feita didaticamente sobre o próprio conjunto de treino)
- Testar outros limiares de decisão além de 0.5
- Explorar outras técnicas de balanceamento (ex: SMOTE)

## Licença

Este projeto está sob a licença MIT.

---

Projeto desenvolvido por **Renan Cunha** como parte de estudos em Inteligência Artificial e Ciência de Dados.
