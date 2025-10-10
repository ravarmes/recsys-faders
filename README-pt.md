<h1 align="center">
    <img alt="RVM" src="https://github.com/ravarmes/recsys-faders/blob/main/assets/logo.jpg" />
</h1>

<h3 align="center">
  FaDeRS: Justiça e Despolarização em Sistemas de Recomendação
</h3>

<p align="center">WebMedia 2025: 31º Simpósio Brasileiro de Multimídia e Web</p>

<p align="center">
  <img alt="Contagem de linguagens do GitHub" src="https://img.shields.io/github/languages/count/ravarmes/recsys-faders?color=%2304D361">

  <a href="http://www.linkedin.com/in/rafael-vargas-mesquita">
    <img alt="Feito por Rafael Vargas Mesquita" src="https://img.shields.io/badge/made%20by-Rafael%20Vargas%20Mesquita-%2304D361">
  </a>

  <img alt="Licença" src="https://img.shields.io/badge/license-MIT-%2304D361">

  <a href="https://github.com/ravarmes/recsys-faders/stargazers">
    <img alt="Stargazers" src="https://img.shields.io/github/stars/ravarmes/recsys-faders?style=social">
  </a>
</p>

## :page_with_curl: Sobre o projeto <a name="-sobre"/></a>

Contexto: Sistemas de recomendação tornaram-se fundamentais na sociedade moderna ao ajudarem a navegar pela vasta quantidade de dados disponíveis, permitindo que usuários encontrem informações, produtos ou serviços de forma mais eficiente e personalizada. Eles impactam diretamente como as pessoas consomem dados, bens e recursos. Problema: Sistemas de recomendação frequentemente carecem de justiça e diversidade, resultando em serviços injustos e aumento da polarização de preferências. Solução: Este trabalho apresenta o FaDeRS (Fairness and Depolarization in Recommender Systems), uma abordagem voltada a aumentar a justiça e a diversidade em sistemas de recomendação. O FaDeRS ajusta predições por meio de perturbações controladas e otimização para mitigar injustiça individual e polarização sem modificar os dados de entrada. Teoria de SI: A pesquisa se relaciona com a teoria sócio-técnica, abordando um dos problemas sócio-algorítmicos, a discriminação algorítmica. Consideramos um conjunto específico de abordagens para codificar comportamentos justos. Método: A pesquisa aplicou um método quantitativo com experimentação utilizando dois datasets em contextos distintos, implementando um algoritmo de pós-processamento baseado na meta-heurística Simulated Annealing. Resumo dos Resultados: O framework proposto demonstrou reduções significativas na polarização (até 78,64%) e na injustiça individual (até 33,97%), com apenas um pequeno aumento no Root Mean Square Error (RMSE), indicando melhoria nas qualidades socialmente desejáveis dos sistemas sem sacrificar indevidamente a acurácia. Contribuições e Impacto na área de SI: A principal contribuição é um mecanismo que equilibra personalização e justiça, abordando simultaneamente polarização e injustiça individual sob a perspectiva dos itens, promovendo uma abordagem mais justa e diversa à recomendação.

### :balance_scale: Medidas de Justiça <a name="-medidas"/></a>

* **Polarização**: Para capturar a polarização, buscamos medir o quanto as avaliações dos usuários discordam. Assim, para medir a polarização dos usuários, consideramos as avaliações estimadas `$\hat{X}$` e definimos a métrica de polarização como a soma normalizada das distâncias euclidianas entre pares de avaliações estimadas dos usuários, isto é, entre linhas de `$\hat{X}$`.

* **Justiça individual**: Para cada item `$j$`, definimos `$\ell_j$`, a perda do item `$j$`, como o erro quadrático médio da estimativa sobre as avaliações conhecidas do item `$j$`.

### :chart_with_upwards_trend: Resultados <a name="-resultados"/></a>

[Link para o arquivo Excel](https://github.com/ravarmes/recsys-depolarize-fair/blob/main/_results-article.xlsx)

### Arquivos

| Arquivo                               | Descrição                                                                                                                                                                                                                                   |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AlgorithmDepolarizeFair              | Classe para implementar justiça e despolarização das recomendações a partir dos algoritmos de sistemas de recomendação.                                                                                                                     |
| AlgorithmUserFairness                | Classes para medir justiça (polarização, justiça individual e justiça de grupo) das recomendações de algoritmos de sistemas de recomendação.                                                                                                |
| RecSys                               | Classe no padrão fábrica para instanciar um sistema de recomendação com base em parâmetros de string.                                                                                                                                       |
| RecSysALS                            | Alternating Least Squares (ALS) para Filtragem Colaborativa é um algoritmo que otimiza iterativamente duas matrizes para melhor prever avaliações de usuários em itens, baseando-se na ideia de fatoração de matrizes.                       |
| RecSysKNN                            | K-Nearest Neighbors para Sistemas de Recomendação é um método que recomenda itens ou usuários com base na proximidade ou similaridade entre eles, usando a técnica dos K vizinhos mais próximos.                                             |
| RecSysNMF                            | Non-Negative Matrix Factorization para Sistemas de Recomendação utiliza a decomposição da matriz de avaliações em duas matrizes fatoriais não-negativas, revelando padrões latentes que podem ser usados para prever avaliações faltantes.   |
| RecSysSGD                            | Stochastic Gradient Descent para Sistemas de Recomendação é uma técnica de otimização que ajusta iterativamente os parâmetros do modelo para minimizar o erro de previsão nas avaliações, por meio de atualizações baseadas em gradientes calculados de forma estocástica. |
| RecSysSVD                            | Singular Value Decomposition para Sistemas de Recomendação é um método que fatora a matriz de avaliações em três matrizes menores, capturando informações essenciais sobre usuários e itens, o que facilita recomendações ao reconstruir a matriz original com dados faltantes preenchidos. |
| RecSysNCF                            | Neural Collaborative Filtering é uma abordagem moderna para filtragem colaborativa que usa redes neurais para modelar interações complexas e não-lineares entre usuários e itens, visando aprimorar a qualidade das recomendações.            |
| TestAlgorithmDepolarizeFair          | Script de teste do algoritmo de despolarização e justiça (AlgorithmDepolarizeFair).                                                                                                                   |

## :hammer_and_wrench: Uso

- Classe principal: src/TestAlgorithmDepolarizeFair.py — basta executar este script para rodar o algoritmo FaDeRS de ponta a ponta.
- Instalar dependências:
  - Windows: `python -m pip install -r requirements.txt`
- Executar os experimentos:
  - `python src/TestAlgorithmDepolarizeFair.py`
- Exemplo mínimo:
  - Crie uma matriz de avaliações estimadas usando um dos algoritmos RecSys e aplique o AlgorithmDepolarizeFair para reduzir a polarização e a injustiça individual mantendo a acurácia aceitável.

```python
from RecSys import RecSys
from AlgorithmUserFairness import Polarization, IndividualLossVariance, RMSE
from AlgorithmDepolarizeFair import AlgorithmDepolarize

# Ler dataset e estimar avaliações
recsys = RecSys(n_users=1000, n_items=1000, top_users=True, top_items=True)
X, users_info, items_info = recsys.read_dataset(1000, 1000, True, True, data_dir="data/MovieLens-1M")
omega = ~X.isnull()
X_est = recsys.compute_X_est(X, algorithm='RecSysSVD')

# Avaliar justiça/precisão
pol = Polarization(); ilv = IndividualLossVariance(X, omega, 0); rmse = RMSE(X, omega)
print(pol.evaluate(X_est), ilv.evaluate(X_est), rmse.evaluate(X_est))

# Aplicar algoritmo de despolarização/imparcialidade
alg = AlgorithmDepolarize(X, omega, 0)
list_X_est = alg.evaluate(X_est, alpha=0.4, h=20)

# Construir matrizes de otimização e recomendação final
losses = [ilv.get_losses(m) for m in list_X_est]
pols = [pol.get_polarizations(m) for m in list_X_est]
ZIL = AlgorithmDepolarize.losses_to_ZIL(losses, n_items=X.shape[1])
ZIP = AlgorithmDepolarize.polarizations_to_ZIP(pols, n_items=X.shape[1])
X_pi = AlgorithmDepolarize.make_matrix_X_pi_annealing(list_X_est, ZIL, ZIP)
```

Notas:
- A exportação para Excel requer o pacote openpyxl (já listado em requirements.txt).
- Se pretende usar RecSysNCF, instale o TensorFlow separadamente e garanta uma versão compatível.

## :email: Contato

Rafael Vargas Mesquita - [GitHub](https://github.com/ravarmes) - [LinkedIn](https://www.linkedin.com/in/rafael-vargas-mesquita) - [Lattes](http://lattes.cnpq.br/6616283627544820) - **ravarmes@hotmail.com**
