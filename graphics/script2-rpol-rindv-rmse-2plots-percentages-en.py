import matplotlib.pyplot as plt
import numpy as np

# Função para calcular a variação percentual em relação ao valor inicial
def percentual_variacao(data):
    return [(val - data[0]) / data[0] * 100 for val in data]

dados_percentual = {
    'MovieLens-1M': {
        'α': [0, 0.2, 0.4, 0.6, 0.8],
        r'$R_{pol}$': percentual_variacao([0.127705059, 0.092975306, 0.065485454, 0.045046727, 0.027279156]),
        r'$R_{indv}$': percentual_variacao([0.034572808, 0.033503484, 0.030058334, 0.026021418, 0.023797359]),
        r'$RMSE$': percentual_variacao([0.875528094, 0.879731532, 0.892053897, 0.910152787, 0.935862058])
    },
    'GoodBooks-10k': {
        'α': [0, 0.2, 0.4, 0.6, 0.8],
        r'$R_{pol}$': percentual_variacao([0.254573171, 0.177911196, 0.128459384, 0.083471240, 0.056132575]),
        r'$R_{indv}$': percentual_variacao([0.058675519, 0.056173741, 0.051789997, 0.044843601, 0.038740893]),
        r'$RMSE$': percentual_variacao([0.855771561, 0.860838742, 0.873177885, 0.895149461, 0.920223485])
    }
}

# Gráficos com a variação percentual
fig, axs = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle('Percentage Variation of $R_{pol}$, $R_{indv}$, and $RMSE$', fontsize=16)

# Iteração pelos dados para criar os gráficos
for i, (title, data) in enumerate(dados_percentual.items()):
    ax = axs[i]

    ax.plot(data['α'], data[r'$R_{pol}$'], marker='o', linestyle='-', color='tab:blue', label=r'$R_{pol}$')
    ax.plot(data['α'], data[r'$R_{indv}$'], marker='o', linestyle='-', color='tab:green', label=r'$R_{indv}$')
    ax.plot(data['α'], data[r'$RMSE$'], marker='o', linestyle='-', color='tab:red', label=r'$RMSE$')

    # Adicionando o título do subplot
    ax.set_title(title, fontsize=14)

    # Define os ticks do eixo X para serem apenas os valores inteiros especificados
    ax.set_xticks(data['α'])
    ax.set_xlabel('Adjustment Factor (α)')
    ax.grid(True, color='lightgray')

    # Ajustar os limites dos eixos y para ambos os gráficos
    ax.set_ylim(-80, 20)

    # Adicionar legendas
    if i == 0:
        ax.set_ylabel('Percentage Variation (%)')

    # Combinar as legendas
    lines_1, labels_1 = ax.get_legend_handles_labels()
    ax.legend(lines_1 , labels_1 , loc='lower left')

# Ajustar o espaçamento entre subplots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
