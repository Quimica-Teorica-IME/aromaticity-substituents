import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from sklearn.linear_model import LinearRegression
from adjustText import adjust_text  # Biblioteca para ajustar rótulos

# Caminho do arquivo
file_path = "hammettxQ.csv"

# Carregar o arquivo CSV com separador ","
try:
    data = pd.read_csv(file_path, sep=",", encoding="utf-8")
except UnicodeDecodeError:
    data = pd.read_csv(file_path, sep=",", encoding="latin1")

# Diagnosticar nomes das colunas
print("Colunas detectadas:", data.columns)

# Remove espaços extras das colunas
data.columns = data.columns.str.strip()

# Verifica se as colunas estão no arquivo
sigma_columns = ['σm', 'σm0', 'σR', 'σI', 'σp', 'σp0', 'σp+', 'σp-']
q_columns = ['Total |Q2| (ring atoms)', 'Total Qzz (ring atoms)', '|Q2| origin', 'Qzz origin', '|Q2|(1)', 'Q2(1)zz']

# Criar diretório para salvar os gráficos
output_dir = "graficos_hammettxQ"
os.makedirs(output_dir, exist_ok=True)

# Função para limpar nomes de arquivos
def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)

# Função para formatar os rótulos com números como subscritos e símbolos + ou - sobrescritos
def format_label(label):
    # Substitui números por subscritos, mantendo o resto do texto
    label = re.sub(r'(\d+)', r'$_{\1}$', label)
    
    # Verifica se o label termina com + ou -
    if label.endswith('+'):
        label = label[:-1] + r'$\mathregular{^{+}}$'  # Formata o símbolo + como superescrito
    elif label.endswith('-'):
        label = label[:-1] + r'$\mathregular{^{-}}$'  # Formata o símbolo - como superescrito
    
    return label

# Loop para criar gráficos
for sigma in sigma_columns:
    for q in q_columns:
        if sigma in data.columns and q in data.columns:
            subset = data[['-X', sigma, q]].dropna()  # Inclui a coluna -X no subset
            if not subset.empty:
                x = subset[sigma].values.reshape(-1, 1)  # Converte para formato adequado para regressão
                y = subset[q].values
                
                # Ajuste linear (regressão linear)
                model = LinearRegression()
                model.fit(x, y)
                y_pred = model.predict(x)
                
                # Coeficientes da reta
                slope = model.coef_[0]
                intercept = model.intercept_
                r2 = model.score(x, y)
                
                # Gráfico de dispersão
                plt.figure(figsize=(10, 8))  # Aumenta o tamanho da figura
                plt.scatter(x, y, color='lightskyblue', edgecolor='black', alpha=0.7, s=500)  # Aumenta o tamanho dos pontos (s=100)
                plt.xlabel(sigma, fontsize=25)  # Aumenta o tamanho do rótulo do eixo X
                plt.ylabel(q, fontsize=25)  # Aumenta o tamanho do rótulo do eixo Y
                plt.xticks(fontsize=25)  # Aumenta o tamanho dos valores do eixo X
                plt.yticks(fontsize=25)  # Aumenta o tamanho dos valores do eixo Y
                plt.grid(True)
                
                # Adiciona os rótulos aos pontos com ajuste automático
                texts = []
                for i, label in enumerate(subset['-X']):
                    # Formata o label para adicionar subscritos e sobrescritos
                    formatted_label = format_label(label)
                    texts.append(plt.text(x[i], y[i], formatted_label, fontsize=25))  # Triplica o tamanho da fonte dos labels
                
                # Ajusta os rótulos para evitar sobreposição e adiciona setas
                adjust_text(
                    texts,
                    only_move={'points': 'y', 'text': 'y'},  # Movimenta apenas no eixo Y
                    expand_text=(1.1, 1.2),  # Expande o espaço disponível para os rótulos
                    arrowprops=dict(arrowstyle="->", color="black", lw=2),  # Setas agora são mais grossas e pretas
                )
                
                # Verifica se o R2 é maior ou igual a 0.6 para plotar a reta
                if r2 >= 0.6:
                    # Traçar a linha de regressão (reta ajustada) - linha pontilhada com maior espaçamento
                    plt.plot(x, y_pred, color='red', linestyle='--', linewidth=2, dashes=(1, 8), label=f'{slope:.2f}x + {intercept:.2f}')
                
                    # Cria a equação e o R² como string
                    equation_text = f'$y = {slope:.2f}x + {intercept:.2f}$\n$R^2 = {r2:.3f}$'
                    
                    # Adiciona a equação e R² no canto inferior esquerdo, a menos que seja um caso especial
                    if not (sigma in ['σp', 'σp-', 'σp0', 'σI'] and q == 'Qzz origin'):
                        plt.text(0.05, 0.05, equation_text, transform=plt.gca().transAxes, fontsize=20,
                                 verticalalignment='bottom', horizontalalignment='left', color='black',
                                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
                
                # Verifica se o gráfico deve ter os rótulos no canto inferior direito
                if sigma in ['σp', 'σp-', 'σp0', 'σI'] and q == 'Qzz origin':
                    # Adiciona a equação e R² no canto inferior direito (não no canto inferior esquerdo)
                    equation_text = f'$y = {slope:.2f}x + {intercept:.2f}$\n$R^2 = {r2:.3f}$'
                    plt.text(0.95, 0.05, equation_text, transform=plt.gca().transAxes, fontsize=20,
                            verticalalignment='bottom', horizontalalignment='right', color='black',
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
                
                # Remover o título do gráfico
                plt.title("")  # Não coloca o título "Relação entre..."
                
                # Salva o gráfico
                plt.tight_layout()
                scatter_filename = sanitize_filename(f"scatter_{q}_vs_{sigma}.png")
                plt.savefig(os.path.join(output_dir, scatter_filename))
                plt.close()

print(f"Gráficos de dispersão gerados e salvos na pasta '{output_dir}'.")