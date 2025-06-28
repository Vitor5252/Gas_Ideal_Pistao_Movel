import numpy as np  # Biblioteca para cálculos numéricos
import tkinter as tk  # Biblioteca para criar a interface gráfica (GUI)
import matplotlib.pyplot as plt  # Biblioteca para plotar gráficos
from matplotlib.animation import FuncAnimation  # Para criar animações com matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Integração do matplotlib com tkinter

# Classe que define a simulação do pistão com massas, tamanhos e colisões
class SimulacaoPistao:
    def __init__(self, num_particulas, massa_pistao, area_pistao, velocidade_particulas, forca_estocastica_std=0.5):
        # Parâmetros do sistema
        self.num_particulas = num_particulas  # Número de partículas na simulação
        self.massa_pistao = massa_pistao  # Massa do pistão
        self.area_pistao = area_pistao  # Área do pistão
        self.velocidade_particulas = velocidade_particulas  # Velocidade inicial das partículas
        self.forca_estocastica_std = forca_estocastica_std  # Intensidade da força estocástica (ruído branco)
        self.gravidade = 9.8  # Aceleração gravitacional
        self.k_b = 1.38e-23  # Constante de Boltzmann
        self.n = 1.0  # Número de moles de gás (fixo)
        self.R = 8.314  # Constante universal dos gases

        # Inicialização das propriedades das partículas
        self.massas = np.random.uniform(0.5, 2.0, num_particulas)  # Massas aleatórias entre 0.5 e 2.0
        self.posicoes = np.random.rand(num_particulas) * 0.9  # Posições aleatórias no intervalo [0, 0.9]
        self.velocidades = np.random.choice([-1, 1], num_particulas) * velocidade_particulas  # Velocidades aleatórias (+/-)
        self.raios = np.random.uniform(0.01, 0.03, num_particulas)  # Raios aleatórios entre 0.01 e 0.03

        # Inicialização do pistão
        self.posicao_pistao = 1.0  # Posição inicial do pistão
        self.dt = 0.001  # Incremento temporal da simulação

        # Variáveis para monitorar o tempo e pressão
        self.tempo = []  # Lista que armazena os tempos
        self.dados_pressao = []  # Lista que armazena os dados de pressão

        # Atualiza a temperatura inicial do sistema
        self.atualizar_temperatura()

    def atualizar_temperatura(self):
        # Calcula a energia cinética média das partículas, agora levando em conta a massa das partículas
        energia_cinetica = 0.5 * np.mean(self.massas * self.velocidades**2)
        # Usa a energia cinética para calcular a temperatura
        self.temperatura = (2 / 3) * (energia_cinetica / self.R)

    def calcular_pressao(self):
        # Calcula o volume do gás com base na posição do pistão
        volume = self.area_pistao * self.posicao_pistao
        if volume <= 0:  # Evita divisão por zero
            volume = 1e-6
        # Usa a equação dos gases ideais para calcular a pressão
        pressao = (self.n * self.R * self.temperatura) / volume
        return pressao

    def tratar_colisoes(self):
        # Verifica colisões entre partículas considerando os raios
        for i in range(self.num_particulas):
            for j in range(i + 1, self.num_particulas):
                if abs(self.posicoes[i] - self.posicoes[j]) < (self.raios[i] + self.raios[j]):  # Soma dos raios
                    # Conservação do momento linear e troca de velocidades
                    mi, mj = self.massas[i], self.massas[j]
                    vi, vj = self.velocidades[i], self.velocidades[j]

                    # Fórmulas para colisões elásticas
                    self.velocidades[i] = (vi * (mi - mj) + 2 * mj * vj) / (mi + mj)
                    self.velocidades[j] = (vj * (mj - mi) + 2 * mi * vi) / (mi + mj)

    def passo_simulacao(self):
        # Adiciona forças estocásticas às partículas
        forca_estocastica = np.random.normal(0, self.forca_estocastica_std, self.num_particulas)
        self.velocidades += forca_estocastica * self.dt

        # Atualiza posições das partículas com base nas velocidades
        self.posicoes += self.velocidades * self.dt

        # Trata colisões com o pistão
        colisao_pistao = self.posicoes + self.raios >= self.posicao_pistao
        self.posicoes[colisao_pistao] = self.posicao_pistao - self.raios[colisao_pistao]
        self.velocidades[colisao_pistao] *= -1

        # Trata colisões com a base
        colisao_base = self.posicoes - self.raios <= 0
        self.posicoes[colisao_base] = self.raios[colisao_base]
        self.velocidades[colisao_base] *= -1

        # Trata colisões entre partículas
        self.tratar_colisoes()

        # Atualiza a temperatura e calcula a pressão
        self.atualizar_temperatura()
        pressao = self.calcular_pressao()

        # Calcula a força no pistão e atualiza sua posição
        forca_no_pistao = pressao * self.area_pistao
        forca_liquida = forca_no_pistao - (self.massa_pistao * self.gravidade)
        self.posicao_pistao += (forca_liquida / self.massa_pistao) * self.dt

        # Impede que o pistão desça abaixo de uma posição mínima
        if self.posicao_pistao < 0.05:
            self.posicao_pistao = 0.05

        # Atualiza o tempo e registra a pressão
        if len(self.tempo) == 0:
            self.tempo.append(0)
        else:
            self.tempo.append(self.tempo[-1] + self.dt)
        self.dados_pressao.append(pressao)

        return pressao

# Classe que define a interface gráfica e a animação
class AplicacaoSimulacao:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulação com Raios e Colisões")

        # Parâmetros iniciais da simulação
        self.num_particulas = 600
        self.massa_pistao = 3.0
        self.area_pistao = 0.1
        self.velocidade_particulas = 5.5

        # Inicializa a simulação
        self.simulacao = SimulacaoPistao(
            self.num_particulas,
            self.massa_pistao,
            self.area_pistao,
            self.velocidade_particulas
        )

        # Configuração do gráfico da simulação
        self.fig, self.ax_simulacao = plt.subplots(figsize=(8, 5))
        self.ax_simulacao.set_xlim(0, 1)
        self.ax_simulacao.set_ylim(0, 1.2)
        self.ax_simulacao.set_title("Simulação com Tamanhos de Partículas")

        # Elementos gráficos: partículas e pistão
        self.particles, = self.ax_simulacao.plot([], [], 'bo', markersize=2)
        self.piston, = self.ax_simulacao.plot([0, 1], [self.simulacao.posicao_pistao, self.simulacao.posicao_pistao], 'r-', lw=2)

        # Integração do matplotlib com tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        # Botão para exibir o histograma final
        self.btn_histograma = tk.Button(self.root, text="Mostrar Histograma Final", command=self.mostrar_histograma_final)
        self.btn_histograma.pack()

        # Configuração da animação
        self.ani = FuncAnimation(self.fig, self.atualizar_animacao, interval=20, blit=False)

    def atualizar_animacao(self, frame):
        # Atualiza a simulação
        self.simulacao.passo_simulacao()

        # Atualiza as posições das partículas e do pistão no gráfico
        self.particles.set_data(
            np.random.rand(self.simulacao.num_particulas), 
            self.simulacao.posicoes
        )
        self.piston.set_ydata([self.simulacao.posicao_pistao, self.simulacao.posicao_pistao])

        # Atualiza o canvas do tkinter
        self.canvas.draw()

    def mostrar_histograma_final(self):
    # Exibe o histograma das velocidades finais (usando valores absolutos)
        fig_histograma, ax = plt.subplots(figsize=(6, 4))
        velocidades_absolutas = np.abs(self.simulacao.velocidades)  # Transforma em valores absolutos
        ax.hist(velocidades_absolutas, bins=20, color='blue', alpha=0.7)
        ax.set_title("Histograma das Velocidades Finais (|v|)")
        ax.set_xlabel("Velocidade")
        ax.set_ylabel("Frequência")
        plt.show()


# Inicializa a aplicação
if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacaoSimulacao(root)
    root.mainloop()
