# Agentes de Reinforcement Learning no Atari Assault (PPO vs QR-DQN)

Treino e análise de agentes de **Deep Reinforcement Learning** no jogo
**Assault** (Atari 2600), com **Stable-Baselines3**. O projeto compara dois
paradigmas de aprendizagem, testa várias formas de ***reward shaping*** e
avalia a robustez dos agentes a ruído visual.

Projeto académico desenvolvido na Licenciatura em Inteligência Artificial e
Ciência de Dados da Faculdade de Ciências da Universidade do Porto (FCUP),
ano letivo 2025/2026.

---

## Objetivos

1. **Superar os 2000 pontos**, com agentes que aprendem sozinhos a gerir o
   sobreaquecimento (*overheat*) da arma e a priorizar alvos.
2. **Comparar dois algoritmos:**
   - **PPO** (*on-policy*), focado na estabilidade da convergência;
   - **QR-DQN** (*off-policy*, distribucional), que modela a distribuição
     completa dos retornos e reutiliza experiência através de um *replay buffer*.
3. **Engenharia de recompensa:** perceber se penalizações artificiais aceleram
   a aprendizagem num ambiente de recompensas esparsas.
4. **Teste de robustez:** medir a degradação dos agentes com ruído gaussiano
   nas observações.

---

## O ambiente

No **Assault** (`ALE/Assault-v5`), o agente controla um veículo na parte de
baixo do ecrã e tem de destruir naves inimigas enquanto se esquiva de
projéteis. A arma **sobreaquece** se disparar sem parar, por isso o agente
tem de aprender a equilibrar ataque, defesa e gestão do tempo de disparo.

---

## Wrappers personalizados

Todas as modificações ao ambiente foram implementadas como *wrappers* do
Gymnasium:

| Wrapper | Observação | Recompensa |
|---|---|---|
| `AssaultVisualWrapper` | Escala de cinzentos (+ ruído gaussiano opcional) | Original, sem alterações |
| `CustomAssaultWrapper` | Escala de cinzentos | Penalização por disparo (−0,5) e por morte (−25) |
| `CustomAssaultWrapperEnemies` | Escala de cinzentos | As anteriores + penalização proporcional ao número de inimigos no ecrã |

No `CustomAssaultWrapperEnemies`, os inimigos são **detetados com visão
computacional** (OpenCV):
- recorta-se a região do ecrã onde aparecem os inimigos;
- binariza-se a imagem e procuram-se componentes ligados;
- filtram-se os *blobs* pela área e pela proporção, para excluir explosões,
  a nave-mãe e os tiros.

---

## Metodologia

- **Treino paralelo** com `SubprocVecEnv` (4 ambientes) e *frame stacking* de
  4 frames.
- **Avaliação justa:** o `EvalCallback` avalia sempre os agentes no **ambiente
  original**, sem *reward shaping*. Assim, as pontuações são comparáveis entre
  todas as configurações.
- **Diagnóstico de treino:**
  - no PPO: KL, *explained variance*, entropia e *clip fraction*;
  - no QR-DQN: *quantile loss*, taxa de exploração e duração dos episódios.
- **Vídeos de cada agente**, para identificar comportamentos emergentes como
  *camping*, *spam* de tiros ou estratégias *kamikaze*.

---

## Experiências e resultados

Todas as pontuações abaixo são de avaliação, no ambiente original e sem
penalizações.

| Ambiente | Algoritmo | Resultado |
|---|---|---|
| Original | PPO (*baseline*) | ~550 pontos |
| Original | PPO (`n_steps` ajustado) | ~700 em média; aprendeu a "acampar" à esquerda |
| Original | QR-DQN | Picos de 1000–1400, com variância elevada |
| Penalização de tiro e morte | PPO | Aprendizagem efetiva, mas com sinais de instabilidade (KL elevado) |
| Penalização de tiro e morte | QR-DQN | **Pico de 3153 pontos**, médias de avaliação perto de 2000 |
| Penalização de tiro e morte | QR-DQN (*tuned*) | Pico de 2999, com convergência mais estável |
| \+ penalização por inimigos | PPO | Convergência prematura ("canto + disparo"); com *tuning*, ~900–1000 |
| \+ penalização por inimigos | QR-DQN (v1 e v2) | Picos de ~2500–3000; a v2 elimina mortes imediatas e *camping* |
| Ruído visual | QR-DQN | Forte degradação dos agentes treinados sem ruído; com treino em ruído, ~800–1000 |

### Principais conclusões

- **O QR-DQN superou claramente o PPO** neste jogo e ultrapassou o objetivo
  dos 2000 pontos. O *replay buffer* e a modelação distribucional lidam
  melhor com a variância do Assault.
- **O *reward shaping* ajudou:** as penalizações por disparo e por morte
  ensinaram o agente a gerir o *overheat* e a priorizar a sobrevivência.
- **Hiperparâmetros como o intervalo de atualização da *target network* e a
  fração de exploração** foram decisivos para estabilizar o treino.
- **Agentes treinados em condições perfeitas não são robustos a ruído.**
  Para cenários com incerteza, é preciso treinar diretamente com ruído.

---

## Vídeos

Todos os vídeos estão na pasta [`videos/`](videos/):

| Ambiente | PPO | QR-DQN |
|---|---|---|
| Original | [v1](videos/ambiente_default_ppo_v1.mp4) · [v2](videos/ambiente_default_ppo_v2.mp4) | [v1](videos/ambiente_default_qrdqn.mp4) |
| Penalização de tiro e morte | [v1](videos/ambiente_wrapper_ppo.mp4) | [v1](videos/ambiente_wrapper_qrdqn_v1.mp4) · [v2](videos/ambiente_wrapper_qrdqn_v2.mp4) |
| \+ penalização por inimigos | [v1](videos/ambiente_enemies_penalty_ppo_v1.mp4) · [v2](videos/ambiente_enemies_penalty_ppo_v2.mp4) | [v1](videos/ambiente_enemies_penalty_qrdqn_v1.mp4) · [v2](videos/ambiente_enemies_penalty_qrdqn_v2.mp4) |
| Ruído visual | — | [sem treino com ruído](videos/ambiente_enemies_penalty_qrdqn_v2_noise.mp4) · [treinado com ruído](videos/ambiente_noise.mp4) |

### QR-DQN com penalizações de tiro e morte

### QR-DQN com penalização por inimigos

### PPO no ambiente original

### Teste de robustez: agente treinado sem ruído 
### Teste de robustez: agente treinado com ruído

---

## Estrutura do repositório

```
.
├── assault_rl_agents.ipynb   # Notebook com wrappers, treino, análise e vídeos
├── requirements.txt
└── README.md
```

As pastas de cada experiência (`Ambiente_*/`), com modelos, *logs* do
TensorBoard e vídeos, são geradas ao correr o notebook e estão excluídas do
repositório (`.gitignore`).

---

## Como executar

Desenvolvido em **Python 3.10**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook assault_rl_agents.ipynb
```

O treino de cada agente (2 milhões de *timesteps*) é exigente. Recomenda-se
GPU e pelo menos 16 GB de RAM, por causa do *replay buffer* do QR-DQN.

---

## Autores

- João Pedro Nunes Rocha
- Paulo Diogo Lopes Pinto
- Rui Filipe Rocha Alvarilhão

Faculdade de Ciências da Universidade do Porto (FCUP), 2025/2026.
