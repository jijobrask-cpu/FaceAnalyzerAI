# 🧠 Face Analyzer AI

> Análise facial geométrica em tempo real via webcam ou imagem.
> ⚠️ **Estimativa não científica** — baseada em proporções geométricas e qualidade de imagem.

---

## 📋 Requisitos

- Python 3.8 ou superior
- Webcam (opcional — também funciona com imagens estáticas)

---

## 🚀 Instalação

```bash
# 1. Clone ou extraia o projeto
cd FaceAnalyzerAI

# 2. (Recomendado) Crie um ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute
python main.py
```

---

## 🖥️ Interface

| Elemento | Descrição |
|---|---|
| Painel esquerdo | Feed da webcam com malha facial (468 pontos) |
| Face Score | Nota de 0 a 10 atualizada em tempo real |
| Barras de métricas | Pontuação individual por critério |
| Dicas | Feedback automático de posicionamento |

---

## 📊 Métricas Analisadas

| Métrica | Peso | Descrição |
|---|---|---|
| Simetria Facial | 25% | Comparação dos lados esquerdo e direito |
| Proporções | 20% | Terços faciais, razão olhos/rosto, boca |
| Harmonia | 15% | Espaçamento interpupilar, posição nariz/boca |
| Estrutura Mandibular | 15% | Contorno jawline, razão largura/altura |
| Qualidade da Imagem | 10% | Nitidez (Laplaciano) + iluminação |
| Centralização | 10% | Posição e tamanho do rosto no frame |
| Estabilidade | 5% | Taxa de detecção nos últimos frames |

---

## 🧵 Arquitetura de Threads

```
Thread Principal (GUI)
    └─ Tkinter event loop + atualização de widgets a 30fps

CameraThread (daemon)
    └─ Captura de frames da webcam via OpenCV

AnalysisThread (daemon)
    └─ Lê frames → FaceAnalyzer.analyze() → score_queue
    └─ score_queue → Thread Principal (GUI update)
```

---

## 📁 Estrutura do Projeto

```
FaceAnalyzerAI/
├── main.py           # Ponto de entrada
├── gui.py            # Interface Tkinter
├── face_analysis.py  # Engine de análise (MediaPipe)
├── camera_thread.py  # Thread de captura
├── utils.py          # Funções auxiliares
├── requirements.txt  # Dependências
├── README.md         # Este arquivo
└── assets/           # Recursos estáticos (ícones, etc.)
```

---

## 🔬 Notas Técnicas

- **MediaPipe Face Mesh**: 468 pontos faciais por frame
- **Smoothing**: Média móvel exponencial (α=0.15) para estabilizar scores
- **Detecção mínima**: `min_detection_confidence=0.5`
- **Screenshots**: Salvos em `screenshots/` com timestamp

---

## ⚠️ Aviso Legal

Este sistema é uma **estimativa geométrica experimental**.
Não mede beleza, atratividade, saúde ou qualquer característica pessoal real.
Os scores são baseados em proporções matemáticas e qualidade técnica da imagem.
