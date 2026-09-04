# Lombada eletrônica

Medição de velocidade e leitura de placa a partir de câmeras IP comuns, sem laço
indutivo, sem radar e sem hardware dedicado. Uma câmera apontada para a via, dois
pontos de referência no chão e a distância entre eles bastam.

O sistema mede a **velocidade média na base** — a mesma grandeza que um radar de
laço duplo mede: o tempo que o veículo leva entre duas linhas transversais a uma
distância conhecida.

> **Este não é um instrumento metrológico.** Leia [Limites legais](#limites-legais)
> antes de qualquer uso. O destino previsto é monitoramento privado: condomínios,
> empresas, pátios e vias internas.

---

## Como a medida funciona

**1. Homografia a partir de quatro pontos.** Os quatro cantos da base marcados na
imagem, mais a distância real entre as linhas, definem uma transformação
perspectiva da imagem para o plano da via. Cada pixel vira uma coordenada em
metros.

A largura da faixa **não precisa ser medida**. Se a largura suposta for `k` vezes
a real, a homografia resultante é `diag(k,1,1) · H`: a coordenada lateral escala,
a longitudinal fica idêntica. E só a longitudinal entra no cálculo de velocidade.
Isso está verificado em [`tests/test_geometry.py`](tests/test_geometry.py).

**2. Ponto de contato, não centro da caixa.** A posição usada é o centro da aresta
inferior da caixa do veículo — onde ele toca o solo. O centroide carrega a altura
do veículo e projeta um erro sistemático que muda de sinal conforme o veículo se
aproxima.

**3. Ajuste quadrático de Y(t).** Sobre uma lombada o veículo desacelera de fato.
Forçar reta faz o resíduo comer a medida: em dados de campo, o ajuste linear ficou
em R² ≈ 0,86 enquanto o quadrático passou de 0,99, e a dispersão da velocidade caiu
de ±13,7% para ±5,2%.

**4. Cruzamentos resolvidos na curva, não no quadro mais próximo.** A 50 km/h e
12 fps há 1,2 m entre quadros. Arredondar o instante de cruzamento para o quadro
mais próximo joga fora vários por cento da medida; resolver a raiz recupera isso.

**5. Recusa explícita.** R² baixo, poucas amostras, base pouco coberta ou
velocidade implausível **não viram medida**. Nenhuma passagem duvidosa entra no
banco — é melhor não medir do que medir errado.

## O que existe para não falhar em silêncio

Os dois modos de falha caros deste tipo de sistema não geram erro nenhum — só
param de produzir eventos:

- **Teto de velocidade por perda de ID.** Rastreadores que dependem da previsão de
  um Kalman precisam de dois ou três quadros para travar. Um carro rápido
  atravessa a base em meio segundo e troca de ID a cada quadro; acima de certa
  velocidade o sistema simplesmente deixa de ver. Aqui a associação tem um
  segundo estágio por distância entre centros, com portão proporcional ao tamanho
  da caixa e restrito à mesma classe, que cobre o quadro em que ainda não há
  velocidade para extrapolar.
- **Cancelamento catastrófico na raiz.** Velocidade constante produz ajuste com
  curvatura numérica na ordem de 1e-14; com a fórmula de Bhaskara direta,
  `-b + √Δ` é a subtração de dois números quase iguais e a velocidade sai errada
  por ordens de grandeza, sem exceção alguma. A fórmula usada é a estável.

Ambos têm teste de regressão.

## Instalação

```bash
pip install -e ".[video,detect,lpr,lpr-trocr,dev]"
```

O núcleo — geometria, velocidade, votação de placa, configuração, banco — depende
só de `numpy` e `PyYAML`. Cada peça pesada é um extra, de modo que a suíte de
testes roda sem GPU e sem baixar modelo.

Com GPU NVIDIA, some `gpu` (troca `onnxruntime` por `onnxruntime-gpu`):

```bash
pip install -e ".[video,detect,lpr,lpr-trocr,gpu]"
```

> **Não instale o `rapidocr` solto.** Ele declara `opencv-python-headless` sem
> fixar versão, e foi por essa porta que o OpenCV 5.0 entrou num parque de
> câmeras e quebrou a decodificação RTSP inteira. O extra `[lpr]` repete o pin
> `<5` de propósito.

## Uso

### Pelo navegador (mais rápido para começar)

```bash
lombada web                   # abre em http://127.0.0.1:8000
```

Cadastra a câmera, sonda o RTSP, calibra a base clicando nos 4 cantos do quadro
e mostra as métricas. A tela **edita o `cameras.yaml` de verdade** — não um
cadastro paralelo —, então o que você configura ali é exatamente o que
`lombada check` valida e `lombada run` executa.

Três coisas que ela faz e o terminal não faz bem:

- **Sonda o fluxo medindo, não perguntando.** Reporta a resolução do quadro
  decodificado e o FPS cronometrado, ao lado do que a câmera *declara*. É como
  se pega o `subtype=1` de Dahua que devolve o fluxo principal em 5 MP, e a
  câmera que anuncia 25 fps entregando 6.
- **Diz se dá para medir a velocidade que te interessa.** Calcula quantas
  amostras cabem na base a 30, 40, 60 e 80 km/h com o FPS real. Abaixo de 4 o
  ajuste não se sustenta e a passagem é recusada — a câmera para de ver os
  carros rápidos sem gerar erro nenhum.
- **Calibra por clique.** Marcar os 4 cantos na imagem é bem menos sujeito a
  erro do que digitar oito números num YAML.

A câmera nasce **desabilitada** e só liga quando a calibração é salva: sem os
4 pontos não existe medida, e deixá-la ligada daria a impressão de um sistema
funcionando que na verdade não mede nada.

O servidor escuta em `127.0.0.1` de propósito — a tela grava e exibe URLs RTSP
com senha. Para expor em outra interface é preciso passar `--host`, e ele avisa.

### Pelo terminal

```bash
cp config/cameras.example.yaml config/cameras.yaml
cp .env.example .env          # senhas das câmeras ficam aqui, fora do git

lombada check                 # valida configuração e calibração
lombada snapshot --camera lombada-01 --output base.jpg
lombada project --camera lombada-01 960 710   # confere um pixel em metros
lombada run                   # roda as câmeras habilitadas
lombada report --days 7
lombada purge                 # aplica a retenção configurada
```

### Calibração em campo

1. `lombada snapshot` para pegar um quadro da câmera.
2. Escolha duas linhas transversais à via — uma faixa pintada, uma junta do
   asfalto, a própria lombada — e **meça a distância entre elas com trena**. Essa
   é a única medida que precisa ser exata.
3. Marque os quatro cantos no quadro, nesta ordem: entrada-esquerda,
   entrada-direita, saída-direita, saída-esquerda. Preencha `base.image_points`.
4. `lombada check` reprojeta os cantos e mostra os metros de volta; e
   `lombada project x y` confere qualquer pixel.

Erro na distância medida vira erro proporcional na velocidade: 10 cm errados em
8 m são 1,25% em toda leitura daquela câmera.

## Configuração

Tudo em [`config/cameras.example.yaml`](config/cameras.example.yaml), comentado.

Senhas nunca ficam no YAML: escreva `${CAM01_SENHA}` e defina no ambiente. A carga
falha se a variável não existir — melhor não subir do que subir sem credencial. O
`cameras.yaml` pode então ir para o git sem vazar nada, e a URL RTSP também sai
mascarada dos logs.

## Arquitetura

```
capture.py    RTSP em thread própria, grab() contínuo + retrieve() na taxa alvo
detect.py     detector de veículos plugável (ultralytics | stub)
track.py      associação em dois estágios: IoU sobre a caixa prevista + portão
geometry.py   homografia imagem -> plano da via
speed.py      ajuste de Y(t) e velocidade média na base
lpr.py        ensemble de motores de OCR + votação posicional (quadros x motores)
evidence.py   quadro anotado, recorte da placa, manifesto com SHA-256
storage.py    SQLite, consultas e retenção
pipeline.py   orquestração por câmera
cli.py        web | check | run | snapshot | project | bench | report | purge
probe.py      sondagem do RTSP: mede o que a camera entrega de fato
registry.py   cadastro de cameras gravado no proprio cameras.yaml
webapp.py     servidor local (so stdlib) da tela de cadastro e teste
```

O laço quente só detecta e rastreia. **OCR não roda dentro do laço**: ele é o erro
clássico deste tipo de sistema — passa a ditar a taxa de quadros, a taxa cai, o
rastreio perde ID em velocidade alta e o sistema deixa de pegar exatamente as
passagens que deveria. O OCR roda uma vez por passagem, sobre os recortes já
guardados, depois que o veículo saiu da base.

## Leitura de placa: ensemble, não "o melhor modelo"

O reconhecimento é um **ensemble de motores que erram diferente**, e a votação
acontece em duas dimensões ao mesmo tempo:

- entre **quadros** — a mesma placa aparece várias vezes na passagem;
- entre **motores** — cada recorte passa por todos eles.

O padrão junta um **CTC** e um **seq2seq com atenção**, que falham em coisas
distintas (borrão, inclinação, sujeira, caractere colado). A interseção dos dois
erros é bem menor que cada um isolado.

| Motor | Papel | O que é |
|---|---|---|
| `rapidocr` | localiza **e** lê | PP-OCR sobre ONNX Runtime, dicionário de caracteres público |
| `trocr` | só lê | `microsoft/trocr-small-printed`, decodificação por atenção |
| `fast_plate_ocr` | localiza e lê | especialista em placa, disponível mas **fora do padrão** |

`engines` é uma lista ordenada: o **primeiro precisa saber localizar**, porque é
a caixa dele que vira o recorte entregue aos demais. TrOCR não localiza nada
sozinho — sem um primário na frente, ele leria o veículo inteiro como uma linha
de texto e devolveria ruído.

Três detalhes que fazem o ensemble valer a pena de fato:

1. **A massa de voto é normalizada por motor.** Cada motor contribui com o mesmo
   total, independente de quantos quadros ele leu. Sem isso, o motor que
   devolveu leitura em seis quadros afogaria o que devolveu em um, e a eleição
   passaria a ser decidida por quem falou mais alto — que é exatamente o erro
   sistemático que o ensemble existe para evitar.
2. **`agreement` é um sinal separado da confiança.** Modelo confiante e errado é
   comum; dois modelos independentes confiantes e errados no mesmo caractere é
   raro. `min_agreement` permite exigir que uma fração dos motores concorde
   antes de gravar a placa — menos placas lidas, quase nenhuma errada.
3. **Um motor que explode não derruba a passagem.** Cada motor roda protegido; a
   velocidade é medida de qualquer jeito, com ou sem placa.

Depois da votação, o formato (`LLLNNNN` antiga, `LLLNLNN` Mercosul) desambigua
`O`/`0`, `I`/`1`, `S`/`5` nas posições cujo tipo ele já determina. A quinta
posição nunca é coagida: é ela que decide qual dos dois formatos é.

O manifesto de evidência e o banco registram **quais motores leram** a placa.

## Evidência

Cada passagem registrada gera, sob `evidence_dir/<câmera>/<dia>/`:

- `*_overview.jpg` — o quadro com a base desenhada, velocidade, horário e placa;
- `*_plate.jpg` — o recorte usado na leitura;
- `*_manifest.json` — a calibração vigente, os parâmetros de qualidade da medida
  (R², nº de amostras, resíduo, extrapolação) e o SHA-256 de cada arquivo.

O manifesto é o que permite discutir uma medida depois. Velocidade sem procedência
não se defende.

## Testes

```bash
pytest
```

A suíte cobre o núcleo determinístico: propriedades da homografia, exatidão do
ajuste em velocidade constante e em desaceleração, os critérios de recusa, a
votação de placa, a associação do rastreador em velocidade alta, a interpolação de
variáveis de ambiente e a retenção.

## Limites legais

**Metrologia.** Medidor de velocidade usado para autuação de trânsito no Brasil
precisa de aprovação de modelo e verificação metrológica pelo Inmetro, instalação
e sinalização conforme a regulamentação do CONTRAN, e a fiscalização só pode ser
feita pelo órgão de trânsito com circunscrição sobre a via. **Nada disso se aplica
a este software** — ele não é aprovado, não é verificado e não produz auto de
infração. A função `apply_legal_tolerance` reproduz a aritmética da tolerância
(7 km/h até 100 km/h, 5% acima) porque ela é a referência prática de comparação;
isso não torna a medida válida para autuar. Confirme a redação vigente das normas
antes de usar o número em qualquer contexto formal.

Usos para os quais isto serve: monitoramento e gestão de tráfego interno,
condomínios e empresas, dado para decidir onde intervir na via, avisos e
advertências privadas.

**LGPD.** Placa é dado pessoal — identifica indiretamente uma pessoa. Operador
privado não tem a exclusão do art. 4º, III (que só alcança o Estado), então está
integralmente sob a lei. O que isso exige na prática:

- base legal definida e registrada antes de ligar (legítimo interesse do art. 7º,
  IX, com teste de balanceamento documentado, é o caminho usual em condomínio);
- sinalização visível informando o monitoramento e a finalidade;
- retenção mínima — daí `retention_days` e o comando `purge`, que apagam o
  registro **e** a evidência em disco;
- acesso restrito e registrado; não compartilhar com terceiros sem base legal.

Um medidor de velocidade que guarda placa sem prazo deixa de ser um medidor de
velocidade e vira um histórico de deslocamento das mesmas pessoas. A retenção não
é enfeite de configuração.

## Licença

MIT — veja [LICENSE](LICENSE).
