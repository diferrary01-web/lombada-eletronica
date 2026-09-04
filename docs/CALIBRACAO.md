# Calibração e orçamento de erro

## Procedimento

1. **Escolha as duas linhas.** Qualquer par de referências transversais à via que
   apareça bem no quadro: faixa pintada, junta de asfalto, a borda da própria
   lombada, ou duas marcas feitas com fita. Elas precisam ser visíveis na imagem
   e alcançáveis a pé.

2. **Meça a distância com trena.** Entre 6 e 12 m funciona bem. Menos que isso e o
   tempo de travessia fica curto demais para acumular amostras; mais que isso e o
   veículo muda de velocidade no meio da base, o que já não é uma média útil.

   Meça no eixo do fluxo, não na diagonal. **Esta é a única medida que precisa ser
   exata.**

3. **Capture um quadro.**

   ```bash
   lombada snapshot --camera lombada-01 --output base.jpg
   ```

4. **Leia os quatro cantos**, em qualquer editor que mostre coordenadas de pixel,
   nesta ordem:

   ```
   0  entrada-esquerda      1  entrada-direita
   3  saída-esquerda        2  saída-direita
   ```

   "Entrada" é onde o veículo entra na base; o sentido do tráfego vai de `Y=0`
   para `Y=distance_m`. Se você inverter, o sistema mede igual e marca a passagem
   como `sentido_invertido`.

5. **Confira.**

   ```bash
   lombada check                                  # reprojeta os 4 cantos
   lombada project --camera lombada-01 960 710    # confere um pixel qualquer
   ```

   Os cantos têm que voltar em `(±largura/2, 0)` e `(±largura/2, distance_m)`. Um
   ponto no meio da base tem que dar um `Y` entre 0 e `distance_m` — e, por causa
   da perspectiva, o meio da *imagem* não é o meio da via: fica mais longe.

## O que precisa ser preciso e o que não precisa

| Parâmetro | Precisão exigida | Por quê |
|---|---|---|
| `distance_m` | **alta** | Entra direto na velocidade: erro relativo aqui é erro relativo em toda leitura da câmera. 10 cm errados em 8 m são 1,25%. |
| `image_points` | média | Erro de alguns pixels desloca as linhas em centímetros. Marque na base do plano da via, não no topo da faixa pintada. |
| `lane_width_m` | **nenhuma** | Escala só a coordenada lateral. A longitudinal, que é a que mede velocidade, não muda. |

Vale insistir no terceiro: a largura suposta produz `H' = diag(k,1,1) · H`, e a
razão que define `Y` é idêntica. Está verificado em `tests/test_geometry.py`.

## Orçamento de erro

Fontes que sobram depois da calibração, em ordem de peso:

1. **Distância medida.** Erro sistemático, proporcional, e igual em todas as
   passagens daquela câmera. Só se corrige medindo de novo.

2. **Ponto de contato.** O sistema usa o centro da aresta inferior da caixa. Se o
   detector corta a caixa no para-choque em vez do pneu, aparece um viés que muda
   com a distância. Compare passagens do mesmo veículo em faixas diferentes para
   detectar.

3. **Resolução temporal.** Os instantes de cruzamento saem da curva ajustada, não
   do quadro mais próximo, então a taxa de captura entra como ruído no ajuste, e
   não como quantização direta. Ainda assim, menos de 4 ou 5 amostras dentro da
   base deixa o ajuste frágil: a `capture_fps` precisa dar pelo menos isso na
   velocidade máxima que interessa medir.

   Amostras dentro da base ≈ `distance_m / velocidade × fps`. A 60 km/h (16,7 m/s)
   numa base de 8 m a 12 fps: 5,8 amostras. A 90 km/h caem para 3,8 — abaixo do
   `min_samples` padrão, e a passagem passa a ser recusada em vez de medida
   errado.

4. **Plano da via.** A homografia supõe que a via é plana entre as duas linhas.
   Sobre uma lombada isso vale bem no trecho de aproximação e pior no lombo. Se
   possível, ponha a base *antes* da lombada, não em cima dela.

## Sinais de calibração ruim

- `lombada check` devolve cantos que não batem com os valores esperados.
- Muitas passagens recusadas por `ajuste ruim` (R² baixo) com trânsito normal:
  em geral são os pontos marcados fora do plano da via.
- Velocidades sistematicamente altas ou baixas contra uma referência conhecida
  (um GPS de celular num carro atravessando a base algumas vezes): erro na
  `distance_m`.
- Muitas recusas por `extrapolados`: a base está parcialmente fora do quadro, ou
  o detector só pega o veículo em parte do trecho. Diminua `distance_m` para o
  que a câmera realmente enxerga, e meça de novo.
