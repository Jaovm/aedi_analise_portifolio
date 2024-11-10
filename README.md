# AEDI - Análise de Risco de Portifolio

Simulação Monte Carlo - Disciplina AEDI UNB

## Introdução

A Simulação de Monte Carlo é uma técnica utilizada para modelar sistemas complexos e incertos, permitindo a análise de resultados em diferentes cenários aleatórios. Neste projeto, utiliza-se a simulação de Monte Carlo
para analisar o risco e retorno de um portifólio de ações. Assim, foi escolhida a distribuição t Student para estimar o Value at Risk (VaR) de um portifólio de ações.

## Fundamentação

A distribuição t de Student é uma distribuição de probabilidade contínua que surge quando se estima a média de uma população normalmente distribuída, mas a variância populacional é desconhecida e substituída pela variância amostral. Ela é particularmente útil em amostras de pequeno tamanho, onde a incerteza sobre a variância populacional é maior.

Matematicamente, a distribuição t de Student com __𝜈__ graus de liberdade é definida pela função de densidade de probabilidade:

$$
f(t) = \frac{\Gamma \left( \frac{\nu + 1}{2} \right)}{\sqrt{\nu \pi} \Gamma \left( \frac{\nu}{2} \right)} \left( 1 + \frac{t^2}{\nu} \right)^{-\frac{\nu + 1}{2}}
$$

onde __Γ__ é a função gama e __𝜈__ representa os graus de liberdade.

Em análises financeiras, o modelo de distribuição normal é frequentemente usado para representar os retornos de ativos. Contudo, dados reais mostram que esses retornos geralmente têm "caudas pesadas", ou seja, eventos extremos (grandes perdas ou ganhos) acontecem com mais frequência do que o previsto pela curva normal.

A distribuição t de Student é uma alternativa melhor nesse caso, pois acomoda essas caudas pesadas, capturando melhor a chance de eventos extremos. Isso leva a estimativas de risco mais precisas, especialmente para métricas como o VaR, que são influenciadas por esses eventos.

O VaR é uma medida estatística que quantifica a perda potencial máxima esperada de um portfólio em um determinado horizonte de tempo, para um dado nível de confiança. Assim, considerando-se um VaR de -0,50 com 95% de confiança para 365 dias, por exemplo, significa que há 95% de confiança de que a perda não excederá 50% do valor do portfólio ao longo dos próximos 365 dias. Da mesma forma, há uma probabilidade de 5% de que a perda seja superior a 50% nesse período.


## Metodologia

Para realizar a análise de risco e retorno do portifólio de ações, foram seguidos os seguintes passos:

1. Definição dos parâmetros da simulação: 
    * Horizonte de Tempo: número de dias para o cálculo dos retornos acumulados da carteira.
    * Graus de liberdade: Graus de liberdade da distribuição t de Student
    * Nível de confiança para o VaR 
    * Número de simulações de Monte Carlo.

2. Coleta dos dados históricos dos ativos: os preços de fechamento ajustados dos ativos foram baixados do Yahoo Finance para o período especificado.

3. Cálculo dos retornos diários dos ativos: os retornos diários são calculados com base nos preços de fechamento ajustados.

4. Estimação dos parâmetros da distribuição t de Student: para cada ativo, foram calculados o retorno médio diário e a volatilidade média diária.

5. Simulação de Monte Carlo: são realizadas simulações de Monte Carlo para gerar cenários de retornos futuros para cada ativo, com base na distribuição t de Student.

6. Cálculo dos retornos diários da carteira: os retornos diários da carteira foram calculados como a soma dos retornos diários dos ativos, ponderados pelos pesos especificados.

7. Cálculo dos retornos acumulados da carteira: os retornos acumulados da carteira para o horizonte de tempo especificado foram calculados.

8. Cálculo do VaR: o VaR para o horizonte de tempo especificado foi calculado com base na distribuição dos retornos acumulados da carteira.

9. Análise dos resultados: os resultados foram apresentados em termos de VaR e distribuição dos retornos acumulados da carteira.

10. A simulação também é feita utilizando-se uma normal permitindo a comparação dos resultados de ambas as distribuições.

## Resultados

A principal diferença observada ao utilizar a distribuição t de Student é o aumento da probabilidade de eventos extremos devido às suas caudas mais pesadas. Isso resulta em um VaR mais conservador (ou seja, uma perda potencial maior) em comparação com a distribuição normal. No contexto da gestão de riscos, isso significa que o modelo está levando em consideração a maior chance de ocorrerem perdas significativas, proporcionando uma estimativa de risco mais realista.