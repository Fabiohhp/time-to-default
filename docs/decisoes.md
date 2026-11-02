# Diário de decisões e achados

Registro cronológico de cada escolha do projeto, das alternativas consideradas e dos problemas encontrados no caminho. Os números citados estão em `reports/metrics.json`, gerado pelos notebooks.

Formato de cada decisão: **o que foi decidido**, *alternativas*, por quê e evidência.

---

## Parte 0 · Escopo e preparação

### D0.1 · Base de dados: LendingClub 2007–2018
- *Alternativas:* LendingClub 2007–2011 (~42 mil contratos, quase sem censura) e Freddie Mac (hipotecas com histórico mensal, exige cadastro e é muito mais pesada).
- **Por quê:** é crédito pessoal, como o de uma cooperativa. São 2,26 milhões de contratos com muitos ainda em aberto, o que dá censura real para a técnica mostrar valor.
- **Como baixar:** o `kagglehub` baixa sem credencial (`wordsforthewise/lending-club`). Só o arquivo `accepted_2007_to_2018Q4.csv.gz` é usado, e o cache de 4,1 GB foi apagado depois.

### D0.2 · Pré-pagamento como risco competitivo
- *Alternativa:* tratar a quitação antecipada como censura comum (abordagem clássica).
- **Por quê:** quem quita deixa de correr risco de default. Como censura, o KM suporia que esse contrato continuaria exposto ao default e superestimaria a perda acumulada. O Cox usa o hazard específico do default, e as probabilidades usam Aalen-Johansen.
- **Evidência (Parte 2):** o KM ingênuo superestima o default no prazo em 43% (36m) e 47% (60m).

### D0.3 · Estrutura de projeto igual à do `score-credito`
- `uv`, código em `src/time_to_default/`, notebooks numerados, `pytest`, `reports/metrics.json` como fonte única dos números e README gerado por script.

### Percalço: o WSL caiu por falta de memória
- **O que houve:** ler o CSV inteiro (1,6 GB, 151 colunas) com `pd.read_csv` estourou os ~7 GB de RAM e derrubou o WSL.
- **Solução:** leitura em blocos de 200 mil linhas, só com as 26 colunas usadas e tipos compactos, salva uma vez em parquet (`data.py`). O pico de memória caiu para 0,75 GB. Rodadas pesadas usam `prlimit --as` para matar só o Python, e não o WSL, se a memória estourar.

---

## Parte 1 · Dados e definição do evento (notebook 01)

### D1.1 · Momento do default = mês da primeira parcela não paga
- *Alternativas:* data do charge-off (não existe na base) ou 90 dias de atraso.
- **Por quê:** é quando o contrato quebra e a cobrança precisa agir. O charge-off (~5 meses depois) e o default regulatório (3 meses depois) são deslocamentos fixos dessa data.
- **Como:** `meses_pagos = last_pymnt_d − issue_d`, e o tempo do default é `meses_pagos + 1`. Quem nunca pagou tem tempo 1.

### D1.2 · Regras de tempo e evento por situação final

| Situação | Evento | Tempo |
|---|---|---|
| Charged Off / Default | default | mês da 1ª parcela não paga |
| Fully Paid antes do prazo | pré-pagamento | mês da quitação |
| Fully Paid no prazo ou depois | censura | prazo |
| Current | censura | fim da observação |
| Late / In Grace | censura | último pagamento |

### D1.3 · Por que "censurado" e não simplesmente y = 0? (pergunta do Fabio)
- Na sobrevivência, cada contrato é um **par** (tempo, evento). `event = 0` significa "sem default *até* esse tempo".
- Para quem quitou no prazo, isso equivale a um y = 0 definitivo, porque o contrato acabou.
- Para um contrato em dia emitido há 8 meses, um classificador o rotularia como bom pagador, quando só sabemos que ele sobreviveu 8 meses. É isso que a censura respeita.

### D1.4 · Pedidos conjuntos fora (120.710 contratos, 5,3%)
- **Percalço:** inicialmente justifiquei dizendo que "renda e DTI somam os dois tomadores". A pedido do Fabio fui verificar, e **estava errado**.
- **Fato verificado:** `annual_inc` e `dti` são só do titular, e os números do casal estão em `annual_inc_joint` e `dti_joint`. O DTI do titular chega a 999 (13,7 mil acima de 50%), enquanto o `dti_joint` é limpo (máximo de 69,5%).
- **Decisão (Fabio):** manter fora. É outro produto (existe só desde out/2015), com outra regra de aprovação e 76% dos contratos ainda em aberto. Incluí-los com as colunas `_joint` seria possível e fica como extensão.

### D1.5 · Outros filtros
- Contratos fora da política de crédito (2.749), moradia ANY/OTHER/NONE (1.185) e renda zero (nenhum depois do filtro de pedidos conjuntos).

### Achado: a taxa de default simples engana
- Na taxa simples, a safra de 2018 teria 1,8% de default, mas 92% dos contratos dela ainda estavam em aberto. A análise de sobrevivência existe justamente para isso.

### Achado: o pré-pagamento domina
- Na base final, 37% dos contratos terminam em pré-pagamento e 12% em default. Isso confirma a decisão D0.2.

---

## Parte 2 · Kaplan-Meier e riscos competitivos (notebook 02)

### D2.1 · Moradia no lugar de "tipo de garantia" (aprovado pelo Fabio)
- Os empréstimos do LendingClub são pessoais e sem garantia. O tipo de moradia (financiada, própria ou aluguel) é o mais próximo de patrimônio.

### D2.2 · Faixas de renda fixas em US$ (aprovado pelo Fabio)
- As faixas são até 40 mil, 40–60, 60–80, 80–100, 100–150 e acima de 150 mil, em vez de quintis.

### D2.3 · Tábua de vida mensal própria, conferida contra o lifelines
- Com o tempo em meses inteiros, KM e Aalen-Johansen saem exatos de contagens mensais e rodam em segundos com 2 milhões de linhas. Os testes comparam o resultado com `KaplanMeierFitter` e `AalenJohansenFitter`.
- O log-rank roda sobre dados agregados com pesos, e há um teste que garante o mesmo resultado do teste linha a linha.

### D2.4 · Grupos comparados dentro de cada prazo
- Contratos de 60 meses têm valor, juros e renda maiores. Misturar os prazos confundiria os efeitos.

### D2.5 · O p-valor do log-rank não serve como critério
- Com mais de 1 milhão de contratos, todo p-valor é zero. A estatística qui² por grau de liberdade é usada para comparar a força de separação das variáveis.

### Achados
- O default real no prazo é de 14,4% em 36m e 27,0% em 60m. O KM ingênuo diz 20,6% e 39,7%.
- Renda: a faixa de até US$ 40 mil tem cerca de 1,9 vez o default da faixa acima de US$ 150 mil.
- Moradia: financiada < própria < aluguel.
- O grade separa muito mais do que qualquer outra variável isolada.

### Percalço: números do texto diferentes dos outputs
- O texto escrito antes de rodar o notebook dizia "superestima uns 40%", mas o output mostrava 43% e 46%. Corrigi, e passei a conferir o texto contra os outputs em toda rodada.

---

## Parte 3 · Período crítico (notebook 03)

### D3.1 · Censura administrativa em nov/2018 (correção de um viés da Parte 1)
- **Percalço:** o hazard da safra de 2017–2018 no mês 24 aparecia em 0,23%, contra 0,7–0,8% nas outras.
- **Causa:** um contrato só vira charge-off uns 5 meses depois da primeira parcela não paga. Quem parou de pagar perto da foto (mar/2019) ainda aparecia como Late ou Current, e a censura dos Late no último pagamento era *informativa*, porque tirava do risco exatamente quem estava para quebrar.
- **Evidência:** contando os defaults pelo mês da 1ª parcela não paga, são cerca de 6–7 mil por mês até nov/2018, que caem para 2,9 mil em dez/2018 e 0,8 mil em mar/2019 (gráfico `chargeoff_reporting_lag.png`).
- **Correção:** a observação de **todos** os contratos termina em nov/2018 (`OBSERVATION_END`), e eventos depois disso viram censura. Os 71.332 contratos emitidos a partir de nov/2018 saíram. A base final tem **2.064.692** contratos.
- **Impacto:** os números da Parte 2 subiram uns 5%, sem mudar nenhuma conclusão. Já o hazard das safras recentes mudou de forma relevante.

### D3.2 · Último mês do contrato fora da busca do pico
- No mês 36, 996 dos 1.937 defaults são contratos que pagaram a última parcela atrasada e foram baixados depois do prazo (o tempo foi truncado no prazo). O pico desse mês é artificial.

### D3.3 · Regra objetiva da janela crítica
- É o maior trecho contínuo em torno do pico em que o hazard (média móvel de 3 meses) fica acima de 90% do pico.

### D3.4 · Duas medidas de "período crítico"
- O **hazard** (risco de quem ainda está ativo) orienta a cobrança. A **distribuição dos defaults** (Aalen-Johansen por janela) orienta a provisão.

### Achados
- Em 36m, o hazard sai de 0,12% para 0,72% ao mês até o 12º mês. A janela crítica vai do 12º ao 30º mês, com pico no 18º. Os meses 7–18 concentram 46% dos defaults.
- Em 60m, a janela crítica vai do 11º ao 30º mês, com pico no 14º e nível cerca de 40% acima.
- O grade muda o *quando*: o pico vem no mês 8 no grade G e no mês 29 no A. A fração dos defaults no 1º ano vai de 26% (A) a 59% (G).
- O risco subiu entre as safras (hazard no 12º mês de 0,48% em 2007–2011 para 0,82% em 2016) e continuou alto em 2017–2018.
- A razão de hazards 60m/36m é estável em cerca de 1,4×, o que sugere riscos proporcionais para o prazo.

---

## Parte 4 · Modelo de Cox (notebook 04)

### D4.1 · Grade fora do modelo (aprovado pelo Fabio)
- O LendingClub define a taxa de juros a partir do sub-grade. Com os dois no modelo, o HR dos juros viraria "+1 p.p. dentro do mesmo grade" e perderia a leitura pedida no projeto.

### D4.2 · Treino nas safras 2007–2015, teste nas safras 2016–2018 (fora do tempo)
- **Condição do Fabio:** precisa ser replicável e não pode distorcer o resultado. A divisão é determinística (por ano de emissão) e a amostra de ajuste tem semente fixa. A estabilidade da amostra é verificada no notebook 04.
- **Revisão (sem amostragem):** o ajuste do Cox na base de treino inteira (883.528 contratos, 151.111 defaults) leva 1,4 s, então **não há amostra no ajuste**. A amostra com semente fixa (100 mil, `random_state=0`) só é usada no teste de Schoenfeld (ver D4.4).

### D4.3 · Covariáveis e unidades
- Juros (+1 p.p.), prazo 60 vs. 36, DTI (+10 p.p.), PTI (+1 p.p.), renda (2× maior, via log₂), FICO (+20 pontos), moradia (aluguel e própria vs. financiada), tempo de emprego (+1 ano, com indicador de não informado), renda verificada, uso do rotativo (+10 p.p.), consultas em 6 meses (limitadas a 6) e atraso de 30+ dias nos últimos 2 anos.
- Tratamentos: PTI e uso do rotativo limitados ao p99,9 (há uso do rotativo de 892%). O tempo de emprego faltante vira indicador, porque a ausência é informativa (HR de 1,15). As 460 linhas sem uso do rotativo saem.
- Ficou de fora a finalidade do empréstimo (14 categorias), para manter a tabela legível.

### D4.4 · Teste de Schoenfeld numa subamostra fixa
- **Percalço:** a primeira tentativa rodou o teste de riscos proporcionais do lifelines na base inteira. O processo passou de 10 minutos e chegou a 5,9 GB de RAM (o mesmo risco que derrubou o WSL), então interrompi.
- **Diagnóstico:** o custo cresce com o quadrado de n (20 mil: 0,4 s; 50 mil: 2,4 s; 100 mil: 12,5 s), e o ajuste em si não era o problema.
- **Solução:** teste formal em 100 mil contratos com semente fixa. O diagnóstico principal passou a ser o **HR por fase do contrato** (um Cox por janela de meses com entrada atrasada), que usa a base inteira e mostra *como* o efeito muda.

### Achados do Cox
- **Juros:** HR de 1,097 por p.p. (+9,7%), e de 1,12 sem controles. É mais forte no início (1,16 nos meses 1–6) e mais fraco depois (1,06 nos meses 37–60).
- **Prazo:** o HR bruto de 1,45 cai para 1,015 com controles, ou seja, o risco maior dos contratos de 60 meses vem do perfil e do preço. O efeito ajustado inverte ao longo do tempo: 0,79 no início e 1,14 nos meses 25–36.
- **DTI:** +13,6% a cada +10 p.p. O PTI acrescenta pouco (+1,7% por p.p.).
- **Moradia:** aluguel +20,6% e imóvel quitado +7,8% vs. financiado.
- **FICO:** −11,5% a cada +20 pontos, mas **sem efeito nos 6 primeiros meses** (HR de 1,006). As quebras precoces são mal previstas pelo score de bureau.
- **Riscos proporcionais violados** por juros, prazo e FICO (maiores estatísticas de Schoenfeld). O modelo único fica como resumo (HR médio no tempo), e o modelo por fase é candidato para previsão (comparação no notebook 05).
- **C-index de treino:** 0,672.

### Percalço: explicações escritas antes de olhar os números
- **Sinais contraintuitivos:** eu tinha escrito que "o juro já precifica" o uso do rotativo e o atraso anterior. A tabela mostrou que, mesmo sem juros, os dois continuam abaixo de 1. Testei e eles só voltam ao sinal esperado quando o modelo perde **juros e FICO juntos** (atraso anterior: 0,94 → 1,12). O FICO é calculado com atrasos e uso de crédito, e o preço é definido em cima dele.
- **Renda verificada:** eu tinha escrito que o efeito caía "ao entrar a taxa de juros". Na verdade, a maior parte da queda (1,45 → 1,17) vem dos outros controles, e os juros levam de 1,17 para 1,10.
- **Forma funcional dos juros:** eu tinha escrito "aproximadamente log-linear". Os decis mostram que é log-linear no miolo (8% a 18%), mas nas pontas a reta exagera o risco. O 1º decil (até 7,7%, 99,7% grade A) é muito mais seguro do que a reta prevê. Mantive o termo linear para preservar a leitura "por p.p." e registrei a ressalva.
- **Lição:** escrever a narrativa só depois de ler os outputs e conferir cada número citado.

---

## Parte 5 · Validação fora do tempo e PCLD (notebooks 05 e 06)

### D5.1 · PCLD completa no estilo da Res. CMN 4.966 (opção escolhida pelo Fabio)
- *Alternativa:* só a PD por horizonte, com LGD assumida (por exemplo, 60%).
- **Escolha:** perda esperada de 12 meses (estágio 1) e da vida toda (estágio 2), com LGD e EAD medidas na base. Para isso entraram 3 colunas novas do CSV (`total_rec_prncp`, `recoveries`, `collection_recovery_fee`). Reler o CSV em blocos levou só 13 s.

### D5.2 · Previsão com dois Cox por causa, estratificados por prazo
- **Por quê:** a PD por horizonte depende também do pré-pagamento. Um Cox para default e outro para pré-pagamento, combinados pela recursão do Aalen-Johansen, dão P(default no mês t) para cada contrato.
- **Por que estratificar:** o prazo vira estrato (curva basal própria para 36 e 60 meses), porque o pré-pagamento muda muito com o prazo e a curva de 36 meses acaba no mês 36. No Cox de interpretação (Parte 4) o prazo continua como covariável, porque o projeto pede o efeito dele.
- **Conferência:** no próprio treino, a PD prevista bate com o Aalen-Johansen (36m: 4,31% previsto contra 4,33% observado em 12 meses).

### Percalço: hazard fantasma no lifelines com entrada atrasada
- Para modelos com `entry_col`, o `predict_cumulative_hazard` devolve, nos tempos **antes** da janela, o valor do primeiro tempo de evento em vez de zero. Somado entre as fases, isso criaria um hazard que não existe.
- **Solução:** os incrementos mensais são montados direto do `baseline_hazard_` × hazard parcial, com zero fora da janela.

### Percalço: memória de novo
- A previsão para o teste montava a matriz completa de 1,18 milhão × 60 meses (540 MB, várias cópias) e estourou o limite.
- **Solução:** cálculo em blocos de 100 mil, guardando só os horizontes pedidos. Na recalibração, as matrizes são cortadas no horizonte.

### D5.3 · Recalibração de nível com a safra mais recente
- **Problema:** fora do tempo, o modelo ordena bem mas subestima as safras novas (safra 2016, 36m, 12 meses: 4,5% previsto contra 5,7% observado, −20%; safra 2018 em 6 meses: −26% e −33%). É o mesmo aumento de risco entre safras visto na Parte 3.
- **Solução:** um multiplicador do hazard de default (1,23) estimado nos **12 primeiros meses da safra de 2016** e avaliado na **safra de 2017**, que ficou fora de tudo.
- **Resultado:** na safra de 2017, a PD média passou de −10% para +9% em relação à observada, e o erro médio por decil caiu de 0,74 para 0,56 p.p. Por prazo o efeito é misto: melhora em 36m (−12% → +7%) e piora em 60m (−8% → +11,5%). O nível muda de safra para safra, então em produção é preciso recalibrar periodicamente.

### D5.4 · Modelo escolhido: Cox único por causa, recalibrado
- O modelo por fase ordena só um pouco melhor (C-index de 0,699 contra 0,698), mas calibra pior por decil (0,87 contra 0,56 p.p.). Para provisão, a calibração pesa mais.
- **Benchmark:** os dois superam o grade da própria LendingClub como score (C-index de 0,682).

### D5.5 · Parâmetros de perda medidos na base
- **LGD de 89,9%**, estimada em 118.950 defaults até 2016, cujas recuperações já amadureceram (em 2018, 51% dos defaults ainda não tinham recuperação nenhuma). É estável em todos os segmentos (89%–90%), então uma LGD única basta.
- **EAD:** saldo da tabela Price depois de t − 1 parcelas, × 1,014 (saldo real no default ÷ agendado).
- **Desconto** pela taxa do contrato (IFRS 9).

### Achados
- **Backtest de exposição em default:** +3,6% (safra 2017, 11 meses) e +4,1% (safra 2016, 23 meses), ou seja, viés pequeno e conservador.
- **Safra de 2018 como carteira nova:**
  - estágio 1 de 4,1% do volume e estágio 2 de 9,4% (2,3×);
  - por prazo: 60m vai de 5,3% para 13,6% (2,6×) e 36m de 3,4% para 6,7% (2,0×);
  - por grade: de A (1,8% → 4,0%) a G (18,5% → 33,9%).
- **A perda esperada mensal tem pico perto do 12º mês**, antes do platô do hazard (12–30), porque o saldo amortiza e a carteira encolhe. Cerca de 47% da perda da vida toda (36m) cai no 1º ano.
- **O decil de menor risco é superestimado** (1,8% previsto contra 0,9% observado), o mesmo efeito da forma funcional dos juros visto na Parte 4. Uma spline nos juros é a melhoria natural.

---

## Parte 6 · Reprodutibilidade

### D6.1 · Um comando reproduz o projeto
- `bash scripts/run_all.sh` faz tudo em sequência:
  1. `uv sync`;
  2. baixa a base (`scripts/download_data.py`);
  3. roda os testes;
  4. **apaga os parquets e o `metrics.json`**, para nenhum resultado antigo vazar para a nova rodada;
  5. executa os 6 notebooks em ordem;
  6. regenera o README.
- **Download:** o `kagglehub` baixa só o arquivo usado (375 MB) direto para `data/raw/` (`path=` + `output_dir=`), sem credencial e sem deixar cache. A dependência `kaggle` (CLI), instalada no início e nunca usada, foi trocada por `kagglehub`.

### D6.2 · README sem número digitado à mão
- `scripts/build_readme.py` monta o README a partir do `reports/metrics.json`, como no projeto de credit scoring.
- **Percalço:** o `metrics.json` não guardava o erro por safra antes da recalibração (−20%), que o README precisava citar. A métrica `by_vintage_error_original` foi adicionada ao notebook 05.

### D6.3 · Revisão final
- `ruff` (erros e nomes não usados) passa limpo. Removi uma constante de cor que não era usada.
- Os notebooks foram gerados por scripts auxiliares durante o desenvolvimento. A versão de referência são os próprios `.ipynb` versionados.
