"""Gera o README.md a partir de reports/metrics.json.

Nenhum número do README é digitado à mão: todos vêm das métricas salvas pelos
notebooks. Rode depois de executar os notebooks (scripts/run_all.sh faz isso).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
m = json.loads((ROOT / "reports" / "metrics.json").read_text())


def num(x: float, decimals: int = 0) -> str:
    return f"{x:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, decimals: int = 0) -> str:
    return num(100 * x, decimals) + "%"


def signed_pct(x: float, decimals: int = 0) -> str:
    return ("+" if x >= 0 else "−") + pct(abs(x), decimals)


d, km, cp, cox, val, prov = (m[k] for k in ("data", "kaplan_meier", "critical_period", "cox", "validation", "provisioning"))
k36, k60 = km["term_36"], km["term_60"]
c36, c60 = cp["term_36"], cp["term_60"]
hr, by_period = cox["hazard_ratios"], cox["hr_by_period"]
book = prov["book_2018"]
overstate_36 = k36["naive_km_default_term"] / k36["cif_default_term"] - 1
overstate_60 = k60["naive_km_default_term"] / k60["cif_default_term"] - 1
orig_2016 = val["by_vintage_error_original"]["2016_36m_12m"]["erro modelo único"]
pred17, obs17 = val["pd12_2017_pred_vs_obs"]["modelo único recalibrado"]
pred17_orig, _ = val["pd12_2017_pred_vs_obs"]["modelo único (original)"]
c_test = val["c_index_test"]
share_7_18 = c36["default_share_by_window"]["7–12"] + c36["default_share_by_window"]["13–18"]
removed_joint = next(f["removidos"] for f in d["funnel"] if f["etapa"] == "pedido individual")
removed_late = next(f["removidos"] for f in d["funnel"] if f["etapa"] == "emitido antes do fim da observação")

COX_ORDER = ["int_rate", "dti", "home_rent", "home_own", "fico", "log2_income", "inquiries", "emp_missing",
             "pti", "term_60", "verified", "revol_util", "past_delinq", "emp_years"]


def cox_rows() -> str:
    rows = []
    for c in COX_ORDER:
        r, ph = hr[c], by_period[c]
        early, late = ph.get("1–6"), ph.get("25–36")
        rows.append(f"| {r['reading']} | {num(r['HR'], 3)} | {num(r['low'], 3)}–{num(r['high'], 3)} "
                    f"| {num(early, 2)} | {num(late, 2)} |")
    return "\n".join(rows)


def grade_rows() -> str:
    return "\n".join(
        f"| {g} | {pct(v['stage1_pct'], 1)} | {pct(v['stage2_pct'], 1)} | {num(v['stage2_pct'] / v['stage1_pct'], 1)}× |"
        for g, v in book["by_grade"].items()
    )


readme = f"""# Tempo até a inadimplência: análise de sobrevivência para cobrança e PCLD

Não basta saber **se** um associado vai inadimplir: para provisionar capital (PCLD) e planejar a cobrança, a cooperativa precisa saber **quando**. Este projeto modela o tempo até o default de {num(d["loans"] / 1e6, 1).replace(",0", "")} milhões de empréstimos pessoais com curvas de Kaplan-Meier, riscos competitivos, o modelo de Cox e uma provisão no estilo da Res. CMN 4.966 (IFRS 9).

> **O risco de um contrato de 36 meses sobe {num(c36["hazard_month_12"] / c36["hazard_month_1"], 0)} vezes no primeiro ano e fica no máximo do {c36["critical_window"][0]}º ao {c36["critical_window"][1]}º mês.** Cada +1 p.p. de juros aumenta o risco relativo de default em {pct(hr["int_rate"]["HR"] - 1, 1)} ({pct(by_period["int_rate"]["1–6"] - 1)} nos 6 primeiros meses). Na provisão, passar um contrato do estágio 1 para o estágio 2 multiplica a reserva por {num(book["stage2_pct"] / book["stage1_pct"], 1)}.

![Hazard mensal de default](reports/figures/hazard_by_term.png)

## O problema

Uma cooperativa de crédito precisa de três respostas para cada contrato: **qual a probabilidade de quebra em cada horizonte** (12 meses para o estágio 1 da PCLD, a vida toda para o estágio 2), **em que fase do contrato a cobrança deve agir** e **quais características aceleram a quebra**. Um classificador de default responde só a primeira, e para um único horizonte fixo.

**Dados:** [LendingClub 2007–2018 (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club), empréstimos pessoais sem garantia. Depois dos filtros de escopo ficam **{num(d["loans"])} contratos** de pedidos individuais. Ao fim da observação, {pct(d["event_share"]["default"], 1)} tinham dado default, {pct(d["event_share"]["pré-pagamento"], 1)} tinham sido quitados antes do prazo e {pct(d["event_share"]["censurado"], 1)} ainda estavam em aberto ou tinham terminado no prazo (censura).

## Resultados

### 1. Riscos competitivos: o Kaplan-Meier sozinho superestima a perda

Quase 40% dos contratos são quitados antes do prazo, e quem quita deixa de correr risco. O Kaplan-Meier trata esses contratos como se continuassem expostos, então superestima o default acumulado. A incidência acumulada (Aalen-Johansen) dá a fração que de fato inadimple:

| | 36 meses | 60 meses |
|---|---|---|
| Default no prazo, 1 − Kaplan-Meier | {pct(k36["naive_km_default_term"], 1)} | {pct(k60["naive_km_default_term"], 1)} |
| **Default no prazo, Aalen-Johansen** | **{pct(k36["cif_default_term"], 1)}** | **{pct(k60["cif_default_term"], 1)}** |
| Superestimação do Kaplan-Meier | {signed_pct(overstate_36)} | {signed_pct(overstate_60)} |
| Pré-pagamento no prazo | {pct(k36["cif_prepay_term"], 1)} | {pct(k60["cif_prepay_term"], 1)} |

![Riscos competitivos](reports/figures/competing_risks_overall.png)

As curvas de Kaplan-Meier por renda, tipo de moradia e grade estão no notebook 02. Como os empréstimos não têm garantia, o tipo de moradia entra no lugar de "tipo de garantia": em 36 meses, o default vai de {pct(km["by_group_36m_cif_default_36m"]["home"]["financiada"], 1)} (imóvel financiado) a {pct(km["by_group_36m_cif_default_36m"]["home"]["aluguel"], 1)} (aluguel).

### 2. Período crítico: quando o contrato quebra

| | 36 meses | 60 meses |
|---|---|---|
| Hazard no 1º mês | {num(c36["hazard_month_1"], 2)}% a.m. | {num(c60["hazard_month_1"], 2)}% a.m. |
| Hazard no 12º mês | {num(c36["hazard_month_12"], 2)}% a.m. | {num(c60["hazard_month_12"], 2)}% a.m. |
| Pico | mês {c36["peak_month"]} ({num(c36["peak_hazard_pct_month"], 2)}%) | mês {c60["peak_month"]} ({num(c60["peak_hazard_pct_month"], 2)}%) |
| **Janela crítica** (hazard acima de 90% do pico) | **meses {c36["critical_window"][0]}–{c36["critical_window"][1]}** | **meses {c60["critical_window"][0]}–{c60["critical_window"][1]}** |

- **O risco acelera no primeiro ano e fica num platô até perto do 30º mês.** Em perdas, os meses 7 a 18 concentram {pct(share_7_18)} de todos os defaults dos contratos de 36 meses.
- **Quanto pior o grade, mais cedo a quebra.** O pico vem no mês {cp["grade_36m_peak_month"]["G"]} no grade G e no mês {cp["grade_36m_peak_month"]["A"]} no grade A. A fração dos defaults que acontece no 1º ano vai de {pct(cp["grade_36m_share_first_year"]["A"])} (A) a {pct(cp["grade_36m_share_first_year"]["G"])} (G). Para a cobrança, isso significa régua preventiva desde os primeiros meses nos grades piores.

![Formato do hazard por grade](reports/figures/hazard_shape_by_grade.png)

### 3. Modelo de Cox: a tabela interpretada

Hazard específico do default (o pré-pagamento é censura), ajustado em {num(cox["train_loans"])} contratos das safras de {cox["train_vintages"][0]} a {cox["train_vintages"][1]}. O grade ficou de fora porque a LendingClub define o juro a partir dele. O HR é o efeito médio no contrato, e as duas últimas colunas mostram o HR estimado só no começo e só no meio do contrato.

| Leitura (com as demais variáveis fixas) | HR | IC 95% | HR meses 1–6 | HR meses 25–36 |
|---|---|---|---|---|
{cox_rows()}

- **O prazo, sozinho, quase não explica nada.** O HR bruto de 60 vs. 36 meses é {num(cox["univariate_hr"]["term_60"], 2)} e cai para {num(hr["term_60"]["HR"], 2)} com controles. O risco maior dos contratos longos vem de quem os contrata e do juro cobrado.
- **Os riscos não são proporcionais.** Juros, prazo e FICO têm efeitos que mudam ao longo do contrato (resíduos de Schoenfeld e um Cox por fase). O FICO não tem efeito nos 6 primeiros meses (HR de {num(by_period["fico"]["1–6"], 2)}).
- **Uso do rotativo e atraso anterior com HR abaixo de 1** não querem dizer que eles protegem. O score FICO e o preço já usam essa informação: os dois só voltam ao sinal esperado quando o modelo perde juros e FICO juntos.

![HR por fase do contrato](reports/figures/cox_hr_by_period.png)

### 4. Validação fora do tempo (safras de 2016 a 2018)

A previsão combina dois Cox por causa (default e pré-pagamento), estratificados por prazo, pela recursão do Aalen-Johansen.

| C-index no teste | |
|---|---|
| Grade da própria LendingClub, usado como score | {num(c_test["grade da LendingClub"], 3)} |
| **Modelo de sobrevivência** | **{num(c_test["modelo único"], 3)}** |

O modelo ordena bem, mas **subestima as safras novas**, cujo risco subiu: nos contratos de 36 meses da safra de 2016, a PD de 12 meses prevista ficou {signed_pct(orig_2016)} em relação à observada. Para corrigir, o nível do hazard foi recalibrado nos 12 primeiros meses da safra de 2016 (multiplicador de {num(val["default_scale"]["modelo único"], 2)}). Na safra de 2017, que ficou fora de tudo, a PD média passou de {pct(pred17_orig, 1)} para {pct(pred17, 1)} prevista, contra {pct(obs17, 1)} observada: o viés passou a ser conservador.

![Calibração por decil](reports/figures/calibration_deciles_2017.png)

### 5. PCLD: estágio 1 × estágio 2

Perda esperada = Σ P(default no mês t) × EAD(t) × LGD, descontada pela taxa do contrato. A **LGD ({pct(prov["lgd"], 1)})** e o **EAD** (saldo da tabela de amortização × {num(prov["ead_factor"], 3)}) foram medidos nos próprios defaults. No backtest, a exposição em default prevista ficou {signed_pct(prov["backtest"]["2017_11m"]["ead_error"], 1)} em relação à realizada na safra de 2017 (11 meses).

Na safra de 2018, tratada como uma carteira nova de US$ {num(book["volume_usd_mi"] / 1000, 1)} bilhões:

| Grade | Estágio 1 (12 meses) | Estágio 2 (vida toda) | Multiplicador |
|---|---|---|---|
{grade_rows()}
| **Carteira** | **{pct(book["stage1_pct"], 1)}** | **{pct(book["stage2_pct"], 1)}** | **{num(book["stage2_pct"] / book["stage1_pct"], 1)}×** |

Nos contratos de 60 meses, a provisão sobe de {pct(book["by_term"]["60"]["stage1_pct"], 1)} para {pct(book["by_term"]["60"]["stage2_pct"], 1)} ({num(book["by_term"]["60"]["stage2_pct"] / book["by_term"]["60"]["stage1_pct"], 1)}×). Para uma cooperativa, o critério de aumento significativo de risco, que move o contrato para o estágio 2, é uma das decisões de maior impacto no resultado.

![Provisão por grade](reports/figures/provision_by_grade.png)

## O que eu destacaria

**1. Um viés que só aparece olhando o calendário.** Um contrato só vira charge-off uns 5 meses depois da primeira parcela não paga. Quem parou de pagar perto da data da foto ainda aparece como "em atraso" ou até "em dia", e os defaults dos últimos meses estavam incompletos. Sem corrigir isso, a safra mais recente parecia muito melhor do que era. A solução foi encerrar a observação de todos os contratos em {d["observation_end"].replace("2018-11", "nov/2018")} (censura administrativa), o que tirou {num(removed_late)} contratos emitidos depois disso.

![Atraso do charge-off](reports/figures/chargeoff_reporting_lag.png)

**2. Riscos competitivos não são um detalhe.** Com 40% de pré-pagamento, ignorar o risco competitivo inflaria a perda acumulada em {pct(overstate_36)} nos contratos de 36 meses.

**3. O HR único é uma média.** O efeito dos juros cai de {num(by_period["int_rate"]["1–6"], 2)} para {num(by_period["int_rate"]["37–60"], 2)} ao longo do contrato. A tabela do modelo único é um resumo honesto, mas a validação compara também um modelo com efeitos por fase.

**4. Toda decisão registrada.** O arquivo [`docs/decisoes.md`](docs/decisoes.md) traz cada escolha do projeto, as alternativas consideradas e os erros que encontrei e corrigi no caminho.

## Limitações

- **Sem histórico mensal de pagamentos.** O tempo do default é reconstruído a partir do último pagamento, e a PCLD é calculada na concessão. Com dados de comportamento (dias de atraso na data de referência), o mesmo modelo alocaria cada contrato no estágio certo.
- **O risco muda entre safras.** O nível precisa de recalibração periódica, e a calibração fora do tempo é boa na média, mas não em todos os segmentos.
- **Forma funcional dos juros.** O efeito é log-linear no miolo da distribuição, e nas pontas a reta exagera o risco (o decil de juros mais baixos é mais seguro do que o modelo prevê). Uma spline seria a melhoria natural.
- **Pedidos conjuntos fora** ({num(removed_joint)} contratos): renda e DTI do titular não são comparáveis aos de pedidos individuais.
- Os dados são de tomadores dos EUA, em dólares. Os números demonstram o método, não servem como referência de mercado brasileiro.

## Estrutura

```
notebooks/
  01_data_and_events.ipynb   definição do tempo e do evento, filtros e censura administrativa
  02_kaplan_meier.ipynb      Kaplan-Meier, Aalen-Johansen e log-rank por prazo, renda, moradia e grade
  03_critical_period.ipynb   hazard mensal, janela crítica, formato por grade e por safra
  04_cox.ipynb               Cox, tabela interpretada, forma funcional e riscos proporcionais
  05_validation.ipynb        validação fora do tempo, calibração e recalibração
  06_provisioning.ipynb      LGD, EAD, backtest e provisão de estágio 1 e 2
src/time_to_default/         código usado pelos notebooks (eventos, tábua de vida, Cox, previsão, perda esperada)
tests/                       testes unitários (pytest)
reports/                     figuras e metrics.json (fonte dos números deste README)
docs/decisoes.md             diário de decisões, achados e percalços
scripts/                     download_data.py, run_all.sh (reproduz tudo) e build_readme.py (gera este README)
```

## Como reproduzir

Requer [uv](https://docs.astral.sh/uv/) e Python 3.12. Uns 8 GB de RAM bastam: a base é lida em blocos.

```bash
bash scripts/run_all.sh
```

O script baixa a base do Kaggle (375 MB, sem precisar de credencial), instala as dependências, roda os testes, executa os notebooks em ordem e regenera este README.

**Stack:** pandas, lifelines, numpy, matplotlib/seaborn, pytest.

**Licença:** código sob a [licença MIT](LICENSE). Os dados pertencem às fontes e seguem os termos delas.
"""

(ROOT / "README.md").write_text(readme)
print("README.md gerado")
