# Casus a

## CpG eiland

- 200 - 500 bp lang
- cg-% is op zn minst 50%
- ratio van waargenomen CpG / verwachte aantal CpG > 60%

## Hidden Markov Model

Een reeks van observeringen "emissies" bijvoorbeeld:

- Nucleotiden: AAAGCTACTGCA
- Reeks verborgen toestanden: wel of geen CpG

### Hoe werkt dit nou?

-,-,+,+,+
waarbij -: geen CpG-i en +: wel CpG-i
staten: - en +
Als ik in + zit, wat is de kans dat ik bij de volgende stap nog in een eiland zit?
Groter want het een eiland is lang. p = 0.99. En de kans dat ik van + naar - ga? p = 0.01.
Van - naar -? p = 0.999, en van - naar +? p = 0.001.

Dit zijn de transitie kansen.

HMM:

- Aantal toestanden
- Transitie kansen

Tabel:

|   | -     | +     |
|---|-------|-------|
| - | 0.999 | 0.001 |
| + | 0.99  | 0.01  |

#### Kans

In elke staat heb je een kans op een nucleotide:
Pn waarbij n een nucleotide is.

Pn voor +:

- Pa: 0.25
- Pc: 0.25
- Pg: 0.25
- Pt: 0.25

Pn voor -:

- Pa: 0.30
- Pc: 0.20
- Pg: 0.20
- Pt: 0.30

Dit zijn de emissie kansen. Ook hier is een table voor:

|   | A    | C    | G    | T    |
|---|------|------|------|------|
| - | 0.3  | 0.2  | 0.2  | 0.3  |
| + | 0.25 | 0.25 | 0.25 | 0.25 |

## Deel 2

|       Tafel: |     ❶    |     ❷    |     ❸    |
|-------------:|:--------:|:--------:|:--------:|
|  Grabbelton: | 6x blauw | 2x blauw | 1x blauw |
|              |  3x geel |  6x geel |  0x geel |
|              | 1x groen | 2x groen | 6x groen |
|              |  2x rood |  2x rood |  5x rood |
| Dobbelsteen: |    ⚀→①   |    ⚀→①   |    ⚀→①   |
|              |    ⚁→②   |    ⚁→②   |    ⚁→①   |
|              |    ⚂→②   |    ⚂→②   |    ⚂→①   |
|              |    ⚃→②   |    ⚃→③   |    ⚃→①   |
|              |    ⚄→③   |    ⚄→③   |    ⚄→②   |
|              |    ⚅→③   |    ⚅→③   |    ⚅→③   |

### Kansrekening

| Tafel | kleur |
|-------|-------|
| 2     | rood  |
| 1     | blauw |
| 2     | geel  |
| 2     | geel  |
| 2     | groen |

HMM:
- Start kansen: P((1..3)) = 1/3
- Overgangs kansen (transition prob):
-- = van
| = naar

  | staat |  ❶  |  ❷  |  ❸  |
  |------:|:---:|:---:|:---:|
  | ❶     | 1/6 | 1/6 | 2/3 |
  | ❷     | 1/2 | 1/3 | 1/6 |
  | ❸     | 1/3 | 1/2 | 1/6 |

- Emissie tabel:

  | staat |  bl  |  gl |  gr  | rd   |
  |------:|:----:|:---:|:----:|------|
  | ❶     | 1/2  | 1/4 | 1/12 | 1/6  |
  | ❷     | 1/6  | 1/2 | 1/6  | 1/6  |
  | ❸     | 1/12 | 0   | 1/2  | 5/12 |
    
1. P(data) = P(2~1~) * P(rood~1~ | 2~1~) * P(1~2~ | 2~1~) * P(blauw~2~ | 1~1~) * P(1~3~ | 1~2~) =
- 1/3 * 1/6 * 1/6 * 1/2 * 1/2 etc = 1/93312

voor mijn data:
p = 1/3 * 1/12 * 1/2 * 1/2 * 1/2 * 5/12 * 2/3 * 1/4 * 1/3 * 1/2 = 4.018775720164609e-05
ln(p) = log(1/3) +log(1/12) + log( 1/2) + log(1/2) + log( 1/2) + log( 5/12) + log( 2/3)+ log(1/4)+
⋮ log(1/3) + log(1/2)
Out[3]: -10.121948155945955

