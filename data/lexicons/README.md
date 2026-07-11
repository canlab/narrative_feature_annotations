# Psycholinguistic norm tables (optional)

Drop CSV files here to enable per-word affective/lexical norm channels in the
`language/lexical/` group. Each file is `<field>.csv` with a `word,value` header
(extra columns ignored). Missing files → those channels are emitted as `NaN`
(the annotation shape is unchanged), so this directory can stay empty.

| File | Channel | Standard source to convert from | License note |
|------|---------|---------------------------------|--------------|
| `valence.csv` | language/lexical/valence | Warriner, Kuperman & Brysbaert (2013) VAD norms, `V.Mean.Sum` | research use |
| `arousal.csv` | language/lexical/arousal | Warriner et al. (2013), `A.Mean.Sum` | research use |
| `dominance.csv` | language/lexical/dominance | Warriner et al. (2013), `D.Mean.Sum` | research use |
| `concreteness.csv` | language/lexical/concreteness | Brysbaert, Warriner & Kuperman (2014), `Conc.M` | research use |
| `aoa.csv` | language/lexical/aoa | Kuperman, Stadthagen-Gonzalez & Brysbaert (2012), `Rating.Mean` | research use |

Example conversion (Warriner → valence.csv):

```python
import pandas as pd
w = pd.read_csv("Ratings_Warriner_et_al.csv")
w[["Word", "V.Mean.Sum"]].to_csv("data/lexicons/valence.csv", index=False, header=["word", "value"])
```

These norm sets are freely available for research from the original authors' supplements;
they are not redistributed in this repo for licensing cleanliness.
