# Dataset source

`credit_default_raw.csv.gz` is the **Default of Credit Card Clients** dataset: 30,000 credit card clients in Taiwan (April to September 2005), with a real label for whether each client defaulted on the next month's payment.

- Original: Yeh, I. C. and Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473-2480. UCI Machine Learning Repository, https://doi.org/10.24432/C55S3H (CC BY 4.0).
- Copy used here: Hugging Face dataset `scikit-learn/credit-card-clients`, file `UCI_Credit_Card.csv` at revision `f809522649eaf948ad40a11ed575c4bb4d4460c1` (CC0 1.0).
- SHA-256 of the original CSV: `a0f0ab49d6326671d6cd83be5c88dcf18007025fe9a53ecd699119c871176ca1`. The file here is that CSV gzipped, unchanged.

It is committed to the repo so the app, CI and the live demo never need to download it.

`make_data.py` reads it, keeps 10 of the 23 features (see the model card), renames them, and splits the rows into:

- `train.csv`: 20,000 rows for training model v1.
- `stream_pool.csv`: 10,000 rows the model never sees in training. Simulated monitoring batches are drawn from these.

`SEX` is left out of the model on purpose. `age` is kept and flagged in the model card's fairness note.
