1. correct extraction from PDF, still combines multiple transactions on one line.

Continue with this prompt:

```
there are still cases which are not extracted correctly, see this description:

"FNB OB Pmt Jicamasalariwie8508 74,602.52Cr 77,700.76Cr Payment To Investment Maint Back",26000.0,51700.76

Should be 2 transactions
1. FNB OB Pmt Jicamasalariwie8508, 74,602.52, 77,700.76
2. Cr Payment To Investment Maint Back",26000.0,51700.76
```


2. with correct extracted csv file, train model again.
    - use LLM again to categorise
    - verify categories
    - train model again


3. UI uppload bank statement and show categories