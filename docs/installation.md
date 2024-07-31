## Installation


To install karray, you can use pip:

```bash
pip install karray
```

karray has the following dependencies:
- NumPy

Optional dependencies:

1. io

```bash
pip install "karray[io]"
```

- pandas (for interoperability with pandas DataFrames)
- polars (for interoperability with polars DataFrames)
- pyarrow (for saving and loading arrays using the Feather format)


2. pandas

```bash
pip install "karray[pandas]"
```

- pandas
- pyarrow

3. polars

```bash
pip install "karray[polars]"
```

- polars
- pyarrow

4. sparse

```bash
pip install "karray[sparse]"
```

- sparse (for interoperability with N-Dimensional sparse arrays)

Alternatively, you can install the full version of karray using:

```bash
pip install "karray[all]"
```
