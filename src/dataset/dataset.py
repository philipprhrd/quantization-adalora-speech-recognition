from datasets import Dataset, Audio, Features, Value, DatasetDict, load_from_disk
import pandas as pd
from pathlib import Path


def create_common_voice_dataset(path: Path, streaming: bool = False) -> DatasetDict:
    """Load Common Voice dataset from a local directory.

    With streaming=False (default): loads metadata into memory, audio is decoded
    lazily on access via memory-mapped Arrow files (RAM-safe for most use cases).

    With streaming=True: fully lazy IterableDataset — no data is loaded until
    iteration. Use this when even the metadata is too large for RAM.

    Args:
        path: Root directory containing validated.tsv and clips/*.mp3
        streaming: Whether to use fully lazy IterableDataset (no random access)
    """
    df = pd.read_csv(path / "validated.tsv", sep="\t", usecols=["path", "sentence"])
    df["audio"] = df["path"].apply(lambda x: str(path / "clips" / x))
    df = df[["audio", "sentence"]]

    if streaming:
        # Fully lazy: audio files are read only during iteration, no data in RAM
        features = Features({
            "audio": Audio(sampling_rate=16000),
            "sentence": Value("string"),
        })

        def _generate(rows: pd.DataFrame):
            for _, row in rows.iterrows():
                yield {"audio": row["audio"], "sentence": row["sentence"]}

        # Shuffle before splitting since IterableDataset has no random access
        df = df.sample(frac=1, random_state=2).reset_index(drop=True)
        n = len(df)
        n_test = int(n * 0.1)

        from datasets import IterableDataset, IterableDatasetDict
        return IterableDatasetDict({
            "train":      IterableDataset.from_generator(_generate, gen_kwargs={"rows": df.iloc[: n - 2 * n_test]},      features=features),
            "validation": IterableDataset.from_generator(_generate, gen_kwargs={"rows": df.iloc[n - 2 * n_test : n - n_test]}, features=features),
            "test":       IterableDataset.from_generator(_generate, gen_kwargs={"rows": df.iloc[n - n_test :]},           features=features),
        })

    # Default: metadata in RAM (~few MB), audio decoded lazily on access
    ds = Dataset.from_dict(
        {"audio": df["audio"].tolist(), "sentence": df["sentence"].tolist()},
        features=Features({"audio": Audio(sampling_rate=16000), "sentence": Value("string")}),
    )

    # fixed seed for reproducibility
    train_testvalid = ds.train_test_split(test_size=0.2, seed=2, shuffle=True)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5, seed=2, shuffle=True)

    return DatasetDict({
        "train":      train_testvalid["train"],
        "validation": test_valid["train"],
        "test":       test_valid["test"],
    })

def create_asr_bundestag_dataset(path: Path) -> DatasetDict:
    ds = load_from_disk(path)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds[["audio", "text"]]

    ds_dict = DatasetDict(
        {
            "train": ds["train"],
            "validation": ds["validation"],
            "test": ds["test"], 
        }
    )

    return ds_dict