from datasets import load_from_disk, concatenate_datasets
import os
import numpy as np

BASE = "/work/hdd/benb/bremy/sbi_lens_million"

dataset = concatenate_datasets([
        load_from_disk(os.path.join(BASE, f"job_{i}")) for i in range(4)
    ])

dataset = dataset.with_format("numpy")

dataset = dataset.filter(
        lambda example: (
            not np.isnan(example["map"]).any()
            and not np.isnan(example["theta"]).any()
        )
    )

dataset.save_to_disk("/work/hdd/benb/bremy/sbi_lens_million_full")
