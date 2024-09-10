import os
from utils import cache
import importlib
module = importlib.import_module("multimodal-clinic-v2.multimodal_clinical_pretraining.data.utils")
UseLastTransform         = getattr(module, "UseLastTransform")
multimodal_pad_collate_fn = getattr(module, "multimodal_pad_collate_fn")

module = importlib.import_module("multimodal-clinic-v2.multimodal_clinical_pretraining.data")
MIMICIIINoteDataset      = getattr(module, "MIMICIIINoteDataset")
MIMICIIIBenchmarkDataset = getattr(module, "MIMICIIIBenchmarkDataset")

from torchvision import transforms
from types import SimpleNamespace

@cache
def get_datasets(split):
    # You can you any of these by uncommenting the notes you want
    used_note_types = [
        # "Echo",
        # "ECG",
        "Nursing",
        "Physician ",
        # "Rehab Services",
        # "Case Management ",
        # "Respiratory ",
        # "Nutrition",
        # "General",
        # "Social Work",
        # "Pharmacy",
        # "Consult",
        # "Radiology",
        # "Nursing/other",
        "Discharge summary",
    ]    

    args = SimpleNamespace(
        mimic_root              = "/ssd-data/datasets/MIMIC-III/1.4/",
        mimic_benchmark_root    = "/ssd-data/datasets/mimic3-benchmarks/",
        measurement_max_seq_len = 256,
        notes_max_seq_len       = 256
    )

    transform = transforms.Compose([UseLastTransform(args.measurement_max_seq_len)])

    measurement_dataset = MIMICIIIBenchmarkDataset(
        mimic3_benchmark_root=os.path.join(args.mimic_benchmark_root, "root"),
        mimic3_root=args.mimic_root,
        split=split,
        transform=transform,
    )    
    note_dataset = MIMICIIINoteDataset(
        root=args.mimic_root,
        split=split,
        max_seq_len=args.notes_max_seq_len,
        mask_rate=0,
        collate_fn=multimodal_pad_collate_fn,
        max_instances=10000,
        used_note_types=used_note_types,
        measurement_dataset=measurement_dataset,
    )
    return measurement_dataset, note_dataset