from transformers import AutoModelForSpeechSeq2Seq

model = AutoModelForSpeechSeq2Seq.from_pretrained("UsefulSensors/moonshine-tiny")

# Alle Linear-Layer ausgeben (das sind die LoRA-Kandidaten)
for name, module in model.named_modules():
    if "Linear" in type(module).__name__:
        print(name)
