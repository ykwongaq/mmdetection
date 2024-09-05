_base_ = "./dino-4scale_r50_8xb2-12e_coco.py"
max_epochs = 36
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30],
        gamma=0.1,
    )
]

work_dir = "/mnt/hdd/davidwong/models/dino"

classes = (
    "Teleostei",
    "Malacostraca",
    "Elasmobranchii",
    "Gastropoda",
    "Actinopterygii",
    "Bivalvia",
    "Reptilia",
    "Hexacorallia",
    "Aves",
    "Echinoidea",
    "Octocorallia",
    "Polychaeta",
    "Hydrozoa",
    "Chondrichthyes",
    "Demospongiae",
    "Dipneusti",
    "Cubozoa",
    "Hexapoda",
    "Ophiuroidea",
    "Coelacanthi",
    "Nuda",
    "Petromyzonti",
    "Ascidiacea",
    "Florideophyceae",
)

model = dict(
    bbox_head=dict(num_classes=len(classes)),
)
