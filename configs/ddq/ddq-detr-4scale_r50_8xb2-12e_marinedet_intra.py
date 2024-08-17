_base_ = ["../_base_/datasets/coco_detection.py", "../_base_/default_runtime.py"]
classes = (
    "megaptera novaeangliae",
    "whitespotted boxfish",
    "black dragonfish",
    "chionoecetes opilio",
    "mugil cephalus",
    "sleeper shark",
    "fish storm",
    "turbot",
    "saccostrea cucullata",
    "largemouth bass",
    "pygoscelis papua",
    "abdopus aculeatus",
    "conomurex luhuanus",
    "seadragon",
    "monachu",
    "copperband butterflyfish",
    "nurse shark",
    "whelk",
    "redtail butterflyfish",
    "razor clam",
    "serranidae",
    "teira batfish",
    "halibut",
    "blue tang surgeonfish",
    "razorfish",
    "whip coral",
    "pleurotomaria",
    "fire coral",
    "coral reef",
    "oyster",
    "bubble tip anemone",
    "dermochelys coriacea",
    "pufferfish",
    "dottyback",
    "anemone hermit crab",
    "goatfish",
    "meyer's butterflyfish",
    "diadema antillarum",
    "deep-sea dragonfish",
    "fur seal",
    "seal",
    "echinometra mathaei",
    "sepioteuthis sepioidea",
    "yellow mask surgeonfish",
    "chelydra serpentina",
    "graeffe's sea cucumber",
    "upside-down jellyfish",
    "chelonia mydas",
    "snake eel",
    "sabellidae",
    "carpet shark",
    "shark",
    "magnificent sea anemone",
    "carpet anemone",
    "pederson cleaner shrimp",
    "elephant ear sponge",
    "blackfin tuna",
    "turbinella pyrum",
    "thenus orientalis",
    "mantis shrimp",
    "pterois volitans",
    "lagenodelphis hosei",
    "tube-dwelling anemone",
    "black swallower",
    "squilla empusa",
    "malabar grouper",
    "variable coral crab",
    "green turtle",
    "lantern shark",
    "barracuda",
    "gorgonian coral",
    "giant manta ray",
    "pink anemone fish",
    "callorhinus ursinus",
    "striped marlin",
    "bullhead",
    "black rock shark",
    "cushion star",
    "handfish",
    "steno bredanensis",
    "hammerhead shark",
    "tiger shark",
    "blackspotted puffer",
    "blanket octopus",
    "pencil sea urchin",
    "little tunny",
    "bigeye tuna",
    "spot-fin porcupinefish",
    "sepioteuthis australis",
    "celestial goldfish",
    "gulper shark",
    "sea fan",
    "maja squinado",
    "crowntail betta",
    "tiger tail seahorse",
    "esox lucius",
    "humphead wrasse",
    "drumfish",
    "american lobster",
    "bull shark",
    "cusk",
    "bearded scorpionfish",
    "longnose batfish",
    "bonito tuna",
    "plaice",
    "sepioloidea lineolata",
    "cuttlefish",
    "eudyptula minor",
    "alligator gar",
    "eudyptes robustus",
    "blue whiting",
    "red king crab",
    "cancer pagurus",
    "dendronephthya",
    "black sea bass",
    "sea snake",
    "eel",
    "argopecten irradians",
    "marsh crab",
    "yellow mask angelfish",
    "blacktip grouper",
    "tilefish",
    "great white shark",
    "emperor angelfish",
    "eudyptes chrysolophus",
    "painted spiny lobster",
    "feather duster worm",
    "parastichopus parvimensis",
    "lobster",
    "durban hinge-beak shrimp",
    "blue marlin",
    "red tailed butterfly fish",
    "scallop",
    "halichoerus grypus",
    "blue triggerfish",
    "barred-fin moray",
    "bighead gurnard",
    "sepia latimanus",
    "white shark",
    "dugong",
    "actinopyga mauritiana",
    "mackerel shark",
    "spiny eel",
    "menhaden",
    "guillemot",
    "fireworm",
    "xenophora pallidula",
    "stenella coeruleoalba",
    "lampfish",
    "conus marmoreus",
    "cypraea histrio",
    "map puffer",
    "panulirus argus",
    "chrysaora fuscescens",
    "patinopecten yessoensis",
    "dragonet",
    "blue crab",
    "soldierfish",
    "lemon shark",
    "frilled shark",
    "spider crab",
    "red snapper",
    "chambered nautilus",
    "stone crab",
    "cucumaria frondosa",
    "catfish",
    "port jackson shark",
    "pipefish",
    "orca",
    "sea whip",
    "physalia physalis",
    "angelshark",
    "histioteuthis reversa",
    "chicoreus ramosus",
    "strombus gigas",
    "gymnothorax javanicus",
    "coral",
    "caridina multidentata",
    "bare-tailed goatfish",
    "white marlin",
    "grey reef shark",
    "cypraea moneta",
    "otaria flavescens",
    "cypraea aurantium",
    "coryphaena hippurus",
    "encrusting sponge",
    "octopus tetricus",
    "turbo marmoratus",
    "gadus morhua",
    "koi fish",
    "sergeant major",
    "sunfish",
    "bluespotted cornetfish",
    "aeolid nudibranchs",
    "testudo hermanni",
    "snake pipefish",
    "coral grouper",
    "zebra crab",
    "bigfin reef squid",
    "pilchard",
    "sea turtle",
    "leafy sea dragon",
    "ostrea edulis",
    "spaghetti eel",
    "searobin",
    "mako shark",
    "doryteuthis opalescens",
    "epitonium clathrus",
    "pinnate batfish",
    "hake",
    "sea snail",
    "cleaner shrimp",
    "cypraea onyx",
    "cypraea talpa",
    "tube anemone",
    "seagrass",
    "marlin",
    "mud crab",
    "anchovy",
    "red lionfish",
    "dolphin",
    "napolean wrasse",
    "sepioteuthis lessoniana",
    "pygoscelis antarcticus",
    "flame scallop",
    "sousa chinensis",
    "oncorhynchus mykiss",
    "octopus maorum",
    "flashlight fish",
    "atlantic blue crab",
    "osteoglossum bicirrhosum",
    "barreleye",
    "crystal jellyfish",
    "trumpetfish",
    "grampus griseus",
    "yellowfin tuna",
    "fimbriated moray",
    "strongylocentrotus franciscanus",
    "pistol shrimp",
    "swell shark",
    "titan triggerfish",
    "macrobrachium rosenbergii",
    "carcharodon carcharias",
    "chromis",
    "oncorhynchus kisutch",
    "harpiosquilla harpax",
    "freshwater mussel",
    "bohadschia marmorata",
    "cymbiola vespertilio",
    "black ghost knifefish",
    "regal tang fish",
    "gonatus fabricii",
    "ray",
    "skipjack tuna",
    "cornetfish",
    "seagull",
    "honeycomb morays",
    "sawshark",
    "balaenoptera acutorostrata",
    "ribbon eel",
    "bass",
    "clownfish",
    "adamussium colbecki",
    "devil ray",
    "thelenota ananas",
    "cheilinus undulatus",
    "sponge",
    "bluefin tuna",
    "fish",
    "shrimp goby",
    "bubble coral",
    "pizza anemone",
    "crappie",
    "killer whale",
    "zebra moray",
    "triplefin",
    "blue-ringed octopus",
    "mole crab",
    "seahorse",
    "goby",
    "wrasse",
    "beluga whale",
    "palechin morays",
    "giant moray",
    "neophoca cinerea",
    "oreochromis niloticus",
    "pygmy seahorse",
    "banggai cardinalfish",
    "atlantic triton",
    "megadyptes antipodes",
    "sole",
    "enteroctopus dofleini",
    "apostichopus japonicus",
    "nudibranch",
    "table coral",
    "cleaner wrasse",
    "black-blotched porcupinefish",
    "clark's anemonefish",
    "blue whale",
    "olindias formosa",
    "halocaridina rubra",
    "zidona dufresnei",
    "sea lion",
    "aequipecten opercularis",
    "snapping turtle",
    "oliva porphyria",
    "arctocephalus pusillus",
    "anemonefish",
    "pagophilus groenlandicus",
    "crown-of-thorns starfish",
    "lantern fish",
    "croaker",
    "smooth shore crab",
    "flathead",
    "starfish",
    "batfish",
    "ocellaris clownfish",
    "rhopilema esculentum",
    "yellowmargin triggerfish",
    "palinurus delagoae",
    "delphinus delphis",
    "small swimming crab",
    "gudgeon",
    "squirrelfish",
    "bicolor cleaner wrasse",
    "whale",
    "electric eel",
    "angelfish",
    "ember parrotfish",
    "astropyga radiata",
    "cheeklined wrasse",
    "clown triggerfish",
    "whiting",
    "goblin shark",
    "voluta musica",
    "sepiola atlantica",
    "great barracuda",
    "cassiopea andromeda",
    "concholepa",
    "panulirus interruptus",
    "jellyfish",
    "cypraea caurica",
    "shrimpfish",
    "carassius auratus",
    "ornate spiny lobster",
    "salp",
    "dogfish shark",
    "sea slug",
    "chrysaora melanaster",
    "thresher shark",
    "pearsonothuria graeffei",
    "cardinalfish",
    "black marlin",
    "moringa eel",
    "stomolophus meleagris",
    "cypraea annulus",
    "day octopus",
    "firefish",
    "planospira scalaris",
    "firefly squid",
    "giant trevally",
    "squat lobster",
    "aurelia labiata",
    "cypraea cribraria",
    "triggerfish",
    "monodon monoceros",
    "powder blue tang fish",
    "bluefin trevally",
    "emys orbicularis",
    "oarfish",
    "tube sponge",
    "megamouth shark",
    "flounder",
    "snapper",
    "sevengill shark",
    "conch",
    "walleye",
    "crassostrea sikamea",
    "sepia officinalis",
    "dosidicus gigas",
    "heliocidaris tuberculata",
    "frogfish",
    "bluespotted ribbontail ray",
    "fiddler crab",
    "dungeness crab",
    "gurnard",
    "aurelia aurita",
    "delphinapterus leucas",
    "pike",
    "winter flounder",
    "blacknose shark",
    "pagrus major",
    "octopus",
    "vampire squid",
    "emperor fish",
    "salvelinus fontinalis",
    "tuna",
    "coral banded shrimp",
    "callinectes sapidus",
    "indo-pacific sergeant",
    "bichir",
    "cyprinus carpio",
    "sixgill shark",
    "grouper",
    "cypraea vitellus",
    "bluestreak cleaner wrasse",
    "chlamys islandica",
    "odontodactylus scyllarus",
    "goldfish",
    "crab",
    "pecten maximus",
    "cypraea mappa",
    "butterfly goldfish",
    "squid",
    "apteronotus albifrons",
    "arrow crab",
    "yellow tang fish",
    "cypraea tigris",
    "echinus esculentus",
    "staghorn coral",
    "cyanea lamarckii",
    "wobbegong shark",
    "striped bass",
    "devil scorpionfish",
    "sea hare",
    "horn-nosed boxfish",
    "euchelus asper",
    "fusinus colus",
    "scaleless dragonfish",
    "viperfish",
    "peacock mantis shrimp",
    "ocean sunfish",
    "coconut crab",
    "panulirus ornatus",
    "eudyptes schlegeli",
    "alpheus bellulus",
    "mussel",
    "mahi-mahi",
    "fan worm",
    "manatee",
    "yellow-edged moray",
    "albacore",
    "pomacanthus imperator",
    "blacktip reef shark",
    "moon jellyfish",
    "bittium reticulatum",
    "spotted sharpnose puffer",
    "starry flounder",
    "antarctic toothfish",
    "cookiecutter shark",
    "corydoras aeneus",
    "gulper eel",
    "banded sea krait",
    "hatchetfish",
    "eriocheir sinensis",
    "striped surgeonfish",
    "octopus mimus",
    "spotted eagle rays",
    "porcupinefish",
    "loosejaw",
    "amberjack",
    "clown anemonefish",
    "mushroom leather coral",
    "echidna nebulosa",
    "unicornfish",
    "pegasus sea moth",
    "manta ray",
    "fangtooth fish",
    "epitonium scalare",
    "kingfish",
    "eumetopias jubatus",
    "blue ring angelfish",
    "eretmochelys imbricata",
    "christmas tree worm",
    "guitarfish",
    "tonna galea",
    "cypraea erosa",
    "cobia",
    "sculpin",
    "harlequin sweetlip",
    "flatfish",
    "stingray",
)
model = dict(
    type="DDQDETR",
    num_queries=900,  # num_matching_queries
    # ratio of num_dense queries to num_queries
    dense_topk_ratio=1.5,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type="ChannelMapper",
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=4,
    ),
    # encoder class name: DeformableDetrTransformerEncoder
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_levels=4, dropout=0.0
            ),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0,
            ),
        ),
    ),  # 0.1 for DeformDETR
    # decoder class name: DDQTransformerDecoder
    decoder=dict(
        # `num_layers` >= 2, because attention masks of the last
        #   `num_layers` - 1 layers are used for distinct query selection
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0.0
            ),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(
                embed_dims=256, num_levels=4, dropout=0.0
            ),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0,
            ),
        ),  # 0.1 for DeformDETR
        post_norm_cfg=None,
    ),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20  # -0.5 for DeformDETR
    ),  # 10000 for DeformDETR
    bbox_head=dict(
        type="DDQDETRHead",
        num_classes=80,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
    ),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100),
    ),
    dqs_cfg=dict(type="nms", iou_threshold=0.8),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="FocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.0),
            ],
        )
    ),
    test_cfg=dict(max_per_img=300),
)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=_base_.backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="RandomChoiceResize",
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type="PackDetInputs"),
]

work_dir = "/mnt/hdd/davidwong/models/ddq/intra"

dataset_type = "CocoDataset"
data_root = "/mnt/hdd/davidwong/data/marinedet/"

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/intra_class_train_no_negative.json",
        data_prefix=dict(img=""),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        backend_args=_base_.backend_args,
        metainfo=dict(classes=classes),
    )
)

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0002, weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
)

# learning policy
max_epochs = 12
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(type="LinearLR", start_factor=0.0001, by_epoch=False, begin=0, end=2000),
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1,
    ),
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

val_evaluator = dict(
    metric="bbox",
    ann_file=data_root + "annotations/intra_class_val_seen_no_negative.json",
)
test_evaluator = val_evaluator

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/intra_class_val_seen_no_negative.json",
        data_prefix=dict(img=""),
        metainfo=dict(classes=classes),
    )
)
test_dataloader = val_dataloader
