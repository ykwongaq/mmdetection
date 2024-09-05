_base_ = ["../_base_/datasets/coco_detection.py", "../_base_/default_runtime.py"]
classes = (
    "abalone",
    "acanthopagrus schlegelii",
    "adamussium colbecki",
    "aeolid nudibranchs",
    "aequipecten opercularis",
    "african lungfish",
    "african pompano",
    "albacore",
    "alligator gar",
    "alpheus bellulus",
    "amberjack",
    "american lobster",
    "amphiprioninae",
    "anchovy",
    "anemone hermit crab",
    "anemone shrimp",
    "anemonefish",
    "angelfish",
    "angelshark",
    "anglerfish",
    "antarctic toothfish",
    "aptenodytes forsteri",
    "apteronotus albifrons",
    "arbacia punctulata",
    "argopecten irradians",
    "arrow crab",
    "astropyga radiata",
    "atlantic blue crab",
    "atlantic triton",
    "australian lungfish",
    "bamboo shark",
    "banded sea krait",
    "banggai cardinalfish",
    "bare-tailed goatfish",
    "barnacle",
    "barracuda",
    "barramundi cod",
    "barred-fin moray",
    "barrel sponge",
    "barreleye",
    "basket star",
    "basking shark",
    "bass",
    "basslet",
    "batfish",
    "beaked coral fish",
    "bearded scorpionfish",
    "bichir",
    "bicolor cleaner wrasse",
    "bigeye",
    "bigeye tuna",
    "bighead gurnard",
    "bittium reticulatum",
    "black dragonfish",
    "black ghost knifefish",
    "black marlin",
    "black rock shark",
    "black sea bass",
    "black swallower",
    "black-blotched porcupinefish",
    "blackfin tuna",
    "blacknose shark",
    "blackspotted puffer",
    "blacktip grouper",
    "blacktip reef shark",
    "blenny",
    "blue crab",
    "blue land crab",
    "blue marlin",
    "blue ring angelfish",
    "blue tang surgeonfish",
    "blue triggerfish",
    "blue whiting",
    "bluefin trevally",
    "bluefin tuna",
    "bluegill",
    "bluespotted cornetfish",
    "bluespotted ribbontail ray",
    "bluestreak cleaner wrasse",
    "bonito tuna",
    "bowfin",
    "box jellyfish",
    "boxfish",
    "brain coral",
    "brittle star",
    "bubble coral",
    "bubble eye goldfish",
    "bubble tip anemone",
    "bull shark",
    "bullhead",
    "butterfly goldfish",
    "butterflyfish",
    "callinectes sapidus",
    "calliostoma zizyphinum",
    "cancer irroratus",
    "cancer pagurus",
    "carassius auratus",
    "carcharodon carcharias",
    "cardinalfish",
    "cardisoma crab",
    "caridina multidentata",
    "carp",
    "carpet anemone",
    "carpet shark",
    "carybdea marsupialis",
    "catfish",
    "celestial goldfish",
    "cerithium vulgatum",
    "cheeklined wrasse",
    "cheilinus undulatus",
    "chelonia mydas",
    "chelydra serpentina",
    "chicoreus ramosus",
    "chimaera",
    "chionoecetes opilio",
    "chlamys islandica",
    "christmas tree worm",
    "chromis",
    "clam",
    "clark's anemonefish",
    "cleaner shrimp",
    "cleaner wrasse",
    "clown anemonefish",
    "clown frogfish",
    "clown triggerfish",
    "clown wrasse",
    "clownfish",
    "cobia",
    "coconut crab",
    "cod",
    "coelacanth",
    "comb jelly",
    "conch",
    "concholepa",
    "conge",
    "conger eel",
    "conomurex luhuanus",
    "conus marmoreus",
    "cookiecutter shark",
    "copperband butterflyfish",
    "coral",
    "coral banded shrimp",
    "coral grouper",
    "coral reef",
    "coral reef shrimp",
    "cornetfish",
    "corydoras aeneus",
    "coryphaena hippurus",
    "cownose ray",
    "crab",
    "crappie",
    "crassostrea gigas",
    "crassostrea sikamea",
    "croaker",
    "crowntail betta",
    "crystal jellyfish",
    "cusk",
    "cymbiola vespertilio",
    "cypraea annulus",
    "cypraea aurantium",
    "cypraea caurica",
    "cypraea cribraria",
    "cypraea erosa",
    "cypraea histrio",
    "cypraea mappa",
    "cypraea moneta",
    "cypraea onyx",
    "cypraea pantherina",
    "cypraea talpa",
    "cypraea tigris",
    "cypraea vitellus",
    "cypraea zebra",
    "cyprinus carpio",
    "damsel fish",
    "decorator crab",
    "deep-sea dragonfish",
    "delta tail betta",
    "dendronephthya",
    "dermochelys coriacea",
    "devil ray",
    "devil scorpionfish",
    "diadema antillarum",
    "dogfish",
    "dogfish shark",
    "dolphin",
    "dottyback",
    "dragonet",
    "dragonfish",
    "drumfish",
    "dungeness crab",
    "durban hinge-beak shrimp",
    "eagle ray",
    "echidna nebulosa",
    "echinometra mathaei",
    "echinus esculentus",
    "eel",
    "eggs",
    "electric eel",
    "electric ray",
    "elephant ear sponge",
    "elephant fish",
    "ember parrotfish",
    "emperor angelfish",
    "emperor fish",
    "emys orbicularis",
    "encrusting sponge",
    "epinephelus marginatus",
    "epitonium clathrus",
    "epitonium scalare",
    "eretmochelys imbricata",
    "eriocheir sinensis",
    "esox lucius",
    "euchelus asper",
    "eudyptes chrysolophus",
    "eudyptes robustus",
    "eudyptes schlegeli",
    "eudyptula minor",
    "european green crab",
    "fan worm",
    "fangtooth fish",
    "feather duster worm",
    "fiddler crab",
    "filefish",
    "fimbriated moray",
    "fire coral",
    "firefish",
    "fireworm",
    "fish",
    "fish storm",
    "flame scallop",
    "flashlight fish",
    "flat needlefish",
    "flatfish",
    "flathead",
    "flounder",
    "freshwater mussel",
    "frilled shark",
    "frogfish",
    "fusinus colus",
    "gadus morhua",
    "giant manta ray",
    "giant moray",
    "giant trevally",
    "goatfish",
    "goblin shark",
    "goby",
    "goldfish",
    "gorgonian coral",
    "great barracuda",
    "great white shark",
    "green sea turtle",
    "green shore crab",
    "green turtle",
    "grey reef shark",
    "grouper",
    "gudgeon",
    "guillemot",
    "guitarfish",
    "gulper eel",
    "gulper shark",
    "gurnard",
    "gymnothorax javanicus",
    "hake",
    "halfmoon",
    "halibut",
    "halocaridina rubra",
    "hammerhead shark",
    "handfish",
    "harlequin shrimp",
    "harlequin sweetlip",
    "harpiosquilla harpax",
    "hatchetfish",
    "hawkfish",
    "heliocidaris tuberculata",
    "hermit crab",
    "heteroconger hassi",
    "hippocampus abdominalis",
    "hippocampus whitei",
    "honeycomb morays",
    "horn-nosed boxfish",
    "horned bannerfish",
    "horrid elbow crab",
    "humphead wrasse",
    "hydrocynus vittatus",
    "ictalurus punctatus",
    "indian ocean oriental sweetlips",
    "indo-pacific sergeant",
    "jellyfish",
    "john dory",
    "kingfish",
    "koi fish",
    "lampfish",
    "lamprey",
    "lantern fish",
    "lantern shark",
    "largemouth bass",
    "leafy sea dragon",
    "leatherback sea turtle",
    "lemon shark",
    "leopard shark",
    "lionfish",
    "little tunny",
    "lizardfish",
    "lobster",
    "loggerhead sea turtle",
    "long-finned pike",
    "longfin eel",
    "longlegged spiny lobster",
    "longnose batfish",
    "loosejaw",
    "mackerel shark",
    "macrobrachium rosenbergii",
    "magnificent sea anemone",
    "mahi-mahi",
    "maja squinado",
    "mako shark",
    "malabar grouper",
    "manta ray",
    "mantis shrimp",
    "many spotted sweetlip",
    "map puffer",
    "marlin",
    "marsh crab",
    "maurolicus",
    "megadyptes antipodes",
    "megamouth shark",
    "menhaden",
    "meyer's butterflyfish",
    "mobula ray",
    "mole crab",
    "moorish idol",
    "moray eel",
    "moringa eel",
    "mud crab",
    "mugil cephalus",
    "mushroom leather coral",
    "mussel",
    "napolean wrasse",
    "needlefish",
    "nudibranch",
    "nurse shark",
    "oarfish",
    "ocean sunfish",
    "ocellaris clownfish",
    "ocellate phyllidia",
    "odontodactylus scyllarus",
    "olindias formosa",
    "oliva porphyria",
    "oncorhynchus kisutch",
    "oncorhynchus mykiss",
    "orca",
    "oreochromis niloticus",
    "ornate spiny lobster",
    "osteoglossum bicirrhosum",
    "ostrea edulis",
    "otter",
    "oyster",
    "pagrus major",
    "painted spiny lobster",
    "palechin morays",
    "palinurus delagoae",
    "panulirus argus",
    "panulirus interruptus",
    "panulirus ornatus",
    "parrotfish",
    "patinopecten yessoensis",
    "peacock mantis shrimp",
    "pecten maximus",
    "pederson cleaner shrimp",
    "pegasus sea moth",
    "pencil sea urchin",
    "perca flavescens",
    "perch",
    "physalia physalis",
    "pike",
    "pilchard",
    "pink anemone fish",
    "pinnate batfish",
    "pipefish",
    "pistol shrimp",
    "pizza anemone",
    "plaice",
    "plankto",
    "planospira scalaris",
    "pleurotomaria",
    "pomacanthus imperator",
    "porcelain crab",
    "porcupinefish",
    "port jackson shark",
    "portuguese man o' war",
    "portunus trituberculatus",
    "potato grouper",
    "powder blue tang fish",
    "predatory sharks",
    "pterois volitans",
    "pufferfish",
    "pumpkinseed sunfish",
    "purple shore crab",
    "pygmy seahorse",
    "pygoscelis adeliae",
    "pygoscelis antarcticus",
    "pygoscelis papua",
    "ray",
    "razor clam",
    "razorfish",
    "red king crab",
    "red lionfish",
    "red snapper",
    "red tailed butterfly fish",
    "redtail butterflyfish",
    "regal tang fish",
    "rhinomuraena quaesita",
    "ribbon eel",
    "ribbonfish",
    "sabellidae",
    "saccostrea cucullata",
    "sailfish",
    "salmo salar",
    "salp",
    "salvelinus fontinalis",
    "sawshark",
    "scaleless dragonfish",
    "scallop",
    "scorpionfish",
    "sculpin",
    "sea anemone",
    "sea bass",
    "sea fan",
    "sea grape",
    "sea hare",
    "sea slug",
    "sea snail",
    "sea snake",
    "sea turtle",
    "sea urchin",
    "sea whip",
    "sea anemone",
    "seadragon",
    "seagrass",
    "seagull",
    "seahorse",
    "searobin",
    "seashell",
    "seaweed",
    "sergeant major",
    "serranidae",
    "sevengill shark",
    "shark",
    "sharksucker",
    "shrimp",
    "shrimpfish",
    "shrimp goby",
    "sixgill shark",
    "skipjack tuna",
    "skunk clownfish",
    "sleeper goby",
    "sleeper shark",
    "slipper lobster",
    "small swimming crab",
    "smallmouth bass",
    "smooth shore crab",
    "snake eel",
    "snake pipefish",
    "snake-like morays",
    "snakehead",
    "snapper",
    "snapping turtle",
    "snow crab",
    "soft coral",
    "soldierfish",
    "sole",
    "south american lungfish",
    "spaghetti eel",
    "spheniscus magellanicus",
    "spider crab",
    "spiny eel",
    "spiny lobster",
    "sponge",
    "sponge crab",
    "spot-fin porcupinefish",
    "spotted eagle rays",
    "spotted sharpnose puffer",
    "squat lobster",
    "squilla empusa",
    "squirrelfish",
    "staghorn coral",
    "stargazer fish",
    "starry flounder",
    "stingray",
    "stone crab",
    "stonefish",
    "striped bass",
    "striped marlin",
    "striped surgeonfish",
    "strombus gigas",
    "strongylocentrotus franciscanus",
    "sunfish",
    "surgeonfish",
    "swell shark",
    "symbiotic shrimp",
    "table coral",
    "teira batfish",
    "testudo hermanni",
    "thenus orientalis",
    "thresher shark",
    "thunnus thynnus",
    "tiger shark",
    "tiger tail seahorse",
    "tilefish",
    "titan triggerfish",
    "tonna galea",
    "trachemys scripta elegans",
    "triggerfish",
    "triplefin",
    "trumpetfish",
    "tube anemone",
    "tube sponge",
    "tube-dwelling anemone",
    "tuna",
    "turbinella pyrum",
    "turbo marmoratus",
    "turbot",
    "turritella communis",
    "turtle",
    "uca pugnax",
    "unicornfish",
    "variable coral crab",
    "viperfish",
    "voluta musica",
    "walleye",
    "whale shark",
    "whelk",
    "whip coral",
    "white marlin",
    "white shark",
    "whitemouth morays",
    "whitespotted boxfish",
    "whitetip reef shark",
    "whiting",
    "winter flounder",
    "wobbegong shark",
    "wolf eel",
    "wrasse",
    "xenophora pallidula",
    "yellow boxfish",
    "yellow mask angelfish",
    "yellow mask surgeonfish",
    "yellow tang fish",
    "yellow-edged moray",
    "yellowfin tuna",
    "yellowmargin triggerfish",
    "zebra crab",
    "zebra moray",
    "zebra shark",
    "zidona dufresnei",
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
        num_classes=555,
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
    ),
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
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=5)

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