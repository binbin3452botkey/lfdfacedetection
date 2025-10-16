project structure:

lfdfacedetection/
├── main.py
├── ui.py
├── WIDERFACE_LFD_XS.py
├── face_output/
│   ├── face_0.jpg
│   └── face_1.jpg
├── facemodel/
│   └── epoch_1000.pth
├── lfd/
│   ├── __init__.py
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── augmentation/
│   │   │   ├── __init__.py
│   │   │   ├── augmentation_pipeline.py
│   │   │   └── new_augmentations.py
│   │   ├── data_loader/
│   │   │   ├── __init__.py
│   │   │   └── data_loader.py
│   │   ├── dataset/
│   │   │   ├── __init__.py
│   │   │   ├── base_parser.py
│   │   │   ├── coco_parser.py
│   │   │   ├── dataset.py
│   │   │   ├── sample.py
│   │   │   ├── tt100k_parser.py
│   │   │   ├── widerface_parser.py
│   │   │   └── utils/
│   │   │       ├── __init__.py
│   │   │       └── turbojpeg.py
│   │   ├── pack/
│   │   │   ├── __init__.py
│   │   │   ├── pack_coco.py
│   │   │   ├── pack_tt100k.py
│   │   │   └── pack_widerface.py
│   │   └── sampler/
│   │       ├── __init__.py
│   │       ├── dataset_sampler.py
│   │       └── region_sampler.py
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   └── tensorrt/
│   │       ├── __init__.py
│   │       ├── build_engine.py
│   │       ├── inference.py
│   │       └── inference_latency_evaluation.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── base_evaluator.py
│   │   ├── coco_evaluator.py
│   │   └── README.md
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── executor.py
│   │   ├── utils.py
│   │   └── hooks/
│   │       ├── __init__.py
│   │       ├── checkpoint_hook.py
│   │       ├── evaluation_hook.py
│   │       ├── hook.py
│   │       ├── logger_hook.py
│   │       ├── lr_scheduler_hook.py
│   │       ├── optimizer_hook.py
│   │       └── speed_hook.py
│   └── model/
│       ├── __init__.py
│       ├── fcos.py
│       ├── lfd.py
│       ├── lfdv2.py
│       ├── backbone/
│       │   ├── __init__.py
│       │   ├── lfd_resnet.py
│       │   ├── resnet.py
│       │   └── pretrained_backbone_weights/
│       │       └── README.md
│       ├── head/
│       │   ├── __init__.py
│       │   ├── fcos_head.py
│       │   └── lfd_head.py
│       ├── losses/
│       │   ├── __init__.py
│       │   ├── bce_with_logits_loss.py
│       │   ├── cross_entropy_loss.py
│       │   ├── focal_loss.py
│       │   ├── gfocal_loss.py
│       │   ├── iou_loss.py
│       │   ├── mse_loss.py
│       │   ├── smooth_l1_loss.py
│       │   └── utils.py
│       ├── neck/
│       │   ├── __init__.py
│       │   ├── fpn.py
│       │   ├── simple_fpn.py
│       │   └── simple_neck.py
│       └── utils/
│           ├── __init__.py
│           └── nms.py
├── lib/
│   ├── summertts.so
│   └── testtts.py
├── monitor/
│   ├── moni_0.mp4
│   └── moni_1.mp4
└── ttsmodel/
    ├── single_speaker_english_fast.bin
    └── single_speaker_fast.bin

comment:
Among them, main.py is the main program. ui.py defines various components of the graphical interface and their positions. lfd is the core module, which contains all algorithm implementations and processing workflows.
The weight file epoch_1000.pth is stored in the facemodel directory and used for face detection. single_speaker_english_fast.bin and single_speaker_fast.bin are stored in the ttsmodel directory—these are two models 
for text-to-speech (TTS) synthesis, where the former is used for English synthesis and the latter for Chinese synthesis.
summertts.so is a C++ module compilable for Python via pybind. By importing this module, you can use the functions related to TTS synthesis. It is placed in the lib directory together with the sample file testtts.py.
The monitor directory is used to store videos captured in monitoring mode, while the face_output directory is used to store faces saved when the face-saving function of the application is used.
