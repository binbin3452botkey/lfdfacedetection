project structure:

```text
lfdfacedetection/
├── main.py
├── ui.py
├── WIDERFACE_LFD_XS.py
├── face_output/
├── facemodel/
│   └── epoch_1000.pth
├── lfd/
│   ├── data_pipeline/
│   ├── data_loader/
│   ├── dataset/
│   ├── pack/
│   ├── sampler/
│   ├── deployment/
│   ├── evaluation/
│   ├── execution/
│   ├── model/
│   │   ├── backbone/
│   │   ├── head/
│   │   ├── losses/
│   │   ├── neck/
│   │   └── utils/
│   └── __init__.py
├── lib/
│   ├── summertts.so
│   └── testtts.py
├── monitor/
└── ttsmodel/
    ├── single_speaker_english_fast.bin
    └── single_speaker_fast.bin
```



comment:

Among them, main.py is the main program. ui.py defines various components of the graphical interface and their positions. 
lfd is the core module, which contains all algorithm implementations and processing workflows.
The weight file epoch_1000.pth is stored in the facemodel directory and used for face detection. 
single_speaker_english_fast.bin and single_speaker_fast.bin are stored in the ttsmodel directory—these are two models for 
text-to-speech (TTS) synthesis, where the former is used for English synthesis and the latter for Chinese synthesis.
summertts.so is a C++ module compilable for Python via pybind. By importing this module, you can use the functions related 
to TTS synthesis. It is placed in the lib directory together with the sample file testtts.py.
The monitor directory is used to store videos captured in monitoring mode, while the face_output directory is used to store 
faces saved when the face-saving function of the application is used.
