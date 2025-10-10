# -*- coding: utf-8 -*-
import summertts
summertts.init_tts("../ttsmodel/single_speaker_fast.bin")
summertts.play_voice("用户表面问题是声卡没声音，但深层需求可能是尽快确定问题根源，避免浪费时间。现在声卡在其他设备上也不工作，得考虑硬件故障。不过不能立马下结论，因为还有其他可能性，比如兼容性或设置问题。",1.0)
summertts.release_tts()
